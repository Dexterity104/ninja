#!/usr/bin/env python3
"""
Multi-file SWE coding agent for the tau subnet.

Validator contract:
    The validator imports this file and calls:

        solve(
            repo_path="/tmp/task_repo",
            issue="Fix the bug...",
            model="validator-managed-model",
            api_base="http://validator-proxy/v1",
            api_key="per-run-proxy-token"
        )

    It returns a dict with patch, logs, steps, cost, and success.

Layout:
    agent.py             validator-owned contract + thin solve() wiring
    agent/prompts.py     system and task prompt templates
    agent/model.py       stdlib OpenAI-compatible chat client with retries
    agent/environment.py fresh-subshell bash executor
    agent/agent_loop.py  the query -> act -> observe step loop
    agent/repo_diff.py   harness-compatible patch collection
    agent/criteria.py    acceptance checklist from issue bullets
    agent/guards.py      patch quality signals for repair gate

All inference uses only the validator-provided api_base/api_key; there are no
third-party dependencies and no sampling overrides (the validator proxy owns
sampling).
"""

from __future__ import annotations

import os
import json
import subprocess
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from agent.agent_loop import AgentRunConfig, run_agent_loop
from agent.prompts import build_task_prompt
from agent.repo_diff import collect_repo_patch
from agent.criteria import extract_criteria, format_checklist
from agent.guards import extended_repair_reason, patch_acceptable

# -----------------------------
# Config
# -----------------------------

DEFAULT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "50"))
# Allow a single command enough time to run a small reproduction or assertion
# that demonstrates the fix is correct. Still far under the per-round wall
# budget so the loop finishes and reports its own patch.
DEFAULT_COMMAND_TIMEOUT = int(os.environ.get("AGENT_COMMAND_TIMEOUT", "40"))

# VALIDATOR CONTRACT: These defaults are only fallbacks for local testing and
# validator wiring. During real validation the validator passes model, api_base,
# and api_key into solve(). Keep this code compatible with that path.
DEFAULT_MODEL = os.environ.get("AGENT_MODEL") or os.environ.get("NINJA_MODEL", "")
DEFAULT_API_BASE = (
    os.environ.get("AGENT_API_BASE")
    or os.environ.get("NINJA_INFERENCE_BASE_URL")
    or os.environ.get("OPENAI_BASE_URL", "")
)
DEFAULT_API_KEY = (
    os.environ.get("AGENT_API_KEY")
    or os.environ.get("NINJA_INFERENCE_API_KEY")
    or os.environ.get("OPENAI_API_KEY", "")
)
DEFAULT_MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "8192"))

MAX_OBSERVATION_CHARS = int(os.environ.get("AGENT_MAX_OBSERVATION_CHARS", "16000"))
MAX_TOTAL_LOG_CHARS = int(os.environ.get("AGENT_MAX_TOTAL_LOG_CHARS", "260000"))

# Stay under the validator's per-round budget so the loop can finish gracefully
# and report its own patch instead of relying on the kill path. The validator
# now exports its real per-round budget as TAU_AGENT_TIMEOUT_SECONDS; honor it
# (leaving a margin for diff collection) so a looser budget actually lets the
# agent keep working. Falls back to the conservative 280s when unset.
def _wall_clock_limit_seconds() -> float:
    budget = os.environ.get("TAU_AGENT_TIMEOUT_SECONDS")
    if budget:
        try:
            return max(60.0, float(int(budget)) - 20.0)
        except ValueError:
            pass
    return 280.0


WALL_CLOCK_LIMIT_SECONDS = _wall_clock_limit_seconds()

# Headroom kept before the wall limit so a repair pass leaves time for the
# final diff collection instead of being killed mid-write.
WALL_CLOCK_RESERVE_SECONDS = 10.0


def _normalize_api_base(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")]
    if base.endswith("/v1"):
        return base
    return base + "/v1"


def _resolve_inference_config(
    model: Optional[str],
    api_base: Optional[str],
    api_key: Optional[str],
) -> Tuple[str, str, str]:
    model_name = (model or DEFAULT_MODEL).strip()
    base = (api_base or DEFAULT_API_BASE).strip()
    key = (api_key if api_key is not None else DEFAULT_API_KEY).strip()

    if not model_name:
        raise ValueError("model is required; validators must pass the centrally managed model id")
    if not base:
        raise ValueError("api_base is required; validators must pass the managed inference proxy URL")
    if not key:
        raise ValueError("api_key is required; validators must pass the per-run proxy token")

    return model_name, _normalize_api_base(base), key


def build_initial_user_prompt(issue: str, repo_summary: str, preloaded_context: str = "") -> str:
    checklist = format_checklist(extract_criteria(issue))
    base = build_task_prompt(task_text=issue, repo_summary=repo_summary, preloaded_context=preloaded_context)
    return base + checklist if checklist else base


# Minimum wall-clock headroom (seconds) needed to attempt a repair pass; below
# this we keep the first patch rather than start work we cannot finish.
VERIFY_REPAIR_MIN_BUDGET_SECONDS = 45.0
VERIFY_REPAIR_MAX_STEPS = 14


def _changed_py_files(patch_text: str) -> list:
    """Python files touched by the patch (parsed from its `+++ b/` headers)."""
    paths = []
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            path = line[len("+++ b/"):].strip()
            if path.endswith(".py") and path not in paths:
                paths.append(path)
    return paths


def _py_syntax_errors(repo_dir: str, patch_text: str) -> list:
    """Changed .py files whose current on-disk content does not parse."""
    broken = []
    for rel in _changed_py_files(patch_text):
        full = os.path.join(repo_dir, rel)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as handle:
                source = handle.read()
        except OSError:
            continue
        try:
            compile(source, rel, "exec")
        except SyntaxError as exc:
            broken.append(f"{rel}: line {exc.lineno}: {exc.msg}")
        except (ValueError, TypeError):
            broken.append(f"{rel}: could not be parsed")
    return broken


def _changed_source_files(patch_text: str, exts: tuple) -> list:
    """Files with the given extensions touched by the patch (`+++ b/` headers)."""
    paths = []
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            path = line[len("+++ b/"):].strip()
            if path.endswith(exts) and path not in paths:
                paths.append(path)
    return paths


def _run_check(cmd: list, cwd: str) -> Optional[str]:
    """Run an external syntax checker. Return a short error string only on a
    CONFIRMED failure; return None if it passes OR the tool is unavailable, so a
    missing tool never produces a false repair trigger."""
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=20)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if proc.returncode == 0:
        return None
    msg = (proc.stderr or proc.stdout or "").strip()
    return (msg.splitlines()[0][:200] if msg else "failed syntax check")


def _syntax_errors(repo_dir: str, patch_text: str) -> list:
    """Changed files that are definitely unparseable -- a POLYGLOT extension of
    the base king's Python-only check (its blind spot: it ships broken
    non-Python patches unrepaired). Every checker is conservative: a missing
    tool or any ambiguity yields nothing, so repair only fires on a real break.
    The repair adopt-gate re-runs this, so even a false positive can never
    worsen the kept patch -- worst case is a wasted repair pass."""
    broken = []
    # Python -- stdlib compile (identical to the base agent).
    for rel in _changed_source_files(patch_text, (".py",)):
        full = os.path.join(repo_dir, rel)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                source = fh.read()
        except OSError:
            continue
        try:
            compile(source, rel, "exec")
        except SyntaxError as exc:
            broken.append(f"{rel}: line {exc.lineno}: {exc.msg}")
        except (ValueError, TypeError):
            broken.append(f"{rel}: could not be parsed")
    # JSON -- stdlib, always available, zero false positives.
    for rel in _changed_source_files(patch_text, (".json",)):
        full = os.path.join(repo_dir, rel)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except OSError:
            continue
        try:
            json.loads(content)
        except ValueError as exc:
            broken.append(f"{rel}: invalid JSON: {str(exc)[:120]}")
    # Plain JS -- `node --check` parses .js/.mjs/.cjs (skip .jsx/.ts; node would
    # false-flag JSX/TS syntax). Skips silently when node is absent.
    for rel in _changed_source_files(patch_text, (".js", ".mjs", ".cjs")):
        err = _run_check(["node", "--check", rel], repo_dir)
        if err:
            broken.append(f"{rel}: {err}")
    # Go -- `gofmt -e` parses Go. Skips silently when gofmt is absent.
    for rel in _changed_source_files(patch_text, (".go",)):
        err = _run_check(["gofmt", "-e", rel], repo_dir)
        if err:
            broken.append(f"{rel}: {err}")
    return broken


def _all_changed_files(patch_text: str) -> list:
    """Every file the patch touches (`+++ b/` headers), excluding /dev/null."""
    out = []
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            p = line[len("+++ b/"):].strip()
            if p and p != "/dev/null" and p not in out:
                out.append(p)
    return out


def _is_test_path(path: str) -> bool:
    p = path.lower()
    base = p.rsplit("/", 1)[-1]
    if any(seg in ("test", "tests", "spec", "specs", "__tests__") for seg in p.split("/")[:-1]):
        return True
    if base.endswith(".py") and (base.startswith("test_") or base.endswith("_test.py") or base.startswith("test")):
        return True
    if ".test." in base or ".spec." in base or base.endswith("_spec.rb") or base.endswith("_test.go"):
        return True
    return False


def _source_files(patch_text: str) -> set:
    """Non-test files the patch changes -- the actual fix surface."""
    return {p for p in _all_changed_files(patch_text) if not _is_test_path(p)}


def _python_test_outcome(repo_dir: str, patch_text: str) -> str:
    """'none' (no python test added), 'pass', 'fail' (a definitive pytest exit-1
    failure), or 'unknown'. Conservative + time-bounded: runs ONLY the first
    added python test, and treats anything ambiguous (collection/import/usage
    error, no pytest) as 'unknown' so it never falsely declares a fix wrong."""
    tests = [p for p in _all_changed_files(patch_text)
             if _is_test_path(p) and p.endswith(".py")
             and os.path.isfile(os.path.join(repo_dir, p))]
    if not tests:
        return "none"
    rel = tests[0]
    for exe in ("python", "python3"):
        try:
            proc = subprocess.run(
                [exe, "-m", "pytest", rel, "-x", "-q", "-p", "no:cacheprovider"],
                cwd=repo_dir, capture_output=True, text=True, timeout=25,
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0:
            return "pass"
        if proc.returncode == 1:
            return "fail"
        return "unknown"  # 2/3/4/5 = collection/usage/no-tests -> ambiguous
    return "unknown"


def _repair_reason(
    repo_dir: str,
    patch_text: str,
    issue_text: str = "",
    check_tests: bool = True,
):
    """(kind, message) when the first patch should be repaired, else None."""
    if not (patch_text or "").strip():
        return ("empty", "the current change set is empty; no fix was produced yet")
    broken = _syntax_errors(repo_dir, patch_text)
    if broken:
        return ("syntax", "the edited files contain syntax errors that must be fixed:\n- " + "\n- ".join(broken[:8]))
    quality = extended_repair_reason(issue_text, patch_text)
    if quality:
        return ("quality", quality)
    if check_tests:
        outcome = _python_test_outcome(repo_dir, patch_text)
        if outcome == "fail":
            return ("test_fail", "your own regression test currently FAILS, so the fix is wrong or incomplete; correct the fix until that test passes (never weaken the test).")
        if outcome == "none" and _source_files(patch_text):
            return ("no_test", "the fix changes source but includes no test proving it works; ADD one focused regression test that fails on the original bug and passes with your fix, and KEEP the existing source fix in place.")
    return None


def _build_repair_task(issue_text: str, reason: str) -> str:
    return (
        "A previous attempt to solve the task below left the repository in an "
        "incomplete or broken state. " + reason + "\n\n"
        "Inspect the current state of the repository, then finish and correct "
        "the change so it fully and correctly solves the task. Re-read each "
        "edited region to confirm it is syntactically valid before submitting.\n\n"
        "Original task:\n" + issue_text
    )


def solve(
    repo_path: str,
    issue: str,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        model_name, base_url, proxy_token = _resolve_inference_config(model, api_base, api_key)
        run_config = AgentRunConfig(
            repo_dir=repo_path,
            model_name=model_name,
            base_url=base_url,
            auth_token=proxy_token,
            max_steps=max_steps,
            command_timeout=command_timeout,
            max_tokens=max_tokens,
            max_observation_chars=MAX_OBSERVATION_CHARS,
            max_log_chars=MAX_TOTAL_LOG_CHARS,
            wall_clock_limit=WALL_CLOCK_LIMIT_SECONDS,
        )
        outcome = run_agent_loop(
            config=run_config,
            task=build_initial_user_prompt(issue, "", ""),
        )

        # Repair pass: empty/broken/quality-guard/test-gated patches get one retry.
        repair_note = ""
        try:
            remaining = WALL_CLOCK_LIMIT_SECONDS - (time.monotonic() - started)
            # Behavioral probes run a test, so only run them when there is budget
            # for a repair afterwards -- never spend a round we cannot improve.
            can_repair = remaining >= VERIFY_REPAIR_MIN_BUDGET_SECONDS
            reason = _repair_reason(
                repo_path, outcome.patch, issue_text=issue, check_tests=can_repair
            )
            if reason is not None and can_repair:
                kind, message = reason
                orig_sources = _source_files(outcome.patch)
                repair_config = AgentRunConfig(
                    repo_dir=repo_path,
                    model_name=model_name,
                    base_url=base_url,
                    auth_token=proxy_token,
                    max_steps=min(max_steps, VERIFY_REPAIR_MAX_STEPS),
                    command_timeout=command_timeout,
                    max_tokens=max_tokens,
                    max_observation_chars=MAX_OBSERVATION_CHARS,
                    max_log_chars=MAX_TOTAL_LOG_CHARS,
                    wall_clock_limit=remaining - WALL_CLOCK_RESERVE_SECONDS,
                )
                repaired = run_agent_loop(
                    config=repair_config,
                    task=build_initial_user_prompt(_build_repair_task(issue, message), "", ""),
                )
                rp = repaired.patch
                # Adopt-gate -- strictly safe: only replace the first patch when the
                # repair is DEMONSTRABLY better, never when it could be worse.
                if rp.strip() and not _syntax_errors(repo_path, rp):
                    rtest = _python_test_outcome(repo_path, rp)
                    if kind in ("empty", "syntax", "test_fail", "quality"):
                        adopt = rtest != "fail" and patch_acceptable(rp)
                    else:  # no_test
                        adopt = (
                            rtest == "pass"
                            and orig_sources.issubset(_source_files(rp))
                            and patch_acceptable(rp)
                        )
                    if adopt:
                        outcome = repaired
                        repair_note = " (repair adopted: %s)" % kind
        except Exception:
            repair_note = " (repair pass skipped after error)"

        elapsed = time.monotonic() - started
        return {
            "patch": outcome.patch,
            "logs": outcome.logs,
            "steps": outcome.steps,
            "cost": outcome.cost,
            "success": outcome.success,
            "message": f"{outcome.exit_status}: {outcome.message} in {elapsed:.1f}s{repair_note}",
        }
    except Exception:
        fallback_patch = collect_repo_patch(repo_path)
        return {
            "patch": fallback_patch,
            "logs": traceback.format_exc()[-8000:],
            "steps": 0,
            "cost": None,
            "success": bool(fallback_patch.strip()),
            "message": "agent crashed; returning the on-disk repository diff",
        }
