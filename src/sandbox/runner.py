"""
Sandbox runner: applies a unified diff to the base files, writes generated tests,
then executes both inside an ephemeral Docker container.

Security constraints applied to every container:
  --network none     no outbound network access
  --memory 256m      capped RAM
  --cpus 0.5         capped CPU
  --read-only        immutable container filesystem (writable /tmp via tmpfs)
  --rm               auto-removed on exit
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.agents.tools import BASE_FILES
from src.state import AgentMessage, AgentState, TestResult
from src.utils.compress import strip_ansi, strip_pytest_noise

# Packages pre-installed in the sandbox. Phase 3: derive from the target repo's
# requirements.txt or pyproject.toml instead of hardcoding.
_SANDBOX_PACKAGES = "pytest fastapi httpx"

_DOCKER_IMAGE = "python:3.12-slim"

_MAX_RUNTIME_SECONDS = 60


def _write_base_files(tmp: Path, base_files: dict[str, str]) -> None:
    for rel_path, content in base_files.items():
        dest = tmp / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)


def _apply_diff(tmp: Path, diff: str) -> tuple[bool, str]:
    """Run `patch -p1` in tmp. Returns (success, stderr)."""
    if not diff or diff.startswith("(generator"):
        return False, "No diff to apply."

    diff_file = tmp / "_patch.diff"
    diff_file.write_text(diff)

    result = subprocess.run(
        ["patch", "-p1", "--input", str(diff_file)],
        cwd=tmp,
        capture_output=True,
        text=True,
    )
    diff_file.unlink(missing_ok=True)
    return result.returncode == 0, result.stderr


def _run_docker(tmp: Path, test_code: str) -> TestResult:
    test_file = tmp / "test_generated.py"
    test_file.write_text(test_code)

    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--memory", "256m",
        "--cpus", "0.5",
        "--read-only",
        "--tmpfs", "/tmp",
        "-v", f"{tmp}:/workspace:ro",
        "-w", "/workspace",
        _DOCKER_IMAGE,
        "sh", "-c",
        f"pip install {_SANDBOX_PACKAGES} -q 2>&1 && pytest test_generated.py -v 2>&1",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_MAX_RUNTIME_SECONDS,
        )
        return TestResult(
            exit_code=proc.returncode,
            stdout=strip_pytest_noise(proc.stdout),
            stderr=strip_ansi(proc.stderr),
        )
    except subprocess.TimeoutExpired:
        return TestResult(
            exit_code=1,
            stdout="",
            stderr=f"Sandbox timed out after {_MAX_RUNTIME_SECONDS}s.",
        )
    except FileNotFoundError:
        return TestResult(
            exit_code=1,
            stdout="",
            stderr=(
                "Docker not found. Install Docker Desktop (or docker-ce) "
                "and ensure the daemon is running."
            ),
        )


def get_patched_files(diff: str, base_files: dict[str, str]) -> tuple[dict[str, str], str]:
    """
    Apply a unified diff to base_files and return the resulting file contents.
    Used by the git agent to build the commit payload without running tests.
    Returns (patched_files_dict, error_message). On failure, error_message is non-empty.
    """
    with tempfile.TemporaryDirectory(prefix="mac_patch_") as tmp_str:
        tmp = Path(tmp_str)
        _write_base_files(tmp, base_files)
        patched, err = _apply_diff(tmp, diff)
        if not patched:
            return {}, err
        result: dict[str, str] = {}
        for f in tmp.rglob("*"):
            if f.is_file() and not f.name.startswith("_"):
                rel = str(f.relative_to(tmp))
                result[rel] = f.read_text(encoding="utf-8", errors="replace")
        return result, ""


def docker_runner_node(state: AgentState) -> dict:
    with tempfile.TemporaryDirectory(prefix="mac_sandbox_") as tmp_str:
        tmp = Path(tmp_str)

        _write_base_files(tmp, BASE_FILES)

        patched, patch_err = _apply_diff(tmp, state["current_diff"] or "")
        if not patched:
            result = TestResult(
                exit_code=1,
                stdout="",
                stderr=f"patch failed: {patch_err}",
            )
        else:
            result = _run_docker(tmp, state["generated_tests"] or "")

    verdict = "PASSED" if result.passed else f"FAILED (exit {result.exit_code})"
    next_status = "approved" if result.passed else "debugging"

    return {
        "test_result": result,
        "status": next_status,
        "messages": [
            AgentMessage(
                role="test_generator",
                content=f"Sandbox {verdict}. "
                        + (result.stdout.splitlines()[-1] if result.stdout else result.stderr[:120]),
            )
        ],
    }
