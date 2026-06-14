"""
MCP server exposing the multi-agent code generation pipeline as tools.

Wraps the FastAPI server (src/main.py) so any MCP-compatible host (Claude
Desktop, Cursor, etc.) can trigger pipeline runs, poll status, approve/reject
diffs, and retrieve traces without touching the REST API directly.

Usage:
    # Start the FastAPI server first:
    uvicorn src.main:app --port 8000

    # Then run this server (stdio transport for MCP hosts):
    PYTHONPATH=. .venv/bin/python src/mcp_server.py

    # Or via HTTP for debugging:
    PYTHONPATH=. .venv/bin/python src/mcp_server.py --transport http --port 8001
"""

import os
import httpx
import fastmcp

BASE_URL = os.environ.get("PIPELINE_API_URL", "http://localhost:8000")

mcp = fastmcp.FastMCP(
    name="multi-agent-code-generator",
    instructions=(
        "Tools for driving the multi-agent code generation pipeline. "
        "Typical flow: generate_code → poll get_run_status until "
        "'awaiting_approval' → inspect the diff → approve_run or reject_run."
    ),
)


def _api(method: str, path: str, **kwargs) -> dict:
    with httpx.Client(base_url=BASE_URL, timeout=300.0) as client:
        resp = client.request(method, path, **kwargs)
        resp.raise_for_status()
        return resp.json()


@mcp.tool
def generate_code(feature_request: str) -> dict:
    """
    Start a new code generation run for the given feature request.

    Launches the plan → generate → review → test pipeline in the background
    and returns a run_id immediately.  Poll get_run_status with that ID.

    Args:
        feature_request: Natural-language description of the feature to implement.

    Returns:
        {"run_id": "<uuid>"}
    """
    return _api("POST", "/generate", json={"feature_request": feature_request})


@mcp.tool
def get_run_status(run_id: str) -> dict:
    """
    Poll the current state of a pipeline run.

    Key fields in the response:
      - status: one of planning / generating / reviewing / testing / debugging /
                awaiting_approval / pr_created / rejected / failed
      - diff: the proposed code change (unified diff)
      - plan: structured plan (files to change, functions to add)
      - review: reviewer verdict and issues
      - test_result: sandbox exit code and output
      - pr_url: GitHub PR URL (set after approve_run completes)
      - running: True while the background thread is still active

    When status == "awaiting_approval" the diff has passed review and tests —
    call approve_run or reject_run to proceed.

    Args:
        run_id: The UUID returned by generate_code.
    """
    return _api("GET", f"/runs/{run_id}")


@mcp.tool
def approve_run(run_id: str) -> dict:
    """
    Approve a run that is awaiting human sign-off.

    Resumes the pipeline: the git agent creates a branch and opens a GitHub PR.
    Returns the final run state including pr_url.

    Only valid when get_run_status returns status == "awaiting_approval".

    Args:
        run_id: The UUID returned by generate_code.
    """
    return _api("POST", f"/runs/{run_id}/approve")


@mcp.tool
def reject_run(run_id: str) -> dict:
    """
    Reject a run that is awaiting human sign-off.

    Terminates the run without creating a branch or PR.
    Returns the final run state with status == "rejected".

    Only valid when get_run_status returns status == "awaiting_approval".

    Args:
        run_id: The UUID returned by generate_code.
    """
    return _api("POST", f"/runs/{run_id}/reject")


@mcp.tool
def get_run_trace(run_id: str) -> dict:
    """
    Retrieve the full observability trace for a completed run.

    Returns per-node timing, token counts, and estimated cost broken down by
    agent (planner, generator, reviewer, test_generator, docker_runner,
    debugger, git_agent), plus run-level totals.

    Args:
        run_id: The UUID returned by generate_code.
    """
    return _api("GET", f"/runs/{run_id}/trace")


@mcp.tool
def get_stats() -> dict:
    """
    Aggregate metrics across all pipeline runs in the current server session.

    Returns total runs, success/rejection counts, average cost, average
    duration, and token totals.
    """
    return _api("GET", "/stats")


if __name__ == "__main__":
    mcp.run()
