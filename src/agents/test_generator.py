from dotenv import load_dotenv

from src import telemetry
from src.agents._client import get_client
from src.state import AgentMessage, AgentState

load_dotenv()

_MAX_DIFF_CHARS = 6000

_SYSTEM = """\
You are a senior engineer writing pytest tests to validate a feature implementation.

Context on the sandbox environment:
- Tests run from the repo root inside a Docker container.
- Available packages: pytest, fastapi, httpx.
- Source files are importable from the repo root (e.g. `from src.main import app`).
- Use `fastapi.testclient.TestClient` for HTTP endpoint tests.
- Do not use fixtures that require a running server — TestClient handles that.

Write the minimal set of tests that prove the feature works correctly.
Cover the happy path and at least one edge case.\
"""

_TOOLS = [
    {
        "name": "create_tests",
        "description": "Output the complete pytest test file as a single string.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete, runnable pytest file content.",
                },
                "rationale": {
                    "type": "string",
                    "description": "One sentence explaining what the tests verify.",
                },
            },
            "required": ["code"],
        },
    }
]


def _build_prompt(state: AgentState) -> str:
    plan = state["plan"]
    diff = state["current_diff"] or "(empty)"
    if len(diff) > _MAX_DIFF_CHARS:
        diff = diff[:_MAX_DIFF_CHARS] + f"\n... [trimmed to {_MAX_DIFF_CHARS} chars]"
    return "\n".join([
        f"Feature request: {state['feature_request']}",
        "",
        "Implemented diff:",
        "```diff",
        diff,
        "```",
        "",
        f"Files changed: {', '.join(plan.files_to_change)}",
        "Functions added: " + (
            ", ".join(f"{f.name}{f.signature}" for f in plan.functions_to_add) or "none"
        ),
        "",
        "Write pytest tests that validate this implementation.",
    ])


def test_generator_node(state: AgentState) -> dict:
    response = get_client().chat(
        system=_SYSTEM,
        history=[{"role": "user", "content": _build_prompt(state)}],
        tools=_TOOLS,
        max_tokens=4096,
        force_tool="create_tests",
    )

    telemetry.record_tokens(response.usage_input, response.usage_output)
    raw = response.tool_calls[0].input
    code = raw["code"]
    rationale = raw.get("rationale", "Tests generated.")

    return {
        "generated_tests": code,
        "status": "testing",
        "messages": [AgentMessage(role="test_generator", content=rationale)],
    }
