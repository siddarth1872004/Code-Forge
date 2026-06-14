from dotenv import load_dotenv

from src import telemetry
from src.agents._client import get_client
from src.state import AgentMessage, AgentState

load_dotenv()

_SYSTEM = """\
You are a debugging specialist. Given a diff and the failing test output from a sandbox run,
identify the root cause and describe exactly what the generator must change.

Be specific: name the file, function, and line where the fix should go.
Do not rewrite the diff yourself — produce a clear, actionable fix description only.\
"""

_TOOLS = [
    {
        "name": "create_fix_request",
        "description": "Describe the exact fix the generator must make to pass the tests.",
        "parameters": {
            "type": "object",
            "properties": {
                "root_cause": {
                    "type": "string",
                    "description": "One sentence: what is actually wrong.",
                },
                "fix_description": {
                    "type": "string",
                    "description": "Specific, actionable description of the change needed.",
                },
                "relevant_location": {
                    "type": "string",
                    "description": "File and function/line where the fix should be applied.",
                },
            },
            "required": ["root_cause", "fix_description"],
        },
    }
]


def _build_prompt(state: AgentState) -> str:
    result = state["test_result"]
    combined_output = "\n".join(filter(None, [result.stdout, result.stderr]))
    return "\n".join([
        f"Feature request: {state['feature_request']}",
        "",
        "Diff under test:",
        "```diff",
        state["current_diff"] or "(empty)",
        "```",
        "",
        "Test code that was run:",
        "```python",
        state["generated_tests"] or "(none)",
        "```",
        "",
        f"Sandbox exit code: {result.exit_code}",
        "Test output:",
        "```",
        combined_output[:3000],
        "```",
        "",
        "Identify the root cause and describe the exact fix.",
    ])


def debugger_node(state: AgentState) -> dict:
    response = get_client().chat(
        system=_SYSTEM,
        history=[{"role": "user", "content": _build_prompt(state)}],
        tools=_TOOLS,
        max_tokens=1024,
        force_tool="create_fix_request",
    )

    telemetry.record_tokens(response.usage_input, response.usage_output)
    raw = response.tool_calls[0].input

    location = raw.get("relevant_location", "")
    feedback = f"Root cause: {raw['root_cause']}\n\nFix: {raw['fix_description']}"
    if location:
        feedback += f"\n\nLocation: {location}"

    return {
        "debug_feedback": feedback,
        "debug_count": state["debug_count"] + 1,
        "status": "generating",
        "messages": [AgentMessage(role="debugger", content=raw["root_cause"])],
    }
