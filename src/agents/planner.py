from dotenv import load_dotenv

from src import telemetry
from src.agents._client import get_client
from src.state import AgentMessage, AgentState, Plan

load_dotenv()

_SYSTEM = """\
You are a senior software architect. Given a feature request, produce a minimal, concrete implementation plan.

Focus on:
- The smallest set of files that need to change
- Clear function signatures for any new code
- Existing patterns or constraints the generator must respect

Do not write code — only plan.\
"""

_TOOLS = [
    {
        "name": "create_plan",
        "description": "Output a structured implementation plan for the feature request.",
        "parameters": {
            "type": "object",
            "properties": {
                "files_to_change": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File paths to modify or create, relative to repo root.",
                },
                "functions_to_add": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "signature": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["name", "signature", "description"],
                    },
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Style, pattern, or architectural constraints the generator must follow.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "One paragraph explaining the approach.",
                },
            },
            "required": ["files_to_change", "reasoning"],
        },
    }
]


def planner_node(state: AgentState) -> dict:
    response = get_client().chat(
        system=_SYSTEM,
        history=[{"role": "user", "content": state["feature_request"]}],
        tools=_TOOLS,
        max_tokens=1024,
        force_tool="create_plan",
    )

    telemetry.record_tokens(response.usage_input, response.usage_output)
    tc = response.tool_calls[0]
    plan = Plan.model_validate(tc.input)

    summary = (
        f"Plan: change {len(plan.files_to_change)} file(s), "
        f"add {len(plan.functions_to_add)} function(s). {plan.reasoning}"
    )

    return {
        "plan": plan,
        "status": "generating",
        "messages": [AgentMessage(role="planner", content=summary)],
    }
