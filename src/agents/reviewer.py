from dotenv import load_dotenv

from src import telemetry
from src.agents._client import get_client
from src.state import AgentMessage, AgentState, Issue, ReviewFeedback

load_dotenv()

_MAX_DIFF_CHARS = 8000

_SYSTEM = """\
You are a senior code reviewer. Review a proposed diff against the implementation plan.

Classify issues as:
- bug: incorrect logic, missing error handling, broken behaviour
- security: injection, auth bypass, data exposure, unsafe deserialization
- style: naming, formatting, convention violations

Block approval only on bugs and security issues. Style findings are informational.
If there are no bugs or security issues, set approved=true even if style issues exist.\
"""

_TOOLS = [
    {
        "name": "submit_review",
        "description": "Submit a structured code review decision.",
        "parameters": {
            "type": "object",
            "properties": {
                "approved": {"type": "boolean"},
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "severity": {"type": "string", "enum": ["style", "security", "bug"]},
                            "description": {"type": "string"},
                            "location": {"type": "string"},
                        },
                        "required": ["severity", "description"],
                    },
                },
                "summary": {"type": "string", "description": "One sentence verdict."},
            },
            "required": ["approved", "summary"],
        },
    }
]


def _build_prompt(state: AgentState) -> str:
    plan = state["plan"]
    diff = state["current_diff"] or "(empty)"
    if len(diff) > _MAX_DIFF_CHARS:
        diff = diff[:_MAX_DIFF_CHARS] + f"\n... [diff trimmed to {_MAX_DIFF_CHARS} chars]"
    lines = [
        f"Feature request: {state['feature_request']}",
        "",
        "Plan:",
        f"  Files: {', '.join(plan.files_to_change)}",
    ]
    if plan.constraints:
        lines.append(f"  Constraints: {'; '.join(plan.constraints)}")
    lines += ["", "Proposed diff:", "```diff", diff, "```"]
    return "\n".join(lines)


def reviewer_node(state: AgentState) -> dict:
    response = get_client().chat(
        system=_SYSTEM,
        history=[{"role": "user", "content": _build_prompt(state)}],
        tools=_TOOLS,
        max_tokens=1024,
        force_tool="submit_review",
    )

    telemetry.record_tokens(response.usage_input, response.usage_output)
    raw = response.tool_calls[0].input

    feedback = ReviewFeedback(
        approved=raw["approved"],
        summary=raw["summary"],
        issues=[Issue.model_validate(i) for i in raw.get("issues", [])],
    )
    blockers = [i for i in feedback.issues if i.severity in ("bug", "security")]
    verdict = "APPROVED" if feedback.approved else f"REJECTED ({len(blockers)} blocker(s))"

    return {
        "review_feedback": feedback,
        "iteration_count": state["iteration_count"] + 1,
        "status": "approved" if feedback.approved else "generating",
        "messages": [AgentMessage(role="reviewer", content=f"{verdict}: {feedback.summary}")],
    }
