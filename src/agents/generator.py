from dotenv import load_dotenv

from src import telemetry
from src.agents._client import get_client
from src.agents.tools import TOOL_DISPATCH
from src.state import AgentMessage, AgentState
from src.utils.compress import compress_diff

load_dotenv()

MAX_TOOL_ROUNDS = 10
_MAX_TOOL_RESULT = 8000

_SYSTEM = """\
You are a senior software engineer implementing a feature. You have tools to inspect the codebase.

Workflow:
1. Use read_file and search_codebase to understand the existing code before writing anything.
2. When you have enough context, call propose_diff with a valid unified diff.

Rules:
- The diff must be minimal — only what the plan requires.
- Match the style and patterns you observe in the existing files.
- Do not apply changes yourself; propose_diff is the only output that matters.\
"""

_TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file from the target repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to repo root."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_codebase",
        "description": "Semantic search over the target repository for relevant patterns or symbols.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "propose_diff",
        "description": "Submit the final unified diff implementing the feature. Call exactly once when ready.",
        "parameters": {
            "type": "object",
            "properties": {
                "diff": {
                    "type": "string",
                    "description": "Unified diff (--- a/file / +++ b/file format).",
                },
                "explanation": {
                    "type": "string",
                    "description": "One paragraph explaining the implementation choices.",
                },
            },
            "required": ["diff", "explanation"],
        },
    },
]


def _trim(text: str, max_chars: int = _MAX_TOOL_RESULT) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [trimmed to {max_chars} chars]"


def _build_user_message(state: AgentState) -> str:
    plan = state["plan"]
    review = state["review_feedback"]
    debug = state["debug_feedback"]

    plan_block = "\n".join([
        f"Files to change: {', '.join(plan.files_to_change)}",
        f"Functions to add: {', '.join(f.name for f in plan.functions_to_add) or 'none'}",
        f"Constraints: {'; '.join(plan.constraints) or 'none'}",
        f"Reasoning: {plan.reasoning}",
    ])

    if review is None and debug is None:
        return (
            f"Feature request: {state['feature_request']}\n\n"
            f"Plan:\n{plan_block}\n\n"
            "Use the tools to explore the codebase, then call propose_diff."
        )

    sections = [
        f"Feature request: {state['feature_request']}",
        f"\nPlan:\n{plan_block}",
        f"\nPrevious diff:\n```diff\n{compress_diff(_trim(state['current_diff'] or '', 4000))}\n```",
    ]

    if debug:
        sections += [
            "\nThe diff was approved by code review but tests failed in the sandbox.",
            f"Debugger analysis:\n{debug}",
            "\nFix the identified issue and call propose_diff with the corrected diff.",
        ]
    else:
        blockers = [
            f"  [{i.severity.upper()}] {i.description}" + (f" ({i.location})" if i.location else "")
            for i in review.issues if i.severity in ("bug", "security")
        ]
        style_notes = [
            f"  [STYLE] {i.description}"
            for i in review.issues if i.severity == "style"
        ]
        sections.append(f"\nReview verdict: {review.summary}")
        if blockers:
            sections.append("Blockers to fix:\n" + "\n".join(blockers))
        if style_notes:
            sections.append("Style notes (informational):\n" + "\n".join(style_notes))
        sections.append("\nAddress the blockers and call propose_diff with the revised diff.")

    return "\n".join(sections)


def generator_node(state: AgentState) -> dict:
    history = [{"role": "user", "content": _build_user_message(state)}]
    diff: str | None = None
    explanation: str = ""

    for _ in range(MAX_TOOL_ROUNDS):
        response = get_client().chat(
            system=_SYSTEM,
            history=history,
            tools=_TOOLS,
            max_tokens=4096,
        )

        telemetry.record_tokens(response.usage_input, response.usage_output)

        history.append({
            "role": "assistant",
            "text": response.text,
            "tool_calls": response.tool_calls,
        })

        if response.stop_reason == "end_turn" or not response.tool_calls:
            break

        results = []
        for tc in response.tool_calls:
            if tc.name == "propose_diff":
                diff = tc.input["diff"]
                explanation = tc.input.get("explanation", "")
                results.append({"id": tc.id, "content": "Diff accepted."})
            else:
                fn = TOOL_DISPATCH.get(tc.name)
                raw = fn(**tc.input) if fn else f"Unknown tool: {tc.name}"
                results.append({"id": tc.id, "content": _trim(str(raw))})

        history.append({"role": "tool_results", "results": results})

        if diff is not None:
            break

    return {
        "current_diff": diff or "(generator did not produce a diff)",
        "status": "reviewing",
        "messages": [
            AgentMessage(
                role="generator",
                content=explanation or "Diff proposed (no explanation provided).",
            )
        ],
    }
