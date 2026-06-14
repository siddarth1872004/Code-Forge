from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src import telemetry
from src.agents.debugger import debugger_node
from src.agents.generator import generator_node
from src.agents.git_agent import git_agent_node
from src.agents.planner import planner_node
from src.agents.reviewer import reviewer_node
from src.agents.test_generator import test_generator_node
from src.sandbox.runner import docker_runner_node
from src.state import AgentState

MAX_REVIEW_ITERATIONS = 2
MAX_DEBUG_ITERATIONS = 1


def _route_after_review(state: AgentState) -> str:
    if state["status"] == "approved":
        return "test_generator"
    if state["iteration_count"] >= MAX_REVIEW_ITERATIONS:
        return END
    return "generator"


def _route_after_tests(state: AgentState) -> str:
    if state["status"] == "approved":
        return "git_agent"
    if state["debug_count"] >= MAX_DEBUG_ITERATIONS:
        return END
    return "debugger"


def build_graph():
    builder = StateGraph(AgentState)

    # Every node is wrapped with telemetry.timed() so that:
    #   - a span opens before the node runs (capturing start time)
    #   - agents call telemetry.record_tokens() mid-execution to fill in usage
    #   - the span closes after the node returns (capturing end time)
    builder.add_node("planner",        telemetry.timed("planner",        planner_node))
    builder.add_node("generator",      telemetry.timed("generator",      generator_node))
    builder.add_node("reviewer",       telemetry.timed("reviewer",       reviewer_node))
    builder.add_node("test_generator", telemetry.timed("test_generator", test_generator_node))
    builder.add_node("docker_runner",  telemetry.timed("docker_runner",  docker_runner_node))
    builder.add_node("debugger",       telemetry.timed("debugger",       debugger_node))
    builder.add_node("git_agent",      telemetry.timed("git_agent",      git_agent_node))

    builder.set_entry_point("planner")

    builder.add_edge("planner", "generator")
    builder.add_edge("generator", "reviewer")
    builder.add_conditional_edges(
        "reviewer",
        _route_after_review,
        {"generator": "generator", "test_generator": "test_generator", END: END},
    )
    builder.add_edge("test_generator", "docker_runner")
    builder.add_conditional_edges(
        "docker_runner",
        _route_after_tests,
        {"git_agent": "git_agent", "debugger": "debugger", END: END},
    )
    builder.add_edge("debugger", "generator")
    builder.add_edge("git_agent", END)

    return builder.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["git_agent"],
    )
