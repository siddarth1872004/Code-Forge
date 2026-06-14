from __future__ import annotations

from datetime import datetime, UTC
from typing import Annotated, Literal, TypedDict
import operator

from pydantic import BaseModel, Field


class FunctionSpec(BaseModel):
    name: str
    signature: str
    description: str


class Plan(BaseModel):
    files_to_change: list[str]
    functions_to_add: list[FunctionSpec] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    reasoning: str


class Issue(BaseModel):
    severity: Literal["style", "security", "bug"]
    description: str
    location: str | None = None


class ReviewFeedback(BaseModel):
    approved: bool
    issues: list[Issue] = Field(default_factory=list)
    summary: str


class TestResult(BaseModel):
    exit_code: int
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.exit_code == 0


class AgentMessage(BaseModel):
    role: Literal["planner", "generator", "reviewer", "test_generator", "debugger"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentState(TypedDict):
    run_id: str
    feature_request: str
    plan: Plan | None
    current_diff: str | None
    review_feedback: ReviewFeedback | None
    generated_tests: str | None
    test_result: TestResult | None
    debug_feedback: str | None
    git_branch: str | None
    pr_url: str | None
    # operator.add appends each node's new messages rather than overwriting
    messages: Annotated[list[AgentMessage], operator.add]
    iteration_count: int
    debug_count: int
    status: Literal[
        "planning",
        "generating",
        "reviewing",
        "approved",
        "testing",
        "debugging",
        "needs_human",
        "awaiting_approval",
        "rejected",
        "pr_created",
    ]
