"""What an application must provide to plug into the framework."""
from __future__ import annotations

from typing import Protocol, Sequence

from ..model import Plan, Task
from ..operators import OperatorLibrary
from ..segmentation import StepLabeler


class Application(Protocol):
    name: str
    operators: OperatorLibrary
    labeler: StepLabeler
    system_prompt: str  # how an LLM consumer should act in this application

    def plan(self, task: Task) -> Plan:
        """The task plan P_t with target bindings where known."""

    def reset(self, task: Task) -> str:
        """Start the task and return the initial observation."""

    def step(self, action: str) -> tuple[str, bool, bool]:
        """Execute an action; return (observation, done, success)."""

    def admissible_actions(self) -> Sequence[str]:
        """Actions valid in the current state."""

    def max_steps(self, task: Task) -> int:
        """Step budget for the task."""
