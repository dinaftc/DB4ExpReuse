from __future__ import annotations

from typing import Protocol, Sequence

from ..model import Experience, Plan, Task

# (action, observation) pairs so far; the first entry is ("", initial observation).
History = Sequence[tuple[str, str]]


class Consumer(Protocol):
    def begin(self, task: Task, plan: Plan, experience: Experience,
              cross_application: bool = False) -> None:
        """Receive the answer of the reuse query before acting on a task."""

    def act(self, history: History, admissible: Sequence[str]) -> str:
        """Choose the next action."""
