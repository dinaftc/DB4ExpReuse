"""The application's own expert as a producer: it plays the action the environment's expert recommends."""
from __future__ import annotations

from typing import Sequence

from ..model import Experience, Plan, Task
from .base import History


class ExpertConsumer:
    """Follows `app.expert_action()`; plays `fallback` when the expert has no recommendation."""

    def __init__(self, app, fallback: str = "look"):
        if not hasattr(app, "expert_action"):
            raise ValueError(f"{app.name} provides no expert")
        self.app, self.fallback = app, fallback

    def begin(self, task: Task, plan: Plan, experience: Experience, cross_application: bool = False) -> None:
        pass

    def act(self, history: History, admissible: Sequence[str]) -> str:
        return self.app.expert_action() or self.fallback
