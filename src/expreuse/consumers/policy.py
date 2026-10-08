"""Trained-policy consumer: inference-time shaping with returned steps.

    a* = argmax_a [ log p(a | s) + lambda * Q_ITD(s, a) ]
    Q_ITD(s, a) = sum_{j in TopM(s)} w_j [a_j = a] / sum_{j in TopM(s)} w_j
    w_j = cosine(s, s_j)

where s_j is the state preceding returned step j. A state is the task goal
with its two preceding actions and observations.
"""
from __future__ import annotations

from typing import Callable, Protocol, Sequence

import numpy as np

from ..embedding import Embedder
from ..model import Experience, Plan, Task
from .base import History


class BasePolicy(Protocol):
    def log_probs(self, task: Task, history: History, admissible: Sequence[str]
                  ) -> Sequence[float]:
        """log p(a | s) for each admissible action."""


def state_text(goal: str, previous: Sequence[tuple[str, str]]) -> str:
    recent = [part for a, o in previous[-2:] for part in (a, o) if part]
    return " | ".join([goal, *recent])


class ShapedPolicy:
    """`operator_of` maps a target-application action to its canonical operator
    (None if it realizes none); it is required for cross-application reuse,
    where a source step supports the admissible actions realizing the same
    shared operator instead of an identical action string."""

    def __init__(self, base: BasePolicy, embedder: Embedder, m: int = 1,
                 lam: float = 1.0,
                 operator_of: Callable[[str], str | None] | None = None):
        self.base, self.embedder = base, embedder
        self.m, self.lam, self.operator_of = m, lam, operator_of
        self._task: Task | None = None
        self._steps = []
        self._states = np.zeros((0, embedder.dim), dtype=np.float32)
        self._cross = False

    def begin(self, task: Task, plan: Plan, experience: Experience,
              cross_application: bool = False) -> None:
        if cross_application and self.operator_of is None:
            raise ValueError("cross-application reuse needs operator_of")
        self._task, self._cross = task, cross_application
        self._steps = experience.steps()
        if self._steps:
            self._states = self.embedder.encode(
                [state_text(s.goal, s.previous) for s in self._steps])

    def q_itd(self, history: History, admissible: Sequence[str]) -> np.ndarray:
        q = np.zeros(len(admissible))
        if not self._steps:
            return q
        state = self.embedder.encode([state_text(self._task.goal, history)])[0]
        # Negative similarities carry no support; clamping keeps Q in [0, 1].
        weights = np.maximum(self._states @ state, 0.0)
        top = np.argsort(-weights, kind="stable")[:self.m]
        total = weights[top].sum()
        if total <= 0:
            return q
        operators = ([self.operator_of(a) for a in admissible] if self._cross else None)
        for j in top:
            step = self._steps[j]
            for i, action in enumerate(admissible):
                if self._cross:
                    match = step.operator is not None and operators[i] == step.operator
                else:
                    match = action == step.action
                if match:
                    q[i] += weights[j]
        return q / total

    def act(self, history: History, admissible: Sequence[str]) -> str:
        scores = np.asarray(self.base.log_probs(self._task, history, admissible),
                            dtype=float)
        scores = scores + self.lam * self.q_itd(history, admissible)
        return admissible[int(np.argmax(scores))]
