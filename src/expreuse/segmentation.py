"""Segment an interaction trace into operator-level subtraces (Section 3.2).

The decomposition itself is application-specific: an application supplies a
`StepLabeler` that says, for each step, which operator the action belongs to
and whether the environment completed the current subgoal. This module turns
those labels into Step and Subtrace rows.
"""
from __future__ import annotations

from typing import Protocol, Sequence

from .model import Plan, Step, Subtrace


class StepLabeler(Protocol):
    def operator_of(self, action: str) -> str | None:
        """Canonical operator the action realizes; None for actions that
        belong to no operator on their own (navigation, look, open, ...)."""

    def completes(self, operator: str, binding: dict[str, str], action: str,
                  observation: str) -> bool:
        """Whether this step completes a subgoal with the given operator."""

    def binding(self, operator: str, steps: Sequence[tuple[str, str]]) -> dict[str, str]:
        """Entities bound to the operator's roles in these (action, observation)
        steps."""


def segment(trace_id: str, plan: Plan, raw: Sequence[tuple[str, str]],
            labeler: StepLabeler) -> tuple[tuple[Step, ...], tuple[Subtrace, ...]]:
    """Assign each (action, observation) to the plan position being worked on.

    Position i owns the steps from the completion of position i-1 up to and
    including the step that completes it. Steps after the last completed
    position stay with the next, unfinished one; steps after the whole plan is
    completed belong to no subgoal. A subtrace `fits` when none of its steps
    realizes a different operator.
    """
    steps: list[Step] = []
    spans: list[list[tuple[str, str]]] = [[] for _ in plan]
    finished = [False] * len(plan)
    position = 0
    for index, (action, observation) in enumerate(raw):
        current = position if position < len(plan) else None
        steps.append(Step(trace_id, index, current, action, observation))
        if current is None:
            continue
        spans[current].append((action, observation))
        target = plan[current]
        if labeler.completes(target.operator, target.binding, action, observation):
            finished[current] = True
            position += 1
    subtraces = []
    for i, span in enumerate(spans):
        if not span:
            continue
        operator = plan[i].operator
        fits = all(labeler.operator_of(a) in (None, operator) for a, _ in span)
        subtraces.append(Subtrace(trace_id, i, operator, labeler.binding(operator, span),
                                  finished[i], fits))
    return tuple(steps), tuple(subtraces)
