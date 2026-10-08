"""Logical data model (Section 3.1) and the shapes returned by reuse queries."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class Task:
    task_id: str
    application: str
    goal: str
    embedding: np.ndarray | None = None


@dataclass(frozen=True)
class Trace:
    trace_id: str
    task_id: str
    producer: str
    success: bool


@dataclass(frozen=True)
class Step:
    trace_id: str
    step_index: int
    subgoal_id: int | None  # None: the step belongs to no operator of the plan
    action: str
    observation: str


@dataclass(frozen=True)
class Subtrace:
    trace_id: str
    subgoal_id: int  # position of the operator instance in the task plan
    operator: str  # canonical operator name
    binding: dict[str, str]  # role -> concrete entity
    finished: bool  # the environment completed the subgoal
    fits: bool  # contains only steps belonging to its operator


@dataclass(frozen=True)
class StoredTrace:
    """One completed execution: the unit a producer inserts into the database."""

    task: Task
    trace: Trace
    steps: tuple[Step, ...]
    subtraces: tuple[Subtrace, ...]

    @property
    def length(self) -> int:
        return len(self.steps)

    def subtrace_steps(self, subgoal_id: int) -> tuple[Step, ...]:
        return tuple(s for s in self.steps if s.subgoal_id == subgoal_id)


@dataclass(frozen=True)
class PlanStep:
    operator: str  # canonical operator name
    binding: dict[str, str] = field(default_factory=dict)  # target entities, if known


Plan = tuple[PlanStep, ...]


@dataclass(frozen=True)
class ReuseRequest:
    """A reuse query for a target task.

    `applications` is the application constraint defining the eligible stored
    tasks V; `producer` is r; `plan` lists the canonical operators of P_t.
    """

    embedding: np.ndarray
    producer: str
    applications: tuple[str, ...]
    plan: tuple[str, ...] = ()
    k: int = 1


@dataclass
class TraceHit:
    trace_id: str
    task_id: str
    application: str
    producer: str
    goal: str
    similarity: float
    steps: list[Step]
    subtraces: list[Subtrace]

    @property
    def length(self) -> int:
        return len(self.steps)


@dataclass
class SubtraceHit:
    trace_id: str
    subgoal_id: int
    application: str
    producer: str
    goal: str
    similarity: float
    operator: str
    binding: dict[str, str]
    steps: list[Step]

    @property
    def length(self) -> int:
        return len(self.steps)


@dataclass(frozen=True)
class ExperienceStep:
    """A returned step as seen by a consumer, with the state that preceded it."""

    goal: str
    action: str
    observation: str
    operator: str | None
    previous: tuple[tuple[str, str], ...]  # up to two preceding (action, observation)


@dataclass
class Experience:
    """Answer of a reuse query. `subtraces[i]` holds the hits for plan position i."""

    route: Literal["whole", "subtraces", "none"] = "none"
    traces: list[TraceHit] = field(default_factory=list)
    subtraces: list[list[SubtraceHit]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.route != "none"

    @property
    def uncovered(self) -> list[int]:
        return [i for i, hits in enumerate(self.subtraces) if not hits]

    def sequences(self) -> list[tuple[str, list[Step], dict[int | None, str]]]:
        """Returned step sequences as (source goal, steps, subgoal_id -> operator)."""
        if self.route == "whole":
            return [(t.goal, t.steps, {s.subgoal_id: s.operator for s in t.subtraces})
                    for t in self.traces]
        return [(h.goal, h.steps, {h.subgoal_id: h.operator})
                for hits in self.subtraces for h in hits]

    def steps(self) -> list[ExperienceStep]:
        out = []
        for goal, steps, operators in self.sequences():
            for j, s in enumerate(steps):
                previous = tuple((p.action, p.observation) for p in steps[max(0, j - 2):j])
                out.append(ExperienceStep(goal, s.action, s.observation,
                                          operators.get(s.subgoal_id), previous))
        return out
