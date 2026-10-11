"""In-memory reference realization: a direct transcription of the query definitions, used as the oracle the
other realizations are checked against."""
from __future__ import annotations

import numpy as np

from ..embedding import normalize
from ..model import ReuseRequest, StoredTrace, SubtraceHit, TraceHit
from .base import ExperienceStore


class MemoryStore(ExperienceStore):
    def __init__(self):
        self._items: list[StoredTrace] = []

    def setup(self, reset: bool = False) -> None:
        if reset:
            self._items = []

    def insert(self, item: StoredTrace) -> None:
        if item.task.embedding is None:
            raise ValueError("task has no goal embedding")
        self._items.append(item)

    def _eligible(self, request: ReuseRequest):
        q = normalize(request.embedding)
        for item in self._items:
            if (item.task.application in request.applications
                    and item.trace.producer == request.producer):
                yield float(normalize(item.task.embedding) @ q), item

    def whole(self, request: ReuseRequest) -> list[TraceHit]:
        candidates = [(sim, item) for sim, item in self._eligible(request)
                      if item.trace.success]
        candidates.sort(key=lambda c: (-c[0], c[1].length, c[1].trace.trace_id))
        return [TraceHit(i.trace.trace_id, i.task.task_id, i.task.application,
                         i.trace.producer, i.task.goal, sim, list(i.steps),
                         list(i.subtraces))
                for sim, i in candidates[:request.k]]

    def subtraces_for(self, request, operators):
        candidates = {op: [] for op in operators}
        for sim, item in self._eligible(request):
            for sub in item.subtraces:
                if sub.operator in candidates and sub.finished and sub.fits:
                    steps = list(item.subtrace_steps(sub.subgoal_id))
                    candidates[sub.operator].append(SubtraceHit(
                        sub.trace_id, sub.subgoal_id, item.task.application,
                        item.trace.producer, item.task.goal, sim, sub.operator,
                        dict(sub.binding), steps))
        for hits in candidates.values():
            hits.sort(key=lambda h: (-h.similarity, h.length, h.trace_id, h.subgoal_id))
        return {op: hits[:request.k] for op, hits in candidates.items()}

    def counts(self) -> dict[str, int]:
        return {
            "tasks": len({i.task.task_id for i in self._items}),
            "traces": len(self._items),
            "subtraces": sum(len(i.subtraces) for i in self._items),
            "steps": sum(i.length for i in self._items),
        }
