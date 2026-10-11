"""Reuse queries (Section 3.3) over a physical realization of the data model."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from ..model import Experience, ReuseRequest, StoredTrace, SubtraceHit, TraceHit


class ExperienceStore(ABC):
    """A physical realization. All realizations must return the same experience
    for the same request: candidates are ordered by goal similarity
    (descending), then length (ascending), then identifier."""

    @abstractmethod
    def setup(self, reset: bool = False) -> None:
        """Create the physical schema; drop existing data first if `reset`."""

    @abstractmethod
    def insert(self, item: StoredTrace) -> None:
        """Store a completed trace and its subtraces atomically."""

    @abstractmethod
    def whole(self, request: ReuseRequest) -> list[TraceHit]:
        """WHOLE: top-K successful complete traces of the requested producer."""

    @abstractmethod
    def subtraces_for(self, request: ReuseRequest, operators: list[str]
                      ) -> dict[str, list[SubtraceHit]]:
        """Top-K finished, fitting subtraces of the producer for each operator."""

    @abstractmethod
    def counts(self) -> dict[str, int]:
        """Number of stored tasks, traces, subtraces and steps."""

    def close(self) -> None:
        pass

    def subtraces(self, request: ReuseRequest) -> list[list[SubtraceHit]]:
        """SUBTRACES: hits for each plan position, in plan order. A position
        whose operator has no eligible subtrace is left empty (uncovered)."""
        found = self.subtraces_for(request, list(dict.fromkeys(request.plan)))
        return [list(found.get(op, [])) for op in request.plan]

    def insert_many(self, items: Iterable[StoredTrace]) -> None:
        """Store several completed traces; a realization may load them in bulk."""
        for item in items:
            self.insert(item)

    def cascade(self, request: ReuseRequest, theta: float) -> Experience:
        """CASCADE: WHOLE if its best trace reaches `theta`, else SUBTRACES."""
        traces = self.whole(request)
        best = traces[0].similarity if traces else None
        if traces and best >= theta:
            return Experience("whole", traces=traces, similarity=best)
        return self._subtraces_or_nothing(request, best)

    def _subtraces_or_nothing(self, request: ReuseRequest, similarity: float | None = None) -> Experience:
        hits = self.subtraces(request)
        if any(hits):
            return Experience("subtraces", subtraces=hits, similarity=similarity)
        return Experience("none", similarity=similarity)

    def query(self, mode: str, request: ReuseRequest, theta: float | None = None
              ) -> Experience:
        if mode == "none":
            return Experience("none")
        if mode == "whole":
            traces = self.whole(request)
            return Experience("whole", traces=traces, similarity=traces[0].similarity) if traces else Experience("none")
        if mode == "subtraces":
            return self._subtraces_or_nothing(request)
        if mode == "cascade":
            if theta is None:
                raise ValueError("cascade requires a similarity threshold")
            return self.cascade(request, theta)
        raise ValueError(f"unknown reuse mode: {mode}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
