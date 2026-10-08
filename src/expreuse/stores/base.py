"""Reuse queries (Section 3.3) over a physical realization of the data model."""
from __future__ import annotations

from abc import ABC, abstractmethod

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

    def cascade(self, request: ReuseRequest, theta: float) -> Experience:
        """CASCADE: WHOLE if its best trace reaches `theta`, else SUBTRACES."""
        traces = self.whole(request)
        if traces and traces[0].similarity >= theta:
            return Experience("whole", traces=traces)
        return self._subtraces_or_nothing(request)

    def _subtraces_or_nothing(self, request: ReuseRequest) -> Experience:
        hits = self.subtraces(request)
        if any(hits):
            return Experience("subtraces", subtraces=hits)
        return Experience("none")

    def query(self, mode: str, request: ReuseRequest, theta: float | None = None
              ) -> Experience:
        if mode == "none":
            return Experience("none")
        if mode == "whole":
            traces = self.whole(request)
            return Experience("whole", traces=traces) if traces else Experience("none")
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
