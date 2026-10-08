"""Partial subtrace coverage: withhold a share of the subtraces of a SUBTRACES answer before the consumer sees it.

A withholding level f removes round(f * n) of the n covered plan positions from the answer; the removed positions
become uncovered, as if the store held no subtrace for them. Two removal schedules: every possible subset of that
size (`all_removals`, averaged by the caller; for short answers), or one random ordering per task, nested across
levels so that what is removed at 25% stays removed at 50% (`nested_removals`; for long answers).
"""
from __future__ import annotations

import hashlib
import random
from copy import deepcopy
from itertools import combinations
from typing import Sequence

from .model import Experience

LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


def covered_positions(experience: Experience) -> list[int]:
    return [i for i, hits in enumerate(experience.subtraces) if hits]


def removal_count(n: int, fraction: float) -> int:
    return int(n * fraction + 0.5)


def all_removals(experience: Experience, fraction: float) -> list[tuple[int, ...]]:
    """Every subset of covered positions of the size the level asks for."""
    covered = covered_positions(experience)
    return list(combinations(covered, removal_count(len(covered), fraction)))


def nested_removals(experience: Experience, fraction: float, key: str) -> tuple[int, ...]:
    """The first round(f * n) positions of one random ordering of the covered positions, the ordering fixed by
    `key` (the task id), so that higher levels remove supersets."""
    covered = covered_positions(experience)
    rng = random.Random(int(hashlib.sha1(key.encode()).hexdigest(), 16))
    order = covered[:]
    rng.shuffle(order)
    return tuple(order[: removal_count(len(covered), fraction)])


def withhold(experience: Experience, positions: Sequence[int]) -> Experience:
    """The answer without the subtraces at the given plan positions."""
    out = deepcopy(experience)
    for i in positions:
        out.subtraces[i] = []
    if out.route == "subtraces" and not any(out.subtraces):
        out.route = "none"
    return out
