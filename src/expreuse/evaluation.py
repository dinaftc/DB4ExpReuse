"""Task success with bootstrap standard errors over task resamples (Section 4.1)."""
from __future__ import annotations

import numpy as np


def success_rate(outcomes, resamples: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """`outcomes` is (runs, tasks) or (tasks,) of 0/1. Returns the mean success
    and its bootstrap standard error, both in percent. Runs of the same task
    are averaged first, so tasks, not runs, are resampled."""
    per_task = np.atleast_2d(np.asarray(outcomes, dtype=float)).mean(axis=0)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(per_task), size=(resamples, len(per_task)))
    means = per_task[index].mean(axis=1)
    return 100 * per_task.mean(), 100 * means.std(ddof=1)
