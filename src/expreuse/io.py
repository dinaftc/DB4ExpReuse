"""JSON Lines serialization of stored traces (one trace per line)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .embedding import Embedder
from .model import Step, StoredTrace, Subtrace, Task, Trace


def to_json(item: StoredTrace) -> dict:
    return {
        "task_id": item.task.task_id, "application": item.task.application,
        "goal": item.task.goal, "trace_id": item.trace.trace_id,
        "producer": item.trace.producer, "success": item.trace.success,
        "steps": [{"subgoal_id": s.subgoal_id, "action": s.action,
                   "observation": s.observation} for s in item.steps],
        "subtraces": [{"subgoal_id": s.subgoal_id, "operator": s.operator,
                       "binding": s.binding, "finished": s.finished, "fits": s.fits}
                      for s in item.subtraces],
    }


def from_json(d: dict, embedding: np.ndarray | None = None) -> StoredTrace:
    tid = d["trace_id"]
    return StoredTrace(
        Task(d["task_id"], d["application"], d["goal"], embedding),
        Trace(tid, d["task_id"], d["producer"], bool(d["success"])),
        tuple(Step(tid, i, s["subgoal_id"], s["action"], s["observation"])
              for i, s in enumerate(d["steps"])),
        tuple(Subtrace(tid, s["subgoal_id"], s["operator"], dict(s["binding"]),
                       bool(s["finished"]), bool(s["fits"])) for s in d["subtraces"]))


def write_jsonl(path: str | Path, items: Iterable[StoredTrace]) -> None:
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(to_json(item)) + "\n")


def read_jsonl(path: str | Path, embedder: Embedder | None = None, batch: int = 512) -> Iterator[StoredTrace]:
    """Embeddings are not serialized; pass an embedder to recompute them. The distinct goals of the file are
    embedded in batches, so a large file costs one model call per `batch` goals rather than one per trace."""
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    vectors: dict[str, np.ndarray] = {}
    if embedder is not None:
        goals = list(dict.fromkeys(d["goal"] for d in rows))
        for i in range(0, len(goals), batch):
            vectors.update(zip(goals[i:i + batch], embedder.encode(goals[i:i + batch])))
    for d in rows:
        yield from_json(d, vectors.get(d["goal"]))
