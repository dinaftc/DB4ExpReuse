"""Shared pieces of the system benchmarks."""
from __future__ import annotations

import argparse
import random
import re
from dataclasses import replace
from typing import Iterator

from ..embedding import Embedder, HashEmbedder
from ..grounding import substitute
from ..io import read_jsonl
from ..model import ReuseRequest, Step, StoredTrace, Subtrace, Task, Trace
from ..stores import make_store
from ..synthetic import make_traces


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--realization", default="relational",
                        help="relational, jsonb, qdrant or memory; the latency benchmark takes a comma-separated "
                             "list and checks that the realizations answer alike")
    parser.add_argument("--dsn", help="PostgreSQL connection string")
    parser.add_argument("--url", help="Qdrant endpoint")
    parser.add_argument("--traces", help="JSONL of base traces (default: synthetic)")
    parser.add_argument("--synthetic", type=int, default=500,
                        help="number of synthetic base traces when --traces is absent")
    parser.add_argument("--embedder", default="hash", choices=["hash", "mpnet"])
    parser.add_argument("--theta", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)


def make_embedder(name: str) -> Embedder:
    if name == "mpnet":
        from ..embedding import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder()
    return HashEmbedder(dim=768)


def load_base(args, embedder: Embedder) -> list[StoredTrace]:
    if args.traces:
        return list(read_jsonl(args.traces, embedder))
    return make_traces(args.synthetic, seed=args.seed, embedder=embedder)


def open_store(args, embedder: Embedder, kind: str | None = None):
    return make_store(kind or args.realization, embedder.dim, dsn=args.dsn, url=args.url)


def _entity_class(entity: str) -> str:
    return re.sub(r"[\s_]*\d+$", "", entity)


def scaled(base: list[StoredTrace], embedder: Embedder, seed: int = 0,
           tag: str = "", include_base: bool = True) -> Iterator[StoredTrace]:
    """Endless stream of stored traces: the base traces, then copies."""
    if include_base:
        yield from base
    rng = random.Random(seed)
    classes = sorted({_entity_class(e) for item in base for s in item.subtraces
                      for e in s.binding.values()})
    for copy in range(1, 1 << 62):
        for item in base:
            own = sorted({_entity_class(e) for s in item.subtraces
                          for e in s.binding.values()})
            mapping = dict(zip(own, rng.sample(classes, len(own))))
            sub = lambda text: substitute(text, mapping)
            trace_id = f"{item.trace.trace_id}#{tag}{copy}"
            task_id = f"{item.task.task_id}#{tag}{copy}"
            goal = sub(item.task.goal)
            yield StoredTrace(
                Task(task_id, item.task.application, goal, embedder.encode([goal])[0]),
                Trace(trace_id, task_id, item.trace.producer, item.trace.success),
                tuple(Step(trace_id, s.step_index, s.subgoal_id, sub(s.action),
                           sub(s.observation)) for s in item.steps),
                tuple(replace(s, trace_id=trace_id,
                              binding={r: sub(e) for r, e in s.binding.items()})
                      for s in item.subtraces))


def take(stream: Iterator[StoredTrace], stored: int, target: int) -> tuple[list[StoredTrace], int]:
    """The next traces of the stream up to `target` stored subtraces, and the new count."""
    batch = []
    while stored < target:
        item = next(stream)
        batch.append(item)
        stored += len(item.subtraces)
    return batch, stored


def grow(stores, stream: Iterator[StoredTrace], stored: int, target: int, chunk: int = 20_000) -> int:
    """Load the same traces into every store until `target` subtraces are stored, `chunk` subtraces at a time."""
    while stored < target:
        batch, stored = take(stream, stored, min(target, stored + chunk))
        for store in stores:
            store.insert_many(batch)
    return stored


def signature(experience) -> tuple:
    """What an answer returned, independent of the realization."""
    return (experience.route, tuple(t.trace_id for t in experience.traces),
            tuple(tuple((h.trace_id, h.subgoal_id) for h in hits) for hits in experience.subtraces))


def requests_from(items: list[StoredTrace], n: int, seed: int = 0, k: int = 1
                  ) -> list[ReuseRequest]:
    """Reuse requests for the goals and plans of sampled traces."""
    rng = random.Random(seed)
    return [ReuseRequest(i.task.embedding, i.trace.producer, (i.task.application,),
                         tuple(s.operator for s in i.subtraces), k)
            for i in rng.choices(items, k=n)]
