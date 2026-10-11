"""Retrieval and insertion latency as the database grows.

    python -m expreuse.bench.latency --realization relational,jsonb,qdrant --dsn postgresql:///expreuse \
    --url http://localhost:6333 --sizes 2000,13000,100000,1000000
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

from .common import (add_arguments, grow, load_base, make_embedder, open_store, requests_from, scaled,
                     signature)


def timed(fn, arguments) -> tuple[float, list]:
    """Median milliseconds of `fn` over `arguments`, and its results."""
    samples, results = [], []
    for a in arguments:
        start = time.perf_counter()
        results.append(fn(a))
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples), results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(parser)
    parser.add_argument("--sizes", default="2000,13000,100000,1000000",
                        help="stored subtraces at which to measure")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--inserts", type=int, default=200)
    args = parser.parse_args()

    embedder = make_embedder(args.embedder)
    base = load_base(args, embedder)
    requests = requests_from(base, args.queries, args.seed)
    stream = scaled(base, embedder, args.seed)
    stores = {kind: open_store(args, embedder, kind) for kind in args.realization.split(",")}
    for store in stores.values():
        store.setup(reset=True)
    stored, answers = 0, {}
    for size in map(int, args.sizes.split(",")):
        stored = grow(list(stores.values()), stream, stored, size)
        fresh = None
        for kind, store in stores.items():
            for r in requests[:20]:
                store.cascade(r, args.theta)
            row = {"realization": kind, "subtraces": stored}
            row["whole_ms"], whole = timed(store.whole, requests)
            row["subtraces_ms"], subtraces = timed(store.subtraces, requests)
            row["cascade_ms"], cascade = timed(lambda r: store.cascade(r, args.theta), requests)
            answers[kind] = ([tuple(t.trace_id for t in hits) for hits in whole],
                             [tuple(tuple((h.trace_id, h.subgoal_id) for h in hits) for hits in a)
                              for a in subtraces], [signature(a) for a in cascade])
            row["identical"] = answers[kind] == answers[next(iter(stores))]
            if fresh is None:
                fresh = [next(stream) for _ in range(args.inserts)]
            row["insert_ms"], _ = timed(store.insert, fresh)
            print(json.dumps(row), flush=True)
        stored += sum(len(i.subtraces) for i in fresh)
    for store in stores.values():
        store.close()


if __name__ == "__main__":
    main()
