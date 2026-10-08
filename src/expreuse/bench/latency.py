"""Retrieval and insertion latency as the database grows (Figure 3a, Table 6).

    python -m expreuse.bench.latency --realization relational --dsn postgresql:///expreuse \
        --sizes 2000,13000,100000,1000000
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

from .common import (add_arguments, grow, load_base, make_embedder, open_store,
                     requests_from, scaled)


def median_ms(fn, arguments) -> float:
    samples = []
    for a in arguments:
        start = time.perf_counter()
        fn(a)
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


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
    with open_store(args, embedder) as store:
        store.setup(reset=True)
        stored = 0
        for size in map(int, args.sizes.split(",")):
            stored = grow(store, stream, stored, size)
            for r in requests[:20]:  # warm up
                store.cascade(r, args.theta)
            row = {
                "realization": args.realization, "subtraces": stored,
                "whole_ms": median_ms(store.whole, requests),
                "subtraces_ms": median_ms(store.subtraces, requests),
                "cascade_ms": median_ms(lambda r: store.cascade(r, args.theta), requests),
            }
            fresh = [next(stream) for _ in range(args.inserts)]
            row["insert_ms"] = median_ms(store.insert, fresh)
            stored += sum(len(i.subtraces) for i in fresh)
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
