"""Throughput under concurrent consumers and producers, on a database already populated by bench.latency.

    python -m expreuse.bench.concurrency --dsn postgresql:///expreuse --clients 1,4,16,32,64 --insert-fraction 0.1
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
import uuid

from .common import (add_arguments, load_base, make_embedder, open_store,
                     requests_from, scaled)


def client(args, embedder, requests, fresh, lock, deadline, seed, out):
    rng = random.Random(seed)
    queries, inserts, failures, latencies = 0, 0, 0, []
    with open_store(args, embedder) as store:
        while time.perf_counter() < deadline:
            try:
                if rng.random() < args.insert_fraction:
                    with lock:
                        item = next(fresh)
                    store.insert(item)
                    inserts += 1
                else:
                    start = time.perf_counter()
                    store.cascade(rng.choice(requests), args.theta)
                    latencies.append((time.perf_counter() - start) * 1000)
                    queries += 1
            except Exception:
                failures += 1
                if hasattr(store, "conn"):
                    store.conn.rollback()
    out.append((queries, inserts, failures, latencies))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(parser)
    parser.add_argument("--clients", default="1,4,16,32,64")
    parser.add_argument("--insert-fraction", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    embedder = make_embedder(args.embedder)
    base = load_base(args, embedder)
    requests = requests_from(base, 1000, args.seed)
    fresh = scaled(base, embedder, args.seed, tag=uuid.uuid4().hex[:8] + "-",
                   include_base=False)
    lock = threading.Lock()
    for n in map(int, args.clients.split(",")):
        out: list = []
        deadline = time.perf_counter() + args.seconds
        threads = [threading.Thread(target=client, args=(
            args, embedder, requests, fresh, lock, deadline, args.seed + i, out))
            for i in range(n)]
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start
        latencies = [x for o in out for x in o[3]]
        print(json.dumps({
            "clients": n, "insert_fraction": args.insert_fraction,
            "queries_per_s": sum(o[0] for o in out) / elapsed,
            "inserts_per_s": sum(o[1] for o in out) / elapsed,
            "failed": sum(o[2] for o in out),
            "median_query_ms": statistics.median(latencies) if latencies else None,
        }), flush=True)


if __name__ == "__main__":
    main()
