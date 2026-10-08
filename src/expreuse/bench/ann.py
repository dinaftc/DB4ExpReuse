"""Exact versus HNSW goal-similarity lookup with pgvector (Section 4.5).

Compares, for a set of query goals, the latency of exact search and of HNSW at
several ef_search values, and how often HNSW returns the same nearest goal, the
same top-10 set and the same CASCADE route (best similarity >= theta).

    python -m expreuse.bench.ann --dsn postgresql:///expreuse --goals 1000000 --ef 40,400

Without --goal-file, goals are synthetic: base goal embeddings plus Gaussian
noise, renormalized.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

from ..embedding import normalize
from ..stores._pg import ensure_pgvector, vector_literal
from .common import add_arguments, load_base, make_embedder

TOP = "SELECT goal_id, 1 - (embedding <=> %(q)s::vector) FROM goal ORDER BY embedding <=> %(q)s::vector LIMIT 10"


def search(conn, queries):
    results, samples = [], []
    for q in queries:
        start = time.perf_counter()
        rows = conn.execute(TOP, {"q": q}).fetchall()
        samples.append((time.perf_counter() - start) * 1000)
        results.append(rows)
    return results, statistics.median(samples)


def main() -> None:
    import psycopg

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(parser)
    parser.add_argument("--goals", type=int, default=100_000)
    parser.add_argument("--goal-file", help="text file with one distinct goal per line")
    parser.add_argument("--queries", type=int, default=283)
    parser.add_argument("--ef", default="40,400")
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--schema", default="expreuse_ann")
    args = parser.parse_args()

    embedder = make_embedder(args.embedder)
    rng = np.random.default_rng(args.seed)
    base = load_base(args, embedder)
    seeds = np.stack([i.task.embedding for i in base])
    if args.goal_file:
        with open(args.goal_file) as f:
            vectors = embedder.encode([line.strip() for line in f if line.strip()])
    else:
        picks = rng.integers(0, len(seeds), args.goals)
        vectors = normalize(seeds[picks] + args.noise * rng.standard_normal(
            (args.goals, embedder.dim)).astype(np.float32))
    queries = [vector_literal(seeds[i]) for i in rng.integers(0, len(seeds), args.queries)]

    with psycopg.connect(args.dsn, autocommit=True) as conn:
        ensure_pgvector(conn)
        conn.execute(f'DROP SCHEMA IF EXISTS "{args.schema}" CASCADE')
        conn.execute(f'CREATE SCHEMA "{args.schema}"')
        conn.execute(f'SET search_path TO "{args.schema}", public')
        conn.execute(f"CREATE TABLE goal (goal_id bigserial PRIMARY KEY, "
                     f"embedding vector({embedder.dim}) NOT NULL)")
        conn.execute("ALTER TABLE goal ALTER COLUMN embedding SET STORAGE PLAIN")
        with conn.cursor() as cur, cur.copy("COPY goal (embedding) FROM STDIN") as copy:
            for v in vectors:
                copy.write_row((vector_literal(v),))
        conn.execute("ANALYZE goal")

        exact, exact_ms = search(conn, queries)
        print(json.dumps({"search": "exact", "goals": len(vectors),
                          "median_ms": exact_ms}), flush=True)
        conn.execute("CREATE INDEX ON goal USING hnsw (embedding vector_cosine_ops)")
        for ef in map(int, args.ef.split(",")):
            conn.execute(f"SET hnsw.ef_search = {ef}")
            approx, ms = search(conn, queries)
            pairs = list(zip(exact, approx))
            share = lambda same: sum(map(same, pairs)) / len(pairs)
            print(json.dumps({
                "search": "hnsw", "ef_search": ef, "median_ms": ms,
                "same_nearest": share(lambda p: p[0][0][0] == p[1][0][0]),
                "same_top10": share(lambda p: {r[0] for r in p[0]} == {r[0] for r in p[1]}),
                "same_route": share(lambda p: (p[0][0][1] >= args.theta)
                                    == (p[1][0][1] >= args.theta)),
            }), flush=True)


if __name__ == "__main__":
    main()
