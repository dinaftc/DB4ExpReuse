#!/usr/bin/env bash
# Latency, concurrency and approximate-search benchmarks. Set TRACES to start from another trace file.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}; STORES=${STORES:-stores}; RESULTS=${RESULTS:-results}
DSN=${DSN:-postgresql:///expreuse}
TRACES=${TRACES:-$STORES/alfworld_expert.jsonl}
OUT="$RESULTS/systems"; mkdir -p "$OUT"
if [[ $TRACES == "$STORES/alfworld_expert.jsonl" ]]; then
  $PY -m expreuse.experiments.reproduce expert_store --stores "$STORES" --results "$RESULTS"
fi
base=(--embedder mpnet --traces "$TRACES")
$PY -m expreuse.bench.latency --realization relational,jsonb,qdrant --dsn "$DSN" ${QDRANT_URL:+--url "$QDRANT_URL"} \
  "${base[@]}" --sizes 2000,13000,100000,1000000 > "$OUT/latency.jsonl"
$PY -m expreuse.bench.concurrency --dsn "$DSN" "${base[@]}" --clients 1,4,16,32,64 > "$OUT/concurrency_query.jsonl"
$PY -m expreuse.bench.concurrency --dsn "$DSN" "${base[@]}" --clients 1,4,16,32,64 --insert-fraction 0.1 \
  > "$OUT/concurrency_mixed.jsonl"
$PY -m expreuse.bench.ann --dsn "$DSN" "${base[@]}" --goals 1000000 --ef 40,400 > "$OUT/ann.jsonl"
