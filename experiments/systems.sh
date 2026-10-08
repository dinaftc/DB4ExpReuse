#!/usr/bin/env bash
# Physical realizations: retrieval and insertion latency as the store grows, throughput under concurrent clients,
# and approximate goal search. The store is a producer's trace file grown synthetically to each size
# (bench.common.grow); without TRACES the benchmarks run on synthetic traces.
set -euo pipefail
cd "$(dirname "$0")/.."; source experiments/common.sh
DSN=${DSN:-postgresql:///expreuse}
TRACES=${TRACES:-$STORES/alfworld_gemma.jsonl}
OUT="$RESULTS/systems"; mkdir -p "$OUT"
base=(--embedder mpnet ${TRACES:+--traces "$TRACES"})
for realization in relational jsonb; do
  $PY -m expreuse.bench.latency --realization "$realization" --dsn "$DSN" "${base[@]}" \
    --sizes 2000,13000,100000,1000000 > "$OUT/latency_${realization}.jsonl"
done
$PY -m expreuse.bench.latency --realization qdrant ${QDRANT_URL:+--url "$QDRANT_URL"} "${base[@]}" \
  --sizes 2000,13000,100000,1000000 > "$OUT/latency_qdrant.jsonl"
$PY -m expreuse.bench.concurrency --dsn "$DSN" "${base[@]}" --clients 1,4,16,32,64 > "$OUT/concurrency_query.jsonl"
$PY -m expreuse.bench.concurrency --dsn "$DSN" "${base[@]}" --clients 1,4,16,32,64 --insert-fraction 0.1 \
  > "$OUT/concurrency_mixed.jsonl"
$PY -m expreuse.bench.ann --dsn "$DSN" "${base[@]}" --goals 100000 --ef 40,400 > "$OUT/ann.jsonl"
