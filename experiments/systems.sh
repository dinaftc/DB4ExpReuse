#!/usr/bin/env bash
# Physical realizations: retrieval and insertion latency as the store grows, throughput under concurrent clients,
# and approximate goal search (Section 4.5). The base store is the traces ALFWorld's handcoded expert leaves on the
# public training games (about 3.6K traces), collected here once; it is then grown by copying traces to each size
# (bench.common.scaled). Set TRACES to start from another trace file.
set -euo pipefail
cd "$(dirname "$0")/.."; source experiments/common.sh
DSN=${DSN:-postgresql:///expreuse}
TRACES=${TRACES:-$STORES/alfworld_expert.jsonl}
OUT="$RESULTS/systems"; mkdir -p "$OUT"
if [[ $TRACES == "$STORES/alfworld_expert.jsonl" ]]; then  # resumes if interrupted; the plan needs no bindings here
  $RUN --application alfworld --consumer expert --mode none --tasks all --bindings none \
    --store-as --store-out "$TRACES" --out "$RESULTS/collect/alfworld_expert.jsonl"
fi
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
$PY -m expreuse.bench.ann --dsn "$DSN" "${base[@]}" --goals 1000000 --ef 40,400 > "$OUT/ann.jsonl"
