#!/usr/bin/env bash
# Cross-agent reuse: a consumer reads another producer's store (the query names that producer). Each line is
# consumer, store producer; the no-reuse baselines are those of same_application.sh.
set -euo pipefail
cd "$(dirname "$0")/.."; source experiments/common.sh
OUT="$RESULTS/cross_agent"
pairs=(
  "alfworld gemma ministral"
  "alfworld ministral gemma"
  "alfworld dagger gemma"
  "alfworld dagger ministral"
  "scienceworld gemma ministral"
  "scienceworld ministral gemma"
  "scienceworld imitation gemma"
  "scienceworld imitation ministral"
)
for pair in "${pairs[@]}"; do
  set -- $pair; app=$1 consumer=$2 source=$3
  for mode in whole subtraces cascade; do
    $RUN --application "$app" --consumer "$consumer" --mode "$mode" --store "$STORES/${app}_${source}.jsonl" \
      --producer "$(producer "$source")" --out "$OUT/${app}_${consumer}_from_${source}_${mode}.jsonl"
  done
done
$PY -m expreuse.experiments.summarize "$OUT"/*.jsonl --csv "$OUT/summary.csv"
