#!/usr/bin/env bash
# Cross-agent reuse (Table 4): a consumer reads the store of a producer of the other agent type, on the same
# tasks and application. Gemma reads the policy's store, with the plan shown and hidden; each policy reads Gemma's
# store. The "own" rows and the no-reuse baselines are those of same_application.sh.
set -euo pipefail
cd "$(dirname "$0")/.."; source experiments/common.sh
OUT="$RESULTS/cross_agent"; mkdir -p "$OUT"
for pair in "alfworld dagger" "scienceworld imitation"; do
  set -- $pair; app=$1 policy=$2
  for mode in whole subtraces cascade; do
    $RUN --application "$app" --consumer gemma --mode "$mode" --store "$STORES/${app}_${policy}.jsonl" \
      --producer "$(producer "$policy")" --out "$OUT/${app}_gemma_plan_from_${policy}_${mode}.jsonl"
    $RUN --application "$app" --consumer gemma --mode "$mode" --store "$STORES/${app}_${policy}.jsonl" \
      --producer "$(producer "$policy")" --hide-plan --out "$OUT/${app}_gemma_noplan_from_${policy}_${mode}.jsonl"
    $RUN --application "$app" --consumer "$policy" --mode "$mode" --store "$STORES/${app}_gemma.jsonl" \
      --producer "$(producer gemma)" --out "$OUT/${app}_${policy}_from_gemma_${mode}.jsonl"
  done
done
$PY -m expreuse.experiments.summarize "$OUT"/*.jsonl --csv "$OUT/summary.csv"
