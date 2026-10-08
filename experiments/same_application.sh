#!/usr/bin/env bash
# Same-application reuse: each consumer reads its own store on the evaluation tasks, under the three queries and
# without reuse. Language models run with the plan shown and, in the no-plan setting, with the plan hidden and the
# answers left ungrounded; the plan still shapes the Subtraces query. Policies run once (deterministic).
set -euo pipefail
cd "$(dirname "$0")/.."; source experiments/common.sh
OUT="$RESULTS/same_application"
for app in alfworld scienceworld; do
  for consumer in gemma ministral; do
    for mode in none whole subtraces cascade; do
      $RUN --application "$app" --consumer "$consumer" --mode "$mode" --store "$STORES/${app}_${consumer}.jsonl" \
        --out "$OUT/${app}_${consumer}_plan_${mode}.jsonl"
      $RUN --application "$app" --consumer "$consumer" --mode "$mode" --store "$STORES/${app}_${consumer}.jsonl" \
        --hide-plan --no-ground --out "$OUT/${app}_${consumer}_noplan_${mode}.jsonl"
    done
  done
done
for mode in none whole subtraces cascade; do
  $RUN --application alfworld --consumer dagger --mode "$mode" --store "$STORES/alfworld_dagger.jsonl" \
    --out "$OUT/alfworld_dagger_${mode}.jsonl"
  $RUN --application scienceworld --consumer imitation --mode "$mode" --store "$STORES/scienceworld_imitation.jsonl" \
    --out "$OUT/scienceworld_imitation_${mode}.jsonl"
done
$PY -m expreuse.experiments.summarize "$OUT"/*.jsonl --csv "$OUT/summary.csv"
