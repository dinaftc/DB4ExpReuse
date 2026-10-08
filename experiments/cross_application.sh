#!/usr/bin/env bash
# Cross-application reuse (Table 5) on the tasks whose plans contain operators both applications share (take,
# put, heat, cool): a consumer of one application reads traces of the other. Answers are shown as recorded (no
# grounding across applications). Whole and Subtraces only: no foreign trace reaches theta, so Cascade always
# falls back to Subtraces. Each line of `rows` is: application, consumer, store's application, store's producer.
set -euo pipefail
cd "$(dirname "$0")/.."; source experiments/common.sh
OUT="$RESULTS/cross_application"; mkdir -p "$OUT"
other() { [[ $1 == alfworld ]] && echo scienceworld || echo alfworld; }

# No-reuse baselines on the cross-application task lists
for app in alfworld scienceworld; do
  tasks="data/tasks/${app}_cross_application.txt"
  $RUN --application "$app" --consumer gemma --mode none --tasks "$tasks" --out "$OUT/${app}_gemma_plan_none.jsonl"
  $RUN --application "$app" --consumer gemma --mode none --hide-plan --tasks "$tasks" \
    --out "$OUT/${app}_gemma_noplan_none.jsonl"
done
$RUN --application alfworld --consumer dagger --mode none --tasks data/tasks/alfworld_cross_application.txt \
  --out "$OUT/alfworld_dagger_none.jsonl"
$RUN --application scienceworld --consumer imitation --mode none --tasks data/tasks/scienceworld_cross_application.txt \
  --out "$OUT/scienceworld_imitation_none.jsonl"

rows=(
  "alfworld gemma gemma"
  "alfworld dagger gemma"
  "alfworld dagger imitation"
  "scienceworld gemma gemma"
  "scienceworld imitation gemma"
  "scienceworld imitation dagger"
)
for row in "${rows[@]}"; do
  set -- $row; app=$1 consumer=$2 source=$3
  from=$(other "$app"); tasks="data/tasks/${app}_cross_application.txt"
  for mode in whole subtraces; do
    common=(--application "$app" --consumer "$consumer" --mode "$mode" --store "$STORES/${from}_${source}.jsonl"
            --cross-application "$from" --producer "$(producer "$source")" --tasks "$tasks")
    if [[ $consumer == gemma ]]; then
      $RUN "${common[@]}" --out "$OUT/${app}_gemma_plan_from_${from}_${source}_${mode}.jsonl"
      $RUN "${common[@]}" --hide-plan --out "$OUT/${app}_gemma_noplan_from_${from}_${source}_${mode}.jsonl"
    else
      $RUN "${common[@]}" --out "$OUT/${app}_${consumer}_from_${from}_${source}_${mode}.jsonl"
    fi
  done
done
$PY -m expreuse.experiments.summarize "$OUT"/*.jsonl --csv "$OUT/summary.csv"
