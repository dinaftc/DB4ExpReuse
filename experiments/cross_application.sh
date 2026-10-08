#!/usr/bin/env bash
# Cross-application reuse on the tasks whose plans contain operators both applications share (take, put, heat,
# cool): an ALFWorld consumer reads ScienceWorld traces and the reverse. Answers are shown as recorded (no
# grounding across applications); the Cascade falls to Subtraces since no foreign trace reaches theta. The
# same-application answers on the same task lists are the reference.
set -euo pipefail
cd "$(dirname "$0")/.."; source experiments/common.sh
OUT="$RESULTS/cross_application"
for mode in none whole subtraces cascade; do
  # reference: own application, own store, on the cross-application task lists
  $RUN --application alfworld --consumer gemma --mode "$mode" --store "$STORES/alfworld_gemma.jsonl" \
    --tasks data/tasks/alfworld_cross_application.txt --out "$OUT/alfworld_gemma_own_${mode}.jsonl"
  $RUN --application scienceworld --consumer gemma --mode "$mode" --store "$STORES/scienceworld_gemma.jsonl" \
    --tasks data/tasks/scienceworld_cross_application.txt --out "$OUT/scienceworld_gemma_own_${mode}.jsonl"
  $RUN --application alfworld --consumer dagger --mode "$mode" --store "$STORES/alfworld_dagger.jsonl" \
    --tasks data/tasks/alfworld_cross_application.txt --out "$OUT/alfworld_dagger_own_${mode}.jsonl"
done
for mode in whole subtraces cascade; do
  $RUN --application alfworld --consumer gemma --mode "$mode" --store "$STORES/scienceworld_gemma.jsonl" \
    --cross-application scienceworld --tasks data/tasks/alfworld_cross_application.txt \
    --out "$OUT/alfworld_gemma_from_scienceworld_${mode}.jsonl"
  $RUN --application scienceworld --consumer gemma --mode "$mode" --store "$STORES/alfworld_gemma.jsonl" \
    --cross-application alfworld --tasks data/tasks/scienceworld_cross_application.txt \
    --out "$OUT/scienceworld_gemma_from_alfworld_${mode}.jsonl"
  for source in gemma imitation; do
    $RUN --application alfworld --consumer dagger --mode "$mode" --store "$STORES/scienceworld_${source}.jsonl" \
      --cross-application scienceworld --producer "$(producer "$source")" \
      --tasks data/tasks/alfworld_cross_application.txt --out "$OUT/alfworld_dagger_from_scienceworld_${source}_${mode}.jsonl"
  done
done
$PY -m expreuse.experiments.summarize "$OUT"/*.jsonl --csv "$OUT/summary.csv"
