#!/usr/bin/env bash
# Producers' stores: every producer solves the store task list once, without experience and without being shown
# the task plan (the plan is only used to segment the trace it leaves into subtraces). One file per
# (application, producer).
set -euo pipefail
cd "$(dirname "$0")/.."; source experiments/common.sh
for app in alfworld scienceworld; do
  for consumer in gemma ministral; do
    $RUN --application "$app" --consumer "$consumer" --mode none --hide-plan --runs 1 --tasks "data/tasks/${app}_store.txt" \
      --store-as --store-out "$STORES/${app}_${consumer}.jsonl" --out "$RESULTS/collect/${app}_${consumer}.jsonl"
  done
done
$RUN --application alfworld --consumer dagger --mode none --tasks data/tasks/alfworld_store.txt \
  --store-as --store-out "$STORES/alfworld_dagger.jsonl" --out "$RESULTS/collect/alfworld_dagger.jsonl"
$RUN --application scienceworld --consumer imitation --mode none --tasks data/tasks/scienceworld_store.txt \
  --store-as --store-out "$STORES/scienceworld_imitation.jsonl" --out "$RESULTS/collect/scienceworld_imitation.jsonl"
