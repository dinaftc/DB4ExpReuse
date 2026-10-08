#!/usr/bin/env bash
# Ablations on the same-application setting: Cascade's threshold, retrieval depth K, partial subtrace coverage,
# and grounding on/off. Reads config.K_LEVELS, config.COVERAGE_LEVELS and config.THETA_LEVELS.
set -euo pipefail
cd "$(dirname "$0")/.."; source experiments/common.sh
SAME="$RESULTS/same_application"; OUT="$RESULTS/ablations"
levels() { $PY -c "from expreuse import config; print(*config.$1)"; }

# Threshold theta: assembled from the Whole, Subtraces and no-reuse conditions (exact for the policies)
for pair in "alfworld dagger" "scienceworld imitation" "alfworld gemma" "scienceworld gemma"; do
  set -- $pair; app=$1 consumer=$2
  base="$SAME/${app}_${consumer}"; [[ $consumer == gemma || $consumer == ministral ]] && base="${base}_plan"
  $PY -m expreuse.experiments.theta_sweep --whole "${base}_whole.jsonl" --subtraces "${base}_subtraces.jsonl" \
    --none "${base}_none.jsonl" | tee "$OUT/theta_${app}_${consumer}.txt"
done

# Retrieval depth K
for k in $(levels "K_LEVELS['llm']"); do
  for mode in whole subtraces; do
    $RUN --application scienceworld --consumer ministral --mode "$mode" --k "$k" \
      --store "$STORES/scienceworld_ministral.jsonl" --out "$OUT/depth_scienceworld_ministral_${mode}_k${k}.jsonl"
  done
done
for k in $(levels "K_LEVELS['policy']"); do
  for mode in whole subtraces; do
    $RUN --application alfworld --consumer dagger --mode "$mode" --k "$k" \
      --store "$STORES/alfworld_dagger.jsonl" --out "$OUT/depth_alfworld_dagger_${mode}_k${k}.jsonl"
    $RUN --application scienceworld --consumer imitation --mode "$mode" --k "$k" \
      --store "$STORES/scienceworld_imitation.jsonl" --out "$OUT/depth_scienceworld_imitation_${mode}_k${k}.jsonl"
  done
done

# Partial coverage: withhold a share of the Subtraces answer. ALFWorld plans are short, so every subset of that
# size is played and averaged; ScienceWorld plans are long, so one nested random ordering per task is used.
for f in $(levels COVERAGE_LEVELS); do
  $RUN --application alfworld --consumer gemma --mode subtraces --store "$STORES/alfworld_gemma.jsonl" \
    --withhold "$f" --withhold-schedule all --out "$OUT/coverage_alfworld_gemma_${f}.jsonl"
  $RUN --application alfworld --consumer dagger --mode subtraces --store "$STORES/alfworld_dagger.jsonl" \
    --withhold "$f" --withhold-schedule all --out "$OUT/coverage_alfworld_dagger_${f}.jsonl"
  $RUN --application scienceworld --consumer gemma --mode subtraces --store "$STORES/scienceworld_gemma.jsonl" \
    --withhold "$f" --withhold-schedule nested --out "$OUT/coverage_scienceworld_gemma_${f}.jsonl"
  $RUN --application scienceworld --consumer imitation --mode subtraces --store "$STORES/scienceworld_imitation.jsonl" \
    --withhold "$f" --withhold-schedule nested --out "$OUT/coverage_scienceworld_imitation_${f}.jsonl"
done

# Grounding off (the grounded runs are those of same_application.sh)
for app in alfworld scienceworld; do
  for consumer in gemma ministral; do
    for mode in whole subtraces cascade; do
      $RUN --application "$app" --consumer "$consumer" --mode "$mode" --store "$STORES/${app}_${consumer}.jsonl" \
        --no-ground --out "$OUT/ungrounded_${app}_${consumer}_${mode}.jsonl"
    done
  done
done
$PY -m expreuse.experiments.summarize "$OUT"/*.jsonl --csv "$OUT/summary.csv"
