#!/usr/bin/env bash
# Sensitivity and ablation (Section 4.6) on the same-application setting: Cascade's threshold (Table 7),
# retrieval depth K, partial subtrace coverage (Figure 4), and grounding on/off. Reads config.THETA_LEVELS,
# config.K_LEVELS and config.COVERAGE_LEVELS. Run same_application.sh first: the threshold sweep of the policies
# and the grounded side of the grounding comparison come from its results.
set -euo pipefail
cd "$(dirname "$0")/.."; source experiments/common.sh
SAME="$RESULTS/same_application"; OUT="$RESULTS/ablations"; mkdir -p "$OUT"
levels() { $PY -c "from expreuse import config; print(*config.$1)"; }
plan_flag() { [[ $1 == noplan ]] && echo "--hide-plan" || true; }

# Threshold theta. A policy is deterministic, so its Cascade at any theta is assembled exactly from its Whole,
# Subtraces and no-reuse runs; a language model plays Cascade at every theta.
for pair in "alfworld dagger" "scienceworld imitation"; do
  set -- $pair; app=$1 consumer=$2; base="$SAME/${app}_${consumer}"
  $PY -m expreuse.experiments.theta_sweep --whole "${base}_whole.jsonl" --subtraces "${base}_subtraces.jsonl" \
    --none "${base}_none.jsonl" | tee "$OUT/theta_${app}_${consumer}.txt"
done
for app in alfworld scienceworld; do
  for plan in plan noplan; do
    for theta in $(levels THETA_LEVELS); do
      $RUN --application "$app" --consumer gemma --mode cascade --theta "$theta" $(plan_flag $plan) \
        --store "$STORES/${app}_gemma.jsonl" --out "$OUT/theta_${app}_gemma_${plan}_${theta}.jsonl"
    done
  done
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

# Partial coverage: withhold a share of the Subtraces answer. ALFWorld answers are short, so every subset of that
# size is played and averaged; ScienceWorld answers are long, so one nested random ordering per task is used.
for pair in "alfworld dagger all" "scienceworld imitation nested"; do
  set -- $pair; app=$1 policy=$2 schedule=$3
  for f in $(levels COVERAGE_LEVELS); do
    for plan in plan noplan; do
      $RUN --application "$app" --consumer gemma --mode subtraces --store "$STORES/${app}_gemma.jsonl" \
        $(plan_flag $plan) --withhold "$f" --withhold-schedule "$schedule" \
        --out "$OUT/coverage_${app}_gemma_${plan}_${f}.jsonl"
    done
    $RUN --application "$app" --consumer "$policy" --mode subtraces --store "$STORES/${app}_${policy}.jsonl" \
      --withhold "$f" --withhold-schedule "$schedule" --out "$OUT/coverage_${app}_${policy}_${f}.jsonl"
  done
done

# Grounding off on ALFWorld (the grounded runs are those of same_application.sh)
for mode in whole subtraces cascade; do
  for consumer in gemma ministral; do
    for plan in plan noplan; do
      $RUN --application alfworld --consumer "$consumer" --mode "$mode" --store "$STORES/alfworld_${consumer}.jsonl" \
        $(plan_flag $plan) --no-ground --out "$OUT/ungrounded_alfworld_${consumer}_${plan}_${mode}.jsonl"
    done
  done
  $RUN --application alfworld --consumer dagger --mode "$mode" --store "$STORES/alfworld_dagger.jsonl" \
    --no-ground --out "$OUT/ungrounded_alfworld_dagger_${mode}.jsonl"
done
$PY -m expreuse.experiments.summarize "$OUT"/*.jsonl --csv "$OUT/summary.csv"
