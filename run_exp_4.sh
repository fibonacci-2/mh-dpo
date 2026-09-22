#!/usr/bin/env bash
# Experiment 4 (ex4 / "Experiment A" in outline.md): train a CBT-DP-DPO LoRA
# adapter -- DPO with a per-example clinical-risk margin Delta_DP(m) and a
# per-example skill-deficit loss weight w_k(x), computed from the ex3
# ablated-negative dataset -- on English CBT-Bench data only, then evaluate
# zero-shot transfer to Arabic (shifaa) against the existing single-objective
# DPO model, the same way ex2 compares modpo against dpo.
#
# w_k(x) defaults to 1.0 for every skill (no skill-based reweighting; see
# src/prepare_cbtdp_data.py) -- override with SKILL_WEIGHTS to turn it on
# once a real per-skill ErrorRate_base(k) is measured, e.g. the outline's
# worked example: SKILL_WEIGHTS='{"SQ":1.5,"EV":1.0,"CR":1.2}'.
#
# Every run is namespaced by a required RUN_TAG, so different SKILL_WEIGHTS
# configs (or any other rerun) never clobber each other's training
# data/model/results -- e.g. the already-committed flat-weights run lives at
# the unsuffixed results/ex4-cbtdp-dpo/ (commit "simdpo run 1"); a finegrained
# rerun with real per-skill weights would be:
#   RUN_TAG=finegrained SKILL_WEIGHTS='{"SQ":1.5,"EV":1.0,"CR":1.2}' ./run_exp_4.sh
# which writes to data/cbtdp_dpo_train-finegrained.csv,
# models/cbtdp-dpo-llama3.1-8b-lora-finegrained/,
# results/ex4-cbtdp-dpo-finegrained/, logs/exp_4-finegrained.log, leaving the
# flat run's outputs untouched.
#
# Usage: RUN_TAG=<name> ./run_exp_4.sh
# Override other defaults via env vars, e.g.: RUN_TAG=finegrained TRAIN_EPOCHS=2 ./run_exp_4.sh
# Everything (all 7 steps) is logged to a single file: logs/exp_4-<RUN_TAG>.log
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ~/code/venv/bin/activate

RUN_TAG="${RUN_TAG:-}"
if [ -z "$RUN_TAG" ]; then
  echo "ERROR: RUN_TAG is required, so each run's data/model/results stay separate." >&2
  echo "Usage: RUN_TAG=<name> [SKILL_WEIGHTS='{...}'] ./run_exp_4.sh" >&2
  echo "e.g.:  RUN_TAG=finegrained SKILL_WEIGHTS='{\"SQ\":1.5,\"EV\":1.0,\"CR\":1.2}' ./run_exp_4.sh" >&2
  exit 1
fi

# -- question sampling + generation knobs (same defaults as run_exp_1/2.sh,
#    for an apples-to-apples comparison against the ex2 dpo generations) --
N="${N:-50}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

# -- CBT-DP-DPO data/training knobs --
# NOTE: don't rewrite this as SKILL_WEIGHTS="${SKILL_WEIGHTS:-{...}}" -- bash's
# brace-matching for ${VAR:-default} breaks when the default itself contains
# braces, silently appending a stray "}" onto an already-set SKILL_WEIGHTS.
if [ -z "${SKILL_WEIGHTS:-}" ]; then
  SKILL_WEIGHTS='{"SQ":1.0,"EV":1.0,"CR":1.0}'
fi
DELTA_BASE="${DELTA_BASE:-0.2}"
GAMMA="${GAMMA:-0.2}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
TRAIN_LR="${TRAIN_LR:-5e-6}"
TRAIN_BETA="${TRAIN_BETA:-0.1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

NEGATIVES=results/ex3-negatives/dpo_negatives_cbtbench_verified.csv
CBTDP_TRAIN="data/cbtdp_dpo_train-${RUN_TAG}.csv"
CBTDP_DIR="models/cbtdp-dpo-llama3.1-8b-lora-${RUN_TAG}"

# Shared across RUN_TAGs on purpose: every variant is evaluated on the same
# sampled Arabic questions, so results stay comparable to each other (and to
# ex2's dpo-on-shifaa run) rather than each re-sampling its own set.
QUESTIONS=data/sampled_questions_shifaa.csv

RESULTS_DIR="results/ex4-cbtdp-dpo-${RUN_TAG}"
GEN_DPO="$RESULTS_DIR/gen_dpo_shifaa.csv"
GEN_CBTDP="$RESULTS_DIR/gen_cbtdp_shifaa.csv"
JUDGE_PAIRWISE="$RESULTS_DIR/judge_pairwise_shifaa.csv"
JUDGE_ABSOLUTE="$RESULTS_DIR/judge_absolute_shifaa.csv"
SUMMARY="$RESULTS_DIR/summary_shifaa.md"

mkdir -p logs "$RESULTS_DIR"
LOG_FILE="logs/exp_4-${RUN_TAG}.log"
: > "$LOG_FILE"  # start a fresh log for this run

echo "== RUN_TAG=$RUN_TAG, SKILL_WEIGHTS=$SKILL_WEIGHTS =="
echo "== 1/7: preparing CBT-DP-DPO training data from ex3 negatives =="
python3 src/prepare_cbtdp_data.py --negatives "$NEGATIVES" --output "$CBTDP_TRAIN" \
  --skill-weights "$SKILL_WEIGHTS" --delta-base "$DELTA_BASE" --gamma "$GAMMA" \
  --log-file "$LOG_FILE"

echo "== 2/7: training CBT-DP-DPO model (LoRA) =="
python3 src/train_cbtdp_dpo.py --train-data "$CBTDP_TRAIN" --output-dir "$CBTDP_DIR" \
  --epochs "$TRAIN_EPOCHS" --lr "$TRAIN_LR" --beta "$TRAIN_BETA" \
  --batch-size "$TRAIN_BATCH_SIZE" --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --seed "$SEED" \
  --log-file "$LOG_FILE"

echo "== 3/7: sampling Arabic (shifaa) questions =="
python3 src/prepare_data.py --dataset shifaa --n "$N" --seed "$SEED" --output "$QUESTIONS" \
  --log-file "$LOG_FILE"

echo "== 4/7: generating DPO-model responses =="
python3 src/generate.py --model dpo --questions "$QUESTIONS" \
  --out "$GEN_DPO" --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" \
  --log-file "$LOG_FILE"

echo "== 5/7: generating CBT-DP-DPO-model responses =="
python3 src/generate.py --model cbtdp --adapter-path "$CBTDP_DIR" --questions "$QUESTIONS" \
  --out "$GEN_CBTDP" --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" \
  --log-file "$LOG_FILE"

echo "== 6/7: judging (cbtdp vs dpo) =="
python3 src/judge.py --a-file "$GEN_DPO" --a-name dpo --b-file "$GEN_CBTDP" --b-name cbtdp \
  --pairwise-out "$JUDGE_PAIRWISE" --absolute-out "$JUDGE_ABSOLUTE" --seed "$SEED" \
  --log-file "$LOG_FILE"

echo "== 7/7: summarizing =="
python3 src/analyze.py --pairwise "$JUDGE_PAIRWISE" --absolute "$JUDGE_ABSOLUTE" --out "$SUMMARY" \
  --log-file "$LOG_FILE"

echo "Done. Summary written to $SUMMARY. Full log: $LOG_FILE"
