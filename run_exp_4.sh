#!/usr/bin/env bash
# Experiment 4B (ex4 / "Experiment 4 B" in README.md): train a CBT-DP-DPO
# LoRA adapter -- DPO with a per-example clinical-risk margin Delta_DP(m) and
# a per-example skill-deficit loss weight w_k(x), computed from the ex3
# ablated-negative dataset -- on English CBT-Bench data only, then evaluate
# zero-shot transfer to Arabic (shifaa) against the existing single-objective
# DPO model, the same way ex2 compares modpo against dpo.
#
# Judging (step 6) uses src/judge.py's 4-principle CBT rubric (one principle
# per skill our ex3 negatives ablate: SQ, EV, CR-realistic-reframe,
# CR-distortion-challenge) -- NOT the original 7-item generic PsychoCounsel
# rubric -- and defaults to judge.py's own default judge model. Override with
# JUDGE_MODEL to judge with a different model, e.g. to compare judges:
#   JUDGE_MODEL=NousResearch/Meta-Llama-3.1-8B-Instruct
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
# To re-judge an ALREADY-TRAINED adapter under the new 4-principle rubric (or
# a new JUDGE_MODEL) without spending GPU time retraining it, set SKIP_TRAIN=1
# and point CBTDP_DIR_OVERRIDE at the existing adapter directory; results
# still land under the new RUN_TAG, so the old judge's results are kept
# untouched for comparison, e.g. to re-judge the "finegrained" run:
#   RUN_TAG=finegrained-4principle SKIP_TRAIN=1 \
#     CBTDP_DIR_OVERRIDE=models/cbtdp-dpo-llama3.1-8b-lora-finegrained \
#     JUDGE_MODEL=NousResearch/Meta-Llama-3.1-8B-Instruct ./run_exp_4.sh
#
# Usage: RUN_TAG=<name> ./run_exp_4.sh
# Override other defaults via env vars, e.g.: RUN_TAG=finegrained TRAIN_EPOCHS=2 ./run_exp_4.sh
# Everything (all 7 steps, or 5 with SKIP_TRAIN) is logged to a single file:
# logs/exp_4-<RUN_TAG>.log
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
# Only source the default venv if one isn't already active and that path
# actually exists -- on machines where the venv lives elsewhere (or is
# already activated before this script runs, e.g. a Slurm/interactive job
# that starts inside its own venv), the old unconditional `source` here
# would hard-fail the whole script under `set -e` before doing anything.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ~/code/venv/bin/activate ]; then
  source ~/code/venv/bin/activate
fi

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
# CBTDP_DIR_OVERRIDE lets a re-judging run point at an already-trained
# adapter (see "SKIP_TRAIN" above) instead of the RUN_TAG-derived path.
CBTDP_DIR="${CBTDP_DIR_OVERRIDE:-models/cbtdp-dpo-llama3.1-8b-lora-${RUN_TAG}}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
if [ "$SKIP_TRAIN" = "1" ] && [ ! -d "$CBTDP_DIR" ]; then
  echo "ERROR: SKIP_TRAIN=1 but CBTDP_DIR does not exist: $CBTDP_DIR" >&2
  echo "Set CBTDP_DIR_OVERRIDE to an already-trained adapter directory." >&2
  exit 1
fi
# Judge model for step 6 (see src/judge.py --judge-model). Empty means "use
# judge.py's own default", so existing invocations are unaffected.
JUDGE_MODEL="${JUDGE_MODEL:-}"

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

STEPS=7
[ "$SKIP_TRAIN" = "1" ] && STEPS=5

echo "== RUN_TAG=$RUN_TAG, SKILL_WEIGHTS=$SKILL_WEIGHTS, SKIP_TRAIN=$SKIP_TRAIN, JUDGE_MODEL=${JUDGE_MODEL:-<judge.py default>} =="

if [ "$SKIP_TRAIN" = "1" ]; then
  echo "== (1-2)/$STEPS: skipped -- reusing already-trained adapter at $CBTDP_DIR =="
else
  echo "== 1/$STEPS: preparing CBT-DP-DPO training data from ex3 negatives =="
  python3 src/prepare_cbtdp_data.py --negatives "$NEGATIVES" --output "$CBTDP_TRAIN" \
    --skill-weights "$SKILL_WEIGHTS" --delta-base "$DELTA_BASE" --gamma "$GAMMA" \
    --log-file "$LOG_FILE"

  echo "== 2/$STEPS: training CBT-DP-DPO model (LoRA) =="
  python3 src/train_cbtdp_dpo.py --train-data "$CBTDP_TRAIN" --output-dir "$CBTDP_DIR" \
    --epochs "$TRAIN_EPOCHS" --lr "$TRAIN_LR" --beta "$TRAIN_BETA" \
    --batch-size "$TRAIN_BATCH_SIZE" --grad-accum-steps "$GRAD_ACCUM_STEPS" \
    --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --seed "$SEED" \
    --log-file "$LOG_FILE"
fi

echo "== 3/$STEPS: sampling Arabic (shifaa) questions =="
python3 src/prepare_data.py --dataset shifaa --n "$N" --seed "$SEED" --output "$QUESTIONS" \
  --log-file "$LOG_FILE"

echo "== 4/$STEPS: generating DPO-model responses =="
python3 src/generate.py --model dpo --questions "$QUESTIONS" \
  --out "$GEN_DPO" --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" \
  --log-file "$LOG_FILE"

echo "== 5/$STEPS: generating CBT-DP-DPO-model responses =="
python3 src/generate.py --model cbtdp --adapter-path "$CBTDP_DIR" --questions "$QUESTIONS" \
  --out "$GEN_CBTDP" --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" \
  --log-file "$LOG_FILE"

echo "== 6/$STEPS: judging (cbtdp vs dpo) with the 4-principle CBT rubric =="
JUDGE_MODEL_ARGS=()
[ -n "$JUDGE_MODEL" ] && JUDGE_MODEL_ARGS=(--judge-model "$JUDGE_MODEL")
python3 src/judge.py --a-file "$GEN_DPO" --a-name dpo --b-file "$GEN_CBTDP" --b-name cbtdp \
  --pairwise-out "$JUDGE_PAIRWISE" --absolute-out "$JUDGE_ABSOLUTE" --seed "$SEED" \
  "${JUDGE_MODEL_ARGS[@]}" --log-file "$LOG_FILE"

echo "== 7/$STEPS: summarizing =="
python3 src/analyze.py --pairwise "$JUDGE_PAIRWISE" --absolute "$JUDGE_ABSOLUTE" --out "$SUMMARY" \
  --log-file "$LOG_FILE"

echo "Done. Summary written to $SUMMARY. Full log: $LOG_FILE"
