#!/usr/bin/env bash
# Experiment 4A (see README.md's "Experiment 4 A: Train on English test on
# English"): the same CBT-DP-DPO objective as ex4B (run_exp_4.sh) -- DPO with
# a per-example clinical-risk margin Delta_DP(m) and a per-example
# skill-deficit loss weight w_k(x) -- but isolating the objective's effect
# from any cross-lingual transfer question. Two differences from ex4B:
#
#   1. Trains ONLY on the TRAIN split of the English CBT-Bench negatives
#      (results/ex3-negatives/dpo_negatives_cbtbench_verified_TRAINSPLIT.csv,
#      124 of 156 exercises), not the full 624-row file ex4B trains on.
#      This matters here in a way it doesn't for ex4B: ex4B tests on Arabic
#      (Shifaa/MentalQA), data the model never saw in training regardless of
#      which English rows it trained on. ex4A tests on the ENGLISH held-out
#      exercises -- if we trained on all 156 and then "tested" on 32 of them,
#      that's leakage, not a real test. So ex4A needs its own, separately
#      trained adapter, restricted to the other 124 exercises.
#   2. Tests on the English CBT-DP test split's 32 client statements
#      (data/cbtdp_test_clients_en.csv) instead of Arabic data, and both
#      generation (src/generate.py --lang en) and judging (src/judge.py
#      --lang en) run in English rather than Arabic.
#
# The 4-principle CBT rubric (step 6) is unchanged from ex4B -- same 4
# principles, same skill definitions, just judged in English against English
# responses (see src/judge.py's PRINCIPLES_EN).
#
# Like run_exp_4.sh, every run is namespaced by a required RUN_TAG. If ex4B's
# train-split adapter already exists (e.g. you're also testing this same
# split's Arabic translation separately), set SKIP_TRAIN=1 and
# CBTDP_DIR_OVERRIDE to reuse it here instead of retraining -- ex4A and any
# such Arabic-translation variant can share one train-split-only adapter,
# since it's the same underlying training run either way.
#
# Usage: RUN_TAG=<name> ./run_exp_4a.sh
# Override other defaults via env vars, e.g.: RUN_TAG=trainsplit TRAIN_EPOCHS=2 ./run_exp_4a.sh
# Everything (all 6 steps, or 4 with SKIP_TRAIN) is logged to a single file:
# logs/exp_4a-<RUN_TAG>.log
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
# Only source the default venv if one isn't already active and that path
# actually exists (see run_exp_4.sh for why this is conditional).
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ~/code/venv/bin/activate ]; then
  source ~/code/venv/bin/activate
fi

RUN_TAG="${RUN_TAG:-}"
if [ -z "$RUN_TAG" ]; then
  echo "ERROR: RUN_TAG is required, so each run's data/model/results stay separate." >&2
  echo "Usage: RUN_TAG=<name> [SKILL_WEIGHTS='{...}'] ./run_exp_4a.sh" >&2
  exit 1
fi

BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SEED="${SEED:-42}"

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

# TRAIN split only -- see header comment for why this must differ from
# ex4B's NEGATIVES (which uses the full, unsplit file).
NEGATIVES="${NEGATIVES:-results/ex3-negatives/dpo_negatives_cbtbench_verified_TRAINSPLIT.csv}"
CBTDP_TRAIN="data/cbtdp_dpo_train-${RUN_TAG}.csv"
CBTDP_DIR="${CBTDP_DIR_OVERRIDE:-models/cbtdp-dpo-llama3.1-8b-lora-${RUN_TAG}}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
if [ "$SKIP_TRAIN" = "1" ] && [ ! -d "$CBTDP_DIR" ]; then
  echo "ERROR: SKIP_TRAIN=1 but CBTDP_DIR does not exist: $CBTDP_DIR" >&2
  echo "Set CBTDP_DIR_OVERRIDE to an already-trained adapter directory." >&2
  exit 1
fi

# The English test split's 32 unique client statements (held out from
# NEGATIVES above -- see results/ex3-negatives/*_TESTSPLIT.csv for the full
# rejected/chosen pairs this was deduplicated from).
QUESTIONS="${QUESTIONS:-data/cbtdp_test_clients_en.csv}"

JUDGE_MODEL="${JUDGE_MODEL:-}"

RESULTS_DIR="results/ex4a-cbtdp-dpo-${RUN_TAG}"
GEN_DPO="$RESULTS_DIR/gen_dpo_en.csv"
GEN_CBTDP="$RESULTS_DIR/gen_cbtdp_en.csv"
JUDGE_PAIRWISE="$RESULTS_DIR/judge_pairwise_en.csv"
JUDGE_ABSOLUTE="$RESULTS_DIR/judge_absolute_en.csv"
SUMMARY="$RESULTS_DIR/summary_en.md"

mkdir -p logs "$RESULTS_DIR"
LOG_FILE="logs/exp_4a-${RUN_TAG}.log"
: > "$LOG_FILE"

STEPS=6
[ "$SKIP_TRAIN" = "1" ] && STEPS=4

echo "== ex4A RUN_TAG=$RUN_TAG, SKILL_WEIGHTS=$SKILL_WEIGHTS, SKIP_TRAIN=$SKIP_TRAIN, JUDGE_MODEL=${JUDGE_MODEL:-<judge.py default>} =="
echo "== training data: $NEGATIVES (train-split only) =="
echo "== test questions: $QUESTIONS (English, held-out test split) =="

if [ "$SKIP_TRAIN" = "1" ]; then
  echo "== (1-2)/$STEPS: skipped -- reusing already-trained adapter at $CBTDP_DIR =="
else
  echo "== 1/$STEPS: preparing CBT-DP-DPO training data from the TRAIN split =="
  python3 src/prepare_cbtdp_data.py --negatives "$NEGATIVES" --output "$CBTDP_TRAIN" \
    --skill-weights "$SKILL_WEIGHTS" --delta-base "$DELTA_BASE" --gamma "$GAMMA" \
    --log-file "$LOG_FILE"

  echo "== 2/$STEPS: training CBT-DP-DPO model (LoRA), train-split only =="
  python3 src/train_cbtdp_dpo.py --train-data "$CBTDP_TRAIN" --output-dir "$CBTDP_DIR" \
    --epochs "$TRAIN_EPOCHS" --lr "$TRAIN_LR" --beta "$TRAIN_BETA" \
    --batch-size "$TRAIN_BATCH_SIZE" --grad-accum-steps "$GRAD_ACCUM_STEPS" \
    --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --seed "$SEED" \
    --log-file "$LOG_FILE"
fi

echo "== 3/$STEPS: generating DPO-model (baseline) responses, English =="
python3 src/generate.py --model dpo --lang en --questions "$QUESTIONS" \
  --out "$GEN_DPO" --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" \
  --log-file "$LOG_FILE"

echo "== 4/$STEPS: generating CBT-DP-DPO-model responses, English =="
python3 src/generate.py --model cbtdp --lang en --adapter-path "$CBTDP_DIR" --questions "$QUESTIONS" \
  --out "$GEN_CBTDP" --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" \
  --log-file "$LOG_FILE"

echo "== 5/$STEPS: judging (cbtdp vs dpo), English, 4-principle CBT rubric =="
JUDGE_MODEL_ARGS=()
[ -n "$JUDGE_MODEL" ] && JUDGE_MODEL_ARGS=(--judge-model "$JUDGE_MODEL")
python3 src/judge.py --a-file "$GEN_DPO" --a-name dpo --b-file "$GEN_CBTDP" --b-name cbtdp \
  --lang en --pairwise-out "$JUDGE_PAIRWISE" --absolute-out "$JUDGE_ABSOLUTE" --seed "$SEED" \
  "${JUDGE_MODEL_ARGS[@]}" --log-file "$LOG_FILE"

echo "== 6/$STEPS: summarizing =="
python3 src/analyze.py --pairwise "$JUDGE_PAIRWISE" --absolute "$JUDGE_ABSOLUTE" --out "$SUMMARY" \
  --log-file "$LOG_FILE"

echo "Done. Summary written to $SUMMARY. Full log: $LOG_FILE"
