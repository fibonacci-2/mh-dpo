#!/usr/bin/env bash
# Experiment 2 (RQ2): does Multi-Objective DPO (MODPO), trained on the
# PsyCoPref preference dataset using its 7 rated principles as auxiliary
# objectives, produce better-balanced Arabic CBT-style responses than the
# existing single-objective DPO model (Psychotherapy-LLM/PsyCoPref-Llama3-8B)?
#
# Usage: ./run_exp_2.sh
# Override defaults via env vars, e.g.: TRAIN_EPOCHS=2 ./run_exp_2.sh
# Everything (all 7 steps) is logged to a single file: logs/exp_2.log
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ~/code/venv/bin/activate

# -- question sampling + generation knobs (same defaults as run_exp_1.sh, for
#    an apples-to-apples comparison against the ex1 dpo/base generations) --
N="${N:-50}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

# -- MODPO training knobs --
AUX_WEIGHT="${AUX_WEIGHT:-0.5}"        # total weight on the 7 auxiliary principles; remainder goes to the primary chosen/rejected term
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
TRAIN_LR="${TRAIN_LR:-5e-6}"
TRAIN_BETA="${TRAIN_BETA:-0.1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

QUESTIONS=data/sampled_questions_shifaa.csv
PSYCOPREF_TRAIN=data/psycopref_train.csv
MODPO_DIR=models/modpo-llama3.1-8b-lora

RESULTS_DIR=results/ex2-modpo
GEN_DPO="$RESULTS_DIR/gen_dpo_shifaa.csv"
GEN_MODPO="$RESULTS_DIR/gen_modpo_shifaa.csv"
JUDGE_PAIRWISE="$RESULTS_DIR/judge_pairwise_shifaa.csv"
JUDGE_ABSOLUTE="$RESULTS_DIR/judge_absolute_shifaa.csv"
SUMMARY="$RESULTS_DIR/summary_shifaa.md"

mkdir -p logs "$RESULTS_DIR"
LOG_FILE=logs/exp_2.log
: > "$LOG_FILE"  # start a fresh log for this run

echo "== 1/7: sampling questions =="
python3 src/prepare_data.py --dataset shifaa --n "$N" --seed "$SEED" --output "$QUESTIONS" \
  --log-file "$LOG_FILE"

echo "== 2/7: preparing PsyCoPref MODPO training data =="
python3 src/prepare_modpo_data.py --output "$PSYCOPREF_TRAIN" \
  --log-file "$LOG_FILE"

echo "== 3/7: training MODPO model (LoRA) =="
python3 src/train_modpo.py --train-data "$PSYCOPREF_TRAIN" --output-dir "$MODPO_DIR" \
  --aux-weight "$AUX_WEIGHT" --epochs "$TRAIN_EPOCHS" --lr "$TRAIN_LR" --beta "$TRAIN_BETA" \
  --batch-size "$TRAIN_BATCH_SIZE" --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --seed "$SEED" \
  --log-file "$LOG_FILE"

echo "== 4/7: generating DPO-model responses =="
python3 src/generate.py --model dpo --questions "$QUESTIONS" \
  --out "$GEN_DPO" --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" \
  --log-file "$LOG_FILE"

echo "== 5/7: generating MODPO-model responses =="
python3 src/generate.py --model modpo --adapter-path "$MODPO_DIR" --questions "$QUESTIONS" \
  --out "$GEN_MODPO" --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" \
  --log-file "$LOG_FILE"

echo "== 6/7: judging (modpo vs dpo) =="
python3 src/judge.py --a-file "$GEN_DPO" --a-name dpo --b-file "$GEN_MODPO" --b-name modpo \
  --pairwise-out "$JUDGE_PAIRWISE" --absolute-out "$JUDGE_ABSOLUTE" --seed "$SEED" \
  --log-file "$LOG_FILE"

echo "== 7/7: summarizing =="
python3 src/analyze.py --pairwise "$JUDGE_PAIRWISE" --absolute "$JUDGE_ABSOLUTE" --out "$SUMMARY" \
  --log-file "$LOG_FILE"

echo "Done. Summary written to $SUMMARY. Full log: $LOG_FILE"
