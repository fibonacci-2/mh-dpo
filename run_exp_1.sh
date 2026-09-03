#!/usr/bin/env bash
# Experiment 1 (RQ1): does English DPO preference-tuning in
# Psychotherapy-LLM/PsyCoPref-Llama3-8B transfer to Arabic CBT-style
# responses, evaluated on the Shifaa dataset?
#
# Usage: ./run_exp_1.sh
# Override defaults via env vars, e.g.: N=100 ./run_exp_1.sh
# Everything (all 4 steps) is logged to a single file: logs/exp_1.log
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ~/code/venv/bin/activate

N="${N:-50}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

QUESTIONS=data/sampled_questions_shifaa.csv
GEN_BASE=results/gen_base_shifaa.csv
GEN_DPO=results/gen_dpo_shifaa.csv
JUDGE_PAIRWISE=results/judge_pairwise_shifaa.csv
JUDGE_ABSOLUTE=results/judge_absolute_shifaa.csv
SUMMARY=results/summary_shifaa.md

mkdir -p logs
LOG_FILE=logs/exp_1.log
: > "$LOG_FILE"  # start a fresh log for this run

echo "== 1/4: sampling questions =="
python3 src/prepare_data.py --dataset shifaa --n "$N" --seed "$SEED" --output "$QUESTIONS" \
  --log-file "$LOG_FILE"

echo "== 2/4: generating base-model responses =="
python3 src/generate.py --model base --questions "$QUESTIONS" \
  --out "$GEN_BASE" --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" \
  --log-file "$LOG_FILE"

echo "== 3/4: generating DPO-model responses =="
python3 src/generate.py --model dpo --questions "$QUESTIONS" \
  --out "$GEN_DPO" --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" \
  --log-file "$LOG_FILE"

echo "== 4/4: judging + summarizing =="
python3 src/judge.py --base "$GEN_BASE" --dpo "$GEN_DPO" \
  --pairwise-out "$JUDGE_PAIRWISE" --absolute-out "$JUDGE_ABSOLUTE" --seed "$SEED" \
  --log-file "$LOG_FILE"

python3 src/analyze.py --pairwise "$JUDGE_PAIRWISE" --absolute "$JUDGE_ABSOLUTE" --out "$SUMMARY" \
  --log-file "$LOG_FILE"

echo "Done. Summary written to $SUMMARY. Full log: $LOG_FILE"
