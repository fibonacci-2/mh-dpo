#!/usr/bin/env bash
# Experiment 3: generate ablated negative responses (y_l) for DPO from
# CBT-Bench reference responses (y_w), one per CBT component removed --
# Socratic Questioning, Empathetic Validation, and two Cognitive Reframing
# failure modes (toxic positivity, distortion reinforcement). See "ex 3" in
# outline.md for the rationale and prompt design.
#
# Generation calls the OpenAI API (gpt-4 by default; OPENAI_API_KEY read from
# .env) -- see src/generate_negatives.py. Verification is a local
# Llama-3B-Instruct LLM-judge pass (finegrained per-component prompts, see
# src/llm-as-judge/prompts.md) plus a random sample written out for manual
# human annotation -- see src/llm-as-judge/verify_negatives.py.
#
# Usage: ./run_exp_3.sh
# Override defaults via env vars, e.g.: MODEL=gpt-4o BATCH_SIZE=32 ./run_exp_3.sh
# Everything (all 3 steps) is logged to a single file: logs/exp_3.log
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ~/code/venv/bin/activate

SEED="${SEED:-42}"
MODEL="${MODEL:-gpt-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-400}"
COMPONENTS="${COMPONENTS:-all}"  # or a comma list, e.g. sq_ablation,ev_ablation
SAMPLE_N="${SAMPLE_N:-100}"

PAIRS=data/cbtbench_pairs.csv

RESULTS_DIR=results/ex3-negatives
NEGATIVES="$RESULTS_DIR/dpo_negatives_cbtbench.csv"
VERIFIED="$RESULTS_DIR/dpo_negatives_cbtbench_verified.csv"
SUMMARY="$RESULTS_DIR/summary_negatives.md"
SAMPLE="$RESULTS_DIR/human_annotation_sample.csv"

mkdir -p logs "$RESULTS_DIR"
LOG_FILE=logs/exp_3.log
: > "$LOG_FILE"  # start a fresh log for this run

# echo "== 1/3: preparing CBT-Bench client_statement/y_w pairs =="
# python3 src/prepare_cbtbench_data.py --output "$PAIRS" \
#   --log-file "$LOG_FILE"

# echo "== 2/3: generating ablated negative responses (y_l) via OpenAI ($MODEL) =="
# python3 src/generate_negatives.py --pairs "$PAIRS" --components "$COMPONENTS" \
#   --out "$NEGATIVES" --model "$MODEL" --batch-size "$BATCH_SIZE" \
#   --max-new-tokens "$MAX_NEW_TOKENS" --seed "$SEED" --log-file "$LOG_FILE"

echo "== 3/3: LLM-judge verification + human-annotation sample =="
python3 src/llm-as-judge/verify_negatives.py --negatives "$NEGATIVES" --out "$VERIFIED" --summary "$SUMMARY" \
  --sample-out "$SAMPLE" --sample-n "$SAMPLE_N" --seed "$SEED" --log-file "$LOG_FILE"

echo "Done. DPO-ready negatives: $VERIFIED. Human-annotation sample: $SAMPLE. Summary: $SUMMARY. Full log: $LOG_FILE"
