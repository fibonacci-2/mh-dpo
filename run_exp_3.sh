#!/usr/bin/env bash
# Experiment 3: generate ablated negative responses (y_l) for DPO from
# CBT-Bench reference responses (y_w), one per CBT component removed --
# Socratic Questioning, Empathetic Validation, and two Cognitive Reframing
# failure modes (toxic positivity, distortion reinforcement). See "ex 3" in
# outline.md for the rationale and prompt design.
#
# Generation uses the same local model as run_exp_1.sh/run_exp_2.sh's "base"
# policy (NousResearch/Meta-Llama-3.1-8B-Instruct) rather than an external
# GPT/Claude API, since no external API key is configured in this
# environment (see src/generate_negatives.py).
#
# Usage: ./run_exp_3.sh
# Override defaults via env vars, e.g.: BATCH_SIZE=32 ./run_exp_3.sh
# Everything (all 3 steps) is logged to a single file: logs/exp_3.log
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ~/code/venv/bin/activate

SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-400}"
COMPONENTS="${COMPONENTS:-all}"  # or a comma list, e.g. sq_ablation,ev_ablation

PAIRS=data/cbtbench_pairs.csv

RESULTS_DIR=results/ex3-negatives
NEGATIVES="$RESULTS_DIR/dpo_negatives_cbtbench.csv"
VERIFIED="$RESULTS_DIR/dpo_negatives_cbtbench_verified.csv"
SUMMARY="$RESULTS_DIR/summary_negatives.md"

mkdir -p logs "$RESULTS_DIR"
LOG_FILE=logs/exp_3.log
: > "$LOG_FILE"  # start a fresh log for this run

echo "== 1/3: preparing CBT-Bench client_statement/y_w pairs =="
python3 src/prepare_cbtbench_data.py --output "$PAIRS" \
  --log-file "$LOG_FILE"

echo "== 2/3: generating ablated negative responses (y_l) =="
python3 src/generate_negatives.py --pairs "$PAIRS" --components "$COMPONENTS" \
  --out "$NEGATIVES" --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" \
  --seed "$SEED" --log-file "$LOG_FILE"

echo "== 3/3: QC pass + summary =="
python3 src/verify_negatives.py --negatives "$NEGATIVES" --out "$VERIFIED" --summary "$SUMMARY" \
  --log-file "$LOG_FILE"

echo "Done. DPO-ready negatives: $VERIFIED. Summary: $SUMMARY. Full log: $LOG_FILE"
