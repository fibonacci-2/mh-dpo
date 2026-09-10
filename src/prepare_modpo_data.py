"""RQ2 data prep: download the PsyCoPref preference dataset used to train the
Multi-Objective DPO (MODPO) model.

Each row already has a question, a chosen/rejected response pair, and both
responses independently rated 1-5 on 7 principles (empathy, relevance,
clarity, safety, exploration, autonomy, staging). Those ratings are used
directly as the per-objective reward scores in src/train_modpo.py's MODPO
loss, so no separate reward models need to be trained.
"""
import argparse
from pathlib import Path

from datasets import load_dataset

from logging_utils import setup_logging

DATASET_ID = "Psychotherapy-LLM/PsyCoPref"

PRINCIPLES = ["empathy", "relevance", "clarity", "safety", "exploration", "autonomy", "staging"]

RATING_COLUMNS = [f"{side}_{dim}_rating" for side in ("chosen", "rejected") for dim in PRINCIPLES]

KEEP_COLUMNS = ["question", "chosen", "rejected"] + RATING_COLUMNS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--output", default="data/psycopref_train.csv")
    parser.add_argument("--n", type=int, default=None, help="Optional cap on rows (default: use the full split)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-file", default="logs/prepare_modpo_data.log")
    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    logger.info(f"Downloading {DATASET_ID} split={args.split}")

    ds = load_dataset(DATASET_ID, split=args.split)
    df = ds.to_pandas()[KEEP_COLUMNS]
    logger.info(f"{len(df)} preference pairs available")

    if args.n is not None and args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
        logger.info(f"Subsampled to {len(df)} rows (seed={args.seed})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
