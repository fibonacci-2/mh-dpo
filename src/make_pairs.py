"""Build a readable side-by-side file of (question, base response, dpo response,
judge winner) pairs for manual eyeballing of a generation run's results.
"""
import argparse

import pandas as pd

from logging_utils import default_log_path, setup_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="results/gen_base.csv")
    parser.add_argument("--dpo", default="results/gen_dpo.csv")
    parser.add_argument("--pairwise", default="results/judge_pairwise.csv")
    parser.add_argument("--out", default="results/pairs.md")
    parser.add_argument("--n", type=int, default=10, help="Number of pairs to include")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    logger = setup_logging(args.log_file or default_log_path(args.out))

    base_df = pd.read_csv(args.base)
    dpo_df = pd.read_csv(args.dpo)
    merged = base_df.merge(dpo_df, on=["qid", "Question"], suffixes=("_base", "_dpo"))

    try:
        pairwise = pd.read_csv(args.pairwise)
        merged = merged.merge(pairwise[["qid", "winner"]], on="qid", how="left")
    except FileNotFoundError:
        logger.info(f"No pairwise file at {args.pairwise}, skipping winner column")
        merged["winner"] = None

    n = min(args.n, len(merged))
    sample = merged.sample(n=n, random_state=args.seed).sort_values("qid")

    lines = [f"# Base vs DPO response pairs ({n} of {len(merged)})\n"]
    for _, row in sample.iterrows():
        lines.append(f"## {row['qid']} (judge winner: {row['winner']})\n")
        lines.append(f"**Question:**\n\n{row['Question']}\n")
        lines.append(f"**Base:**\n\n{row['response_base']}\n")
        lines.append(f"**DPO:**\n\n{row['response_dpo']}\n")
        lines.append("---\n")

    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Wrote {n} pairs to {args.out}")


if __name__ == "__main__":
    main()
