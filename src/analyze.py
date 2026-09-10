"""RQ1/RQ2: summarize the judge outputs (any two models) into a win-rate and
per-principle comparison."""
import argparse

import pandas as pd

from logging_utils import default_log_path, setup_logging

PRINCIPLES = [
    "empathy",
    "personalization",
    "self_exploration",
    "clarity",
    "autonomy",
    "harm_avoidance",
    "stage_sensitivity",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairwise", default="results/judge_pairwise.csv")
    parser.add_argument("--absolute", default="results/judge_absolute.csv")
    parser.add_argument("--out", default="results/summary.md")
    parser.add_argument("--log-file", default=None, help="Default: logs/<out filename stem>.log")
    args = parser.parse_args()

    logger = setup_logging(args.log_file or default_log_path(args.out))

    pairwise = pd.read_csv(args.pairwise)
    absolute = pd.read_csv(args.absolute)

    model_names = list(pd.unique(absolute["model"]))
    if len(model_names) != 2:
        raise ValueError(f"Expected exactly 2 models in {args.absolute}, found {model_names}")
    name_a, name_b = model_names

    n_total = len(pairwise)
    n_valid = pairwise["winner"].notna().sum()
    win_counts = pairwise["winner"].value_counts()
    a_wins = int(win_counts.get(name_a, 0))
    b_wins = int(win_counts.get(name_b, 0))
    b_win_rate = b_wins / n_valid * 100 if n_valid else float("nan")

    lines = []
    lines.append(f"# Results: {name_b} vs {name_a}\n")
    lines.append(
        f"Pairwise judge comparisons: {n_valid}/{n_total} parsed. "
        f"{name_b} wins: {b_wins}, {name_a} wins: {a_wins}, "
        f"{name_b} win rate: {b_win_rate:.1f}%\n"
    )

    means = (
        absolute.dropna(subset=PRINCIPLES, how="all")
        .groupby("model")[PRINCIPLES]
        .mean()
        .round(2)
        .loc[[name_a, name_b]]
    )
    lines.append("## Mean Likert scores per principle (1-5, judge-rated)\n")
    lines.append(means.to_markdown())
    lines.append("")

    delta = (means.loc[name_b] - means.loc[name_a]).round(2)
    lines.append(f"## Delta ({name_b} - {name_a})\n")
    lines.append(delta.to_frame("delta").to_markdown())
    lines.append("")

    n_scored = absolute.dropna(subset=PRINCIPLES, how="all").groupby("model").size()
    lines.append("## Judge parse coverage\n")
    lines.append(
        f"Absolute scores parsed: {n_scored.to_dict()} out of "
        f"{absolute.groupby('model').size().to_dict()} per model\n"
    )

    report = "\n".join(lines)
    with open(args.out, "w") as f:
        f.write(report)
    logger.info(f"Wrote summary to {args.out}")
    logger.info("\n" + report)


if __name__ == "__main__":
    main()
