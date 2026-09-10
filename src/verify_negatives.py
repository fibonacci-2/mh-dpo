"""Ex3 QC pass: flag generated negatives (see generate_negatives.py) that look
like the ablation didn't actually take -- e.g. an "SQ ablation" response that
still asks open-ended questions, or an "EV ablation" response that's still
visibly empathetic.

These are cheap keyword/length heuristics, not a model judgment -- treat
flags as "worth a human look," not ground truth. There's no local judge model
run here (unlike judge.py) since the checks are about surface form (does a
question mark appear at all, etc.), which regex handles more reliably than
prompting a small model would.
"""
import argparse
from pathlib import Path

import pandas as pd

from logging_utils import default_log_path, setup_logging

EMPATHY_MARKERS = [
    "sorry to hear", "sorry you're", "understand how", "understand that",
    "that sounds", "i hear you", "must be hard", "must be difficult",
    "i can imagine", "makes sense that you", "it's okay to feel",
    "your feelings are valid", "that must feel",
]

TOXIC_POSITIVITY_MARKERS = [
    "look on the bright side", "everything happens for a reason",
    "don't worry", "just think positive", "it could be worse",
    "at least", "silver lining", "stay positive", "it'll all work out",
    "no need to worry", "just be happy", "just be grateful",
    "game-changer", "game changer", "isn't it wonderful", "isn't it exciting",
    "so exciting", "amazing", "incredible", "let's focus on the positive",
    "focus on the positive", "who needs to", "totally normal", "so brave",
    "skip the past", "dwell on the past", "just think about",
]

CORRECTIVE_MARKERS = [
    "however", "but let's consider", "alternative perspective",
    "let's challenge", "is that really true", "evidence for",
    "evidence against", "another way to look at", "let's examine",
    "let's question",
]

REFUSAL_MARKERS = [
    "i cannot", "i can't help", "i can't fulfill", "i cant fulfill",
    "i'm afraid i can't", "i'm not able to", "as an ai", "i won't",
    "i can't generate", "i can't create content", "i can't provide",
]


def contains_any(text, markers):
    low = text.lower()
    return any(m in low for m in markers)


def check_row(component, chosen, rejected):
    flags = []
    chosen, rejected = str(chosen), str(rejected)

    if rejected.strip() == "" :
        flags.append("empty_response")
        return flags
    if contains_any(rejected, REFUSAL_MARKERS):
        flags.append("looks_like_refusal")
    if rejected.strip() == chosen.strip():
        flags.append("identical_to_chosen")
    if len(rejected.split()) < 0.3 * max(len(chosen.split()), 1):
        flags.append("much_shorter_than_chosen")

    if component == "sq_ablation" and "?" in rejected:
        flags.append("still_contains_questions")
    elif component == "ev_ablation" and contains_any(rejected, EMPATHY_MARKERS):
        flags.append("still_sounds_empathetic")
    elif component == "cr_toxic_positivity" and not contains_any(rejected, TOXIC_POSITIVITY_MARKERS):
        flags.append("not_clearly_toxic_positive")
    elif component == "cr_distortion_reinforcement" and contains_any(rejected, CORRECTIVE_MARKERS):
        flags.append("still_challenges_distortion")

    return flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--negatives", default="results/dpo_negatives_cbtbench.csv")
    parser.add_argument("--out", default="results/dpo_negatives_cbtbench_verified.csv")
    parser.add_argument("--summary", default="results/summary_negatives.md")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    logger = setup_logging(args.log_file or default_log_path(args.out))

    df = pd.read_csv(args.negatives)
    logger.info(f"Loaded {len(df)} generated negatives from {args.negatives}")

    df["flags"] = [
        ";".join(check_row(row["component_ablated"], row["chosen"], row["rejected"]))
        for _, row in df.iterrows()
    ]
    df["flagged"] = df["flags"] != ""

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    logger.info(f"Wrote {len(df)} rows with QC flags to {args.out}")

    per_component = df.groupby("component_ablated")["flagged"].agg(["count", "sum"])
    per_component["pass_rate"] = 1 - per_component["sum"] / per_component["count"]

    lines = ["# Ex3 negative-generation QC summary\n"]
    lines.append(f"Total generated rows: {len(df)}. Flagged (heuristic, needs review): {int(df['flagged'].sum())}.\n")
    lines.append("Flags are surface-form heuristics (keyword/length checks), not a model judgment -- "
                  "treat a flag as \"worth a human look,\" not a confirmed failure.\n")
    lines.append("## Pass rate by component\n")
    lines.append("| component | n | flagged | pass rate |")
    lines.append("|---|---|---|---|")
    for component, row in per_component.iterrows():
        lines.append(f"| {component} | {int(row['count'])} | {int(row['sum'])} | {row['pass_rate']:.0%} |")

    lines.append("\n## Sample flagged rows\n")
    sample = df[df["flagged"]].head(5)
    for _, row in sample.iterrows():
        lines.append(f"### {row['qid']} / {row['component_ablated']} (flags: {row['flags']})\n")
        lines.append(f"**Client statement:** {row['client_statement']}\n")
        lines.append(f"**Chosen (y_w):** {row['chosen']}\n")
        lines.append(f"**Rejected (y_l):** {row['rejected']}\n")
        lines.append("---\n")

    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Wrote QC summary to {args.summary}")
    logger.info(f"Pass rate by component:\n{per_component}")


if __name__ == "__main__":
    main()
