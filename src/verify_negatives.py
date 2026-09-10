"""Ex3 verification pass for generated negatives (see generate_negatives.py).

Two things only, per the actual QC protocol -- no keyword/length heuristics:

  1. LLM-as-judge: a local judge model (Llama-3B-Instruct) is shown the
     client statement plus the chosen (y_w) and rejected (y_l) responses,
     order randomized, and picks the better one -- `judge_winner` is one of
     "chosen", "rejected", or "tie".
  2. A random sample of rows (default 100) is written out untouched, with
     a blank `human_label` column for a human annotator to fill in by hand,
     using the SAME three-way scheme ("chosen"/"rejected"/"tie") -- pure
     random sampling, nothing derived or pre-filled.

Judge and human deliberately answer the identical question with the
identical label set, so src/compare_judgments.py can compare them directly
(exact agreement, kappa) instead of mapping between two different schemes.
"""
import argparse
import os
import random
import re
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import default_log_path, setup_logging

load_dotenv()

JUDGE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

JUDGE_SYSTEM = (
    "You are an expert clinical supervisor evaluating the quality of CBT "
    "(Cognitive Behavioral Therapy) therapist responses. You will be shown a "
    "client's statement and two candidate therapist responses (A and B). "
    "Judge which response better reflects sound CBT practice -- balancing "
    "empathetic validation, Socratic questioning, and realistic cognitive "
    "reframing, without giving unsolicited directive advice or reinforcing "
    "the client's cognitive distortion. Reply with a single letter: A if "
    "response A is clearly better overall, B if response B is clearly "
    "better overall, or T if they are about equally good / you can't tell. "
    "Do not explain."
)


def load_judge():
    hf_token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL, device_map="auto", dtype=torch.bfloat16, low_cpu_mem_usage=True, token=hf_token
    )
    model.eval()
    return tokenizer, model


def ask(tokenizer, model, system, user, max_new_tokens):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def parse_winner(raw):
    match = re.search(r"\b([ABT])\b", raw.upper())
    return match.group(1) if match else None


def judge_row(tokenizer, model, row):
    chosen, rejected = str(row["chosen"]), str(row["rejected"])
    if random.random() < 0.5:
        a, b, a_label, b_label = chosen, rejected, "chosen", "rejected"
    else:
        a, b, a_label, b_label = rejected, chosen, "rejected", "chosen"

    user = f"Client statement:\n{row['client_statement']}\n\nResponse A:\n{a}\n\nResponse B:\n{b}"
    raw = ask(tokenizer, model, JUDGE_SYSTEM, user, max_new_tokens=8)
    letter = parse_winner(raw)
    if letter == "T":
        winner_label = "tie"
    else:
        winner_label = {"A": a_label, "B": b_label}.get(letter)
    return raw, winner_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--negatives", default="results/dpo_negatives_cbtbench.csv")
    parser.add_argument("--out", default="results/dpo_negatives_cbtbench_verified.csv")
    parser.add_argument("--summary", default="results/summary_negatives.md")
    parser.add_argument("--sample-out", default="results/human_annotation_sample.csv")
    parser.add_argument("--sample-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    logger = setup_logging(args.log_file or default_log_path(args.out))
    random.seed(args.seed)

    df = pd.read_csv(args.negatives)
    logger.info(f"Loaded {len(df)} generated negatives from {args.negatives}")

    # 1) LLM-as-judge: does the judge still prefer chosen (y_w) over the
    # generated negative (rejected, y_l) in a randomized pairwise comparison?
    logger.info(f"Loading judge model: {JUDGE_MODEL}")
    tokenizer, model = load_judge()

    judge_rows = []
    for _, row in df.iterrows():
        raw, winner_label = judge_row(tokenizer, model, row)
        judge_rows.append(
            {
                **row.to_dict(),
                "judge_raw": raw,
                "judge_winner": winner_label,
            }
        )
        logger.info(f"[judge] {row['qid']} ({row['component_ablated']}): winner={winner_label}")

    judged_df = pd.DataFrame(judge_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    judged_df.to_csv(args.out, index=False)
    logger.info(f"Wrote {len(judged_df)} rows with judge verdicts to {args.out}")

    # 2) Random sample for manual human annotation, using the SAME
    # "chosen"/"rejected"/"tie" scheme the judge uses -- no heuristics, no
    # derived columns beyond blank fields for the annotator.
    sample_n = min(args.sample_n, len(df))
    sample_df = df.sample(n=sample_n, random_state=args.seed).copy()
    sample_df["human_label"] = ""
    sample_df["human_notes"] = ""
    Path(args.sample_out).parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(args.sample_out, index=False)
    logger.info(f"Wrote {len(sample_df)} randomly sampled rows to {args.sample_out} for human annotation")

    # Summary: judge verdict distribution by component.
    winner_counts = (
        judged_df.groupby("component_ablated")["judge_winner"]
        .value_counts(dropna=False)
        .unstack(fill_value=0)
    )
    for col in ["chosen", "rejected", "tie"]:
        if col not in winner_counts.columns:
            winner_counts[col] = 0
    winner_counts = winner_counts[["chosen", "rejected", "tie"]]
    winner_rates = winner_counts.div(winner_counts.sum(axis=1), axis=0)

    lines = ["# Ex3 negative-generation verification summary\n"]
    lines.append(
        f"Total generated rows: {len(df)}. Judge model: {JUDGE_MODEL}. "
        f"{sample_n} rows randomly sampled to `{args.sample_out}` for manual human annotation "
        f"using the same chosen/rejected/tie scheme.\n"
    )
    lines.append(
        "Judge verdict = which response the judge preferred in a randomized pairwise "
        "comparison of chosen (y_w) vs. the generated negative (rejected, y_l). A healthy "
        "ablation should show a high `chosen` rate (the judge still prefers the "
        "ground truth).\n"
    )
    lines.append("## Judge verdict distribution by component\n")
    lines.append("| component | n | chosen | rejected | tie |")
    lines.append("|---|---|---|---|---|")
    for component in winner_counts.index:
        n = int(winner_counts.loc[component].sum())
        lines.append(
            f"| {component} | {n} | {winner_rates.loc[component, 'chosen']:.0%} | "
            f"{winner_rates.loc[component, 'rejected']:.0%} | {winner_rates.loc[component, 'tie']:.0%} |"
        )

    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Wrote verification summary to {args.summary}")
    logger.info(f"Judge verdict distribution by component:\n{winner_counts}")


if __name__ == "__main__":
    main()
