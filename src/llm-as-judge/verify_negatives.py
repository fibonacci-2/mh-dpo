"""Ex3 verification pass for generated negatives (see generate_negatives.py).

Two things only, per the actual QC protocol -- no keyword/length heuristics:

  1. LLM-as-judge: a local judge model (Llama-3B-Instruct) is shown the
     client statement plus the chosen (y_w) and rejected (y_l) responses and
     asked the finegrained, component-specific question from `prompts.md`
     (one prompt per `component_ablated` value) -- did `rejected` actually
     succeed at the ablation it was generated for? The judge answers with a
     JSON object `{"is_valid_negative": bool, "reasoning": str}`.
  2. A random sample of rows (default 100) is written out untouched, with
     blank `human_valid` / `human_notes` columns for a human annotator to
     fill in by hand, answering the SAME per-component question the judge
     answered -- pure random sampling, nothing derived or pre-filled.

Judge and human deliberately answer the identical question per row, so
src/compare_judgments.py can compare them directly (exact agreement, kappa)
instead of mapping between two different schemes.
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from logging_utils import default_log_path, setup_logging

load_dotenv()

JUDGE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# One finegrained judge prompt per `component_ablated` value, transcribed
# from prompts.md. Each answers "did the ablation actually take?" for its
# specific failure mode, rather than a generic chosen-vs-rejected preference.
JUDGE_PROMPTS = {
    "sq_ablation": (
        "You are an expert supervisor in Cognitive Behavioral Therapy (CBT). Your task is to "
        "evaluate whether a generated negative response (y_l) successfully ablates Socratic "
        "Questioning from a therapist's intervention.\n\n"
        "Evaluation Target: The candidate response (y_l) is a VALID NEGATIVE if it COMPLETELY "
        "REMOVES all Socratic inquiry (no open-ended questions inviting self-reflection) and "
        "instead gives direct, unsolicited advice, commands, or directive telling.\n\n"
        "Evaluation Criteria:\n"
        "1. Does <y_l> contain ZERO open-ended Socratic questions aimed at guided discovery?\n"
        "2. Does <y_l> substitute questioning with direct advice, telling, or directive instructions?\n"
        "3. Is <y_l> grammatically fluent and coherent?\n\n"
        "Output Format: Return ONLY a JSON object, with no extra text, in exactly this shape:\n"
        '{\n  "is_valid_negative": true,\n  "reasoning": "Concise 1-2 sentence clinical justification."\n}\n'
        "where is_valid_negative is true if y_l successfully ablated Socratic Questioning, false otherwise."
    ),
    "ev_ablation": (
        "You are an expert supervisor in Cognitive Behavioral Therapy (CBT). Your task is to "
        "evaluate whether a generated negative response (y_l) successfully ablates Empathetic "
        "Validation.\n\n"
        "Evaluation Target: The candidate response (y_l) is a VALID NEGATIVE if it COMPLETELY "
        "STRIPS OUT all emotional validation, warmth, and relational empathy, resulting in a "
        "cold, clinical, or overly mechanical response that ignores the client's emotional "
        "state while discussing cognitive logic.\n\n"
        "Evaluation Criteria:\n"
        "1. Does <y_l> completely lack emotional reflection, warmth, or empathetic validation?\n"
        "2. Does <y_l> jump straight into mechanical analysis, logical dissecting, or clinical questioning?\n"
        "3. Is <y_l> coherent and grammatically correct (not random noise)?\n\n"
        "Output Format: Return ONLY a JSON object, with no extra text, in exactly this shape:\n"
        '{\n  "is_valid_negative": true,\n  "reasoning": "Concise 1-2 sentence clinical justification."\n}\n'
        "where is_valid_negative is true if y_l successfully ablated Empathetic Validation, false otherwise."
    ),
    "cr_toxic_positivity": (
        "You are an expert supervisor in Cognitive Behavioral Therapy (CBT). Your task is to "
        "evaluate whether a generated negative response (y_l) exhibits Toxic Positivity or "
        "Dismissive Reframing.\n\n"
        "Evaluation Target: The candidate response (y_l) is a VALID NEGATIVE if it attempts "
        "cognitive reframing using superficial optimism, forced cheerfulness, or dismissive "
        'reassurances ("toxic positivity") that invalidate or minimize the client\'s real '
        "distress.\n\n"
        "Evaluation Criteria:\n"
        '1. Does <y_l> offer superficial, unrealistically cheerful, or dismissive reframing (e.g., "Just look on the bright side!", "Everything happens for a reason!")?\n'
        "2. Does <y_l> fail to guide the client toward a realistic, evidence-based alternative perspective?\n"
        "3. Is <y_l> coherent and natural-sounding dialogue?\n\n"
        "Output Format: Return ONLY a JSON object, with no extra text, in exactly this shape:\n"
        '{\n  "is_valid_negative": true,\n  "reasoning": "Concise 1-2 sentence clinical justification."\n}\n'
        "where is_valid_negative is true if y_l exhibits toxic positivity / dismissive reframing, false otherwise."
    ),
    "cr_distortion_reinforcement": (
        "You are an expert supervisor in Cognitive Behavioral Therapy (CBT). Your task is to "
        "evaluate whether a generated negative response (y_l) inadvertently validates or "
        "reinforces a cognitive distortion.\n\n"
        "Evaluation Target: The candidate response (y_l) is a VALID NEGATIVE if it actively "
        "reinforces, validates, or agrees with the client's maladaptive cognitive distortion "
        "(e.g., confirming their catastrophic prediction, agreeing with mind-reading, or "
        "validating all-or-nothing thinking).\n\n"
        "Evaluation Criteria:\n"
        "1. Does <y_l> confirm or validate the truth of the client's cognitive distortion rather than challenging or reframing it?\n"
        "2. Does <y_l> sound superficially supportive while delivering clinically harmful reinforcement?\n"
        "3. Is <y_l> coherent and relevant to the client statement?\n\n"
        "Output Format: Return ONLY a JSON object, with no extra text, in exactly this shape:\n"
        '{\n  "is_valid_negative": true,\n  "reasoning": "Concise 1-2 sentence clinical justification."\n}\n'
        "where is_valid_negative is true if y_l reinforces the client's cognitive distortion, false otherwise."
    ),
}

USER_TEMPLATE = (
    'Client Statement <T>: "{client_statement}"\n'
    'Target CBT Response <y_w>: "{chosen_response}"\n'
    'Candidate Negative Response <y_l>: "{rejected_response}"'
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


def parse_verdict(raw):
    """Pull `{"is_valid_negative": bool, "reasoning": str}` out of the judge's
    raw text. Returns (None, None) if nothing parseable is found."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None, None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, None
    return obj.get("is_valid_negative"), obj.get("reasoning")


def judge_row(tokenizer, model, row):
    component = row["component_ablated"]
    system = JUDGE_PROMPTS[component]
    user = USER_TEMPLATE.format(
        client_statement=row["client_statement"],
        chosen_response=row["chosen"],
        rejected_response=row["rejected"],
    )
    raw = ask(tokenizer, model, system, user, max_new_tokens=150)
    is_valid, reasoning = parse_verdict(raw)
    return raw, is_valid, reasoning


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

    unknown = sorted(set(df["component_ablated"]) - set(JUDGE_PROMPTS))
    if unknown:
        raise ValueError(f"No judge prompt in JUDGE_PROMPTS for component_ablated value(s): {unknown}")

    # 1) LLM-as-judge: for each row, ask the finegrained judge prompt for
    # that row's component_ablated -- did the generated negative (rejected,
    # y_l) actually succeed at the intended ablation?
    logger.info(f"Loading judge model: {JUDGE_MODEL}")
    tokenizer, model = load_judge()

    judge_rows = []
    for _, row in df.iterrows():
        raw, is_valid, reasoning = judge_row(tokenizer, model, row)
        judge_rows.append(
            {
                **row.to_dict(),
                "judge_raw": raw,
                "judge_valid": is_valid,
                "judge_reasoning": reasoning,
            }
        )
        logger.info(f"[judge] {row['qid']} ({row['component_ablated']}): valid={is_valid}")

    judged_df = pd.DataFrame(judge_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    judged_df.to_csv(args.out, index=False)
    logger.info(f"Wrote {len(judged_df)} rows with judge verdicts to {args.out}")

    # 2) Random sample for manual human annotation, using the SAME
    # per-component is_valid_negative question the judge answers -- no
    # heuristics, no derived columns beyond blank fields for the annotator.
    sample_n = min(args.sample_n, len(df))
    sample_df = df.sample(n=sample_n, random_state=args.seed).copy()
    sample_df["human_valid"] = ""
    sample_df["human_notes"] = ""
    Path(args.sample_out).parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(args.sample_out, index=False)
    logger.info(f"Wrote {len(sample_df)} randomly sampled rows to {args.sample_out} for human annotation")

    # Summary: judge validity distribution by component.
    verdict = judged_df["judge_valid"].map({True: "valid", False: "invalid"}).fillna("unparsed")
    valid_counts = pd.crosstab(judged_df["component_ablated"], verdict)
    for col in ["valid", "invalid", "unparsed"]:
        if col not in valid_counts.columns:
            valid_counts[col] = 0
    valid_counts = valid_counts[["valid", "invalid", "unparsed"]]
    valid_rates = valid_counts.div(valid_counts.sum(axis=1), axis=0)

    lines = ["# Ex3 negative-generation verification summary\n"]
    lines.append(
        f"Total generated rows: {len(df)}. Judge model: {JUDGE_MODEL}, using the finegrained "
        f"per-component prompts in `prompts.md`. {sample_n} rows randomly sampled to "
        f"`{args.sample_out}` for manual human annotation using the same per-component question.\n"
    )
    lines.append(
        "Judge verdict = whether the finegrained judge for this row's `component_ablated` "
        "considers the generated negative (rejected, y_l) a VALID NEGATIVE, i.e. the intended "
        "ablation actually took. A healthy generation pass should show a high `valid` rate. "
        "`unparsed` rows are ones where the judge's raw output couldn't be parsed as the "
        "expected JSON object.\n"
    )
    lines.append("## Judge verdict distribution by component\n")
    lines.append("| component | n | valid | invalid | unparsed |")
    lines.append("|---|---|---|---|---|")
    for component in valid_counts.index:
        n = int(valid_counts.loc[component].sum())
        lines.append(
            f"| {component} | {n} | {valid_rates.loc[component, 'valid']:.0%} | "
            f"{valid_rates.loc[component, 'invalid']:.0%} | {valid_rates.loc[component, 'unparsed']:.0%} |"
        )

    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Wrote verification summary to {args.summary}")
    logger.info(f"Judge verdict distribution by component:\n{valid_counts}")


if __name__ == "__main__":
    main()
