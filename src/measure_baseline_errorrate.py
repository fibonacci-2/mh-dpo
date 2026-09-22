"""Ex4 prep: measure ErrorRate_base(k) for the DPO baseline model, per
outline.md's w_k(x) = 1.0 + ErrorRate_base(k) -- the empirical failure rate
of the baseline (dpo) model on each CBT skill k in {SQ, EV, CR}.

Method: generate the dpo model's own English responses to a sample of
CBT-Bench client statements (data/cbtbench_pairs.csv, from
prepare_cbtbench_data.py), then reuse the exact same finegrained
ablation-detection judges from src/llm-as-judge/verify_negatives.py --
originally built to check "did this synthetic negative successfully ablate
skill k?" -- against the baseline's own (non-synthetic) response instead.
A baseline response that the judge flags as matching the failure-mode
criteria for skill k is a baseline failure on k.

CR is measured with both cr_toxic_positivity and cr_distortion_reinforcement
(the two failure modes prepare_cbtdp_data.py maps to the single "CR" skill
tag) and averaged, since a baseline response could independently exhibit
either failure.

Output: results/ex4-baseline-errorrate/skill_error_rates.md (+ the raw
judged rows), and logs the SKILL_WEIGHTS JSON string ready to pass to
run_exp_4.sh.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logging_utils import default_log_path, setup_logging

sys.path.insert(0, str(Path(__file__).resolve().parent / "llm-as-judge"))
from verify_negatives import JUDGE_PROMPTS, USER_TEMPLATE, JUDGE_MODEL as VERIFY_JUDGE_MODEL  # noqa: E402

GEN_MODEL = "Psychotherapy-LLM/PsyCoPref-Llama3-8B"  # the "dpo" baseline (see src/generate.py)

# Same skill tags/component mapping as prepare_cbtdp_data.py.
SQ_COMPONENT = "sq_ablation"
EV_COMPONENT = "ev_ablation"
CR_COMPONENTS = ["cr_toxic_positivity", "cr_distortion_reinforcement"]

SYSTEM_PROMPT_EN = (
    "You are a psychotherapist specializing in Cognitive Behavioral Therapy (CBT). "
    "A client will send you a statement or psychological concern. Provide a therapeutic "
    "response grounded in CBT principles: show genuine empathy for their feelings, "
    "understand their thoughts and concerns, help them explore these, use Socratic "
    "questioning to guide self-discovery, and suggest practical steps or strategies "
    "they could try. Write clearly, concisely, and respectfully."
)


def generate_baseline_responses(df, batch_size, max_new_tokens, logger):
    logger.info(f"Loading baseline generation model: {GEN_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL, device_map="auto", dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()

    statements = df["client_statement"].tolist()
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT_EN},
                {"role": "user", "content": s},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for s in statements
    ]

    responses = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_only = out[:, inputs["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        responses.extend([d.strip() for d in decoded])
        logger.info(f"[baseline-gen] {min(i + batch_size, len(prompts))}/{len(prompts)} done")

    del model
    torch.cuda.empty_cache()
    return responses


def load_judge():
    tokenizer = AutoTokenizer.from_pretrained(VERIFY_JUDGE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        VERIFY_JUDGE_MODEL, device_map="auto", dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    return tokenizer, model


def ask(tokenizer, model, system, user, max_new_tokens=150):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
    gen = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def parse_verdict(raw):
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0)).get("is_valid_negative")
    except json.JSONDecodeError:
        return None


def judge_component(tokenizer, model, component, client_statement, chosen, candidate):
    system = JUDGE_PROMPTS[component]
    user = USER_TEMPLATE.format(
        client_statement=client_statement, chosen_response=chosen, rejected_response=candidate
    )
    raw = ask(tokenizer, model, system, user)
    return parse_verdict(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cbtbench-pairs", default="data/cbtbench_pairs.csv")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out-rows", default="results/ex4-baseline-errorrate/baseline_judged_rows.csv")
    parser.add_argument("--out-summary", default="results/ex4-baseline-errorrate/skill_error_rates.md")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    logger = setup_logging(args.log_file or default_log_path(args.out_summary))

    df = pd.read_csv(args.cbtbench_pairs)
    df = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    logger.info(f"Sampled {len(df)} CBT-Bench client statements (seed={args.seed})")

    df["baseline_response"] = generate_baseline_responses(
        df, args.batch_size, args.max_new_tokens, logger
    )

    logger.info(f"Loading judge model: {VERIFY_JUDGE_MODEL}")
    j_tok, j_model = load_judge()

    rows = []
    for _, row in df.iterrows():
        verdicts = {}
        for component in [SQ_COMPONENT, EV_COMPONENT] + CR_COMPONENTS:
            verdicts[component] = judge_component(
                j_tok, j_model, component, row["client_statement"], row["positive_response"], row["baseline_response"]
            )
        rows.append({"qid": row["qid"], "client_statement": row["client_statement"], **verdicts})
        logger.info(f"[judge] {row['qid']}: {verdicts}")

    judged = pd.DataFrame(rows)
    Path(args.out_rows).parent.mkdir(parents=True, exist_ok=True)
    judged.to_csv(args.out_rows, index=False)
    logger.info(f"Wrote {len(judged)} judged rows to {args.out_rows}")

    def error_rate(col):
        parsed = judged[col].dropna()
        return float(parsed.mean()) if len(parsed) else None

    err_sq = error_rate(SQ_COMPONENT)
    err_ev = error_rate(EV_COMPONENT)
    err_cr_components = [error_rate(c) for c in CR_COMPONENTS]
    err_cr = sum(err_cr_components) / len(err_cr_components)

    skill_weights = {
        "SQ": round(1.0 + err_sq, 3),
        "EV": round(1.0 + err_ev, 3),
        "CR": round(1.0 + err_cr, 3),
    }

    lines = ["# ErrorRate_base(k) and derived skill weights w_k\n"]
    lines.append(f"Baseline model: `{GEN_MODEL}`. Judge: `{VERIFY_JUDGE_MODEL}`. N={len(judged)} CBT-Bench statements (seed={args.seed}).\n")
    lines.append("w_k = 1.0 + ErrorRate_base(k)\n")
    lines.append("| skill | ErrorRate_base(k) | w_k |")
    lines.append("|---|---:|---:|")
    lines.append(f"| SQ | {err_sq:.3f} | {skill_weights['SQ']} |")
    lines.append(f"| EV | {err_ev:.3f} | {skill_weights['EV']} |")
    lines.append(f"| CR (avg of toxic_positivity={err_cr_components[0]:.3f}, distortion_reinforcement={err_cr_components[1]:.3f}) | {err_cr:.3f} | {skill_weights['CR']} |")
    lines.append(f"\nSKILL_WEIGHTS='{json.dumps(skill_weights)}'\n")

    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_summary, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Wrote summary to {args.out_summary}")
    logger.info(f"SKILL_WEIGHTS='{json.dumps(skill_weights)}'")
    print(json.dumps(skill_weights))


if __name__ == "__main__":
    main()
