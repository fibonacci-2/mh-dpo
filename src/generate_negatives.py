"""Ex3: generate ablated negative responses (y_l) for DPO from CBT-Bench
reference responses (y_w), per the four CBT-component ablations in
outline.md's "ex 3":

  sq_ablation              Socratic Questioning -> unsolicited advice-giving
  ev_ablation              Empathetic Validation -> cold/mechanical logic
  cr_toxic_positivity      Cognitive Reframing -> toxic positivity/dismissal
  cr_distortion_reinforcement   Cognitive Reframing -> reinforces the distortion

outline.md's prompts assume GPT/Claude as the generator; this project has no
external LLM API key configured, so the same local model already used as the
"base" policy in generate.py (NousResearch/Meta-Llama-3.1-8B-Instruct) plays
generator here too, prompted to rewrite y_w into a flawed y_l.

CBT-Bench's reference-response rows (see prepare_cbtbench_data.py) don't carry
a labeled cognitive distortion, so the two prompts that reference one
(sq_ablation, cr_toxic_positivity) get a generic placeholder telling the model
to infer it from context instead of a ground-truth label.

Output: one row per (client_statement, component) pair, with `chosen` = y_w
(unchanged across components) and `rejected` = the generated y_l -- i.e. a
DPO-ready preference table, long-form over components.
"""
import argparse
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import default_log_path, setup_logging

GEN_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"

NO_LABEL_PLACEHOLDER = (
    "not explicitly labeled in the source data -- infer the client's likely "
    "underlying cognitive distortion from their statement"
)

OUTPUT_INSTRUCTION = (
    "Output ONLY the rewritten response text itself, with no preamble, "
    "labels, headers, or surrounding quotation marks."
)

COMPONENTS = {
    "sq_ablation": {
        "label": "Socratic Questioning (SQ) Ablation -> Unsolicited Advice-Giving",
        "system": (
            "You are an AI data generator producing rejected (negative) "
            "therapeutic responses for DPO training on CBT dialogue datasets."
        ),
        "user": (
            "Task:\n"
            "Read the client's statement and the valid CBT therapist response "
            "(which uses Socratic Questioning). Rewrite the therapist response "
            "into a flawed response (y_l) that COMPLETELY REMOVES all "
            "open-ended questions and Socratic dialogue. Instead, make the "
            "therapist give immediate, direct, unsolicited advice and tell the "
            "client exactly what they should do or feel. Maintain a polite "
            "tone, but ensure the response violates the Socratic method.\n\n"
            "Client Statement: \"{client_statement}\"\n"
            "Target Core Belief / Distortion: \"{distortion_label}\"\n"
            "Ground-Truth Response (y_w): \"{positive_response}\"\n\n"
            f"{OUTPUT_INSTRUCTION}"
        ),
    },
    "ev_ablation": {
        "label": "Empathetic Validation (EV) Ablation -> Cold / Mechanical Clinical Logic",
        "system": (
            "You are an AI data generator producing rejected (negative) "
            "therapeutic responses for DPO fine-tuning."
        ),
        "user": (
            "Task:\n"
            "Read the client's statement and the ground-truth therapist "
            "response. Generate a negative candidate (y_l) by completely "
            "ablating all Empathetic Validation (EV). The response must strip "
            "out any emotional acknowledgment, warmth, or validation of the "
            "client's distress. Jump straight into cold, clinical logic, "
            "analyzing their cognitive error as if evaluating an equation.\n\n"
            "Client Statement: \"{client_statement}\"\n"
            "Ground-Truth Response (y_w): \"{positive_response}\"\n\n"
            f"{OUTPUT_INSTRUCTION}"
        ),
    },
    "cr_toxic_positivity": {
        "label": "Cognitive Reframing (CR) Failure -> Toxic Positivity & Dismissal",
        "system": (
            "You are an AI data generator creating preference optimization "
            "pairs for CBT models."
        ),
        "user": (
            "Task:\n"
            "Convert the provided high-quality therapist response into a "
            "flawed Cognitive Reframing attempt (y_l). Instead of helping the "
            "client construct a realistic, evidence-based alternative "
            "perspective, force an unrealistically positive, cheerful, or "
            "dismissive reframe (\"toxic positivity\"). The response should "
            "minimize the client's genuine problem.\n\n"
            "Client Statement: \"{client_statement}\"\n"
            "Target Cognitive Distortion: \"{distortion_label}\"\n"
            "Ground-Truth Response (y_w): \"{positive_response}\"\n\n"
            f"{OUTPUT_INSTRUCTION}"
        ),
    },
    "cr_distortion_reinforcement": {
        "label": "Cognitive Reframing (CR) Failure -> Distortion Reinforcement",
        "system": (
            "You are an AI dataset compiler generating hard-negative samples "
            "(y_l) for LLM therapeutic alignment."
        ),
        "user": (
            "Task:\n"
            "Read the client's statement, which contains a specific cognitive "
            "distortion, and the ground-truth therapist response (for tone "
            "reference only). Generate a negative response (y_l) where the "
            "therapist sounds empathetic and supportive on the surface, but "
            "covertly reinforces and validates the client's cognitive "
            "distortion (e.g. agreeing that their catastrophic prediction is "
            "likely to happen, or agreeing with their mind reading).\n\n"
            "Client Statement: \"{client_statement}\"\n"
            "Cognitive Distortion Present: \"{distortion_label}\"\n"
            "Ground-Truth Response (y_w, tone reference only): \"{positive_response}\"\n\n"
            f"{OUTPUT_INSTRUCTION}"
        ),
    },
}


def build_prompts(tokenizer, component, df):
    spec = COMPONENTS[component]
    prompts = []
    for _, row in df.iterrows():
        user = spec["user"].format(
            client_statement=row["client_statement"],
            positive_response=row["positive_response"],
            distortion_label=NO_LABEL_PLACEHOLDER,
        )
        messages = [
            {"role": "system", "content": spec["system"]},
            {"role": "user", "content": user},
        ]
        prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    return prompts


def generate_batch(tokenizer, model, prompts, batch_size, max_new_tokens, logger, tag):
    responses = []
    t0 = time.time()
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=1536
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
        responses.extend([d.strip().strip('"') for d in decoded])
        elapsed = time.time() - t0
        logger.info(
            f"[{tag}] {min(i + batch_size, len(prompts))}/{len(prompts)} done in {elapsed:.0f}s"
        )
    return responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/cbtbench_pairs.csv")
    parser.add_argument(
        "--components",
        default="all",
        help="Comma-separated subset of " + ",".join(COMPONENTS.keys()) + ", or 'all'",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-file", default=None, help="Default: logs/<out filename stem>.log")
    args = parser.parse_args()

    components = list(COMPONENTS.keys()) if args.components == "all" else args.components.split(",")
    for c in components:
        if c not in COMPONENTS:
            parser.error(f"Unknown component {c!r}; choices are {list(COMPONENTS.keys())}")

    logger = setup_logging(args.log_file or default_log_path(args.out))
    torch.manual_seed(args.seed)

    df = pd.read_csv(args.pairs)
    logger.info(f"Loaded {len(df)} client_statement/y_w pairs from {args.pairs}")

    logger.info(f"Loading generator model: {GEN_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL, device_map="auto", dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()

    all_rows = []
    for component in components:
        logger.info(f"== component: {component} ({COMPONENTS[component]['label']}) ==")
        prompts = build_prompts(tokenizer, component, df)
        rejected = generate_batch(
            tokenizer, model, prompts, args.batch_size, args.max_new_tokens, logger, component
        )
        for (_, row), y_l in zip(df.iterrows(), rejected):
            all_rows.append(
                {
                    "qid": row["qid"],
                    "exercise": row["exercise"],
                    "level": row["level"],
                    "component_ablated": component,
                    "client_statement": row["client_statement"],
                    "chosen": row["positive_response"],
                    "rejected": y_l,
                }
            )

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(args.out, index=False)
    logger.info(f"Wrote {len(out_df)} negative-response rows to {args.out}")


if __name__ == "__main__":
    main()
