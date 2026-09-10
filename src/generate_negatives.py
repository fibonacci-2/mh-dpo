"""Ex3: generate ablated negative responses (y_l) for DPO from CBT-Bench
reference responses (y_w), per the four CBT-component ablations in
outline.md's "ex 3":

  sq_ablation              Socratic Questioning -> unsolicited advice-giving
  ev_ablation              Empathetic Validation -> cold/mechanical logic
  cr_toxic_positivity      Cognitive Reframing -> toxic positivity/dismissal
  cr_distortion_reinforcement   Cognitive Reframing -> reinforces the distortion

outline.md's prompts assume GPT/Claude as the generator. An OpenAI API key
is now configured (OPENAI_API_KEY in .env), so this script calls the OpenAI
API directly -- gpt-4 by default -- instead of the local Llama model used
elsewhere in this repo (e.g. generate.py's "base" policy).

CBT-Bench's reference-response rows (see prepare_cbtbench_data.py) don't carry
a labeled cognitive distortion, so the two prompts that reference one
(sq_ablation, cr_toxic_positivity) get a generic placeholder telling the model
to infer it from context instead of a ground-truth label.

Output: one row per (client_statement, component) pair, with `chosen` = y_w
(unchanged across components) and `rejected` = the generated y_l -- i.e. a
DPO-ready preference table, long-form over components.
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from logging_utils import default_log_path, setup_logging

load_dotenv()

GEN_MODEL = "gpt-4"

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


def build_messages(component, df):
    spec = COMPONENTS[component]
    all_messages = []
    for _, row in df.iterrows():
        user = spec["user"].format(
            client_statement=row["client_statement"],
            positive_response=row["positive_response"],
            distortion_label=NO_LABEL_PLACEHOLDER,
        )
        all_messages.append(
            [
                {"role": "system", "content": spec["system"]},
                {"role": "user", "content": user},
            ]
        )
    return all_messages


def call_model(client, model, messages, max_tokens, temperature, top_p, seed, logger, max_retries=5):
    delay = 2
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
            return resp.choices[0].message.content.strip().strip('"')
        except Exception as e:
            last_err = e
            logger.warning(f"OpenAI call failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
    logger.error(f"Giving up on one row after {max_retries} attempts: {last_err}")
    return ""


def generate_all(client, model, messages_list, concurrency, max_tokens, temperature, top_p, seed, logger, tag):
    results = [None] * len(messages_list)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                call_model, client, model, messages, max_tokens, temperature, top_p, seed, logger
            ): i
            for i, messages in enumerate(messages_list)
        }
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()
            done += 1
            if done % 10 == 0 or done == len(messages_list):
                elapsed = time.time() - t0
                logger.info(f"[{tag}] {done}/{len(messages_list)} done in {elapsed:.0f}s")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/cbtbench_pairs.csv")
    parser.add_argument(
        "--components",
        default="all",
        help="Comma-separated subset of " + ",".join(COMPONENTS.keys()) + ", or 'all'",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=GEN_MODEL, help="OpenAI model name")
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Number of concurrent OpenAI requests"
    )
    parser.add_argument("--max-new-tokens", type=int, default=400, help="max_tokens per response")
    parser.add_argument("--seed", type=int, default=42, help="Passed through as the OpenAI request seed")
    parser.add_argument("--log-file", default=None, help="Default: logs/<out filename stem>.log")
    args = parser.parse_args()

    components = list(COMPONENTS.keys()) if args.components == "all" else args.components.split(",")
    for c in components:
        if c not in COMPONENTS:
            parser.error(f"Unknown component {c!r}; choices are {list(COMPONENTS.keys())}")

    logger = setup_logging(args.log_file or default_log_path(args.out))

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        parser.error("OPENAI_API_KEY not set -- add it to .env or the environment")
    client = OpenAI(api_key=api_key)

    df = pd.read_csv(args.pairs)
    logger.info(f"Loaded {len(df)} client_statement/y_w pairs from {args.pairs}")
    logger.info(f"Using OpenAI generator model: {args.model}")

    all_rows = []
    for component in components:
        logger.info(f"== component: {component} ({COMPONENTS[component]['label']}) ==")
        messages_list = build_messages(component, df)
        rejected = generate_all(
            client,
            args.model,
            messages_list,
            args.batch_size,
            args.max_new_tokens,
            0.7,
            0.9,
            args.seed,
            logger,
            component,
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
