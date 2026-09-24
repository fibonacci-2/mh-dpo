"""RQ1/RQ2/ex4: judge two models' generations against each other with a local
LLM-as-judge (e.g. dpo vs base for RQ1, modpo vs dpo for RQ2, cbtdp vs dpo
for ex4).

Two reference-free evaluation protocols, judged against a **4-principle CBT
rubric** derived directly from the four skills our ex3 negatives ablate
(sq_ablation, ev_ablation, cr_toxic_positivity, cr_distortion_reinforcement
-- see src/prepare_cbtdp_data.py's COMPONENT_META), instead of the PsyCoPref
paper's (https://arxiv.org/abs/2502.19731) 7 generic "PsychoCounsel
Principles". This makes the judge measure the exact skills the CBT-DP-DPO
objective (ex4) is trained to improve, rather than a general
counseling-quality rubric that has no direct link to our ablations.

  1. Pairwise win-rate: the judge is shown a client message and two
     therapist responses (order randomized) and picks the better one
     overall, using the 4 principles below as the rubric.
  2. Absolute Likert scoring: the judge rates each response 1-5 on each
     of the 4 principles independently.

No external LLM API key is required -- a local instruct model stands in as
the judge. Pass --judge-model to swap it (e.g. to compare judges, or to
move up from the original SmolLM3-3B proxy to something stronger); treat
absolute scores as directional, not calibrated, whichever judge is used.
"""
import argparse
import json
import random
import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import default_log_path, setup_logging

# Original proxy judge (kept as the default so already-published ex1/ex2/ex3
# commands stay reproducible byte-for-byte). Pass --judge-model to override,
# e.g. --judge-model NousResearch/Meta-Llama-3.1-8B-Instruct -- already used
# elsewhere in this repo (see src/generate.py's "base" model and
# src/measure_baseline_errorrate.py's GEN_MODEL lineage) so it needs no new
# HF gated-access approval or download on top of what ex1/ex2/ex4 already need.
JUDGE_MODEL = "HuggingFaceTB/SmolLM3-3B"

# 4 CBT principles, one per skill our ex3 negatives ablate (not the 7-item
# generic PsychoCounsel rubric). Arabic wording mirrors the project's fixed
# CBT terminology glossary (الأسئلة السقراطية / التحقق من المشاعر / إعادة
# البناء المعرفي / التشوه المعرفي) so the judge, the ablation prompts, and
# the translation pipeline all describe the same constructs the same way.
# English wording says the same thing, for ex4A (English-only isolation
# test, see README) or any other English-language judging.
#
# Keys are IDENTICAL across languages on purpose: analyze.py imports
# PRINCIPLES purely for its keys (which columns to read from
# judge_absolute.csv), and never sees which language produced them -- so
# the same summary code works whether judge.py was run with --lang ar or
# --lang en.
PRINCIPLES_AR = {
    "sq": (
        "الأسئلة السقراطية: هل يستخدم الرد أسئلة مفتوحة تدعو المستفيد لاستكشاف "
        "أفكاره وافتراضاته بنفسه، بدل إعطائه نصيحة مباشرة أو إخباره بما يجب أن "
        "يفعله أو يشعر به؟"
    ),
    "ev": (
        "التحقق من المشاعر: هل يعترف الرد بمشاعر المستفيد ويتحقق منها فعلياً "
        "بدفء، بدل الانتقال مباشرة إلى منطق بارد أو تحليل جامد يتجاهل الجانب "
        "العاطفي؟"
    ),
    "cr_realistic_reframe": (
        "إعادة البناء المعرفي الواقعية: إذا حاول الرد تقديم منظور بديل، هل هو "
        "واقعي ومتوازن يأخذ صعوبة الموقف على محمل الجد، بدل تفاؤل سطحي أو "
        "إيجابية سامة تتجاهل المشكلة أو تستخف بها؟"
    ),
    "cr_distortion_challenge": (
        "مواجهة التشوه المعرفي: هل يتحدى الرد التشوه المعرفي الظاهر في كلام "
        "المستفيد (كالتهويل أو قراءة الأفكار أو التفكير الثنائي)، بدل أن يعزّزه "
        "أو يوافق عليه ضمنياً؟"
    ),
}

PRINCIPLES_EN = {
    "sq": (
        "Socratic questioning: does the response use open-ended questions that "
        "invite the client to examine their own thoughts and assumptions, "
        "instead of giving direct advice or telling them what to do or feel?"
    ),
    "ev": (
        "Emotional validation: does the response genuinely acknowledge and "
        "validate the client's feelings with warmth, instead of jumping "
        "straight into cold logic or mechanical analysis that ignores the "
        "emotional side?"
    ),
    "cr_realistic_reframe": (
        "Realistic cognitive reframing: if the response offers an alternative "
        "perspective, is it realistic and balanced, taking the difficulty of "
        "the situation seriously, instead of superficial optimism or toxic "
        "positivity that dismisses or minimizes the problem?"
    ),
    "cr_distortion_challenge": (
        "Challenging the distortion: does the response challenge the "
        "cognitive distortion visible in the client's statement (such as "
        "catastrophizing, mind reading, or all-or-nothing thinking), instead "
        "of reinforcing it or implicitly agreeing with it?"
    ),
}

PRINCIPLES_BY_LANG = {"ar": PRINCIPLES_AR, "en": PRINCIPLES_EN}
PRINCIPLES = PRINCIPLES_AR  # backward-compatible default; analyze.py only needs the keys, identical in both languages

PAIRWISE_SYSTEM_AR = (
    "أنت خبير تقييم متخصص في جودة الاستجابات العلاجية النفسية (العلاج السلوكي "
    "المعرفي). سأعطيك رسالة من مستفيد باللغة العربية، ثم استجابتين من معالِجَين "
    "مختلفين (A وB). قارن الاستجابتين وفق المبادئ الأربعة التالية، المشتقة "
    "مباشرة من مهارات العلاج السلوكي المعرفي التي نستهدفها: الأسئلة السقراطية، "
    "التحقق من المشاعر، إعادة البناء المعرفي الواقعية، ومواجهة التشوه المعرفي. "
    "أجب بحرف واحد فقط، A أو B، للإشارة إلى الاستجابة الأفضل إجمالاً، دون أي "
    "شرح إضافي."
)

PAIRWISE_SYSTEM_EN = (
    "You are an expert evaluator specializing in the quality of psychotherapy "
    "responses (Cognitive Behavioral Therapy). I will give you a client's "
    "message, then two therapist responses (A and B). Compare the two "
    "responses on the following 4 principles, derived directly from the CBT "
    "skills we are targeting: Socratic questioning, emotional validation, "
    "realistic cognitive reframing, and challenging the distortion. Reply "
    "with a single letter only, A or B, indicating the overall better "
    "response, with no additional explanation."
)

ABSOLUTE_SYSTEM_AR = (
    "أنت خبير تقييم متخصص في جودة الاستجابات العلاجية النفسية (العلاج السلوكي "
    "المعرفي). سأعطيك رسالة من مستفيد باللغة العربية، واستجابة معالِج واحدة. "
    "قيّم الاستجابة على مقياس من 1 إلى 5 (5 هي الأفضل) لكل مبدأ من المبادئ "
    "التالية، وأعد النتيجة بصيغة JSON فقط بالمفاتيح الإنجليزية التالية بالضبط: "
    + ", ".join(PRINCIPLES_AR.keys())
    + ". لا تكتب أي نص خارج كائن JSON."
)

ABSOLUTE_SYSTEM_EN = (
    "You are an expert evaluator specializing in the quality of psychotherapy "
    "responses (Cognitive Behavioral Therapy). I will give you a client's "
    "message and a single therapist response. Rate the response on a scale "
    "of 1 to 5 (5 is best) for each of the following principles, and return "
    "the result as JSON only, with exactly these English keys: "
    + ", ".join(PRINCIPLES_EN.keys())
    + ". Do not write any text outside the JSON object."
)

PAIRWISE_SYSTEM_BY_LANG = {"ar": PAIRWISE_SYSTEM_AR, "en": PAIRWISE_SYSTEM_EN}
ABSOLUTE_SYSTEM_BY_LANG = {"ar": ABSOLUTE_SYSTEM_AR, "en": ABSOLUTE_SYSTEM_EN}
PAIRWISE_SYSTEM = PAIRWISE_SYSTEM_AR  # backward-compatible default
ABSOLUTE_SYSTEM = ABSOLUTE_SYSTEM_AR  # backward-compatible default

# The user-message wrapper around the client message / responses, per
# language -- kept alongside the system prompts above since which one to
# use depends on the same --lang flag.
USER_LABELS = {
    "ar": {"client": "رسالة المستفيد", "resp_a": "الاستجابة A", "resp_b": "الاستجابة B", "therapist": "استجابة معالِج"},
    "en": {"client": "Client message", "resp_a": "Response A", "resp_b": "Response B", "therapist": "Therapist response"},
}


def load_judge(judge_model):
    tokenizer = AutoTokenizer.from_pretrained(judge_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        judge_model, device_map="auto", dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    return tokenizer, model


def ask(tokenizer, model, system, user, max_new_tokens):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    kwargs = {}
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(
        model.device
    )
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            **kwargs,
        )
    gen = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def parse_winner(raw):
    match = re.search(r"\b([AB])\b", raw.upper())
    return match.group(1) if match else None


def parse_scores(raw):
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    scores = {}
    for key in PRINCIPLES:
        val = obj.get(key)
        try:
            val = float(val)
        except (TypeError, ValueError):
            return None
        scores[key] = val
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-file", required=True, help="Generations CSV for model A (e.g. results/gen_dpo.csv)")
    parser.add_argument("--a-name", required=True, help="Name for model A (e.g. dpo)")
    parser.add_argument("--b-file", required=True, help="Generations CSV for model B (e.g. results/gen_modpo.csv)")
    parser.add_argument("--b-name", required=True, help="Name for model B (e.g. modpo)")
    parser.add_argument("--pairwise-out", default="results/judge_pairwise.csv")
    parser.add_argument("--absolute-out", default="results/judge_absolute.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lang", choices=list(PRINCIPLES_BY_LANG), default="ar",
        help="Language of the client messages/responses being judged, and of "
        "the judge's own prompt (default: ar). Use en for ex4A (English-only "
        "isolation test) or any other English generations. The 4 principle "
        "keys (sq/ev/cr_realistic_reframe/cr_distortion_challenge) are the "
        "same in both languages, so analyze.py needs no changes either way.",
    )
    parser.add_argument(
        "--judge-model",
        default=JUDGE_MODEL,
        help=f"HF model id used as the judge (default: {JUDGE_MODEL}, the original proxy judge). "
        "Pass a stronger local model to judge with a different model, e.g. "
        "NousResearch/Meta-Llama-3.1-8B-Instruct.",
    )
    parser.add_argument(
        "--log-file", default=None, help="Default: logs/<pairwise-out filename stem>.log"
    )
    args = parser.parse_args()

    logger = setup_logging(args.log_file or default_log_path(args.pairwise_out))

    random.seed(args.seed)

    principles = PRINCIPLES_BY_LANG[args.lang]
    pairwise_system = PAIRWISE_SYSTEM_BY_LANG[args.lang]
    absolute_system = ABSOLUTE_SYSTEM_BY_LANG[args.lang]
    labels = USER_LABELS[args.lang]

    a_df = pd.read_csv(args.a_file)
    b_df = pd.read_csv(args.b_file)
    merged = a_df.merge(
        b_df, on=["qid", "Question"], suffixes=(f"_{args.a_name}", f"_{args.b_name}")
    )

    logger.info(f"Loading judge model: {args.judge_model} (lang={args.lang})")
    tokenizer, model = load_judge(args.judge_model)

    principles_desc = "\n".join(f"- {v}" for v in principles.values())

    pairwise_rows = []
    for _, row in merged.iterrows():
        resp_a, resp_b = row[f"response_{args.a_name}"], row[f"response_{args.b_name}"]
        if random.random() < 0.5:
            a, b, a_model, b_model = resp_a, resp_b, args.a_name, args.b_name
        else:
            a, b, a_model, b_model = resp_b, resp_a, args.b_name, args.a_name

        user = (
            f"{labels['client']}:\n{row['Question']}\n\n"
            f"{labels['resp_a']}:\n{a}\n\n{labels['resp_b']}:\n{b}"
        )
        raw = ask(tokenizer, model, pairwise_system, user, max_new_tokens=8)
        winner_letter = parse_winner(raw)
        winner_model = {"A": a_model, "B": b_model}.get(winner_letter)
        pairwise_rows.append(
            {
                "qid": row["qid"],
                "a_model": a_model,
                "b_model": b_model,
                "raw_judgment": raw,
                "winner": winner_model,
                "judge_model": args.judge_model,
            }
        )
        logger.info(f"[pairwise] {row['qid']}: winner={winner_model} raw={raw!r}")

    pd.DataFrame(pairwise_rows).to_csv(args.pairwise_out, index=False)
    logger.info(f"Wrote pairwise judgments to {args.pairwise_out}")

    principles_label = {"ar": "المبادئ", "en": "Principles"}[args.lang]
    absolute_rows = []
    for model_name, df in ((args.a_name, a_df), (args.b_name, b_df)):
        for _, row in df.iterrows():
            user = (
                f"{principles_label}:\n{principles_desc}\n\n"
                f"{labels['client']}:\n{row['Question']}\n\n{labels['therapist']}:\n{row['response']}"
            )
            raw = ask(tokenizer, model, absolute_system, user, max_new_tokens=128)
            scores = parse_scores(raw)
            entry = {"qid": row["qid"], "model": model_name, "raw_judgment": raw, "judge_model": args.judge_model}
            entry.update(scores if scores else {k: None for k in principles})
            absolute_rows.append(entry)
            logger.info(f"[absolute] {row['qid']} ({model_name}): scores={scores}")

    pd.DataFrame(absolute_rows).to_csv(args.absolute_out, index=False)
    logger.info(f"Wrote absolute judgments to {args.absolute_out}")


if __name__ == "__main__":
    main()
