"""RQ1/RQ2: judge two models' generations against each other with a local
LLM-as-judge (e.g. dpo vs base for RQ1, modpo vs dpo for RQ2).

Reimplements, as cheaply as possible, the two reference-free evaluation
protocols from the PsyCoPref paper (https://arxiv.org/abs/2502.19731):

  1. Pairwise win-rate: the judge is shown a client message and two
     therapist responses (order randomized) and picks the better one,
     using the paper's "PsychoCounsel Principles" as the rubric.
  2. Absolute Likert scoring: the judge rates each response 1-5 on each
     of the paper's 7 principles.

The paper's judge was GPT-4o; this environment has no external LLM API
key configured, so a small local instruct model (SmolLM3-3B, which
officially supports Arabic) stands in as the judge instead. This is a
proxy — treat absolute scores as directional, not calibrated.
"""
import argparse
import json
import random
import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import default_log_path, setup_logging

JUDGE_MODEL = "HuggingFaceTB/SmolLM3-3B"

PRINCIPLES = {
    "empathy": "التعاطف والفهم العاطفي: هل يُظهر الرد تعاطفاً حقيقياً مع مشاعر المستفيد؟",
    "personalization": "التخصيص والملاءمة: هل الرد مخصص لتفاصيل حالة المستفيد وليس عاماً؟",
    "self_exploration": "تيسير استكشاف الذات: هل يساعد الرد المستفيد على استكشاف أفكاره ومشاعره؟",
    "clarity": "الوضوح والإيجاز: هل الرد واضح ومنظم وغير مطوّل بلا داعٍ؟",
    "autonomy": "تعزيز الاستقلالية والثقة: هل يشجع الرد المستفيد على اتخاذ خطوات بنفسه بثقة؟",
    "harm_avoidance": "تجنب اللغة الضارة: هل يخلو الرد من أي لغة قد تكون ضارة أو غير مسؤولة؟ (5 = خالٍ تماماً من الضرر)",
    "stage_sensitivity": "الحساسية لمرحلة التغيير: هل يناسب أسلوب الرد المرحلة النفسية التي يبدو المستفيد فيها؟",
}

PAIRWISE_SYSTEM = (
    "أنت خبير تقييم متخصص في جودة الاستجابات العلاجية النفسية (العلاج السلوكي "
    "المعرفي). سأعطيك رسالة من مستفيد باللغة العربية، ثم استجابتين من معالِجَين "
    "مختلفين (A وB). قارن الاستجابتين وفق المبادئ التالية: التعاطف، التخصيص، "
    "تيسير استكشاف الذات، الوضوح والإيجاز، تعزيز الاستقلالية، تجنب اللغة الضارة، "
    "والحساسية لمرحلة التغيير. أجب بحرف واحد فقط، A أو B، للإشارة إلى الاستجابة "
    "الأفضل إجمالاً، دون أي شرح إضافي."
)

ABSOLUTE_SYSTEM = (
    "أنت خبير تقييم متخصص في جودة الاستجابات العلاجية النفسية (العلاج السلوكي "
    "المعرفي). سأعطيك رسالة من مستفيد باللغة العربية، واستجابة معالِج واحدة. "
    "قيّم الاستجابة على مقياس من 1 إلى 5 (5 هي الأفضل) لكل مبدأ من المبادئ "
    "التالية، وأعد النتيجة بصيغة JSON فقط بالمفاتيح الإنجليزية التالية بالضبط: "
    + ", ".join(PRINCIPLES.keys())
    + ". لا تكتب أي نص خارج كائن JSON."
)


def load_judge():
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL, device_map="auto", dtype=torch.bfloat16, low_cpu_mem_usage=True
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
        "--log-file", default=None, help="Default: logs/<pairwise-out filename stem>.log"
    )
    args = parser.parse_args()

    logger = setup_logging(args.log_file or default_log_path(args.pairwise_out))

    random.seed(args.seed)

    a_df = pd.read_csv(args.a_file)
    b_df = pd.read_csv(args.b_file)
    merged = a_df.merge(
        b_df, on=["qid", "Question"], suffixes=(f"_{args.a_name}", f"_{args.b_name}")
    )

    logger.info(f"Loading judge model: {JUDGE_MODEL}")
    tokenizer, model = load_judge()

    principles_desc = "\n".join(f"- {v}" for v in PRINCIPLES.values())

    pairwise_rows = []
    for _, row in merged.iterrows():
        resp_a, resp_b = row[f"response_{args.a_name}"], row[f"response_{args.b_name}"]
        if random.random() < 0.5:
            a, b, a_model, b_model = resp_a, resp_b, args.a_name, args.b_name
        else:
            a, b, a_model, b_model = resp_b, resp_a, args.b_name, args.a_name

        user = (
            f"رسالة المستفيد:\n{row['Question']}\n\n"
            f"الاستجابة A:\n{a}\n\nالاستجابة B:\n{b}"
        )
        raw = ask(tokenizer, model, PAIRWISE_SYSTEM, user, max_new_tokens=8)
        winner_letter = parse_winner(raw)
        winner_model = {"A": a_model, "B": b_model}.get(winner_letter)
        pairwise_rows.append(
            {
                "qid": row["qid"],
                "a_model": a_model,
                "b_model": b_model,
                "raw_judgment": raw,
                "winner": winner_model,
            }
        )
        logger.info(f"[pairwise] {row['qid']}: winner={winner_model} raw={raw!r}")

    pd.DataFrame(pairwise_rows).to_csv(args.pairwise_out, index=False)
    logger.info(f"Wrote pairwise judgments to {args.pairwise_out}")

    absolute_rows = []
    for model_name, df in ((args.a_name, a_df), (args.b_name, b_df)):
        for _, row in df.iterrows():
            user = (
                f"المبادئ:\n{principles_desc}\n\n"
                f"رسالة المستفيد:\n{row['Question']}\n\nاستجابة المعالج:\n{row['response']}"
            )
            raw = ask(tokenizer, model, ABSOLUTE_SYSTEM, user, max_new_tokens=128)
            scores = parse_scores(raw)
            entry = {"qid": row["qid"], "model": model_name, "raw_judgment": raw}
            entry.update(scores if scores else {k: None for k in PRINCIPLES})
            absolute_rows.append(entry)
            logger.info(f"[absolute] {row['qid']} ({model_name}): scores={scores}")

    pd.DataFrame(absolute_rows).to_csv(args.absolute_out, index=False)
    logger.info(f"Wrote absolute judgments to {args.absolute_out}")


if __name__ == "__main__":
    main()
