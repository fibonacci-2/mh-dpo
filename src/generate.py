"""RQ1: generate Arabic CBT-style responses from the DPO model and its base variant.

Run once per model:
    python src/generate.py --model dpo  --questions data/sampled_questions.csv --out results/gen_dpo.csv
    python src/generate.py --model base --questions data/sampled_questions.csv --out results/gen_base.csv
"""
import argparse
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATHS = {
    "dpo": "Psychotherapy-LLM/PsyCoPref-Llama3-8B",
    # Same weights as meta-llama/Llama-3.1-8B-Instruct (the DPO model's base),
    # mirrored without the gated-access requirement.
    "base": "NousResearch/Meta-Llama-3.1-8B-Instruct",
}

SYSTEM_PROMPT = (
    "أنت معالج نفسي متخصص في العلاج السلوكي المعرفي (CBT). "
    "سيرسل لك المستفيد سؤالاً أو مشكلة نفسية بالعربية. قدّم استجابة علاجية "
    "تعتمد على مبادئ العلاج السلوكي المعرفي: أظهر تعاطفاً حقيقياً مع مشاعره، "
    "افهم أفكاره ومخاوفه، ساعده على استكشافها، واقترح خطوات أو استراتيجيات "
    "عملية يمكنه تجربتها. اكتب بأسلوب واضح وموجز ومحترم، بالعربية الفصحى."
)


def build_prompts(tokenizer, questions):
    prompts = []
    for q in questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_PATHS.keys(), required=True)
    parser.add_argument("--questions", default="data/sampled_questions.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    model_path = MODEL_PATHS[args.model]
    df = pd.read_csv(args.questions)

    print(f"Loading tokenizer/model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()

    questions = df["Question"].tolist()
    prompts = build_prompts(tokenizer, questions)

    responses = []
    t0 = time.time()
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_only = out[:, inputs["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        responses.extend([d.strip() for d in decoded])
        elapsed = time.time() - t0
        print(
            f"[{args.model}] {min(i + args.batch_size, len(prompts))}/{len(prompts)} "
            f"done in {elapsed:.0f}s"
        )

    out_df = pd.DataFrame(
        {
            "qid": df["qid"],
            "Question": questions,
            "response": responses,
            "model": args.model,
        }
    )
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} responses to {args.out}")


if __name__ == "__main__":
    main()
