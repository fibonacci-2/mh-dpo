"""RQ1/RQ2: generate Arabic CBT-style responses from the DPO model, its base
variant, or the RQ2 MODPO LoRA adapter.

Run once per model:
    python src/generate.py --model dpo   --questions data/sampled_questions.csv --out results/gen_dpo.csv
    python src/generate.py --model base  --questions data/sampled_questions.csv --out results/gen_base.csv
    python src/generate.py --model modpo --adapter-path models/modpo-llama3.1-8b-lora \\
        --questions data/sampled_questions.csv --out results/gen_modpo.csv
"""
import argparse
import time

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import default_log_path, setup_logging

MODEL_PATHS = {
    "dpo": "Psychotherapy-LLM/PsyCoPref-Llama3-8B-Reward",
    # Same weights as meta-llama/Llama-3.1-8B-Instruct (the DPO model's base,
    # and the base MODPO is LoRA-tuned from in src/train_modpo.py), mirrored
    # without the gated-access requirement.
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
    parser.add_argument("--model", choices=list(MODEL_PATHS.keys()) + ["modpo"], required=True)
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Required when --model modpo: path to the LoRA adapter saved by src/train_modpo.py",
    )
    parser.add_argument("--questions", default="data/sampled_questions.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--log-file", default=None, help="Default: logs/<out filename stem>.log"
    )
    args = parser.parse_args()

    if args.model == "modpo" and not args.adapter_path:
        parser.error("--adapter-path is required when --model modpo")

    logger = setup_logging(args.log_file or default_log_path(args.out))

    # modpo is a LoRA adapter on top of the same base checkpoint as "base",
    # not its own hub id.
    model_path = MODEL_PATHS["base"] if args.model == "modpo" else MODEL_PATHS[args.model]
    df = pd.read_csv(args.questions)

    tokenizer_path = args.adapter_path if args.model == "modpo" else model_path
    logger.info(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    if args.model == "modpo":
        logger.info(f"Loading MODPO LoRA adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
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
        logger.info(
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
    logger.info(f"Wrote {len(out_df)} responses to {args.out}")


if __name__ == "__main__":
    main()
