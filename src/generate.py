"""RQ1/RQ2/ex4: generate Arabic CBT-style responses from the DPO model, its
base variant, the RQ2 MODPO LoRA adapter, or the ex4 CBT-DP-DPO LoRA adapter.

Run once per model:
    python src/generate.py --model dpo   --questions data/sampled_questions.csv --out results/gen_dpo.csv
    python src/generate.py --model base  --questions data/sampled_questions.csv --out results/gen_base.csv
    python src/generate.py --model modpo --adapter-path models/modpo-llama3.1-8b-lora \\
        --questions data/sampled_questions.csv --out results/gen_modpo.csv
    python src/generate.py --model cbtdp --adapter-path models/cbtdp-dpo-llama3.1-8b-lora \\
        --questions data/sampled_questions.csv --out results/gen_cbtdp.csv

Note: cbtdp is loaded on top of the "dpo" checkpoint (it's continued-trained
from there, see src/train_cbtdp_dpo.py), while modpo is loaded on top of
"base" -- see ADAPTER_BASE below.
"""
import argparse
import time

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import default_log_path, setup_logging

MODEL_PATHS = {
    # NOT the "-Reward" repo: that's a reward/classification checkpoint
    # (score.weight head, no lm_head) -- loading it via AutoModelForCausalLM
    # silently random-inits the lm_head, producing fluent-looking garbage
    # instead of an error. This is the actual DPO generation model.
    "dpo": "Psychotherapy-LLM/PsyCoPref-Llama3-8B",
    # Same weights as meta-llama/Llama-3.1-8B-Instruct (the DPO model's base,
    # and the base both MODPO and CBT-DP-DPO are LoRA-tuned from -- see
    # src/train_modpo.py, src/train_cbtdp_dpo.py), mirrored without the
    # gated-access requirement.
    "base": "NousResearch/Meta-Llama-3.1-8B-Instruct",
}

# LoRA-adapter models: which MODEL_PATHS checkpoint each adapter was trained
# from, and must be loaded on top of at inference time. modpo trains from the
# untouched base (see src/train_modpo.py); cbtdp (ex4) continues training
# from the DPO checkpoint itself (see src/train_cbtdp_dpo.py) -- loading a
# cbtdp adapter on top of "base" instead would apply a LoRA delta computed
# relative to dpo's weights onto a different set of weights, silently
# producing an incoherent model instead of an error.
ADAPTER_BASE = {"modpo": "base", "cbtdp": "dpo"}

SYSTEM_PROMPTS = {
    "ar": (
        "أنت معالج نفسي متخصص في العلاج السلوكي المعرفي (CBT). "
        "سيرسل لك المستفيد سؤالاً أو مشكلة نفسية بالعربية. قدّم استجابة علاجية "
        "تعتمد على مبادئ العلاج السلوكي المعرفي: أظهر تعاطفاً حقيقياً مع مشاعره، "
        "افهم أفكاره ومخاوفه، ساعده على استكشافها، واقترح خطوات أو استراتيجيات "
        "عملية يمكنه تجربتها. اكتب بأسلوب واضح وموجز ومحترم، بالعربية الفصحى."
    ),
    # For ex4A (English-only isolation test, see README): same instructions,
    # in English, so the client statement is answered in the language it was
    # written in rather than prompting a language switch. Kept identical in
    # substance to the Arabic prompt above -- same CBT principles, same tone
    # -- so ex4A and ex4B differ only in language, not in what's being asked
    # of the model. Mirrors src/measure_baseline_errorrate.py's SYSTEM_PROMPT_EN.
    "en": (
        "You are a psychotherapist specializing in Cognitive Behavioral Therapy (CBT). "
        "A client will send you a statement or psychological concern. Provide a "
        "therapeutic response grounded in CBT principles: show genuine empathy for "
        "their feelings, understand their thoughts and concerns, help them explore "
        "these, and suggest practical steps or strategies they could try. Write "
        "clearly, concisely, and respectfully."
    ),
}


def build_prompts(tokenizer, questions, lang):
    prompts = []
    for q in questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS[lang]},
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
    parser.add_argument("--model", choices=list(MODEL_PATHS.keys()) + list(ADAPTER_BASE.keys()), required=True)
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Required when --model modpo/cbtdp: path to the LoRA adapter saved by "
        "src/train_modpo.py or src/train_cbtdp_dpo.py respectively",
    )
    parser.add_argument("--questions", default="data/sampled_questions.csv")
    parser.add_argument(
        "--lang", choices=list(SYSTEM_PROMPTS), default="ar",
        help="Language of the system prompt (default: ar). Use en for ex4A "
        "(English-only isolation test) or any other English questions file.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--log-file", default=None, help="Default: logs/<out filename stem>.log"
    )
    args = parser.parse_args()

    is_adapter_model = args.model in ADAPTER_BASE
    if is_adapter_model and not args.adapter_path:
        parser.error(f"--adapter-path is required when --model {args.model}")

    logger = setup_logging(args.log_file or default_log_path(args.out))

    # modpo/cbtdp are LoRA adapters loaded on top of whichever checkpoint they
    # were trained from (see ADAPTER_BASE), not their own hub id.
    model_path = MODEL_PATHS[ADAPTER_BASE[args.model]] if is_adapter_model else MODEL_PATHS[args.model]
    df = pd.read_csv(args.questions)

    tokenizer_path = args.adapter_path if is_adapter_model else model_path
    logger.info(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    if is_adapter_model:
        logger.info(f"Loading {args.model.upper()} LoRA adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    questions = df["Question"].tolist()
    prompts = build_prompts(tokenizer, questions, args.lang)

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
