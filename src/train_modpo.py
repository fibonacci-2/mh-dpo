"""RQ2: train a Multi-Objective DPO (MODPO) LoRA adapter on PsyCoPref, as
outlined in https://arxiv.org/pdf/2602.16053.

MODPO extends DPO with a margin term that also pushes the policy toward
higher reward on auxiliary objectives:

    sigmoid_arg = (beta * logits - w_{-k}^T (r_{-k}(y_w) - r_{-k}(y_l))) / w_k

where `logits` is the usual DPO chosen/rejected log-ratio term, `w_k` is the
weight on the dataset's own chosen/rejected preference (the "primary"
objective), and w_{-k} are the weights on the auxiliary objectives (here, the
7 PsyCoPref principles). PsyCoPref already rates both the chosen and rejected
response on those 7 principles, so r_{-k}(x, y) is read directly from the
dataset instead of training separate reward models.

The auxiliary reward difference for each row is precomputed once as a scalar
`margin` column (see build_dataset) and consumed by MODPOTrainer.dpo_loss,
which reimplements DPOTrainer's sigmoid loss with that extra term. Everything
else (data loading, tokenization, reference log-probs via LoRA
adapter-disabling) is standard trl DPOTrainer behavior.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from trl.trainer.dpo_trainer import DataCollatorForPreference

from logging_utils import default_log_path, setup_logging

BASE_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"

PRINCIPLES = ["empathy", "relevance", "clarity", "safety", "exploration", "autonomy", "staging"]

# English counterpart of generate.py's Arabic SYSTEM_PROMPT: PsyCoPref is
# English, but keeping the same system/user turn structure the model will
# see at Arabic inference time (see src/generate.py) is the point of RQ1/RQ2.
SYSTEM_PROMPT = (
    "You are a psychotherapist specializing in Cognitive Behavioral Therapy "
    "(CBT). A client will send you a message describing a psychological "
    "concern. Respond with a therapeutic reply grounded in CBT principles: "
    "show genuine empathy for their feelings, understand their thoughts and "
    "worries, help them explore those thoughts, and suggest practical steps "
    "or strategies they could try. Write clearly, concisely, and respectfully."
)


def build_dataset(df, aux_weight):
    """Convert PsyCoPref rows into trl's conversational DPO format, with a
    precomputed `margin` column holding w_{-k}^T (r_{-k}(y_w) - r_{-k}(y_l)).
    """
    per_dim_weight = aux_weight / len(PRINCIPLES)

    records = []
    for _, row in df.iterrows():
        diffs = [
            (row[f"chosen_{p}_rating"] - row[f"rejected_{p}_rating"]) / 4.0  # 1-5 scale -> [-1, 1]
            for p in PRINCIPLES
        ]
        margin = per_dim_weight * sum(diffs)
        records.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["question"]},
                ],
                "chosen": [{"role": "assistant", "content": row["chosen"]}],
                "rejected": [{"role": "assistant", "content": row["rejected"]}],
                "margin": margin,
            }
        )
    return Dataset.from_list(records)


class DataCollatorForMODPO(DataCollatorForPreference):
    """DataCollatorForPreference, extended to pass the `margin` column through
    to the batch (the base collator only knows about prompt/chosen/rejected).
    """

    def torch_call(self, examples):
        output = super().torch_call(examples)
        if "margin" in examples[0]:
            output["margin"] = torch.tensor([e["margin"] for e in examples], dtype=torch.float32)
        return output


class MODPOTrainer(DPOTrainer):
    def __init__(self, *args, primary_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.primary_weight = primary_weight

    def concatenated_forward(self, model, batch, is_ref_model=False):
        model_output = super().concatenated_forward(model, batch, is_ref_model=is_ref_model)
        if "margin" in batch:
            model_output["margin"] = batch["margin"].to(model_output["chosen_logps"].device)
        return model_output

    def dpo_loss(
        self,
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        loss_type="sigmoid",
        model_output=None,
    ):
        if loss_type != "sigmoid" or model_output is None or "margin" not in model_output:
            return super().dpo_loss(
                chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, loss_type, model_output
            )

        device = self.accelerator.device
        chosen_logratios = chosen_logps.to(device) - (not self.reference_free) * ref_chosen_logps.to(device)
        rejected_logratios = rejected_logps.to(device) - (not self.reference_free) * ref_rejected_logps.to(device)
        if self.reference_free:
            ref_logratios = torch.zeros_like(chosen_logps.to(device))
        else:
            ref_logratios = ref_chosen_logps.to(device) - ref_rejected_logps.to(device)
        logits = (chosen_logps.to(device) - rejected_logps.to(device)) - ref_logratios

        margin = model_output["margin"].to(device)
        losses = -F.logsigmoid((self.beta * logits - margin) / self.primary_weight)
        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()
        return losses, chosen_rewards, rejected_rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/psycopref_train.csv")
    parser.add_argument("--output-dir", default="models/modpo-llama3.1-8b-lora")
    parser.add_argument("--aux-weight", type=float, default=0.5, help="Total weight on the 7 auxiliary principles; the rest goes to the primary chosen/rejected preference")
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=768)
    parser.add_argument("--save-steps", type=int, default=200, help="Checkpoint every N steps, so an interrupted run can resume instead of restarting from scratch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    logger = setup_logging(args.log_file or default_log_path(args.output_dir))

    if not 0.0 < args.aux_weight < 1.0:
        raise ValueError("--aux-weight must be strictly between 0 and 1")
    primary_weight = 1.0 - args.aux_weight

    logger.info(f"Loading training data from {args.train_data}")
    df = pd.read_csv(args.train_data)
    logger.info(f"{len(df)} preference pairs loaded")

    dataset = build_dataset(df, args.aux_weight)
    logger.info(
        f"primary (chosen/rejected) weight={primary_weight:.4f}, "
        f"per-principle auxiliary weight={args.aux_weight / len(PRINCIPLES):.4f} x {len(PRINCIPLES)} principles"
    )

    logger.info(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Required for gradient checkpointing to work with a frozen (LoRA) base
    # model -- otherwise the backward graph has no path to the input and
    # checkpointing silently produces no gradient.
    model.enable_input_require_grads()

    training_args = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        # max_prompt_length=args.max_prompt_length,
        # max_completion_length=args.max_completion_length,
        max_length=args.max_prompt_length + args.max_completion_length,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,  # keep our custom `margin` column alive through the Trainer
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        dataloader_num_workers=0,
        # The final epoch's last batch is smaller than batch_size, and trl's
        # selective_log_softmax hits a shape-mismatch (zip() strict=True) on
        # that ragged batch when the model is split across GPUs via
        # DataParallel. Dropping it costs at most one example out of 34k.
        dataloader_drop_last=True,
        seed=args.seed,
        report_to="none",
    )

    # No ref_model is passed: DPOTrainer detects the PEFT-wrapped model and
    # computes reference log-probs by disabling the adapter, avoiding a
    # second full model copy (needed to fit on 16GB V100s).
    trainer = MODPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        data_collator=DataCollatorForMODPO(pad_token_id=tokenizer.pad_token_id),
        train_dataset=dataset,
        processing_class=tokenizer,
        primary_weight=primary_weight,
    )

    resume = any(Path(args.output_dir).glob("checkpoint-*"))
    if resume:
        logger.info(f"Found existing checkpoint(s) in {args.output_dir}, resuming")
    else:
        logger.info("Starting MODPO training")
    trainer.train(resume_from_checkpoint=resume)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
