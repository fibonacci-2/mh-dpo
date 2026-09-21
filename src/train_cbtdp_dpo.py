"""Ex4 ("Experiment A" in outline.md): train a CBT-DP-DPO LoRA adapter on the
English CBT-Bench negatives from ex3, then test zero-shot transfer to Arabic
(see run_exp_4.sh).

CBT-DP-DPO extends DPO with a per-example clinical-risk margin and a
per-example skill-deficit loss weight:

    L_CBT-DP-DPO(theta) = -E[ w_k(x) * log sigmoid( R_hat_theta(x, y_w, y_l) - Delta_DP(m) ) ]

where R_hat_theta(x, y_w, y_l) = beta * logits is the usual DPO implicit
reward gap (beta * the chosen/rejected log-ratio term, relative to the
reference model), and:

  Delta_DP(m): the clinical risk margin for this row's failure mode m --
    standard DPO's loss saturates once R_hat > 0; Delta_DP(m) instead forces
    R_hat strictly above Delta_DP(m) before the loss stops penalizing the
    model, more so for clinically hazardous failure modes.
  w_k(x): the skill-deficit weight for this row's ablated skill k -- scales
    the whole per-example loss (and gradient), so skills the base model is
    weaker on get amplified training signal.

Both `margin` and `weight` are precomputed once per row by
src/prepare_cbtdp_data.py and consumed here as dataset columns, the same
"precompute once, trainer just reads columns" pattern as train_modpo.py's
`margin` column.
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

# Same English CBT system prompt as train_modpo.py, kept identical on purpose:
# both RQ2 (MODPO) and ex4 (CBT-DP-DPO) train on English and are then
# evaluated zero-shot on Arabic (src/generate.py), so the prompt/turn
# structure the model sees at train and inference time should match.
SYSTEM_PROMPT = (
    "You are a psychotherapist specializing in Cognitive Behavioral Therapy "
    "(CBT). A client will send you a message describing a psychological "
    "concern. Respond with a therapeutic reply grounded in CBT principles: "
    "show genuine empathy for their feelings, understand their thoughts and "
    "worries, help them explore those thoughts, and suggest practical steps "
    "or strategies they could try. Write clearly, concisely, and respectfully."
)


def build_dataset(df):
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["client_statement"]},
                ],
                "chosen": [{"role": "assistant", "content": row["chosen"]}],
                "rejected": [{"role": "assistant", "content": row["rejected"]}],
                "margin": float(row["margin"]),
                "weight": float(row["weight"]),
            }
        )
    return Dataset.from_list(records)


class DataCollatorForCBTDP(DataCollatorForPreference):
    """DataCollatorForPreference, extended to pass `margin` and `weight`
    through to the batch (the base collator only knows about
    prompt/chosen/rejected)."""

    def torch_call(self, examples):
        output = super().torch_call(examples)
        for key in ("margin", "weight"):
            if key in examples[0]:
                output[key] = torch.tensor([e[key] for e in examples], dtype=torch.float32)
        return output


class CBTDPTrainer(DPOTrainer):
    def concatenated_forward(self, model, batch, is_ref_model=False):
        model_output = super().concatenated_forward(model, batch, is_ref_model=is_ref_model)
        for key in ("margin", "weight"):
            if key in batch:
                model_output[key] = batch[key].to(model_output["chosen_logps"].device)
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
        weight = model_output["weight"].to(device)
        # R_hat_theta(x, y_w, y_l) = beta * logits; loss = -w_k(x) * log sigmoid(R_hat - Delta_DP(m))
        losses = -weight * F.logsigmoid(self.beta * logits - margin)
        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()
        return losses, chosen_rewards, rejected_rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/cbtdp_dpo_train.csv")
    parser.add_argument("--output-dir", default="models/cbtdp-dpo-llama3.1-8b-lora")
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

    logger.info(f"Loading training data from {args.train_data}")
    df = pd.read_csv(args.train_data)
    logger.info(f"{len(df)} preference pairs loaded")

    dataset = build_dataset(df)
    logger.info(
        f"margin range: [{df['margin'].min():.2f}, {df['margin'].max():.2f}], "
        f"weight range: [{df['weight'].min():.2f}, {df['weight'].max():.2f}]"
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
        max_length=args.max_prompt_length + args.max_completion_length,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,  # keep our custom margin/weight columns alive through the Trainer
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        dataloader_num_workers=0,
        # See train_modpo.py: drop the ragged final batch to avoid a
        # selective_log_softmax shape mismatch under DataParallel.
        dataloader_drop_last=True,
        seed=args.seed,
        report_to="none",
    )

    # No ref_model is passed: DPOTrainer detects the PEFT-wrapped model and
    # computes reference log-probs by disabling the adapter, avoiding a
    # second full model copy (needed to fit on 16GB V100s).
    trainer = CBTDPTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        data_collator=DataCollatorForCBTDP(pad_token_id=tokenizer.pad_token_id),
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    resume = any(Path(args.output_dir).glob("checkpoint-*"))
    if resume:
        logger.info(f"Found existing checkpoint(s) in {args.output_dir}, resuming")
    else:
        logger.info("Starting CBT-DP-DPO training")
    trainer.train(resume_from_checkpoint=resume)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
