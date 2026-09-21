"""Ex4 data prep: turn ex3's verified CBT-Bench negatives into a CBT-DP-DPO
training table, per outline.md's "Experiment A" (the L_CBT-DP-DPO loss).

Each row of results/ex3-negatives/dpo_negatives_cbtbench_verified.csv already
carries `component_ablated` and a judge verdict (`judge_valid`) on whether the
generated negative (y_l) actually exhibits the failure mode it was generated
for. This script:

  1. keeps only judge_valid == True rows (the ablation actually took -- a
     clean synthetic clinical flaw, not noise), and
  2. maps component_ablated to the outline's metadata tuple (m, k, S(m)):

       component_ablated             k (skill)  m (failure mode)            S(m)
       sq_ablation                   SQ         Direct_Advice                0
       ev_ablation                   EV         Cold_Logic                   0
       cr_toxic_positivity           CR         Toxic_Positivity             1
       cr_distortion_reinforcement   CR         Distortion_Reinforcement     2

Two derived columns feed straight into train_cbtdp_dpo.py's loss:

  margin = delta_base + gamma * S(m)      -- Delta_DP(m), the clinical risk margin
  weight = --skill-weights[k]             -- w_k(x), the skill-targeted deficit weight

`--skill-weights` defaults to 1.0 for every skill, i.e. no skill-based
reweighting (see outline.md: w_k = 1.0 + ErrorRate_base(k), and 1.0 is
already the floor for a skill with zero measured baseline error). Passing a
real per-skill ErrorRate_base(k)-derived JSON here is the only change needed
to turn skill reweighting on -- the trainer just reads whatever `weight`
column this script writes.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from logging_utils import default_log_path, setup_logging

COMPONENT_META = {
    "sq_ablation": {"skill": "SQ", "failure_mode": "Direct_Advice", "severity": 0},
    "ev_ablation": {"skill": "EV", "failure_mode": "Cold_Logic", "severity": 0},
    "cr_toxic_positivity": {"skill": "CR", "failure_mode": "Toxic_Positivity", "severity": 1},
    "cr_distortion_reinforcement": {"skill": "CR", "failure_mode": "Distortion_Reinforcement", "severity": 2},
}

DEFAULT_SKILL_WEIGHTS = {"SQ": 1.0, "EV": 1.0, "CR": 1.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--negatives", default="results/ex3-negatives/dpo_negatives_cbtbench_verified.csv")
    parser.add_argument("--output", default="data/cbtdp_dpo_train.csv")
    parser.add_argument(
        "--skill-weights",
        default=json.dumps(DEFAULT_SKILL_WEIGHTS),
        help="JSON object mapping skill (SQ/EV/CR) to w_k, e.g. '{\"SQ\":1.5,\"EV\":1.0,\"CR\":1.2}'. "
        "Default: 1.0 for every skill (no skill-based reweighting).",
    )
    parser.add_argument("--delta-base", type=float, default=0.2, help="delta_base in Delta_DP(m) = delta_base + gamma * S(m)")
    parser.add_argument("--gamma", type=float, default=0.2, help="gamma (clinical risk step size) in Delta_DP(m)")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    logger = setup_logging(args.log_file or default_log_path(args.output))

    skill_weights = json.loads(args.skill_weights)
    missing = set(DEFAULT_SKILL_WEIGHTS) - set(skill_weights)
    if missing:
        parser.error(f"--skill-weights is missing skill(s): {sorted(missing)}")
    logger.info(f"Skill weights w_k: {skill_weights}")
    logger.info(f"Margin formula: Delta_DP(m) = {args.delta_base} + {args.gamma} * S(m)")

    df = pd.read_csv(args.negatives)
    logger.info(f"Loaded {len(df)} rows from {args.negatives}")

    unknown = sorted(set(df["component_ablated"]) - set(COMPONENT_META))
    if unknown:
        raise ValueError(f"No skill/failure-mode mapping for component_ablated value(s): {unknown}")

    before = len(df)
    df = df[df["judge_valid"] == True].copy()  # noqa: E712 -- explicit bool compare, not truthiness
    logger.info(f"Kept {len(df)}/{before} rows with judge_valid == True")

    meta = df["component_ablated"].map(COMPONENT_META)
    df["skill"] = meta.apply(lambda m: m["skill"])
    df["failure_mode"] = meta.apply(lambda m: m["failure_mode"])
    df["severity"] = meta.apply(lambda m: m["severity"])
    df["margin"] = args.delta_base + args.gamma * df["severity"]
    df["weight"] = df["skill"].map(skill_weights)

    out = df[
        [
            "qid",
            "client_statement",
            "chosen",
            "rejected",
            "component_ablated",
            "skill",
            "failure_mode",
            "severity",
            "margin",
            "weight",
        ]
    ]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    logger.info(f"Wrote {len(out)} CBT-DP-DPO training rows to {args.output}")
    logger.info(f"Rows per skill:\n{out['skill'].value_counts()}")
    logger.info(f"Rows per failure_mode (margin):\n{out.groupby('failure_mode')['margin'].agg(['first', 'count'])}")


if __name__ == "__main__":
    main()
