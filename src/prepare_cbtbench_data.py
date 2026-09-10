"""Ex3 data prep: pull the "deliberate practice" reference-response configs
out of CBT-Bench (https://huggingface.co/datasets/Psychotherapy-LLM/CBT-Bench)
to use as y_w (the ground-truth positive response) for negative-response
generation in src/generate_negatives.py.

CBT-Bench ships 10 configs named dp_ref_exe_1..10 (one per deliberate-practice
exercise), each a small set of (client_statement, response) rows where
`response` is an expert-written reference reply. There is no separate
distortion-labeled response dataset in CBT-Bench, so this is the one source
used for all four ablation types in generate_negatives.py.
"""
import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from logging_utils import setup_logging

DATASET_ID = "Psychotherapy-LLM/CBT-Bench"
N_EXERCISES = 10


def load_dp_ref(exercise):
    ds = load_dataset(DATASET_ID, f"dp_ref_exe_{exercise}", split="train")
    df = ds.to_pandas()
    df["exercise"] = exercise
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/cbtbench_pairs.csv")
    parser.add_argument("--log-file", default="logs/prepare_cbtbench_data.log")
    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    logger.info(f"Downloading {DATASET_ID} configs dp_ref_exe_1..{N_EXERCISES}")

    dfs = [load_dp_ref(i) for i in range(1, N_EXERCISES + 1)]
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"{len(df)} client_statement/response pairs across {N_EXERCISES} exercises")

    assert not df["id"].duplicated().any(), "unexpected duplicate ids across exercises"

    df = df.rename(columns={"response": "positive_response"})
    df.insert(0, "qid", df["exercise"].apply(lambda e: f"ex{e:02d}") + "_" + df["id"])
    df = df[["qid", "exercise", "level", "client_statement", "positive_response"]]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info(f"Wrote {len(df)} rows to {args.output}")
    logger.info(f"Level breakdown:\n{df['level'].value_counts()}")


if __name__ == "__main__":
    main()
