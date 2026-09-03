"""RQ1 data prep: sample Arabic questions from a source dataset to feed to the
DPO/base models.

Supports two datasets:
  --dataset shifaa    data/shifaa-train.csv - general Arabic medical
                       consultations, filtered down to the mental-health-
                       relevant rows (same filter as data.ipynb: drop the
                       endocrine/hormones category).
  --dataset mentalqa  data/mentalqa-train-dev.tsv - MentalQA, already
                       mental-health-specific Arabic Q&A, no filtering needed.

Output schema is the same regardless of dataset: qid, Question, category
(category is the dataset's own topic/label field, kept only for reporting).
"""
import argparse
from pathlib import Path

import pandas as pd

from logging_utils import setup_logging

EXCLUDE_CATEGORY = "أمراض الغدد والهرمونات - الهرمونات وأثر اضطرابها"


def load_shifaa(path):
    df = pd.read_csv(path)
    df = df[~df["Hierarchical Diagnosis"].str.contains(EXCLUDE_CATEGORY, na=False)]
    df = df.drop_duplicates(subset="Question")
    return df.rename(columns={"Hierarchical Diagnosis": "category"})[["Question", "category"]]


def load_mentalqa(path):
    df = pd.read_csv(path, sep="\t")
    df = df.drop_duplicates(subset="question")
    df = df.rename(columns={"question": "Question", "final_QT": "category"})
    return df[["Question", "category"]]


DATASETS = {
    "shifaa": {"path": "data/shifaa-train.csv", "loader": load_shifaa},
    "mentalqa": {"path": "data/mentalqa-train-dev.tsv", "loader": load_mentalqa},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS.keys(), default="shifaa")
    parser.add_argument(
        "--input", default=None, help="Override the default source path for --dataset"
    )
    parser.add_argument(
        "--output", default=None, help="Default: data/sampled_questions_<dataset>.csv"
    )
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    input_path = args.input or spec["path"]
    output_path = args.output or f"data/sampled_questions_{args.dataset}.csv"
    log_file = args.log_file or f"logs/prepare_data_{args.dataset}.log"

    logger = setup_logging(log_file)
    logger.info(f"Loading dataset={args.dataset} from {input_path}")

    df = spec["loader"](input_path)
    logger.info(f"{len(df)} unique mental-health questions available after filtering/dedup")

    n = min(args.n, len(df))
    sample = df.sample(n=n, random_state=args.seed).reset_index(drop=True)
    sample.insert(0, "qid", [f"q{i:04d}" for i in range(len(sample))])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)

    logger.info(f"Wrote {len(sample)} sampled questions to {output_path}")
    logger.info(f"Category breakdown:\n{sample['category'].value_counts()}")


if __name__ == "__main__":
    main()
