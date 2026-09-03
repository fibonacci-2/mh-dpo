"""RQ1: sample Arabic questions from Shifaa to feed to the DPO/base models.

Filters the Shifaa consultations down to the mental-health-relevant rows
(same filter used in data.ipynb: drop the endocrine/hormones category),
dedupes by question text, and takes a reproducible random sample.
"""
import argparse

import pandas as pd

EXCLUDE_CATEGORY = "أمراض الغدد والهرمونات - الهرمونات وأثر اضطرابها"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/shifaa-train.csv")
    parser.add_argument("--output", default="data/sampled_questions.csv")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df[~df["Hierarchical Diagnosis"].str.contains(EXCLUDE_CATEGORY, na=False)]
    df = df.drop_duplicates(subset="Question")

    n = min(args.n, len(df))
    sample = df.sample(n=n, random_state=args.seed)[["Question", "Hierarchical Diagnosis"]]
    sample = sample.reset_index(drop=True)
    sample.insert(0, "qid", [f"q{i:04d}" for i in range(len(sample))])
    sample.to_csv(args.output, index=False)
    print(f"Wrote {len(sample)} sampled questions to {args.output}")
    print(sample["Hierarchical Diagnosis"].value_counts())


if __name__ == "__main__":
    main()
