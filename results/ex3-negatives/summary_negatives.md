# Ex3 negative-generation verification summary

Total generated rows: 624. Judge model: meta-llama/Llama-3.2-3B-Instruct. 100 rows randomly sampled to `results/ex3-negatives/human_annotation_sample.csv` for manual human annotation using the same chosen/rejected/tie scheme.

Judge verdict = which response the judge preferred in a randomized pairwise comparison of chosen (y_w) vs. the generated negative (rejected, y_l). A healthy ablation should show a high `chosen` rate (the judge still prefers the ground truth).

## Judge verdict distribution by component

| component | n | chosen | rejected | tie |
|---|---|---|---|---|
| cr_distortion_reinforcement | 156 | 83% | 17% | 0% |
| cr_toxic_positivity | 156 | 96% | 4% | 0% |
| ev_ablation | 156 | 72% | 28% | 1% |
| sq_ablation | 156 | 94% | 6% | 0% |