# Ex3 negative-generation verification summary

Total generated rows: 624. Judge model: meta-llama/Llama-3.2-3B-Instruct, using the finegrained per-component prompts in `prompts.md`. 100 rows randomly sampled to `results/ex3-negatives/human_annotation_sample.csv` for manual human annotation using the same per-component question.

Judge verdict = whether the finegrained judge for this row's `component_ablated` considers the generated negative (rejected, y_l) a VALID NEGATIVE, i.e. the intended ablation actually took. A healthy generation pass should show a high `valid` rate. `unparsed` rows are ones where the judge's raw output couldn't be parsed as the expected JSON object.

## Judge verdict distribution by component

| component | n | valid | invalid | unparsed |
|---|---|---|---|---|
| cr_distortion_reinforcement | 156 | 78% | 21% | 1% |
| cr_toxic_positivity | 156 | 81% | 16% | 3% |
| ev_ablation | 156 | 100% | 0% | 0% |
| sq_ablation | 156 | 98% | 0% | 2% |