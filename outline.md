# rq1
to what extent english dpo transfer to arabic cbt dialoges?
for this we need to compare performance on arabic cbt task of a model 
that has gone under dpo and its base variant.

Model: https://huggingface.co/Psychotherapy-LLM/PsyCoPref-Llama3-8B


## data
arabic data: data/shifaa-train.csv
use questions only from here. and ask them to the dpo english model and the
base variant. 

# metrics
use same metrics as these papers. the easiest ones to implement to judge quality of cbt response generated 
by model (on the arabic task)
https://huggingface.co/Psychotherapy-LLM/PsyCoPref-Llama3-8B

https://huggingface.co/Psychotherapy-LLM/PsyCoPref-Llama3-8B


# RQ 2 (CBT Objective Design): 
Does incorporating an explicit Cognitive Behavioral Therapy (CBT) distortion-reframing penalty into the DPO loss prevent models from collapsing into over-validating/sycophantic responses across both languages?

# RQ3 (Multilingual Alignment): 
Does joint multilingual CBT-DPO outperform language-specific DPO without degrading base language generation performance?
train 3 lora dpo variants from the same base (meta-llama/Llama-3.1-8B-Instruct) and compare.

## variants
- english-only: existing Psychotherapy-LLM/PsyCoPref-Llama3-8B (no retraining needed)
- arabic-only: lora dpo on base using arabic-translated preference pairs
- joint: lora dpo on base using english + arabic preference pairs combined

## data
english pairs: Psychotherapy-LLM/PsychoCounsel-Preference (36k pairs, already exists)
arabic pairs: machine-translate the same 36k pairs to arabic (chosen/rejected preserved)

## training
lora dpo (peft + trl DPOTrainer), same hyperparams across all 3 variants for a fair comparison

## metrics
same rq1 harness (pairwise win-rate + 7-principle likert judge), run on:
- arabic cbt quality (shifaa/mentalqa questions) for all 3 variants
- english cbt quality (held-out psychocounsel-preference prompts) for all 3, to check degradation
