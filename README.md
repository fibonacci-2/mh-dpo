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


## ex2
train a new model using Multi Objective DPO as outlined in the paper: https://arxiv.org/pdf/2602.16053
data: psyopref dataset, from gh https://huggingface.co/datasets/Psychotherapy-LLM/PsyCoPref, download. it has questions, responses, reward scores.
we might need a seperate reward model for each.


## ex 3: generate negative examples for cognitive distortion
Generate negative responses (y_l) for DPO from CBT-Bench 
We need to define CBT-components, as we discussed in the meeting, that we optimize or base our negative responses generation on. 
Take y_w (from the data) and pass it to GPT or claude model, where we explicitly ask to remove a specific component of CBT. 
Empathetic Validation (EV): Acknowledging and validating emotional distress.
Distortion Identification (DI): Explicitly or implicitly highlighting cognitive distortions.
Socratic Questioning (SQ): Posing open-ended questions to challenge the belief.
Cognitive Reframing (CR): Guiding the patient toward an alternative, realistic perspective.
Socratic Questioning (SQ) Ablation → Unsolicited Advice-Giving
In CBT, therapists must guide clients through self-discovery. This prompt generates a negative response (y_l) that replaces open Socratic inquiry with direct, directive advice or lecturing.
System Prompt:
You are an AI data generator producing rejected (negative) therapeutic responses for DPO training on CBT dialogue datasets.

Task:
Read the client's statement and the valid CBT therapist response (which uses Socratic Questioning). 
Rewrite the therapist response into a flawed response (y_l) that COMPLETELY REMOVES all open-ended questions and Socratic dialogue. 
Instead, make the therapist give immediate, direct, unsolicited advice and tell the client exactly what they should do or feel. Maintain a polite tone, but ensure the response violates the Socratic method.

Input:
- Client Statement: "{client_statement}"
- Target Core Belief / Distortion: "{cbt_label}"
- Ground-Truth Response (y_w): "{positive_response}"

Generated Negative Response (y_l):

Empathetic Validation (EV) Ablation → Cold / Mechanical Clinical Logic
CBT requires balancing cognitive restructuring with strong relational empathy. This prompt strips away emotional validation, producing a cold, mechanical, or overly clinical response.
Plaintext
System Prompt:
You are an AI data generator producing rejected (negative) therapeutic responses for DPO fine-tuning.
Task:
Read the client's statement and the ground-truth therapist response. 
Generate a negative candidate (y_l) by completely ablating all Empathetic Validation (EV). 
The response must strip out any emotional acknowledgment, warmth, or validation of the client's distress. Jump straight into cold, clinical logic, analyzing their cognitive error as if evaluating an equation.
Input:
- Client Statement: "{client_statement}"
- Ground-Truth Response (y_w): "{positive_response}"
Generated Negative Response (y_l):

3. Cognitive Reframing (CR) Failure → Toxic Positivity & Dismissal
Valid reframing seeks realistic alternatives. This prompt generates a negative response (y_l) that attempts cognitive restructuring via superficial, invalidating optimism or toxic positivity.
Plaintext
System Prompt:
You are an AI data generator creating preference optimization pairs for CBT models.
Task:
Convert the provided high-quality therapist response into a flawed Cognitive Reframing attempt (y_l). 
Instead of helping the client construct a realistic, evidence-based alternative perspective, force an unrealistically positive, cheerful, or dismissive reframe ("toxic positivity"). The response should minimize the client's genuine problem.
Input:
- Client Statement: "{client_statement}"
- Target Cognitive Distortion: "{distortion_label}"
- Ground-Truth Response (y_w): "{positive_response}"
Generated Negative Response (y_l):

4. Cognitive Reframing (CR) Failure → Distortion Reinforcement
A critical failure mode in CBT occurs when a therapist inadvertently validates or feeds into the client's cognitive distortion.
Plaintext
System Prompt:
You are an AI dataset compiler generating hard-negative samples (y_l) for LLM therapeutic alignment.
Task:
Read the client's statement containing a specific cognitive distortion. 
Generate a negative response (y_l) where the therapist sounds empathetic and supportive on the surface, but covertly reinforces and validates the client's cognitive distortion (e.g., agreeing that their catastrophic prediction is likely to happen or agreeing with their mind reading).

Input:
- Client Statement: "{client_statement}"
- Cognitive Distortion Present: "{distortion_label}"
Generated Negative Response (y_l):


# Experiment 4 A: Train on English test on English
We would like to assess in isolation the quality of the new proposed objective function. 
This ex will be exactly like the next one, which we already ran, but different in two ways:
* instead of testing on Arabic data, Shifaa and Mental QA, it will test on the test split of the CBT (Abdulrahman created)
* instead of using the 7 principles from psychopref to judge the function, we will use a new set of 4 principles derived from the four CBT skills we are ablating

# Experiment 4 B: English only, train on English → Test on Arabic
L_CBT-DP-DPO(θ) = -E_{(x, y_w, y_l, m, k) ~ D_EN} [ w_k(x) * log σ( R_hat_θ(x, y_w, y_l) - Δ_DP(m) ) ]
(x, y_w, y_l, m, k): metadata tuple:
x: The client's input statement expressing emotional distress or a cognitive distortion.
y_w: The chosen (winning) therapist response, demonstrating sound CBT practice.
y_l: The rejected (losing) therapist response, containing a specific synthetic clinical flaw.
m: The perturbation failure mode tag used to generate y_l (e.g., direct advice, toxic positivity, distortion reinforcement).
k: The CBT skill tag ablated in y_l (e.g., Socratic Questioning, Empathetic Validation, Cognitive Reframing).

## w_k(x) — The Skill-Targeted Deficit Weight
A dynamic weight multiplier applied to the loss of a specific training sample based on the skill k involved.
w_k = 1.0 + ErrorRate_base(k).
k in {SQ, EV, CR_T, CR_D} is the skill tag associated with the preference pair.
ErrorRate_base(k) in [0.0, 1.0] is the empirical failure rate of the baseline model when tested on skill k.
1.0 is the base multiplier floor ensuring that even perfect skills receive standard gradient weighting.
Pre-trained LLMs naturally possess higher competence in some skills (e.g., empathetic validation) than others (e.g., asking open Socratic questions). If the base model has a 50% error rate on Socratic inquiry (w_SQ = 1.5), this term scales gradient updates on Socratic samples by 1.5x, forcing the model to focus gradient steps on its weakest clinical areas.

## Delta_DP(m) — The Dynamic Clinical Risk Margin
Formula: Delta_DP(m) = delta_base + gamma * S(m)
delta_base = 0.2: The minimum baseline margin for all standard pairs.
gamma = 0.2: The clinical risk step size.
S(m) in {0, 1, 2}: Severity level assigned to failure mode m.
Role: In standard DPO, the loss saturates R_hat > 0. Delta_DP(m) forces the model to achieve an implicit reward gap strictly greater than Delta_DP(m) before the loss stops penalizing the model.
For a low-risk procedural error (S = 0), Delta = 0.2.
For a high-risk error like distortion reinforcement (S = 2), Delta = 0.6, forcing the optimizer to heavily suppress probability mass for clinically hazardous outputs.

## Example for calculations
Prompt Template Used
Skill Tag (k)
Failure Mode (m)
Clinical Severity S(m)
Margin ΔDP​(m)
Skill Weight wk​
SQ Ablation Prompt
SQ
Direct_Advice
0 low
0.2
1.5
EV Ablation Prompt
EV
Cold_Logic
0 low
0.2
1
Toxic Positivity Prompt
CR
Toxic_Positivity
1 Med
0.4
1.2
Distortion Reinforcement Prompt
CR
Distortion_Reinforcement
2 High
0.6
1.2


## Contribution:
Evaluation Innovation: Zero-Shot Cross-Lingual Clinical Generalization. evaluate whether learning risk margins in English transfers zero-shot to Arabic without any Arabic fine-tuning.
The objective function on English (clinical- or CBT-based)
