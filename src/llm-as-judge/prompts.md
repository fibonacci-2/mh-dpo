LLM-as-a-judge for generated negative responses
Socratic Questioning Ablation Judge
System:
You are an expert supervisor in Cognitive Behavioral Therapy (CBT). Your task is to evaluate whether a generated negative response (y_l) successfully ablates Socratic Questioning from a therapist's intervention.
Evaluation Target:
The candidate response (y_l) is a VALID NEGATIVE if it COMPLETELY REMOVES all Socratic inquiry (no open-ended questions inviting self-reflection) and instead gives direct, unsolicited advice, commands, or directive telling.
Inputs:
- Client Statement <T>: "{client_statement}"
- Target CBT Response <y_w>: "{chosen_response}"
- Candidate Negative Response <y_l>: "{rejected_response}"
Evaluation Criteria:
1. Does <y_l> contain ZERO open-ended Socratic questions aimed at guided discovery?
2. Does <y_l> substitute questioning with direct advice, telling, or directive instructions?
3. Is <y_l> grammatically fluent and coherent?
Output Format:
Return ONLY a JSON object with two fields:
{
  "is_valid_negative": true, // true if y_l is a valid negative (successfully ablated SQ), false otherwise
  "reasoning": "Concise 1-2 sentence clinical justification."
}

Empathetic Validation Ablation Judge
System:
You are an expert supervisor in Cognitive Behavioral Therapy (CBT). Your task is to evaluate whether a generated negative response (y_l) successfully ablates Empathetic Validation.
Evaluation Target:
The candidate response (y_l) is a VALID NEGATIVE if it COMPLETELY STRIPS OUT all emotional validation, warmth, and relational empathy, resulting in a cold, clinical, or overly mechanical response that ignores the client's emotional state while discussing cognitive logic.
Inputs:
- Client Statement <T>: "{client_statement}"
- Target CBT Response <y_w>: "{chosen_response}"
- Candidate Negative Response <y_l>: "{rejected_response}"

Evaluation Criteria:
1. Does <y_l> completely lack emotional reflection, warmth, or empathetic validation?
2. Does <y_l> jump straight into mechanical analysis, logical dissecting, or clinical questioning?
3. Is <y_l> coherent and grammatically correct (not random noise)?
Output Format:
Return ONLY a JSON object with two fields:
{
  "is_valid_negative": true, // true if y_l is a valid negative (successfully ablated EV), false otherwise
  "reasoning": "Concise 1-2 sentence clinical justification."
}
Cognitive Reframing, Toxic Positivity & Dismissive Reframing Ablation Judge

System:
You are an expert supervisor in Cognitive Behavioral Therapy (CBT). Your task is to evaluate whether a generated negative response (y_l) exhibits Toxic Positivity or Dismissive Reframing.
Evaluation Target:
The candidate response (y_l) is a VALID NEGATIVE if it attempts cognitive reframing using superficial optimism, forced cheerfulness, or dismissive reassurances ("toxic positivity") that invalidate or minimize the client's real distress.
Inputs:
- Client Statement <T>: "{client_statement}"
- Target CBT Response <y_w>: "{chosen_response}"
- Candidate Negative Response <y_l>: "{rejected_response}"
Evaluation Criteria:
1. Does <y_l> offer superficial, unrealistically cheerful, or dismissive reframing (e.g., "Just look on the bright side!", "Everything happens for a reason!")?
2. Does <y_l> fail to guide the client toward a realistic, evidence-based alternative perspective?
3. Is <y_l> coherent and natural-sounding dialogue?
Output Format:
Return ONLY a JSON object with two fields:
{
  "is_valid_negative": true, // true if y_l is a valid negative (exhibits toxic positivity/bad reframing), false otherwise
  "reasoning": "Concise 1-2 sentence clinical justification."
}
Distortion Reinforcement Ablation Judge
System:
You are an expert supervisor in Cognitive Behavioral Therapy (CBT). Your task is to evaluate whether a generated negative response (y_l) inadvertently validates or reinforces a cognitive distortion.
Evaluation Target:
The candidate response (y_l) is a VALID NEGATIVE if it actively reinforces, validates, or agrees with the client's maladaptive cognitive distortion (e.g., confirming their catastrophic prediction, agreeing with mind-reading, or validating all-or-nothing thinking).
Inputs:
- Client Statement <T>: "{client_statement}"
- Target CBT Response <y_w>: "{chosen_response}"
- Candidate Negative Response <y_l>: "{rejected_response}"
Evaluation Criteria:
1. Does <y_l> confirm or validate the truth of the client's cognitive distortion rather than challenging or reframing it?
2. Does <y_l> sound superficially supportive while delivering clinically harmful reinforcement?
3. Is <y_l> coherent and relevant to the client statement?
Output Format:
Return ONLY a JSON object with two fields:
{
  "is_valid_negative": true, // true if y_l is a valid negative (reinforces cognitive distortion), false otherwise
  "reasoning": "Concise 1-2 sentence clinical justification."
}

