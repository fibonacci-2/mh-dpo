# ErrorRate_base(k) and derived skill weights w_k

Baseline model: `Psychotherapy-LLM/PsyCoPref-Llama3-8B`. Judge: `meta-llama/Llama-3.2-3B-Instruct`. N=150 CBT-Bench statements (seed=42).

w_k = 1.0 + ErrorRate_base(k)

| skill | ErrorRate_base(k) | w_k |
|---|---:|---:|
| SQ | 0.041 | 1.041 |
| EV | 0.011 | 1.011 |
| CR (avg of toxic_positivity=0.000, distortion_reinforcement=0.000) | 0.000 | 1.0 |

SKILL_WEIGHTS='{"SQ": 1.041, "EV": 1.011, "CR": 1.0}'
