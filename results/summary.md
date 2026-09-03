# RQ1 results: does English DPO transfer to Arabic CBT dialogues?

Pairwise judge comparisons: 49/50 parsed. DPO wins: 44, Base wins: 5, DPO win rate: 89.8%

## Mean Likert scores per principle (1-5, judge-rated)

| model   |   empathy |   personalization |   self_exploration |   clarity |   autonomy |   harm_avoidance |   stage_sensitivity |
|:--------|----------:|------------------:|-------------------:|----------:|-----------:|-----------------:|--------------------:|
| base    |      3.94 |              3.49 |               2.82 |      3.98 |       3.82 |             4.35 |                3.71 |
| dpo     |      4.63 |              4.57 |               4.47 |      4.45 |       4.49 |             4.63 |                4.57 |

## Delta (dpo - base)

|                   |   delta |
|:------------------|--------:|
| empathy           |    0.69 |
| personalization   |    1.08 |
| self_exploration  |    1.65 |
| clarity           |    0.47 |
| autonomy          |    0.67 |
| harm_avoidance    |    0.28 |
| stage_sensitivity |    0.86 |

## Judge parse coverage

Absolute scores parsed: {'base': 49, 'dpo': 49} out of {'base': 50, 'dpo': 50} per model
