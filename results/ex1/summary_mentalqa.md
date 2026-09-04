# RQ1 results: does English DPO transfer to Arabic CBT dialogues?

Pairwise judge comparisons: 50/50 parsed. DPO wins: 36, Base wins: 14, DPO win rate: 72.0%

## Mean Likert scores per principle (1-5, judge-rated)

| model   |   empathy |   personalization |   self_exploration |   clarity |   autonomy |   harm_avoidance |   stage_sensitivity |
|:--------|----------:|------------------:|-------------------:|----------:|-----------:|-----------------:|--------------------:|
| base    |      4.46 |              4.28 |               4.04 |      4.22 |       4.22 |             4.58 |                4.36 |
| dpo     |      4.9  |              4.86 |               4.88 |      4.68 |       4.66 |             4.94 |                4.86 |

## Delta (dpo - base)

|                   |   delta |
|:------------------|--------:|
| empathy           |    0.44 |
| personalization   |    0.58 |
| self_exploration  |    0.84 |
| clarity           |    0.46 |
| autonomy          |    0.44 |
| harm_avoidance    |    0.36 |
| stage_sensitivity |    0.5  |

## Judge parse coverage

Absolute scores parsed: {'base': 50, 'dpo': 50} out of {'base': 50, 'dpo': 50} per model
