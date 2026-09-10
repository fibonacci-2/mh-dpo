# RQ1 results: does English DPO transfer to Arabic CBT dialogues?

Pairwise judge comparisons: 50/50 parsed. DPO wins: 1, Base wins: 49, DPO win rate: 2.0%

## Mean Likert scores per principle (1-5, judge-rated)

| model   |   empathy |   personalization |   self_exploration |   clarity |   autonomy |   harm_avoidance |   stage_sensitivity |
|:--------|----------:|------------------:|-------------------:|----------:|-----------:|-----------------:|--------------------:|
| base    |      4.46 |              4.22 |               4.04 |      4.2  |       4.16 |             4.56 |                4.34 |
| dpo     |      2.68 |              1.86 |               2.52 |      1.58 |       2.32 |             3.1  |                2.48 |

## Delta (dpo - base)

|                   |   delta |
|:------------------|--------:|
| empathy           |   -1.78 |
| personalization   |   -2.36 |
| self_exploration  |   -1.52 |
| clarity           |   -2.62 |
| autonomy          |   -1.84 |
| harm_avoidance    |   -1.46 |
| stage_sensitivity |   -1.86 |

## Judge parse coverage

Absolute scores parsed: {'base': 50, 'dpo': 50} out of {'base': 50, 'dpo': 50} per model
