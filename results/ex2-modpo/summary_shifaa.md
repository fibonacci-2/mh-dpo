# Results: modpo vs dpo

Pairwise judge comparisons: 49/50 parsed. modpo wins: 9, dpo wins: 40, modpo win rate: 18.4%

## Mean Likert scores per principle (1-5, judge-rated)

| model   |   empathy |   personalization |   self_exploration |   clarity |   autonomy |   harm_avoidance |   stage_sensitivity |
|:--------|----------:|------------------:|-------------------:|----------:|-----------:|-----------------:|--------------------:|
| dpo     |      4.69 |              4.61 |               4.49 |      4.49 |       4.47 |             4.67 |                4.65 |
| modpo   |      4.18 |              3.76 |               3.35 |      4.14 |       4.08 |             4.47 |                4    |

## Delta (modpo - dpo)

|                   |   delta |
|:------------------|--------:|
| empathy           |   -0.51 |
| personalization   |   -0.85 |
| self_exploration  |   -1.14 |
| clarity           |   -0.35 |
| autonomy          |   -0.39 |
| harm_avoidance    |   -0.2  |
| stage_sensitivity |   -0.65 |

## Judge parse coverage

Absolute scores parsed: {'dpo': 49, 'modpo': 49} out of {'dpo': 50, 'modpo': 50} per model
