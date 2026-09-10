# Results: modpo vs dpo

Pairwise judge comparisons: 50/50 parsed. modpo wins: 42, dpo wins: 8, modpo win rate: 84.0%

## Mean Likert scores per principle (1-5, judge-rated)

| model   |   empathy |   personalization |   self_exploration |   clarity |   autonomy |   harm_avoidance |   stage_sensitivity |
|:--------|----------:|------------------:|-------------------:|----------:|-----------:|-----------------:|--------------------:|
| dpo     |      3.04 |              2.29 |               2.27 |      1.78 |       2.67 |             3.18 |                2.49 |
| modpo   |      4.26 |              4    |               3.48 |      4.28 |       4.14 |             4.5  |                4.06 |

## Delta (modpo - dpo)

|                   |   delta |
|:------------------|--------:|
| empathy           |    1.22 |
| personalization   |    1.71 |
| self_exploration  |    1.21 |
| clarity           |    2.5  |
| autonomy          |    1.47 |
| harm_avoidance    |    1.32 |
| stage_sensitivity |    1.57 |

## Judge parse coverage

Absolute scores parsed: {'dpo': 49, 'modpo': 50} out of {'dpo': 50, 'modpo': 50} per model
