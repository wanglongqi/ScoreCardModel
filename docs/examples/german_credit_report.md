# Scorecard Model Report

## Executive Summary

This report evaluates a credit scorecard model built on 13 features using Logistic Regression with WOE transformation. The model achieves a KS statistic of **48.1%** and an AUC of **0.787** (Accuracy Ratio = 0.575), indicating reasonable discriminatory power between good and bad accounts.

**KS = 48.1%** — the maximum separation between cumulative good and bad distributions. This is considered moderate (acceptable) for credit scorecards.

**AUC = 0.787** — the probability that the model ranks a randomly chosen good account higher than a randomly chosen bad account. An AUC of 0.5 is random; values above 0.9 are excellent.

## Model Performance

The four plots below assess the model's ability to separate good from bad accounts across the entire score range.

### Score Distribution: Good vs Bad

Overlaid density of scores for good (blue) vs bad (red) accounts. Good separation means the two distributions have minimal overlap.

![Score Distribution: Good vs Bad](german_credit_report_plots/score_distribution_good_vs_bad.png)

### KS Curve

Cumulative proportion of goods and bads as we move from high-risk to low-risk scores. The KS statistic is the maximum vertical distance between the two curves.

![KS Curve](german_credit_report_plots/ks_curve.png)

### ROC Curve

Trade-off between True Positive Rate (sensitivity) and False Positive Rate (1 - specificity). The diagonal line represents a random model.

![ROC Curve](german_credit_report_plots/roc_curve.png)

### Cumulative Accuracy Profile (CAP)

Cumulative goods captured as a function of the population fraction, ordered by risk score. The Accuracy Ratio (AR) measures how far the model is from random toward perfect.

![Cumulative Accuracy Profile (CAP)](german_credit_report_plots/cumulative_accuracy_profile_cap.png)

## Feature Analysis

Information Value (IV) measures each feature's predictive power. Industry-standard interpretation: <0.02 useless, 0.02–0.1 weak, 0.1–0.3 medium, 0.3–0.5 strong, >0.5 suspicious (investigate for data leakage).

The model uses 13 features with a total IV of **2.18**. The chart below ranks features by individual IV contribution.

### Feature IV Ranking

![Feature IV Ranking](german_credit_report_plots/feature_iv_ranking.png)

### Top Features by IV

| Feature         |     IV | Monotonicity   | Recommendation   |
|:----------------|-------:|:---------------|:-----------------|
| checking_status | 0.5873 | non-monotonic  | Investigate      |
| credit_history  | 0.3714 | non-monotonic  | Accept           |
| purpose         | 0.2299 | non-monotonic  | Accept           |
| savings_status  | 0.1629 | non-monotonic  | Accept           |
| credit_amount   | 0.1561 | non-monotonic  | Accept           |

## Scorecard

The scorecard translates model log-odds into interpretable point values. Each feature is binned, and each bin is assigned a WOE (Weight of Evidence) and a Points value. Higher points indicate lower risk (more "good"-like). The total score for an applicant is the sum of points across all features plus a base offset.

The table below shows the full scorecard (57 rows across 13 features).

| Variable            | Bin                            |        WOE |   Points |
|:--------------------|:-------------------------------|-----------:|---------:|
| checking_status     | 0<=X<200                       | -0.388748  |    30.84 |
| checking_status     | <0                             | -0.786484  |    22.04 |
| checking_status     | >=200                          |  0.46687   |    49.76 |
| checking_status     | no checking                    |  1.06182   |    62.91 |
| credit_history      | all paid                       | -1.08095   |    18.94 |
| credit_history      | critical/other existing credit |  0.857564  |    55.69 |
| credit_history      | delayed previously             |  0.0915878 |    41.17 |
| credit_history      | existing paid                  | -0.160318  |    36.39 |
| credit_history      | no credits/all paid            | -1.3937    |    13.01 |
| purpose             | business                       | -0.388748  |    29.6  |
| purpose             | domestic appliance             | -0.483646  |    27.2  |
| purpose             | education                      | -0.851371  |    17.9  |
| purpose             | furniture/equipment            |  0.0738349 |    41.3  |
| purpose             | new car                        | -0.433636  |    28.46 |
| purpose             | other                          | -0.851371  |    17.9  |
| purpose             | radio/tv                       |  0.507441  |    52.27 |
| purpose             | repairs                        | -0.0303906 |    38.66 |
| purpose             | retraining                     |  0.614966  |    54.99 |
| purpose             | used car                       |  0.648582  |    55.84 |
| savings_status      | 100<=X<500                     | -0.189315  |    36.08 |
| savings_status      | 500<=X<1000                    |  0.835028  |    54.21 |
| savings_status      | <100                           | -0.230758  |    35.35 |
| savings_status      | >=1000                         |  1.09454   |    58.8  |
| savings_status      | no known savings               |  0.418784  |    46.84 |
| credit_amount       | (-inf, 1381.75]                | -0.128243  |    36.5  |
| credit_amount       | (1381.75, 2332.0]              |  0.388653  |    48.31 |
| credit_amount       | (2332.0, 4226.0]               |  0.388653  |    48.31 |
| credit_amount       | (4226.0, inf]                  | -0.542112  |    27.04 |
| duration            | (-inf, 12.0]                   |  0.362643  |    44.42 |
| duration            | (12.0, 18.0]                   |  0.141373  |    41.38 |
| duration            | (18.0, 24.0]                   |  0.0973548 |    40.77 |
| duration            | (24.0, inf]                    | -0.624209  |    30.85 |
| property_magnitude  | car                            | -0.0442797 |    38.72 |
| property_magnitude  | life insurance                 | -0.0503913 |    38.62 |
| property_magnitude  | no known property              | -0.646959  |    29.02 |
| property_magnitude  | real estate                    |  0.538083  |    48.09 |
| housing             | for free                       | -0.642826  |    31.73 |
| housing             | own                            |  0.218292  |    42.05 |
| housing             | rent                           | -0.387968  |    34.78 |
| age                 | (-inf, 27.0]                   | -0.398229  |    29.56 |
| age                 | (27.0, 33.0]                   | -0.11626   |    36.55 |
| age                 | (33.0, 42.0]                   |  0.224513  |    45    |
| age                 | (42.0, inf]                    |  0.442777  |    50.41 |
| employment          | 1<=X<4                         | -0.0303906 |    39.15 |
| employment          | 4<=X<7                         |  0.0954351 |    40.33 |
| employment          | <1                             | -0.393538  |    35.74 |
| employment          | >=7                            |  0.356298  |    42.77 |
| employment          | unemployed                     | -0.295845  |    36.66 |
| personal_status     | female div/dep/mar             | -0.280286  |    34.82 |
| personal_status     | male div/sep                   | -0.232332  |    35.61 |
| personal_status     | male mar/wid                   |  0.0703104 |    40.59 |
| personal_status     | male single                    |  0.179303  |    42.38 |
| foreign_worker      | no                             |  1.09454   |    65.7  |
| foreign_worker      | yes                            | -0.0342722 |    38.61 |
| other_payment_plans | bank                           | -0.399386  |    32.68 |
| other_payment_plans | none                           |  0.0911966 |    40.97 |
| other_payment_plans | stores                         | -0.340546  |    33.68 |

### Scorecard Points Heatmap

The heatmap provides a bird's-eye view of the scorecard. Green cells = higher points (lower risk), red cells = lower points (higher risk). Consistent color gradients within each feature indicate good monotonicity.

![Scorecard Points Heatmap](german_credit_report_plots/scorecard_points_heatmap.png)

## Calibration & Cutoff

The base event rate in the test set is **69.7%**. The plots below assess probability calibration and help select an optimal decision threshold.

### Calibration Curve

Compares predicted probabilities against observed event rates. A well-calibrated model follows the diagonal. Points above the line mean the model underestimates risk; below the line means it overestimates.

![Calibration Curve](german_credit_report_plots/calibration_curve.png)

### Cutoff Optimization

Shows how approval rate, bad rate, and relative profit change with the score cutoff. The optimal cutoff balances the cost of false positives (approving a bad account) against false negatives (rejecting a good account).

![Cutoff Optimization](german_credit_report_plots/cutoff_optimization.png)
