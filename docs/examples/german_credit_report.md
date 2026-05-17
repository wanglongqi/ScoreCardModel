# Scorecard Model Report

## Executive Summary

This report evaluates a credit scorecard model built on 13 features using Logistic Regression with **adjusted** WOE transformation. The model achieves a KS statistic of **49.1%** and an AUC of **0.787** (Accuracy Ratio = 0.574), indicating reasonable discriminatory power between good and bad accounts.

**KS = 49.1%** — the maximum separation between cumulative good and bad distributions. This is considered moderate (acceptable) for credit scorecards.

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

The model uses 13 features with a total IV of **2.21**. The chart below ranks features by individual IV contribution.

### Feature IV Ranking

![Feature IV Ranking](german_credit_report_plots/feature_iv_ranking.png)

### Top Features by IV

The table below ranks the top 10 features. Note the **Trend Advice**: features with 'Strong Trend (Minor Violations)' are highly predictive but have minor non-monotonic bins. In many practical cases, keeping these features provides significant performance gains compared to enforcing strict monotonicity.

| Feature            |     IV | Monotonicity   | Trend_Advice                    | Recommendation          |
|:-------------------|-------:|:---------------|:--------------------------------|:------------------------|
| checking_status    | 0.5873 | non-monotonic  | Strong Trend (Minor Violations) | Investigate             |
| credit_history     | 0.3714 | non-monotonic  | Irregular                       | Review (Unstable Trend) |
| purpose            | 0.2299 | non-monotonic  | Irregular                       | Review (Unstable Trend) |
| savings_status     | 0.1629 | non-monotonic  | Irregular                       | Review (Unstable Trend) |
| credit_amount      | 0.1561 | non-monotonic  | Irregular                       | Review (Unstable Trend) |
| duration           | 0.1513 | decreasing     | Good                            | Accept                  |
| property_magnitude | 0.1458 | non-monotonic  | Irregular                       | Review (Unstable Trend) |
| housing            | 0.11   | non-monotonic  | Irregular                       | Review (Unstable Trend) |
| age                | 0.1073 | increasing     | Good                            | Accept                  |
| employment         | 0.0662 | non-monotonic  | Irregular                       | Review (Unstable Trend) |

## Scorecard

The scorecard translates model log-odds into interpretable point values. Each feature is binned, and each bin is assigned a WOE (Weight of Evidence) and a Points value. Higher points indicate lower risk (more "good"-like). The total score for an applicant is the sum of points across all features plus a base offset.

The table below shows the full scorecard (57 rows across 13 features).

| Variable            | Bin                            |        WOE |   Points |
|:--------------------|:-------------------------------|-----------:|---------:|
| checking_status     | 0<=X<200                       | -0.389053  |    30.83 |
| checking_status     | <0                             | -0.788869  |    22.06 |
| checking_status     | >=200                          |  0.504014  |    50.42 |
| checking_status     | no checking                    |  1.07118   |    62.86 |
| credit_history      | all paid                       | -1.0905    |    18.72 |
| credit_history      | critical/other existing credit |  0.86754   |    55.78 |
| credit_history      | delayed previously             |  0.105666  |    41.36 |
| credit_history      | existing paid                  | -0.160963  |    36.32 |
| credit_history      | no credits/all paid            | -1.41373   |    12.6  |
| purpose             | business                       | -0.384106  |    29.72 |
| purpose             | domestic appliance             | -0.448645  |    28.11 |
| purpose             | education                      | -0.85411   |    17.93 |
| purpose             | furniture/equipment            |  0.0801993 |    41.37 |
| purpose             | new car                        | -0.43383   |    28.48 |
| purpose             | other                          | -0.85411   |    17.93 |
| purpose             | radio/tv                       |  0.513722  |    52.25 |
| purpose             | repairs                        |  0.0213588 |    39.9  |
| purpose             | retraining                     |  0.937649  |    62.89 |
| purpose             | used car                       |  0.675285  |    56.31 |
| savings_status      | 100<=X<500                     | -0.182016  |    36.27 |
| savings_status      | 500<=X<1000                    |  0.888859  |    54.48 |
| savings_status      | <100                           | -0.231914  |    35.42 |
| savings_status      | >=1000                         |  1.19358   |    59.66 |
| savings_status      | no known savings               |  0.428806  |    46.65 |
| credit_amount       | (-inf, 1381.75]                | -0.126477  |    36.48 |
| credit_amount       | (1381.75, 2332.0]              |  0.394983  |    48.36 |
| credit_amount       | (2332.0, 4226.0]               |  0.394983  |    48.36 |
| credit_amount       | (4226.0, inf]                  | -0.543054  |    26.99 |
| duration            | (-inf, 12.0]                   |  0.366392  |    44.29 |
| duration            | (12.0, 18.0]                   |  0.147339  |    41.34 |
| duration            | (18.0, 24.0]                   |  0.101402  |    40.73 |
| duration            | (24.0, inf]                    | -0.625575  |    30.95 |
| property_magnitude  | car                            | -0.0431797 |    38.68 |
| property_magnitude  | life insurance                 | -0.047634  |    38.6  |
| property_magnitude  | no known property              | -0.647773  |    29.06 |
| property_magnitude  | real estate                    |  0.544923  |    48.03 |
| housing             | for free                       | -0.642801  |    31.68 |
| housing             | own                            |  0.218116  |    41.97 |
| housing             | rent                           | -0.386769  |    34.74 |
| age                 | (-inf, 27.0]                   | -0.398634  |    29.58 |
| age                 | (27.0, 33.0]                   | -0.114006  |    36.57 |
| age                 | (33.0, 42.0]                   |  0.229235  |    44.99 |
| age                 | (42.0, inf]                    |  0.450354  |    50.41 |
| employment          | 1<=X<4                         | -0.0291934 |    39.09 |
| employment          | 4<=X<7                         |  0.101402  |    40.33 |
| employment          | <1                             | -0.392292  |    35.64 |
| employment          | >=7                            |  0.362285  |    42.8  |
| employment          | unemployed                     | -0.286126  |    36.64 |
| personal_status     | female div/dep/mar             | -0.280309  |    34.93 |
| personal_status     | male div/sep                   | -0.212256  |    36    |
| personal_status     | male mar/wid                   |  0.0841597 |    40.69 |
| personal_status     | male single                    |  0.179744  |    42.21 |
| foreign_worker      | no                             |  1.22533   |    67.52 |
| foreign_worker      | yes                            | -0.0356568 |    38.54 |
| other_payment_plans | bank                           | -0.397351  |    32.74 |
| other_payment_plans | none                           |  0.0903517 |    40.87 |
| other_payment_plans | stores                         | -0.323482  |    33.97 |

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
