# Scorecard Model Report

## Executive Summary

This report evaluates a credit scorecard model built on 10 features using Logistic Regression with WOE transformation. The model achieves a KS statistic of **94.7%** and an AUC of **0.995** (Accuracy Ratio = 0.990), indicating strong discriminatory power between good and bad accounts.

**KS = 94.7%** — the maximum separation between cumulative good and bad distributions. A KS above 0.5 is generally considered very strong for credit scorecards.

**AUC = 0.995** — the probability that the model ranks a randomly chosen good account higher than a randomly chosen bad account. An AUC of 0.5 is random; values above 0.9 are excellent.

## Model Performance

The four plots below assess the model's ability to separate good from bad accounts across the entire score range.

### Score Distribution: Good vs Bad

Overlaid density of scores for good (blue) vs bad (red) accounts. Good separation means the two distributions have minimal overlap.

![Score Distribution: Good vs Bad](example_report_plots/score_distribution_good_vs_bad.png)

### KS Curve

Cumulative proportion of goods and bads as we move from high-risk to low-risk scores. The KS statistic is the maximum vertical distance between the two curves.

![KS Curve](example_report_plots/ks_curve.png)

### ROC Curve

Trade-off between True Positive Rate (sensitivity) and False Positive Rate (1 - specificity). The diagonal line represents a random model.

![ROC Curve](example_report_plots/roc_curve.png)

### Cumulative Accuracy Profile (CAP)

Cumulative goods captured as a function of the population fraction, ordered by risk score. The Accuracy Ratio (AR) measures how far the model is from random toward perfect.

![Cumulative Accuracy Profile (CAP)](example_report_plots/cumulative_accuracy_profile_cap.png)

## Feature Analysis

Information Value (IV) measures each feature's predictive power. Industry-standard interpretation: <0.02 useless, 0.02–0.1 weak, 0.1–0.3 medium, 0.3–0.5 strong, >0.5 suspicious (investigate for data leakage).

The model uses 10 features with a total IV of **9.86**. The chart below ranks features by individual IV contribution.

### Feature IV Ranking

![Feature IV Ranking](example_report_plots/feature_iv_ranking.png)

### Top Features by IV

| Feature          |      IV | Monotonicity   | Recommendation   |
|:-----------------|--------:|:---------------|:-----------------|
| worst radius     | 16.7758 | decreasing     | Investigate      |
| worst texture    |  1.2886 | decreasing     | Investigate      |
| worst symmetry   |  0.8829 | decreasing     | Investigate      |
| worst smoothness |  0.8009 | decreasing     | Investigate      |
| mean symmetry    |  0.6387 | decreasing     | Investigate      |

## Scorecard

The scorecard translates model log-odds into interpretable point values. Each feature is binned, and each bin is assigned a WOE (Weight of Evidence) and a Points value. Higher points indicate lower risk (more "good"-like). The total score for an applicant is the sum of points across all features plus a base offset.

The table below shows the full scorecard (50 rows across 10 features).

| Variable                | Bin                |        WOE |   Points |
|:------------------------|:-------------------|-----------:|---------:|
| worst fractal dimension | (-inf, 0.0695]     |  0.507098  |    52.68 |
| worst fractal dimension | (0.0695, 0.0768]   |  0.690441  |    53.62 |
| worst fractal dimension | (0.0768, 0.0831]   |  0.324742  |    51.74 |
| worst fractal dimension | (0.0831, 0.0951]   | -0.234073  |    48.88 |
| worst fractal dimension | (0.0951, inf]      | -1.12173   |    44.34 |
| fractal dimension error | (-inf, 0.00203]    |  0.571393  |    55.64 |
| fractal dimension error | (0.00203, 0.00278] |  0.490148  |    54.85 |
| fractal dimension error | (0.00278, 0.0036]  | -0.110502  |    49.01 |
| fractal dimension error | (0.0036, 0.00479]  | -0.535827  |    44.87 |
| fractal dimension error | (0.00479, inf]     | -0.312649  |    47.04 |
| texture error           | (-inf, 0.784]      |  0.571393  |    56.84 |
| texture error           | (0.784, 1.009]     | -0.284869  |    46.71 |
| texture error           | (1.009, 1.214]     | -0.262646  |    46.97 |
| texture error           | (1.214, 1.562]     | -0.234073  |    47.31 |
| texture error           | (1.562, inf]       |  0.266879  |    53.24 |
| smoothness error        | (-inf, 0.00487]    |  0.15467   |    52.06 |
| smoothness error        | (0.00487, 0.00587] | -0.182919  |    47.74 |
| smoothness error        | (0.00587, 0.00699] | -0.312649  |    46.08 |
| smoothness error        | (0.00699, 0.00878] |  0.0267574 |    50.42 |
| smoothness error        | (0.00878, inf]     |  0.324742  |    54.24 |
| symmetry error          | (-inf, 0.0147]     | -0.362406  |    42.7  |
| symmetry error          | (0.0147, 0.0172]   | -0.0792494 |    48.47 |
| symmetry error          | (0.0172, 0.0198]   |  0.100083  |    52.12 |
| symmetry error          | (0.0198, 0.0244]   |  0.306884  |    56.33 |
| symmetry error          | (0.0244, inf]      |  0.0463659 |    51.03 |
| worst radius            | (-inf, 12.774]     |  4.57058   |   236.44 |
| worst radius            | (12.774, 14.084]   |  2.30923   |   144.24 |
| worst radius            | (14.084, 15.946]   |  0.930384  |    88.02 |
| worst radius            | (15.946, 20.336]   | -1.04841   |     7.33 |
| worst radius            | (20.336, inf]      | -5.59223   |  -177.94 |
| worst texture           | (-inf, 20.098]     |  2.32239   |   132.38 |
| worst texture           | (20.098, 23.524]   |  0.997076  |    85.42 |
| worst texture           | (23.524, 26.502]   | -0.161641  |    44.35 |
| worst texture           | (26.502, 30.748]   | -0.736782  |    23.97 |
| worst texture           | (30.748, inf]      | -1.23188   |     6.43 |
| worst symmetry          | (-inf, 0.243]      |  1.20477   |    74.42 |
| worst symmetry          | (0.243, 0.269]     |  0.746011  |    65.15 |
| worst symmetry          | (0.269, 0.294]     |  0.21023   |    54.33 |
| worst symmetry          | (0.294, 0.322]     | -0.234073  |    45.35 |
| worst symmetry          | (0.322, inf]       | -1.59304   |    17.89 |
| worst smoothness        | (-inf, 0.112]      |  1.28815   |    93.73 |
| worst smoothness        | (0.112, 0.125]     |  0.690441  |    73.48 |
| worst smoothness        | (0.125, 0.137]     |  0.266879  |    59.12 |
| worst smoothness        | (0.137, 0.15]      | -0.485824  |    33.62 |
| worst smoothness        | (0.15, inf]        | -1.34639   |     4.46 |
| mean symmetry           | (-inf, 0.158]      |  1.19028   |    52.58 |
| mean symmetry           | (0.158, 0.172]     |  0.915     |    52    |
| mean symmetry           | (0.172, 0.185]     | -0.212333  |    49.64 |
| mean symmetry           | (0.185, 0.199]     | -0.485824  |    49.06 |
| mean symmetry           | (0.199, inf]       | -0.962811  |    48.06 |

### Scorecard Points Heatmap

The heatmap provides a bird's-eye view of the scorecard. Green cells = higher points (lower risk), red cells = lower points (higher risk). Consistent color gradients within each feature indicate good monotonicity.

![Scorecard Points Heatmap](example_report_plots/scorecard_points_heatmap.png)

## Calibration & Cutoff

The base event rate in the test set is **63.2%**. The plots below assess probability calibration and help select an optimal decision threshold.

### Calibration Curve

Compares predicted probabilities against observed event rates. A well-calibrated model follows the diagonal. Points above the line mean the model underestimates risk; below the line means it overestimates.

![Calibration Curve](example_report_plots/calibration_curve.png)

### Cutoff Optimization

Shows how approval rate, bad rate, and relative profit change with the score cutoff. The optimal cutoff balances the cost of false positives (approving a bad account) against false negatives (rejecting a good account).

![Cutoff Optimization](example_report_plots/cutoff_optimization.png)
