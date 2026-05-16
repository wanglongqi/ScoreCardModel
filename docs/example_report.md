# Scorecard Model Report

## Summary Metrics

| Metric | Value |
|--------|-------|
| KS Statistic | 0.947 |
| AUC | 0.995 |
| Features | 10 |

## Model Performance

### Score Distribution: Good vs Bad

![Score Distribution: Good vs Bad](example_report_plots/score_distribution_good_vs_bad.png)

### KS Curve

![KS Curve](example_report_plots/ks_curve.png)

### ROC Curve

![ROC Curve](example_report_plots/roc_curve.png)

### Cumulative Accuracy Profile (CAP)

![Cumulative Accuracy Profile (CAP)](example_report_plots/cumulative_accuracy_profile_cap.png)

## Feature Analysis

### Feature IV Ranking

![Feature IV Ranking](example_report_plots/feature_iv_ranking.png)

## Scorecard

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

![Scorecard Points Heatmap](example_report_plots/scorecard_points_heatmap.png)

## Calibration & Cutoff

### Calibration Curve

![Calibration Curve](example_report_plots/calibration_curve.png)

### Cutoff Optimization

![Cutoff Optimization](example_report_plots/cutoff_optimization.png)
