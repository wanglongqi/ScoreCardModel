# Examples

## Feature Selection (Step-by-Step)

Start by ranking all features by Information Value:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from ScoreCardModel.analytics.selection import rank_features

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

ranking = rank_features(X_train, y_train)
print(ranking)
```

Output (30 features ranked by IV):

```
                    Feature       IV    IV_Label   Monotonicity  ... Chi2_pvalue Recommendation
0           worst perimeter  17.0221  suspicious  non-monotonic  ...     0.0000    Investigate
1              worst radius  16.7758  suspicious     decreasing  ...     0.0000    Investigate
...
24  worst fractal dimension   0.4379      strong  non-monotonic  ...     0.0000         Accept
25   mean fractal dimension   0.2001      medium  non-monotonic  ...     0.0012         Accept
26  fractal dimension error   0.1916      medium  non-monotonic  ...     0.0017         Accept
27            texture error   0.1189      medium  non-monotonic  ...     0.0314         Accept
28         smoothness error   0.0536        weak  non-monotonic  ...     0.2915         Accept
29           symmetry error   0.0503        weak  non-monotonic  ...     0.3202         Accept
```

The `Recommendation` column guides the decision:
- **Reject** — IV < 0.02 (useless)
- **Accept** — IV 0.02–0.5 (weak to strong)
- **Investigate** — IV > 0.5 (suspicious — review manually)

Filter to "Accept" features, then add "Investigate" ones that are monotonically decreasing (ideal for scorecards). Finally, remove highly correlated features:

```python
from ScoreCardModel.analytics.selection import select_by_correlation

accept = ranking[ranking['Recommendation'] == 'Accept']['Feature'].tolist()
investigate = ranking[
    (ranking['Recommendation'] == 'Investigate') &
    (ranking['Monotonicity'] == 'decreasing')
]['Feature'].tolist()

candidates = accept + investigate
print(f'Candidates: {len(candidates)} features')

final = select_by_correlation(X_train[candidates], max_corr=0.7)
print(f'After correlation filter: {len(final)} features')
print(final)
```

Output:

```
Candidates: 25 features
After correlation filter: 10 features
['worst fractal dimension', 'fractal dimension error', 'texture error',
 'smoothness error', 'symmetry error', 'worst radius', 'worst texture',
 'worst symmetry', 'worst smoothness', 'mean symmetry']
```

## End-to-End Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from ScoreCardModel import BinningTransformer, WOETransformer
from ScoreCardModel.analytics.metrics import calculate_ks

pipeline = Pipeline([
    ('binning', BinningTransformer(strategy='quantile', n_bins=5)),
    ('woe', WOETransformer(method='empirical_logit')),
    ('model', LogisticRegression()),
])
pipeline.fit(X_train[final], y_train)

y_prob = pipeline.predict_proba(X_test[final])[:, 1]
ks = calculate_ks(y_test, y_prob)
print(f'KS: {ks:.3f}')
```

Output:

```
KS: 0.947
```

Export scorecard table:

```python
from ScoreCardModel.score_card.transformers import ScoreCardTransformer

lr = pipeline.named_steps['model']
bt = pipeline.named_steps['binning']
wt = pipeline.named_steps['woe']
sct = ScoreCardTransformer(lr, bt, wt)
card = sct.export_scorecard()
print(card.head(10))
```

Output:

```
              Variable               Bin       WOE   Points
0  worst fractal dimension    (-inf, 0.0803]  0.760721   130.13
1  worst fractal dimension  (0.0803, 0.0902]  0.317370   119.46
2  worst fractal dimension  (0.0902, 0.1039] -0.140957   110.46
3  worst fractal dimension   (0.1039, 0.118] -0.422805   104.22
4  worst fractal dimension     (0.118, inf] -0.757640    96.75
5  worst radius              (-inf, 11.454]  2.731686   183.52
6  worst radius            (11.454, 12.744]  1.985193   167.34
7  worst radius            (12.744, 14.042]  0.952830   145.79
8  worst radius            (14.042, 17.072] -0.845640   109.31
9  worst radius               (17.072, inf] -4.882953     8.13
```

## Visualizations

All plots are generated from a model trained on the breast cancer dataset with features selected via `rank_features()` + `select_by_correlation()` (10 features, quantile binning). The plots below are real output — not mockups.

### Model Discrimination

| KS Curve | ROC Curve | CAP Curve |
|---|---|---|
| ![KS Curve](images/ks_curve.png) | ![ROC Curve](images/roc_curve.png) | ![CAP Curve](images/cap_curve.png) |

KS measures the maximum separation between good and bad populations. ROC shows the trade-off between TPR and FPR. CAP (Cumulative Accuracy Profile) shows the model's cumulative lift over random selection.

### Performance Diagnostics

| Gain / Lift | Score Distribution | Calibration |
|---|---|---|
| ![Gain Lift](images/gain_lift.png) | ![Score Distribution](images/score_distribution.png) | ![Calibration](images/calibration.png) |

Gain/Lift charts show how well the model ranks risk at each decile. Score distribution compares score profiles of good vs bad accounts. Calibration plots assess whether predicted probabilities match observed event rates.

### WOE and IV Analysis

| WOE Pattern | IV Summary |
|---|---|
| ![WOE Pattern](images/woe_pattern.png) | ![IV Summary](images/iv_summary.png) |

WOE pattern plots show the relationship between bins and their Weight of Evidence. IV summary ranks features by Information Value for feature selection.

### Scorecard Interpretation

| Scorecard Waterfall | Scorecard Heatmap |
|---|---|
| ![Scorecard Waterfall](images/scorecard_waterfall.png) | ![Scorecard Heatmap](images/scorecard_heatmap.png) |

Waterfall charts show how each feature contributes to the final score. Heatmaps provide a bird's-eye view of the scorecard table.

### Decision Threshold

| Cutoff Optimization | Confusion Matrix |
|---|---|
| ![Cutoff Optimization](images/cutoff_optimization.png) | ![Confusion Matrix](images/confusion_matrix.png) |

Cutoff optimization identifies the optimal decision threshold by balancing cost/benefit. The confusion matrix shows classification performance at the chosen cutoff.

### Automated Report

```python
from ScoreCardModel.analytics.reporting import generate_report

generate_report(pipeline, X_train, y_train, X_test, y_test,
                output_path="scorecard_report.md")
```

The report is a markdown file with embedded PNG plots and the full scorecard table — suitable for sharing with stakeholders or regulatory review.

[View Example Report](example_report.md)
