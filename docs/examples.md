# Examples

## Feature Selection (Step-by-Step)

Start by ranking all features by Information Value:

```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from ScoreCardModel.analytics.selection import rank_features

COLUMN_MAP = {
    'x1': 'LIMIT_BAL', 'x2': 'SEX', 'x3': 'EDUCATION', 'x4': 'MARRIAGE', 'x5': 'AGE',
    'x6': 'PAY_0', 'x7': 'PAY_2', 'x8': 'PAY_3', 'x9': 'PAY_4', 'x10': 'PAY_5', 'x11': 'PAY_6',
    'x12': 'BILL_AMT1', 'x13': 'BILL_AMT2', 'x14': 'BILL_AMT3', 'x15': 'BILL_AMT4',
    'x16': 'BILL_AMT5', 'x17': 'BILL_AMT6',
    'x18': 'PAY_AMT1', 'x19': 'PAY_AMT2', 'x20': 'PAY_AMT3', 'x21': 'PAY_AMT4',
    'x22': 'PAY_AMT5', 'x23': 'PAY_AMT6',
}

data = fetch_openml('default-of-credit-card-clients', as_frame=True,
                    parser='pandas', version=1)
X = data.data.rename(columns=COLUMN_MAP)
y = (data.target == '0').astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

ranking = rank_features(X_train, y_train, n_bins=5)
print(ranking)
```

Output (23 features ranked by IV):

```
      Feature      IV    IV_Label   Monotonicity  ... Chi2_pvalue Recommendation
0       PAY_0  0.8816  suspicious     decreasing  ...     0.0000    Investigate
1       PAY_2  0.5704  suspicious     decreasing  ...     0.0000    Investigate
2       PAY_3  0.4354      strong  non-monotonic  ...     0.0000         Accept
3       PAY_4  0.3673      strong  non-monotonic  ...     0.0000         Accept
...
12   PAY_AMT5  0.0694        weak     increasing  ...     0.0000         Accept
13   EDUCATION  0.0193     useless  non-monotonic  ...     0.0000         Reject
...
21   MARRIAGE  0.0057     useless  non-monotonic  ...     0.5000         Reject
22        SEX  0.0000     useless     single_bin  ...     1.0000         Reject
```

Feature names are mapped from the original UCI dataset (PAY_0–PAY_6 = repayment status history, LIMIT_BAL = credit limit, PAY_AMT1–PAY_AMT6 = payment amounts, etc.).

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
Candidates: 13 features
After correlation filter: 9 features
['PAY_3', 'LIMIT_BAL', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT6', 'PAY_AMT4', 'PAY_AMT5', 'PAY_0']
```

## End-to-End Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from ScoreCardModel import BinningTransformer, WOETransformer
from ScoreCardModel.analytics.metrics import calculate_ks

pipeline = Pipeline([
    ('binning', BinningTransformer(n_bins=5)),
    ('woe', WOETransformer(method='empirical_logit')),
    ('model', LogisticRegression(max_iter=1000)),
])
pipeline.fit(X_train[final], y_train)

y_prob = pipeline.predict_proba(X_test[final])[:, 1]
ks = calculate_ks(y_test, y_prob)
print(f'KS: {ks:.3f}')
```

Output:

```
KS: 0.398
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
 Variable                  Bin       WOE  Points
    PAY_3          (-1.0, 0.0]  0.304362   62.04
    PAY_3         (-inf, -1.0]  0.353610   62.67
    PAY_3           (0.0, inf] -1.393825   40.22
LIMIT_BAL      (-inf, 50000.0] -0.520152   51.85
LIMIT_BAL (100000.0, 180000.0]  0.159326   60.05
LIMIT_BAL (180000.0, 270000.0]  0.301749   61.77
LIMIT_BAL      (270000.0, inf]  0.589377   65.24
LIMIT_BAL  (50000.0, 100000.0] -0.165176   56.13
 PAY_AMT1        (-inf, 316.0] -0.638325   52.43
 PAY_AMT1     (1714.0, 3000.0]  0.042485   58.50
```

## Visualizations

All plots are generated from a model trained on the Taiwan Credit dataset with 9 features selected via `rank_features()` + `select_by_correlation()`. The plots below are real output — not mockups.

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

[View Taiwan Credit Report](examples/taiwan_credit_report.md)

## Dataset Examples

The following examples demonstrate the full scorecard workflow on four real-world datasets spanning different domains, sizes, and data types.

### German Credit

- **1,000 rows**, 20 features (7 numeric + 13 categorical)
- **30% default rate**
- Demonstrates mixed-type handling — categorical features are auto-detected by `BinningTransformer`
- Full report includes score distribution, KS curve, ROC, scorecard table, and cutoff analysis

```python
data = fetch_openml('credit-g', as_frame=True, parser='pandas')
X, y = data.data, (data.target == 'good').astype(int)

ranking = rank_features(X_train, y_train, n_bins=4)
final = ranking[ranking['Recommendation'].isin(['Accept', 'Investigate'])]['Feature'].tolist()
# 13 features selected
```

```
KS: 0.481
```

[View Report](examples/german_credit_report.md)

### Taiwan Credit (Default of Credit Card Clients)

- **30,000 rows**, 23 numeric features
- **22% default rate**
- Features renamed from x1–x23 to UCI names (PAY_0–PAY_6 = payment status, LIMIT_BAL = credit limit, PAY_AMT1–PAY_AMT6 = payment amounts, etc.)
- Demonstrates large-dataset performance with 9 selected features (after correlation filter)

```python
data = fetch_openml('default-of-credit-card-clients', as_frame=True, parser='pandas', version=1)
X = data.data.rename(columns=COLUMN_MAP)
y = (data.target == '0').astype(int)

ranking = rank_features(X_train, y_train, n_bins=5)
final = select_by_correlation(X_train[candidates], max_corr=0.7)
# 9 features: PAY_0, PAY_3, LIMIT_BAL, PAY_AMT1–PAY_AMT6
```

```
KS: 0.398
```

[View Report](examples/taiwan_credit_report.md)

### Give Me Some Credit

- **150,000 rows**, 10 features (6 integer + 4 float)
- **6.7% default rate** — heavily imbalanced
- Requires missing value imputation for `MonthlyIncome` (20% NaN) and `NumberOfDependents` (2.6% NaN)
- Demonstrates imbalanced dataset handling with 6 selected features

```python
data = fetch_openml('give-me-some-credit', as_frame=True, parser='pandas', version=1)
X, y = data.data, (data.target == '0').astype(int)

X['MonthlyIncome'] = X['MonthlyIncome'].fillna(X['MonthlyIncome'].median())
X['NumberOfDependents'] = X['NumberOfDependents'].fillna(0)

ranking = rank_features(X_train, y_train, n_bins=10)
final = accept + investigate  # 6 features
```

```
KS: 0.368
```

[View Report](examples/give_me_some_credit_report.md)
