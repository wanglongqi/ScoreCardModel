# ScoreCardModel

A professional and modern toolset for scorecard modeling, fully compatible with `scikit-learn`. Designed for credit risk analysts and data scientists who need transparent, regulator-friendly scoring models.

## Features

- **Scikit-learn Compatibility:** Core components (`BinningTransformer`, `WOETransformer`, `ScoreCardTransformer`) work in any `sklearn.pipeline.Pipeline` or `GridSearchCV`.
- **5 WOE Methods:** Standard, Adjusted (Laplace), Empirical Logit, Signed, and Weighted — each with documented use cases.
- **WOE Diagnostics:** Monotonicity checks, IV-by-bin, chi-square significance, midpoint correlation, and full bin statistics for regulatory documentation.
- **Advanced Binning:** Quantile, Uniform, Optimal (via `optbinning`), and Tree-based strategies with NaN handling and validation.
- **Analyst Facade:** `ScoreCardWrapper` provides a simplified API for traditional analyst workflows. `BaseScorecard` / `ConservativeScorecard` templates for quick starts.
- **Professional Visualization:** 16+ plot types: KS, ROC, CAP, Gain/Lift, Calibration, PSI Drift, WOE Pattern, IV Ranking, Scorecard Heatmap, Cutoff Optimization, and more.
- **Automated HTML Report:** One-line `generate_report()` produces a multi-page professional report with all key plots and metrics.
- **Feature Selection:** IV-based filtering, correlation-based deduplication, and ranked feature diagnostics.
- **Modern Tooling:** Managed with `uv`, linted with `ruff`, type-checked with `mypy`, 98+ tests.

## Installation

```bash
pip install scorecard-toolkit
```

From source:

```bash
git clone https://github.com/wanglongqi/ScoreCardModel.git
cd ScoreCardModel
pip install .
```

## Quick Start

### 1. Analyst Facade (Simplified)

```python
from ScoreCardModel import ScoreCardWrapper

sc = ScoreCardWrapper(binning_strategy='quantile', base_points=600, pdo=20)
sc.fit(X_train, y_train)

scores = sc.predict(X_test)
card = sc.export_scorecard()
print(card.head(10))
```

### 2. Scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from ScoreCardModel import BinningTransformer, WOETransformer

pipeline = Pipeline([
    ('binning', BinningTransformer(strategy='tree', n_bins=5)),
    ('woe', WOETransformer(method='empirical_logit')),
    ('model', LogisticRegression())
])
pipeline.fit(X_train, y_train)
```

### 3. Scorecard Templates

```python
from ScoreCardModel.templates import BaseScorecard, ConservativeScorecard

sc = BaseScorecard()
sc.fit(X_train, y_train)
scores = sc.predict(X_test)
```

### 4. Automated Report

```python
from ScoreCardModel.analytics.reporting import generate_report

generate_report(pipeline, X_train, y_train, X_test, y_test,
                output_path="scorecard_report.html")
```

### 5. Feature Analysis

```python
from ScoreCardModel.analytics.selection import rank_features

ranking = rank_features(X_train, y_train)
print(ranking[['Feature', 'IV', 'IV_Label', 'Monotonicity', 'Recommendation']])
```

## Documentation

See the [full documentation](docs/index.md) for API reference, guides, and examples.

- [WOE In-Depth Guide](docs/woe_in_depth.md) — WOE methods, diagnostics, IV interpretation
- [Best Practices](docs/best_practices.md) — Credit risk scorecard development guidelines
- [API Reference](docs/api.md) — Complete class and function reference

## License

MIT
