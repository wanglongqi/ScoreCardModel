# ScoreCardModel

A professional and modern toolset for scorecard modeling, fully compatible with `scikit-learn`.

## Features

- **Scikit-learn Compatibility:** Use core components (`BinningTransformer`, `WOETransformer`, `ScoreCardTransformer`) directly in `sklearn.pipeline.Pipeline`.
- **Advanced Binning Engine:** Support for Quantile, Uniform, Optimal (via `optbinning`), and Tree-based binning.
- **Analyst Facade:** `ScoreCardWrapper` provides a simplified API for traditional analyst workflows.
- **Professional Analytics:** Systematic review suite including KS, ROC, Bin Analysis, IV ranking, and PSI metrics.
- **Modern Tooling:** Managed with `uv`, linted with `ruff`, and type-checked with `mypy`.

## Installation

```bash
uv pip install .
```

## Quick Start

### 1. Analyst Facade (Simplified)

```python
from ScoreCardModel import ScoreCardWrapper

# Initialize and fit
sc = ScoreCardWrapper(binning_strategy='optimal')
sc.fit(X_train, y_train)

# Predict scores
scores = sc.predict(X_test)

# Export scorecard table
card = sc.export_scorecard()
```

### 2. Scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from ScoreCardModel import BinningTransformer, WOETransformer

pipeline = Pipeline([
    ('binning', BinningTransformer(strategy='tree')),
    ('woe', WOETransformer()),
    ('model', LogisticRegression())
])

pipeline.fit(X_train, y_train)
```

### 3. Professional Reports

```python
from ScoreCardModel.analytics.plotting import plot_ks, plot_bin_stats
from ScoreCardModel.analytics.metrics import calculate_ks

y_prob = pipeline.predict_proba(X_test)[:, 1]
plot_ks(y_test, y_prob)
```

## Modernization Plan

See [docs/MODERNIZATION_PLAN.md](docs/MODERNIZATION_PLAN.md) for details on the refactoring and enhancements.

## License

MIT
