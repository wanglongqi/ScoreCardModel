# API Reference

## Core Classes

### BinningTransformer

```python
from ScoreCardModel import BinningTransformer
```

**Strategies:** `quantile`, `uniform`, `optimal`, `tree`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `strategy` | str | `'quantile'` | Binning strategy |
| `n_bins` | int | `5` | Number of bins (ignored by `optimal`) |
| `variables` | list[str] | `None` | Subset of columns to bin |
| `bin_definitions` | dict[str, list[float]] | `{}` | Custom split points per feature |

### WOETransformer

```python
from ScoreCardModel import WOETransformer
```

**Methods:** `standard`, `adjusted`, `empirical_logit`, `signed`, `weighted`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `method` | str | `'adjusted'` | WOE calculation method |
| `laplace_smoothing` | float | `1e-6` | Smoothing for `adjusted` method |
| `rare_lumping` | bool | `False` | Merge rare bins |
| `min_bin_pct` | float | `0.05` | Min bin population % |
| `rare_level_label` | str | `'RARE'` | Label for merged rare bins |

**Properties:**
- `iv` — Information Value per feature
- `woe_maps_` — WOE mapping per feature per bin

### ScoreCardTransformer

```python
from ScoreCardModel import ScoreCardTransformer
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | Any | required | Fitted sklearn model |
| `binning_transformer` | Any | required | Fitted BinningTransformer |
| `woe_transformer` | Any | required | Fitted WOETransformer |
| `base_points` | float | `600` | Base score |
| `base_odds` | float | `50` | Odds at base score |
| `pdo` | float | `20` | Points to double odds |

**Methods:**
- `transform(x)` — Calculate scores
- `export_scorecard()` — Export scorecard table

### ScoreCardWrapper

```python
from ScoreCardModel import ScoreCardWrapper
```

High-level facade combining binning, WOE, logistic regression, and scoring.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `binning_strategy` | str | `'quantile'` | Binning strategy |
| `n_bins` | int | `5` | Number of bins |
| `base_points` | float | `600` | Base score |
| `base_odds` | float | `50` | Odds at base score |
| `pdo` | float | `20` | Points to double odds |

## WOE Methods

| Method | Use Case |
|---|---|
| `standard` | Baseline — `ln(%good / %bad)` |
| `adjusted` | Handles zero-count bins with Laplace smoothing |
| `empirical_logit` | Agresti correction for small datasets |
| `signed` | Preserves direction of rare events |
| `weighted` | Population-weighted WOE |

## Diagnostics

```python
from ScoreCardModel.weight_of_evidence.diagnostics import (
    check_monotonicity, iv_by_bin, woe_chi_square,
    midpoint_correlation, bin_statistics
)
```

## Analytics

### Metrics

```python
from ScoreCardModel.analytics.metrics import (
    calculate_ks, calculate_psi, calculate_accuracy_ratio
)
```

### Plotting (16+ plot types)

```python
from ScoreCardModel.analytics.plotting import (
    plot_ks, plot_roc, plot_cap, plot_gain_lift,
    plot_double_lift, plot_score_distribution, plot_calibration,
    plot_psi_drift, plot_woe_pattern, plot_iv_summary_enhanced,
    plot_event_rate_by_bin, plot_bin_stats, plot_iv_summary,
    plot_scorecard_waterfall, plot_scorecard_heatmap,
    plot_cutoff_optimization, plot_confusion_matrix
)
```

### Feature Selection

```python
from ScoreCardModel.analytics.selection import (
    select_by_iv, select_by_correlation, rank_features
)
```

### Reporting

```python
from ScoreCardModel.analytics.reporting import generate_report
```

## Templates

```python
from ScoreCardModel.templates import BaseScorecard, ConservativeScorecard
```

| Template | Description |
|---|---|
| `BaseScorecard` | Standard development with all features |
| `ConservativeScorecard` | Regulatory submissions, limited feature sets |
