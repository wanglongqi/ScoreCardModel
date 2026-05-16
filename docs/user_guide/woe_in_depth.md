# Weight of Evidence (WOE) — In-Depth Guide

## What is WOE?

Weight of Evidence measures how different a bin's event rate is from the population average:

**Positive WOE** = more good events than expected = lower risk  
**Negative WOE** = more bad events than expected = higher risk  
**WOE of 0** = the bin's risk matches the population average

In scorecard modeling, WOE transforms categorical or binned numerical features into a continuous monotonic scale that has a linear relationship with the log-odds of the target event.

## Why Use WOE in Scorecards?

- **Linearizes log-odds** — enables standard logistic regression
- **Handles missing values** — missing can be treated as its own bin
- **Bin-level interpretability** — each bin gets a transparent risk score
- **Monotonicity** — easier to validate and explain to regulators
- **Handles outliers** — binning + WOE naturally caps extreme values

## WOE vs Other Encoding Methods

| Method | Pros | Cons |
|---|---|---|
| **WOE** | Linearizes log-odds, handles outliers, interpretable | Requires target, loses absolute feature scale |
| **One-Hot** | No target leakage, preserves categories | High dimensionality, no ordinal relationship |
| **Target Encoding** | Simple, captures target correlation | Risk of overfitting, no explicit monotonicity |

## WOE Calculation Methods

All methods take arrays of good and bad counts per bin and return WOE values.

| Method | Formula | When to Use |
|---|---|---|
| **Standard** | `ln(dist_good / dist_bad)` | Default when both classes have sufficient counts in every bin |
| **Adjusted (Laplace)** | `ln((good + s) / (good_total + 2s) / ((bad + s) / (bad_total + 2s)))` | Handles zero-count bins via smoothing. Good default for most use cases. |
| **Empirical Logit** | `ln((good + 0.5) / (bad + 0.5))` | Agresti correction. Standard in SAS-based scorecard development. Good for small bin counts. |
| **Signed** | `sgn(good - bad) * ln(max / min)` | Preserves directional information explicitly. Useful when sign matters for regulatory reporting. |
| **Weighted** | `ln(dist_good / dist_bad) * (bin_total / n_total)` | Downweights small bins to prevent outsized influence from tiny populations. |

Usage:

```python
from ScoreCardModel import WOETransformer

WOETransformer(method='adjusted', laplace_smoothing=1e-6)
WOETransformer(method='empirical_logit')
WOETransformer(method='signed')
WOETransformer(method='weighted')
```

### Choosing a Method

1. **Large dataset, balanced classes** → Standard WOE
2. **Zero-count bins possible** → Adjusted (Laplace) WOE
3. **Small dataset or SAS migration** → Empirical Logit
4. **Regulatory focus on direction** → Signed WOE
5. **Unbalanced bin sizes** → Weighted WOE

## WOE Transformer Features

### Rare Level Lumping

Automatically merge categorical levels with population below `min_bin_pct` into a single `RARE` bin:

```python
WOETransformer(rare_lumping=True, min_bin_pct=0.05)
```

### Unseen Category Handling

Controls how unseen categories in test/validation data are handled:

| Strategy | Behavior |
|---|---|
| `'zero'` | WOE = 0 (no evidence, no adjustment) |
| `'mean'` | WOE = mean of all known WOE values |
| `'min'` | WOE = min of all known WOE values (conservative) |
| `'raise'` | Raise error |

```python
WOETransformer(unseen_strategy='zero')
```

### Missing Value Treatment

| Strategy | Behavior |
|---|---|
| `'separate'` | Treat NaN as its own "MISSING" bin |
| `'mode'` | Impute with most common bin |
| `'raise'` | Fail on missing values |

```python
WOETransformer(missing_strategy='separate')
```

## WOE Diagnostics

Every fitted `WOETransformer` exposes diagnostics for regulatory review:

| Diagnostic | Function | Description |
|---|---|---|
| **Monotonicity** | `check_monotonicity(woe_map, ordered_bins)` | Returns `increasing` / `decreasing` / `non-monotonic`. Uses Spearman rank correlation. |
| **IV Contribution** | `iv_by_bin(good, bad, good_total, bad_total)` | Per-bin IV breakdown. Identifies which bins carry predictive power. |
| **Bin Statistics** | `bin_statistics(bin_series, target)` | Per-bin counts, event rates, WOE values, and population %. |
| **Chi-Square Test** | `woe_chi_square(bin_series, target)` | Test of independence between bins and target. |
| **Midpoint Correlation** | `midpoint_correlation(bin_edges, woe_values)` | For numeric features: correlation between bin midpoints and WOE. High = good linearity. |

```python
from ScoreCardModel.weight_of_evidence.diagnostics import check_monotonicity

wt = WOETransformer().fit(X, y)
for col, woe_map in wt.woe_maps_.items():
    direction, strength = check_monotonicity(woe_map, ordered_bins)
    print(f"{col}: {direction} (strength={strength:.2f})")
```

## Information Value (IV)

IV quantifies a feature's predictive power:

| IV Range | Label | Recommendation |
|---|---|---|
| < 0.02 | Useless | Reject |
| 0.02 – 0.1 | Weak | Accept only with business justification |
| 0.1 – 0.3 | Medium | Good candidate |
| 0.3 – 0.5 | Strong | Excellent candidate |
| > 0.5 | Suspicious | Investigate for data leakage |

IV is accessed directly on a fitted `WOETransformer`:

```python
print(wt.iv)  # dict of {feature_name: iv_value}
```

## Interpreting WOE Patterns

- **Monotonic increasing** — risk decreases as the feature value increases
- **Monotonic decreasing** — risk increases as the feature value increases
- **U-shaped** — both extremes are high-risk (common for age, LTV ratios)
- **Erratic / non-monotonic** — likely overfitting or poor binning. Consider merging adjacent bins.

## Common Pitfalls

1. **Zero-count bins** — Laplace smoothing mitigates this (use `adjusted` method)
2. **Perfect separation** — IV > 0.5 is suspicious; investigate for data leakage
3. **Non-monotonic WOE in production** — monitor for pattern reversal over time
4. **Too many bins** — each bin should have at least 5% of population
5. **Extreme WOE values** — cap at ±3.0 to prevent single bins from dominating scores

## Regulatory Context

WOE-based scorecards are well-suited for regulatory compliance:

- **Basel II/III** — scorecards must be interpretable, stable, and well-documented
- **CCAR** — model use must be clearly defined with performance monitoring
- **FCRA / Reg B** — adverse action reason codes can be derived from per-feature point contributions

The transparency of WOE — each bin maps to a specific point contribution — is a key advantage over black-box models in regulated environments.
