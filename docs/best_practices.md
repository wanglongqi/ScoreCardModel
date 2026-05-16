# Credit Risk Scorecard Best Practices

This guide covers regulatory and industry best practices for developing,
validating, and monitoring scorecard models using ScoreCardModel.

## Feature Selection

**IV Thresholds (Regulatory Standard):**

| IV Range | Label | Recommendation |
|---|---|---|
| < 0.02 | Useless | Reject — no predictive power |
| 0.02 – 0.1 | Weak | Accept only with business justification |
| 0.1 – 0.3 | Medium | Good candidate |
| 0.3 – 0.5 | Strong | Excellent candidate |
| > 0.5 | Suspicious | Investigate for data leakage or overfitting |

- Minimum 3 bins per feature for regulatory acceptance
- At least 5% of population in each bin (merge rare bins)
- No pairwise feature correlation above |r| > 0.7
- 8–15 features recommended for a robust scorecard

## WOE Requirements

**Monotonicity:**
- Monotonic or directionally interpretable WOE patterns are preferred by regulators
- Non-monotonic WOE may indicate overfitting or poor binning
- Use `check_monotonicity()` from the diagnostics module to verify

**WOE Ranges:**
- Cap WOE values at ±3.0 to prevent extreme scores from single bins
- WOE of 0 means the bin's risk matches the population average
- Positive WOE = higher proportion of goods (lower risk)

**Bin Quality:**
- No bin should have zero counts of either good or bad events
- Laplace smoothing (`adjusted` method) handles zero-count bins
- For small datasets, use `empirical_logit` method (Agresti correction)

## Scorecard Construction

**PDO and Base Odds:**
- Common settings: PDO = 20 (doubling odds adds 20 points), Base Odds = 50:1 at 600 points
- Verify factor calculation: `factor = PDO / ln(2)`
- Verify offset: `offset = Base Points - factor * ln(Base Odds)`

**Points Distribution:**
- Feature point ranges should be proportional to their contribution
- A feature contributing 50+ points of range is significant
- Features with < 10 points range may not justify their inclusion

## Model Validation

**Holdout Testing:**
- Minimum 30% of data held out for validation
- Time-based out-of-time (OOT) validation required for regulatory submissions
- KS on validation set should be within 0.05 of training KS

**Stability Monitoring (PSI):**

| PSI Range | Interpretation |
|---|---|
| < 0.1 | No significant shift |
| 0.1 – 0.25 | Minor shift — investigate |
| > 0.25 | Major shift — consider retraining |

- Monitor PSI monthly or quarterly
- Compare current score distribution to development distribution

**Backtesting:**
- Compare predicted vs actual default rates quarterly
- Hosmer-Lemeshow test for calibration assessment
- Track feature-level WOE pattern stability

## Regulatory Context (Basel / CCAR)

**Basel II/III:**
- Scorecards must be interpretable (no black-box models)
- Model documentation must include: development data, variable definitions,
  binning rationale, WOE calculations, and validation results
- Annual model review required

**CCAR (Comprehensive Capital Analysis and Review):**
- Model use must be clearly defined
- Performance monitoring must be ongoing
- Model limitations must be documented

**FCRA / Reg B (Adverse Action):**
- Reason codes must be provided to declined applicants
- The top 2–4 contributing features (in points) should be the reason codes
- Use `plot_scorecard_waterfall()` to visualize per-feature point contributions

## Scorecard Templates

| Template | When to Use |
|---|---|
| `BaseScorecard` | Standard development with all features |
| `ConservativeScorecard` | Regulatory submissions, limited feature sets |
| Custom Pipeline | When you need non-LogisticRegression models (use Pipeline) |
