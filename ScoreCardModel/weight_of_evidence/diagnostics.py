
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr


def check_monotonicity(
    woe_map: dict[str, float],
    ordered_bins: list[str],
    strength_threshold: float = 0.8,
) -> tuple[str, float]:
    """Check if WOE values are monotonic across ordered bins.

    Parameters
    ----------
    woe_map : dict
        Mapping of bin labels to WOE values.
    ordered_bins : list
        Bin labels in the natural order (e.g., sorted by feature value).
    strength_threshold : float, default=0.8
        Minimum Spearman correlation to consider monotonic.

    Returns
    -------
    tuple of (str, float)
        Direction: 'increasing', 'decreasing', 'non-monotonic', or 'single_bin'.
        Strength: absolute Spearman correlation.
    """
    valid_bins = [b for b in ordered_bins if b in woe_map]
    if len(valid_bins) < 3:
        return 'single_bin', 1.0

    values = [woe_map[b] for b in valid_bins]
    ranks = np.arange(len(values))
    corr, _ = spearmanr(ranks, values)
    corr = abs(corr)

    if corr >= strength_threshold:
        diffs = np.diff(values)
        if np.all(diffs >= -1e-6):
            return 'increasing', corr
        if np.all(diffs <= 1e-6):
            return 'decreasing', corr

    return 'non-monotonic', corr


def iv_by_bin(
    good: np.ndarray,
    bad: np.ndarray,
    good_total: float,
    bad_total: float,
) -> np.ndarray:
    """Calculate per-bin Information Value contribution.

    IV_i = (dist_good_i - dist_bad_i) * ln(dist_good_i / dist_bad_i)
    """
    dist_good = good / good_total
    dist_bad = bad / bad_total
    woe = np.log(dist_good / dist_bad)
    return (dist_good - dist_bad) * woe


def woe_chi_square(bin_series: pd.Series, target: pd.Series) -> tuple[float, float]:
    """Chi-square test of independence between bins and target.

    Returns (chi2_statistic, p_value).
    A p-value < 0.05 suggests the feature is significantly related to the target.
    """
    contingency = pd.crosstab(bin_series, target)
    stat, pval, _, _ = chi2_contingency(contingency)
    return float(stat), float(pval)


def midpoint_correlation(
    bin_edges: list[float],
    woe_values: list[float],
) -> float:
    """Spearman correlation between bin midpoints and WOE values.

    High absolute correlation (> 0.9) suggests good linearity between
    the feature value and log-odds.
    """
    if len(bin_edges) < 3 or len(woe_values) < 3:
        return 0.0

    midpoints = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
    n = min(len(midpoints), len(woe_values))
    corr, _ = spearmanr(midpoints[:n], woe_values[:n])
    return float(corr) if not np.isnan(corr) else 0.0


def bin_statistics(
    bin_series: pd.Series,
    target: pd.Series,
) -> pd.DataFrame:
    """Calculate per-bin statistics for regulatory documentation.

    Returns DataFrame with columns: bin, total, pop_pct, good, bad,
    event_rate, woe, iv.
    """
    df = pd.DataFrame({'bin': bin_series, 'target': target})
    counts = df.groupby('bin')['target'].agg(['sum', 'count'])
    counts.columns = ['good', 'total']
    counts['bad'] = counts['total'] - counts['good']
    counts['pop_pct'] = counts['total'] / len(df)
    counts['event_rate'] = counts['good'] / counts['total']

    total_good = target.sum()
    total_bad = len(target) - total_good
    dist_good = counts['good'] / total_good
    dist_bad = counts['bad'] / total_bad
    counts['woe'] = np.log(dist_good / dist_bad)
    counts['iv'] = (dist_good - dist_bad) * counts['woe']

    counts = counts.reset_index()
    return counts[['bin', 'total', 'pop_pct', 'good', 'bad', 'event_rate', 'woe', 'iv']]
