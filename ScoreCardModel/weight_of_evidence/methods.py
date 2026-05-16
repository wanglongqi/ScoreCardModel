import numpy as np


def calculate_standard_woe(
    good: np.ndarray,
    bad: np.ndarray,
    good_total: float = 1.0,
    bad_total: float = 1.0,
) -> np.ndarray:
    """Standard WOE = ln(dist_good / dist_bad).

    dist_good = good / good_total
    dist_bad  = bad / bad_total

    Positive WOE means the bin has a higher proportion of goods than the population.
    """
    dist_good = good / good_total
    dist_bad = bad / bad_total
    return np.log(dist_good / dist_bad)


def calculate_adjusted_woe(
    good: np.ndarray,
    bad: np.ndarray,
    good_total: float,
    bad_total: float,
    smoothing: float = 1e-6,
) -> np.ndarray:
    """Adjusted WOE with Laplace smoothing for zero-count bins.

    dist_good = (good + s) / (good_total + 2s)
    dist_bad  = (bad + s) / (bad_total + 2s)

    The smoothing prevents division by zero and log(0) for bins with
    zero good or zero bad counts. The default s = 1e-6 is negligible
    for all practical cases.
    """
    dist_good = (good + smoothing) / (good_total + 2 * smoothing)
    dist_bad = (bad + smoothing) / (bad_total + 2 * smoothing)
    return np.log(dist_good / dist_bad)


def calculate_empirical_logit_woe(
    good: np.ndarray,
    bad: np.ndarray,
    good_total: float = 1.0,
    bad_total: float = 1.0,
) -> np.ndarray:
    """Empirical logit WOE using Agresti correction (add 0.5).

    dist_good = (good + 0.5) / (good_total + 1.0)
    dist_bad  = (bad + 0.5) / (bad_total + 1.0)

    Standard in SAS-based scorecard development. The 0.5 correction
    provides a more robust estimate for bins with small counts
    compared to standard WOE.
    """
    dist_good = (good + 0.5) / (good_total + 1.0)
    dist_bad = (bad + 0.5) / (bad_total + 1.0)
    return np.log(dist_good / dist_bad)


def calculate_signed_woe(
    good: np.ndarray,
    bad: np.ndarray,
    good_total: float = 1.0,
    bad_total: float = 1.0,
) -> np.ndarray:
    """Signed WOE: preserves direction and magnitude explicitly.

    WOE = sgn(good - bad) * ln(max(dist) / min(dist))

    Unlike standard WOE which can produce asymmetric magnitudes,
    signed WOE ensures symmetric positive/negative values around zero.
    """
    dist_good = good / good_total
    dist_bad = bad / bad_total
    ratio = np.maximum(dist_good, dist_bad) / np.minimum(dist_good, dist_bad)
    woe = np.log(ratio)
    return np.where(dist_good >= dist_bad, woe, -woe)


def calculate_weighted_woe(
    good: np.ndarray,
    bad: np.ndarray,
    good_total: float,
    bad_total: float,
    bin_total: np.ndarray,
    n_total: float,
) -> np.ndarray:
    """Weighted WOE: downweights small bins by population proportion.

    WOE_weighted = ln(dist_good / dist_bad) * (bin_total / n_total)

    Useful when some bins have very small populations that might
    otherwise dominate the scorecard through extreme WOE values.
    Bins with zero total population receive WOE = 0.
    """
    dist_good = good / good_total
    dist_bad = bad / bad_total
    weight = bin_total / n_total
    woe = np.log(dist_good / dist_bad)
    woe[bin_total == 0] = 0.0
    return woe * weight
