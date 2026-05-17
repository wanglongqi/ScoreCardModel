
import numpy as np
import pandas as pd

from ScoreCardModel.binning.transformers import BinningTransformer
from ScoreCardModel.weight_of_evidence.diagnostics import (
    calculate_feature_psi,
    check_monotonicity,
    woe_chi_square,
)
from ScoreCardModel.weight_of_evidence.transformers import WOETransformer


def calculate_psi_report(
    X_expected: pd.DataFrame,
    X_actual: pd.DataFrame,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Calculate PSI for all features in a DataFrame."""
    bt = BinningTransformer(n_bins=n_bins)
    bt.fit(X_expected)
    X_exp_bin = bt.transform(X_expected)
    X_act_bin = bt.transform(X_actual)

    rows = []
    for col in X_expected.columns:
        if col not in X_actual.columns:
            continue
        psi = calculate_feature_psi(X_exp_bin[col], X_act_bin[col])

        if psi < 0.1:
            label = 'stable'
        elif psi < 0.25:
            label = 'warning'
        else:
            label = 'unstable'

        rows.append({'Feature': col, 'PSI': round(psi, 4), 'Status': label})

    return pd.DataFrame(rows).sort_values('PSI', ascending=False).reset_index(drop=True)


def select_by_iv(
    X: pd.DataFrame,
    y: pd.Series,
    min_iv: float = 0.02,
    max_iv: float = 0.5,
    n_bins: int = 5,
) -> list[str]:
    """Select features whose Information Value falls within [min_iv, max_iv]."""
    bt = BinningTransformer(n_bins=n_bins)
    X_bin = bt.fit_transform(X)
    wt = WOETransformer()
    wt.fit(X_bin, y)
    return [feat for feat, iv in wt.iv_.items() if min_iv <= iv <= max_iv]


def select_by_correlation(
    X: pd.DataFrame,
    max_corr: float = 0.7,
) -> list[str]:
    """Remove highly correlated features, keeping first in each pair."""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = set()
    for col in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, col]
            if not np.isnan(val) and val > max_corr:
                to_drop.add(col)
    return [c for c in X.columns if c not in to_drop]


def rank_features(
    X: pd.DataFrame,
    y: pd.Series,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Rank features by IV with diagnostic information.

    Returns a DataFrame with Feature, IV, IV_Label, Monotonicity,
    Mono_Strength, Chi2_pvalue, and Recommendation columns.
    """
    bt = BinningTransformer(n_bins=n_bins)
    X_bin = bt.fit_transform(X)
    wt = WOETransformer()
    wt.fit(X_bin, y)

    rows = []
    for feat in X.columns:
        iv = wt.iv_.get(feat, 0.0)

        if feat in wt.woe_maps_:
            woe_map = wt.woe_maps_[feat]
            ordered_bins = sorted(woe_map.keys())
            mono, strength = check_monotonicity(woe_map, ordered_bins)
        else:
            mono, strength = 'unknown', 0.0

        if feat in X_bin.columns:
            _, pval = woe_chi_square(X_bin[feat], y)
        else:
            pval = 1.0

        if iv < 0.02:
            label = 'useless'
            rec = 'Reject'
        elif iv < 0.1:
            label = 'weak'
            rec = 'Accept'
        elif iv < 0.3:
            label = 'medium'
            rec = 'Accept'
        elif iv < 0.5:
            label = 'strong'
            rec = 'Accept'
        else:
            label = 'suspicious'
            rec = 'Investigate'

        if mono == 'non-monotonic' and strength >= 0.7:
            trend_advice = 'Strong Trend (Minor Violations)'
            if rec == 'Accept':
                rec = 'Accept (Review Trend)'
        elif mono == 'non-monotonic':
            trend_advice = 'Irregular'
            if rec in ['Accept', 'Investigate']:
                rec = 'Review (Unstable Trend)'
        else:
            trend_advice = 'Good'

        rows.append({
            'Feature': feat,
            'IV': round(iv, 4),
            'IV_Label': label,
            'Monotonicity': mono,
            'Mono_Strength': round(strength, 3),
            'Trend_Advice': trend_advice,
            'Chi2_pvalue': round(pval, 4),
            'Recommendation': rec,
        })

    return pd.DataFrame(rows).sort_values('IV', ascending=False).reset_index(drop=True)
