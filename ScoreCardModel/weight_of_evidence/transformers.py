from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ScoreCardModel.weight_of_evidence.methods import (
    calculate_standard_woe,
    calculate_adjusted_woe,
    calculate_empirical_logit_woe,
    calculate_signed_woe,
    calculate_weighted_woe,
)

WOE_METHODS = {
    'standard': calculate_standard_woe,
    'adjusted': calculate_adjusted_woe,
    'empirical_logit': calculate_empirical_logit_woe,
    'signed': calculate_signed_woe,
    'weighted': calculate_weighted_woe,
}


class WOETransformer(BaseEstimator, TransformerMixin):
    """
    Optimized WOE transformer using vectorized pandas operations.

    Parameters
    ----------
    method : str, default='adjusted'
        WOE calculation method. One of: 'standard', 'adjusted', 'empirical_logit',
        'signed', 'weighted'.
    laplace_smoothing : float, default=1e-6
        Smoothing factor for 'adjusted' method to prevent log(0).
    rare_lumping : bool, default=False
        If True, merge rare categories (< min_bin_pct) into a single 'RARE' bin.
    min_bin_pct : float, default=0.05
        Minimum population percentage for a bin (used with rare_lumping).
    rare_level_label : str, default='RARE'
        Label for merged rare categories.

    Attributes:
        woe_maps_: Dictionary mapping features to their WOE values.
        iv_: Dictionary containing Information Value for each feature.
    """

    def __init__(
        self,
        method: str = 'adjusted',
        laplace_smoothing: float = 1e-6,
        rare_lumping: bool = False,
        min_bin_pct: float = 0.05,
        rare_level_label: str = 'RARE',
    ):
        if method not in WOE_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Valid: {sorted(WOE_METHODS)}"
            )
        self.method = method
        self.laplace_smoothing = laplace_smoothing
        self.rare_lumping = rare_lumping
        self.min_bin_pct = min_bin_pct
        self.rare_level_label = rare_level_label

    @property
    def iv(self) -> Dict[str, float]:
        """Expose calculated Information Value for each feature."""
        if not hasattr(self, 'iv_'):
            return {}
        return self.iv_

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "WOETransformer":
        """Calculate WOE maps for each feature."""
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        unique_vals = y.unique()
        if not set(unique_vals).issubset({0, 1}):
            raise ValueError(
                f"Target 'y' must be binary with values 0/1, got: {sorted(unique_vals)}"
            )

        if x.isna().any().any():
            raise ValueError("Input X contains NaN values. Impute before fitting WOE.")

        self.woe_maps_: Dict[str, Dict[str, float]] = {}
        self.iv_: Dict[str, float] = {}

        for col in x.columns:
            df = pd.DataFrame({'bin': x[col], 'target': y})

            if self.rare_lumping:
                counts = df['bin'].value_counts()
                rare_levels = counts[counts / len(df) < self.min_bin_pct].index
                df['bin'] = df['bin'].apply(
                    lambda v: self.rare_level_label if v in rare_levels else v
                )

            grouped = df.groupby('bin')['target'].agg(['sum', 'count'])
            grouped.columns = ['good', 'total']
            grouped['bad'] = grouped['total'] - grouped['good']

            total_good = int(y.sum())
            total_bad = len(y) - total_good

            woe_fn = WOE_METHODS[self.method]
            good_arr = grouped['good'].values.astype(float)
            bad_arr = grouped['bad'].values.astype(float)
            bin_total = grouped['total'].values.astype(float)

            if self.method == 'adjusted':
                woe = woe_fn(good_arr, bad_arr, good_total=total_good, bad_total=total_bad,
                             smoothing=self.laplace_smoothing)
            elif self.method == 'weighted':
                woe = woe_fn(good_arr, bad_arr, good_total=total_good, bad_total=total_bad,
                             bin_total=bin_total, n_total=float(len(y)))
            else:
                woe = woe_fn(good_arr, bad_arr, good_total=total_good, bad_total=total_bad)

            dist_good = good_arr / total_good if total_good > 0 else good_arr
            dist_bad = bad_arr / total_bad if total_bad > 0 else bad_arr
            self.woe_maps_[col] = dict(zip(grouped.index, woe))
            self.iv_[col] = float(((dist_good - dist_bad) * woe).sum())

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Replace bins with their corresponding WOE values."""
        check_is_fitted(self, 'woe_maps_')
        x_out = x.copy()

        for col, woe_map in self.woe_maps_.items():
            x_out[col] = x_out[col].map(woe_map).fillna(0.0)

        return x_out
