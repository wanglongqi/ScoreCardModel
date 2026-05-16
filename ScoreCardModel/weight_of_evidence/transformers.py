from typing import Dict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class WOETransformer(BaseEstimator, TransformerMixin):
    """
    Optimized WOE transformer using vectorized pandas operations.
    
    Attributes:
        woe_maps_: Dictionary mapping features to their WOE values.
        iv_: Dictionary containing Information Value for each feature.
        laplace_smoothing: Small value to prevent division by zero/log of zero.
    """
    
    def __init__(self, laplace_smoothing: float = 1e-6):
        self.laplace_smoothing = laplace_smoothing

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

        # Validate binary target
        unique_vals = y.unique()
        if not set(unique_vals).issubset({0, 1}):
            raise ValueError(
                f"Target 'y' must be binary with values 0/1, got: {sorted(unique_vals)}"
            )

        # Check for NaN in X
        if x.isna().any().any():
            raise ValueError("Input X contains NaN values. Impute before fitting WOE.")

        self.woe_maps_: Dict[str, Dict[str, float]] = {}
        self.iv_: Dict[str, float] = {}
        for col in x.columns:
            # Group by bin and calculate counts of good (1) and bad (0)
            df = pd.DataFrame({'bin': x[col], 'target': y})
            counts = df.groupby('bin')['target'].agg(['sum', 'count'])
            counts.columns = ['good', 'total']
            counts['bad'] = counts['total'] - counts['good']
            
            # Global totals
            total_good = y.sum()
            total_bad = len(y) - total_good
            
            # Distribution of good/bad
            # Apply Laplace smoothing to avoid 0 counts
            dist_good = (counts['good'] + self.laplace_smoothing) / (total_good + 2 * self.laplace_smoothing)
            dist_bad = (counts['bad'] + self.laplace_smoothing) / (total_bad + 2 * self.laplace_smoothing)
            
            # WOE = ln(dist_good / dist_bad)
            # Standard convention for scorecard: positive WOE means higher probability of 'good'
            woe = np.log(dist_good / dist_bad)
            self.woe_maps_[col] = woe.to_dict()
            
            # IV = sum((dist_good - dist_bad) * WOE)
            self.iv_[col] = ((dist_good - dist_bad) * woe).sum()
            
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Replace bins with their corresponding WOE values."""
        check_is_fitted(self, 'woe_maps_')
        x_out = x.copy()
        
        for col, woe_map in self.woe_maps_.items():
            x_out[col] = x_out[col].map(woe_map).fillna(0.0)
            
        return x_out
