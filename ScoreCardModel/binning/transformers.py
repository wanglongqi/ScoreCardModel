from typing import Optional

import numpy as np
import pandas as pd
from optbinning import OptimalBinning
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

VALID_STRATEGIES = {'quantile', 'uniform', 'optimal', 'tree'}


class BinningTransformer(BaseEstimator, TransformerMixin):
    """
    Modern binning transformer supporting multiple strategies.
    
    Strategies:
    - 'quantile': Equal frequency binning.
    - 'uniform': Equal width binning.
    - 'optimal': Optimal binning using optbinning library.
    - 'tree': Decision tree based splits.
    """
    
    def __init__(
        self, 
        strategy: str = 'quantile', 
        n_bins: int = 5, 
        variables: Optional[list[str]] = None,
        bin_definitions: Optional[dict[str, list[float]]] = None
    ):
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Valid: {sorted(VALID_STRATEGIES)}")
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        self.strategy = strategy
        self.n_bins = n_bins
        self.variables = variables
        self.bin_definitions = bin_definitions or {}

    def fit(self, x: pd.DataFrame, y: Optional[pd.Series] = None) -> "BinningTransformer":
        """Fit the binning boundaries."""
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)

        target_variables = self.variables if self.variables is not None else x.columns.tolist()
        self.fitted_bins_ = {}

        for var in target_variables:
            if var not in x.columns:
                continue

            if x[var].isna().any():
                raise ValueError(
                    f"Column '{var}' contains NaN values. "
                    f"Impute missing values before fitting BinningTransformer."
                )

            if not pd.api.types.is_numeric_dtype(x[var]):
                self.fitted_bins_[var] = 'categorical'
                continue

            if var in self.bin_definitions:
                splits = sorted(self.bin_definitions[var])
                if len(splits) < 1:
                    raise ValueError(f"bin_definitions for '{var}' must have at least 1 split value")
                self.fitted_bins_[var] = splits
                continue

            if self.strategy == 'optimal':
                if y is None:
                    raise ValueError("Target 'y' is required for 'optimal' binning strategy.")
                optb = OptimalBinning(name=var, dtype="numerical")
                optb.fit(x[var], y)
                splits = optb.splits
                self.fitted_bins_[var] = list(splits) if splits is not None else []
            elif self.strategy == 'tree':
                if y is None:
                    raise ValueError("Target 'y' is required for 'tree' binning strategy.")
                tree = DecisionTreeClassifier(max_leaf_nodes=self.n_bins, random_state=42)
                tree.fit(x[[var]], y)
                thresholds = sorted(tree.tree_.threshold[tree.tree_.threshold != -2])
                self.fitted_bins_[var] = thresholds
            elif self.strategy == 'quantile':
                _, bins = pd.qcut(x[var], q=self.n_bins, retbins=True, duplicates='drop')
                internal = bins[1:-1]
                self.fitted_bins_[var] = list(internal) if len(internal) > 0 else []
            elif self.strategy == 'uniform':
                _, bins = pd.cut(x[var], bins=self.n_bins, retbins=True)
                internal = bins[1:-1]
                self.fitted_bins_[var] = list(internal) if len(internal) > 0 else []
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform continuous variables into discrete bins."""
        check_is_fitted(self, 'fitted_bins_')
        x_out = x.copy()

        for var, state in self.fitted_bins_.items():
            if var not in x_out.columns:
                continue
            if isinstance(state, str) and state == 'categorical':
                x_out[var] = x_out[var].astype(str)
            else:
                if len(state) == 0:
                    x_out[var] = 'ALL'
                else:
                    bins = [-np.inf] + list(state) + [np.inf]
                    x_out[var] = pd.cut(x_out[var], bins=bins).astype(str)

        return x_out
