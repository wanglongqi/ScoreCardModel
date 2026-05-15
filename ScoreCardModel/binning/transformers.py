from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.tree import DecisionTreeClassifier
from optbinning import OptimalBinning

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
        variables: Optional[List[str]] = None,
        bin_definitions: Optional[Dict[str, List[float]]] = None
    ):
        self.strategy = strategy
        self.n_bins = n_bins
        self.variables = variables
        self.bin_definitions = bin_definitions or {}
        self.fitted_bins_: Dict[str, Any] = {}

    def fit(self, x: pd.DataFrame, y: Optional[pd.Series] = None) -> "BinningTransformer":
        """Fit the binning boundaries."""
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
            
        target_variables = self.variables if self.variables is not None else x.columns
        self.fitted_bins_ = {}
        
        for var in target_variables:
            # If column is not numeric, we treat it as discrete/categorical
            if not pd.api.types.is_numeric_dtype(x[var]):
                self.fitted_bins_[var] = 'categorical'
                continue

            if var in self.bin_definitions:
                self.fitted_bins_[var] = self.bin_definitions[var]
                continue
                
            if self.strategy == 'optimal':
                if y is None:
                    raise ValueError("Target 'y' is required for 'optimal' binning strategy.")
                optb = OptimalBinning(name=var, dtype="numerical")
                optb.fit(x[var], y)
                self.fitted_bins_[var] = optb.splits
            elif self.strategy == 'tree':
                if y is None:
                    raise ValueError("Target 'y' is required for 'tree' binning strategy.")
                tree = DecisionTreeClassifier(max_leaf_nodes=self.n_bins)
                tree.fit(x[[var]], y)
                thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]
                self.fitted_bins_[var] = sorted(thresholds)
            elif self.strategy == 'quantile':
                _, bins = pd.qcut(x[var], q=self.n_bins, retbins=True, duplicates='drop')
                self.fitted_bins_[var] = bins[1:-1] # Only internal splits
            elif self.strategy == 'uniform':
                _, bins = pd.cut(x[var], bins=self.n_bins, retbins=True)
                self.fitted_bins_[var] = bins[1:-1]
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
                
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform continuous variables into discrete bins."""
        check_is_fitted(self)
        x_out = x.copy()
        
        for var, state in self.fitted_bins_.items():
            if isinstance(state, str) and state == 'categorical':
                x_out[var] = x_out[var].astype(str)
            else:
                # state is an array of splits
                bins = [-np.inf] + list(state) + [np.inf]
                x_out[var] = pd.cut(x_out[var], bins=bins).astype(str)
            
        return x_out
