from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ScoreCardTransformer(BaseEstimator, TransformerMixin):
    """
    Maps WOE values and model coefficients to ScoreCard points.
    
    Calculation:
        factor = pdo / ln(2)
        offset = base_points - factor * ln(base_odds)
        score = -(woe * coef + intercept / n_features) * factor + offset / n_features
    """
    
    def __init__(
        self, 
        model: Any, 
        binning_transformer: Any, 
        woe_transformer: Any,
        base_points: float = 600,
        base_odds: float = 50,
        pdo: float = 20
    ):
        self.model = model
        self.binning_transformer = binning_transformer
        self.woe_transformer = woe_transformer
        
        if pdo <= 0:
            raise ValueError(f"PDO must be > 0, got {pdo}")
        if base_points <= 0:
            raise ValueError(f"base_points must be > 0, got {base_points}")
        if base_odds <= 0:
            raise ValueError(f"base_odds must be > 0, got {base_odds}")
        
        self.base_points = base_points
        self.base_odds = base_odds
        self.pdo = pdo
        self.factor_ = pdo / np.log(2)
        self.offset_ = base_points - self.factor_ * np.log(base_odds)

    def fit(self, x: Any = None, y: Any = None) -> "ScoreCardTransformer":
        """ScoreCardTransformer is usually initialized with fitted components."""
        return self

    def transform(self, x: pd.DataFrame) -> pd.Series:
        """Calculate total score for each observation."""
        x_bin = self.binning_transformer.transform(x)
        x_woe = self.woe_transformer.transform(x_bin)
        
        # Ensure feature alignment with model
        if hasattr(self.model, "feature_names_in_"):
            x_woe = x_woe[self.model.feature_names_in_]
        else:
            # Fallback to the order used in WOE transformer fit
            x_woe = x_woe[list(self.woe_transformer.woe_maps_.keys())]
            
        # logit = ln(P(1)/P(0))
        # score = factor * logit + offset
        logit = self.model.decision_function(x_woe)
        scores = self.factor_ * logit + self.offset_
        
        return pd.Series(scores, index=x.index)

    def export_scorecard(self) -> pd.DataFrame:
        """Export a systematic scorecard table."""
        rows = []
        coefs = self.model.coef_[0]
        intercept = self.model.intercept_[0]
        features = list(self.woe_transformer.woe_maps_.keys())
        
        # Base points distribution
        # intercept is distributed across features for the table
        intercept_share = intercept / len(features)
        
        for i, feat in enumerate(features):
            woe_map = self.woe_transformer.woe_maps_[feat]
            coef = coefs[i]
            for bin_name, woe in woe_map.items():
                # Points for this bin = factor * (coef * woe + intercept_share) + offset/n
                points = (self.factor_ * (coef * woe + intercept_share) + 
                         (self.offset_ / len(features)))
                rows.append({
                    'Variable': feat,
                    'Bin': bin_name,
                    'WOE': woe,
                    'Points': round(points, 2)
                })
                
        return pd.DataFrame(rows)
