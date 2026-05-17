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
        
        # Get feature names from model if possible, else from woe_transformer
        if hasattr(self.model, "feature_names_in_"):
            features = list(self.model.feature_names_in_)
        else:
            features = list(self.woe_transformer.woe_maps_.keys())
            
        # Base points distribution
        # intercept is distributed across features for the table
        intercept_share = intercept / len(features)
        
        for i, feat in enumerate(features):
            if feat not in self.woe_transformer.woe_maps_:
                # This should not happen if everything is aligned
                continue
                
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

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        try:
            card = self.export_scorecard()
        except Exception:
            return None
        if card.empty:
            return "<p><i>Empty scorecard</i></p>"
        rows_html = ""
        for _, row in card.iterrows():
            woe_str = f"{row['WOE']:.4f}"
            pts_str = f"{row['Points']:.2f}"
            rows_html += (
                f"<tr style='border-bottom:1px solid #ddd'>"
                f"<td style='padding:4px 8px'>{row['Variable']}</td>"
                f"<td style='padding:4px 8px'>{row['Bin']}</td>"
                f"<td style='padding:4px 8px;text-align:right'>{woe_str}</td>"
                f"<td style='padding:4px 8px;text-align:right'>{pts_str}</td>"
                f"</tr>"
            )
        return (
            "<table style='border-collapse:collapse;width:100%;font-family:monospace;font-size:13px'>"
            "<thead><tr style='background:#f5f5f5;border-bottom:2px solid #ccc'>"
            "<th style='padding:6px 8px;text-align:left'>Variable</th>"
            "<th style='padding:6px 8px;text-align:left'>Bin</th>"
            "<th style='padding:6px 8px;text-align:right'>WOE</th>"
            "<th style='padding:6px 8px;text-align:right'>Points</th>"
            "</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table>"
        )
