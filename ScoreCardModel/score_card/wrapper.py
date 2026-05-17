from typing import Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ScoreCardModel.binning.transformers import BinningTransformer
from ScoreCardModel.score_card.transformers import ScoreCardTransformer
from ScoreCardModel.weight_of_evidence.transformers import WOETransformer


class ScoreCardWrapper:
    """
    Analyst Facade for the modernized ScoreCard library.
    
    Exposes familiar methods like pre_trade and predict while using 
    scikit-learn pipelines.
    """
    
    def __init__(
        self, 
        binning_strategy: str = 'quantile', 
        n_bins: int = 5,
        base_points: float = 600,
        base_odds: float = 50,
        pdo: float = 20,
        model_params: Optional[dict] = None
    ):
        self.binning_strategy = binning_strategy
        self.n_bins = n_bins
        self.base_points = base_points
        self.base_odds = base_odds
        self.pdo = pdo

        params = {}
        if model_params:
            params.update(model_params)

        # Build the internal pipeline
        self.pipeline = Pipeline([
            ('binning', BinningTransformer(strategy=binning_strategy, n_bins=n_bins)),
            ('woe', WOETransformer()),
            ('model', LogisticRegression(**params))
        ])
        self.scorecard_transformer_: Optional[ScoreCardTransformer] = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "ScoreCardWrapper":
        """Fit the entire scorecard pipeline."""
        self.pipeline.fit(x, y)
        
        # Initialize the ScoreCardTransformer for scoring
        self.scorecard_transformer_ = ScoreCardTransformer(
            model=self.pipeline.named_steps['model'],
            binning_transformer=self.pipeline.named_steps['binning'],
            woe_transformer=self.pipeline.named_steps['woe'],
            base_points=self.base_points,
            base_odds=self.base_odds,
            pdo=self.pdo
        )
        return self

    def pre_trade(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform raw data into WOE values (familiar for analysts)."""
        # We only run the binning and woe steps
        x_bin = self.pipeline.named_steps['binning'].transform(x)
        return self.pipeline.named_steps['woe'].transform(x_bin)

    def predict(self, x: pd.DataFrame) -> pd.Series:
        """Predict scores for raw input data."""
        if self.scorecard_transformer_ is None:
            raise ValueError("ScoreCardWrapper must be fitted before calling predict.")
        return self.scorecard_transformer_.transform(x)

    def export_scorecard(self) -> pd.DataFrame:
        """Export the systematic scorecard table."""
        if self.scorecard_transformer_ is None:
            raise ValueError("ScoreCardWrapper must be fitted before exporting.")
        return self.scorecard_transformer_.export_scorecard()

    def export_to_excel(self, x_train: pd.DataFrame, y_train: pd.Series,
                        x_test: Optional[pd.DataFrame] = None,
                        y_test: Optional[pd.Series] = None,
                        output_path: str = "scorecard_specification.xlsx") -> str:
        """Export a professional multi-sheet Excel specification."""
        from ScoreCardModel.analytics.reporting import export_to_excel
        return export_to_excel(self.pipeline, x_train, y_train, x_test, y_test, output_path)
