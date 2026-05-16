from typing import Optional
import pandas as pd
from ScoreCardModel.score_card.wrapper import ScoreCardWrapper
from ScoreCardModel.analytics.selection import select_by_iv


class BaseScorecard:
    """Default scorecard: LogisticRegression, quantile binning, adjusted WOE.

    Parameters
    ----------
    binning_strategy : str, default='quantile'
    n_bins : int, default=5
    base_points : float, default=600
    base_odds : float, default=50
    pdo : float, default=20
    woe_method : str, default='adjusted'
    """

    def __init__(
        self,
        binning_strategy: str = 'quantile',
        n_bins: int = 5,
        base_points: float = 600,
        base_odds: float = 50,
        pdo: float = 20,
        woe_method: str = 'adjusted',
    ):
        self.binning_strategy = binning_strategy
        self.n_bins = n_bins
        self.base_points = base_points
        self.base_odds = base_odds
        self.pdo = pdo
        self.woe_method = woe_method
        self._wrapper: Optional[ScoreCardWrapper] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseScorecard":
        self._wrapper = ScoreCardWrapper(
            binning_strategy=self.binning_strategy,
            n_bins=self.n_bins,
            base_points=self.base_points,
            base_odds=self.base_odds,
            pdo=self.pdo,
        )
        self._wrapper.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._wrapper is None:
            raise ValueError("Scorecard must be fitted before predict.")
        return self._wrapper.predict(X)

    def export_scorecard(self) -> pd.DataFrame:
        if self._wrapper is None:
            raise ValueError("Scorecard must be fitted before export.")
        return self._wrapper.export_scorecard()

    def pre_trade(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._wrapper is None:
            raise ValueError("Scorecard must be fitted before pre_trade.")
        return self._wrapper.pre_trade(X)


class ConservativeScorecard(BaseScorecard):
    """Conservative scorecard: fewer bins, higher IV threshold.

    Pre-filters features by IV to include only predictive features.
    """

    def __init__(
        self,
        n_bins: int = 4,
        min_iv: float = 0.05,
        base_points: float = 600,
        base_odds: float = 50,
        pdo: float = 20,
    ):
        super().__init__(n_bins=n_bins, base_points=base_points,
                         base_odds=base_odds, pdo=pdo)
        self.min_iv = min_iv

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ConservativeScorecard":
        selected = select_by_iv(X, y, min_iv=self.min_iv, n_bins=self.n_bins)
        if not selected:
            raise ValueError(
                f"No features passed IV threshold (min_iv={self.min_iv}). "
                f"Try lowering the threshold or using more bins."
            )
        return super().fit(X[selected], y)
