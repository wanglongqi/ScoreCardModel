import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ScoreCardModel.binning.transformers import BinningTransformer
from ScoreCardModel.score_card.transformers import ScoreCardTransformer
from ScoreCardModel.weight_of_evidence.transformers import WOETransformer


@pytest.fixture
def fitted_components():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({'age': np.random.randint(18, 70, n)})
    y = pd.Series((X['age'] > 40).astype(int).values.ravel())
    bt = BinningTransformer(n_bins=4).fit(X)
    X_bin = bt.transform(X)
    wt = WOETransformer().fit(X_bin, y)
    X_woe = wt.transform(X_bin)
    lr = LogisticRegression().fit(X_woe, y)
    return X, y, bt, wt, lr


def test_scorecard_transform_shape(fitted_components):
    X, y, bt, wt, lr = fitted_components
    sct = ScoreCardTransformer(lr, bt, wt)
    scores = sct.transform(X)
    assert len(scores) == len(X)
    assert scores.mean() > 0


def test_scorecard_export_format(fitted_components):
    X, y, bt, wt, lr = fitted_components
    sct = ScoreCardTransformer(lr, bt, wt)
    card = sct.export_scorecard()
    assert all(col in card.columns for col in ['Variable', 'Bin', 'WOE', 'Points'])
    assert len(card) > 0


def test_scorecard_monotonic_with_probability(fitted_components):
    X, y, bt, wt, lr = fitted_components
    sct = ScoreCardTransformer(lr, bt, wt)
    scores = sct.transform(X)
    X_woe = wt.transform(bt.transform(X))
    from scipy.stats import spearmanr
    probs = lr.predict_proba(X_woe)[:, 1]
    corr, _ = spearmanr(probs, scores)
    assert corr > 0.99


def test_scorecard_invalid_pdo_raises():
    bt = BinningTransformer()
    wt = WOETransformer()
    lr = LogisticRegression()
    with pytest.raises(ValueError, match='PDO'):
        ScoreCardTransformer(lr, bt, wt, pdo=0)


def test_scorecard_invalid_base_points_raises():
    bt = BinningTransformer()
    wt = WOETransformer()
    lr = LogisticRegression()
    with pytest.raises(ValueError, match='base_points'):
        ScoreCardTransformer(lr, bt, wt, base_points=0)


def test_scorecard_invalid_base_odds_raises():
    bt = BinningTransformer()
    wt = WOETransformer()
    lr = LogisticRegression()
    with pytest.raises(ValueError, match='base_odds'):
        ScoreCardTransformer(lr, bt, wt, base_odds=0)
