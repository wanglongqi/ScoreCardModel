import numpy as np
import pandas as pd
import pytest

from ScoreCardModel.score_card.wrapper import ScoreCardWrapper


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        'age': np.random.randint(18, 70, n),
        'income': np.random.normal(50000, 15000, n)
    })
    y = pd.Series((X['age'] > 40).astype(int).values.ravel())
    return X, y


def test_wrapper_fit_and_predict(sample_data):
    X, y = sample_data
    sc = ScoreCardWrapper(binning_strategy='quantile', n_bins=4)
    sc.fit(X, y)
    scores = sc.predict(X)
    assert len(scores) == len(X)
    assert scores.isna().sum() == 0


def test_wrapper_predict_unfitted_raises(sample_data):
    X, y = sample_data
    sc = ScoreCardWrapper()
    with pytest.raises(ValueError, match='fitted'):
        sc.predict(X)


def test_wrapper_export_scorecard(sample_data):
    X, y = sample_data
    sc = ScoreCardWrapper(n_bins=3).fit(X, y)
    card = sc.export_scorecard()
    assert isinstance(card, pd.DataFrame)
    assert len(card) > 0


def test_wrapper_export_unfitted_raises(sample_data):
    sc = ScoreCardWrapper()
    with pytest.raises(ValueError):
        sc.export_scorecard()


def test_wrapper_pre_trade(sample_data):
    X, y = sample_data
    sc = ScoreCardWrapper().fit(X, y)
    woe_df = sc.pre_trade(X)
    assert isinstance(woe_df, pd.DataFrame)
    assert woe_df.shape[1] == 2  # two features -> two WOE columns
