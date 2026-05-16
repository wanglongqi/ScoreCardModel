import numpy as np
import pandas as pd
import pytest
from ScoreCardModel.templates.presets import BaseScorecard, ConservativeScorecard


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        'age': np.random.randint(18, 70, n),
        'income': np.random.normal(50000, 15000, n),
    })
    y = pd.Series((X['age'] > 40).astype(int).values.ravel())
    return X, y


def test_base_scorecard(sample_data):
    X, y = sample_data
    sc = BaseScorecard()
    sc.fit(X, y)
    scores = sc.predict(X)
    assert len(scores) == len(X)


def test_base_scorecard_export(sample_data):
    X, y = sample_data
    sc = BaseScorecard()
    sc.fit(X, y)
    card = sc.export_scorecard()
    assert isinstance(card, pd.DataFrame)
    assert len(card) > 0


def test_conservative_scorecard(sample_data):
    X, y = sample_data
    sc = ConservativeScorecard(min_iv=0.01)
    sc.fit(X, y)
    scores = sc.predict(X)
    assert len(scores) == len(X)
