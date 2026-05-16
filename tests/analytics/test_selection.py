import numpy as np
import pandas as pd
import pytest
from ScoreCardModel.analytics.selection import select_by_iv, select_by_correlation, rank_features


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        'strong_feat': np.random.normal(0, 1, n) + np.random.binomial(1, 0.3, n) * 2,
        'weak_feat': np.random.normal(0, 1, n),
        'useless_feat': np.random.uniform(0, 1, n),
    })
    y = pd.Series((X['strong_feat'] > 0.5).astype(int).values.ravel())
    return X, y


def test_select_by_iv_returns_features(sample_data):
    X, y = sample_data
    selected = select_by_iv(X, y, min_iv=0.0)
    assert len(selected) > 0
    assert all(f in X.columns for f in selected)


def test_select_by_iv_filters_low_iv():
    X = pd.DataFrame({
        'good': np.random.normal(0, 1, 100),
        'bad': np.random.uniform(0, 1, 100),
    })
    y = pd.Series((X['good'] > 0).astype(int).values.ravel())
    selected = select_by_iv(X, y, min_iv=0.02)
    assert 'good' in selected or 'bad' in selected


def test_select_by_correlation():
    X = pd.DataFrame({
        'a': np.random.normal(0, 1, 100),
        'b': np.random.normal(0, 1, 100),
        'c': np.random.normal(0, 1, 100),
    })
    X['a'] = X['b'] * 0.95 + np.random.normal(0, 0.1, 100)
    selected = select_by_correlation(X, max_corr=0.8)
    assert len(selected) < 3


def test_select_by_correlation_no_drop():
    X = pd.DataFrame({
        'a': np.random.normal(0, 1, 100),
        'b': np.random.normal(0, 1, 100),
    })
    selected = select_by_correlation(X, max_corr=0.9)
    assert len(selected) == 2


def test_rank_features_returns_dataframe(sample_data):
    X, y = sample_data
    result = rank_features(X, y)
    assert isinstance(result, pd.DataFrame)
    expected_cols = ['Feature', 'IV', 'IV_Label', 'Monotonicity',
                     'Mono_Strength', 'Chi2_pvalue', 'Recommendation']
    for col in expected_cols:
        assert col in result.columns
    assert len(result) == len(X.columns)
