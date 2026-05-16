import numpy as np
import pandas as pd
import pytest
from ScoreCardModel.weight_of_evidence.transformers import WOETransformer


@pytest.fixture
def sample_woe_data():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        'bin_feat': pd.cut(np.random.normal(0, 1, n), bins=4).astype(str)
    })
    y = pd.Series(np.random.binomial(1, 0.4, n))
    return X, y


def test_woe_basic_fit_transform(sample_woe_data):
    X, y = sample_woe_data
    wt = WOETransformer().fit(X, y)
    X_woe = wt.transform(X)
    assert not X_woe.isna().any().any()
    assert len(wt.woe_maps_) == 1


def test_woe_iv_value_range(sample_woe_data):
    X, y = sample_woe_data
    wt = WOETransformer().fit(X, y)
    iv = list(wt.iv_.values())[0]
    assert 0 <= iv <= 10


def test_woe_zero_count_bins():
    X = pd.DataFrame({'bin': ['A', 'A', 'B', 'B', 'C', 'C']})
    y = pd.Series([1, 1, 1, 1, 0, 0])
    wt = WOETransformer(laplace_smoothing=1e-6).fit(X, y)
    X_woe = wt.transform(X)
    assert not X_woe.isna().any().any()
    assert not np.isinf(X_woe['bin']).any()


def test_woe_non_binary_target_raises():
    X = pd.DataFrame({'bin': ['A', 'B', 'C']})
    y = pd.Series([0, 1, 2])
    with pytest.raises(ValueError, match='binary'):
        WOETransformer().fit(X, y)


def test_woe_nan_in_x_raises():
    X = pd.DataFrame({'bin': ['A', 'B', None]})
    y = pd.Series([0, 1, 0])
    with pytest.raises(ValueError, match='NaN'):
        WOETransformer().fit(X, y)


def test_woe_unseen_in_transform(sample_woe_data):
    X, y = sample_woe_data
    wt = WOETransformer().fit(X, y)
    X_test = pd.DataFrame({'bin_feat': ['UNSEEN_LABEL']})
    X_woe = wt.transform(X_test)
    assert 'bin_feat' in X_woe.columns


def test_woe_iv_property(sample_woe_data):
    X, y = sample_woe_data
    wt = WOETransformer().fit(X, y)
    assert hasattr(wt, 'iv')
    assert wt.iv == wt.iv_


def test_woe_transform_unfitted_raises():
    wt = WOETransformer()
    with pytest.raises(Exception):
        wt.transform(pd.DataFrame({'bin': ['A']}))
