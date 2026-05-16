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


def test_woe_method_standard():
    X = pd.DataFrame({'bin': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']})
    y = pd.Series([1, 1, 0, 1, 0, 0, 1, 1, 0])
    wt = WOETransformer(method='standard').fit(X, y)
    for vals in wt.woe_maps_.values():
        assert not np.isinf(list(vals.values())).any()
        assert not np.isnan(list(vals.values())).any()


def test_woe_method_empirical_logit():
    X = pd.DataFrame({'bin': ['A', 'A', 'B', 'B']})
    y = pd.Series([1, 1, 0, 0])
    wt = WOETransformer(method='empirical_logit').fit(X, y)
    assert not np.isinf(list(wt.woe_maps_['bin'].values())).any()


def test_woe_method_signed():
    X = pd.DataFrame({'bin': ['A', 'A', 'B', 'B']})
    y = pd.Series([1, 1, 0, 0])
    wt = WOETransformer(method='signed').fit(X, y)
    vals = list(wt.woe_maps_['bin'].values())
    assert vals[0] > 0  # more goods -> positive
    assert vals[1] < 0  # more bads -> negative
    assert abs(vals[0]) == pytest.approx(abs(vals[1]), abs=1e-6)


def test_woe_invalid_method_raises():
    with pytest.raises(ValueError, match='method'):
        WOETransformer(method='invalid')


def test_woe_rare_lumping():
    X = pd.DataFrame({'cat': ['A', 'A', 'A', 'A', 'A', 'B', 'C']})
    y = pd.Series([1, 1, 1, 1, 1, 0, 0])
    wt = WOETransformer(rare_lumping=True, min_bin_pct=0.2).fit(X, y)
    assert 'RARE' in wt.woe_maps_['cat']  # 'B' and 'C' merged into RARE
    assert len(wt.woe_maps_['cat']) == 2  # A + RARE


def test_woe_methods_produce_different_results():
    X = pd.DataFrame({'bin': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']})
    y = pd.Series([1, 1, 0, 1, 0, 0, 1, 1, 0])
    wt_std = WOETransformer(method='standard').fit(X, y)
    wt_adj = WOETransformer(method='adjusted').fit(X, y)
    for k in wt_std.woe_maps_['bin']:
        assert not np.isinf(wt_std.woe_maps_['bin'][k])
        assert wt_std.woe_maps_['bin'][k] == pytest.approx(wt_adj.woe_maps_['bin'][k], abs=1e-1)
    wt = WOETransformer()
    with pytest.raises(Exception, match="fitted"):
        wt.transform(pd.DataFrame({'bin': ['A']}))
