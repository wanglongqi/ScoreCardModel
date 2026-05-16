import numpy as np
import pandas as pd
import pytest
from ScoreCardModel.binning.transformers import BinningTransformer


def test_quantile_binning():
    X = pd.DataFrame({'age': np.random.randint(18, 70, 200)})
    bt = BinningTransformer(strategy='quantile', n_bins=4).fit(X)
    X_bin = bt.transform(X)
    assert X_bin['age'].nunique() == 4


def test_uniform_binning():
    X = pd.DataFrame({'age': np.random.randint(18, 70, 200)})
    bt = BinningTransformer(strategy='uniform', n_bins=3).fit(X)
    X_bin = bt.transform(X)
    assert X_bin['age'].nunique() == 3


def test_categorical_passthrough():
    X = pd.DataFrame({'color': ['red', 'blue', 'green', 'red']})
    bt = BinningTransformer().fit(X)
    X_bin = bt.transform(X)
    assert pd.api.types.is_string_dtype(X_bin['color'])


def test_nan_raises():
    X = pd.DataFrame({'age': [20, 30, np.nan, 40]})
    bt = BinningTransformer()
    with pytest.raises(ValueError, match='NaN'):
        bt.fit(X)


def test_invalid_strategy_raises():
    with pytest.raises(ValueError, match='Unknown strategy'):
        BinningTransformer(strategy='invalid')


def test_n_bins_too_small():
    with pytest.raises(ValueError, match='n_bins'):
        BinningTransformer(n_bins=1)


def test_bin_definitions_override():
    X = pd.DataFrame({'age': range(100)})
    bt = BinningTransformer(bin_definitions={'age': [30, 60]}).fit(X)
    X_bin = bt.transform(X)
    assert X_bin['age'].nunique() == 3


def test_tree_strategy():
    X = pd.DataFrame({'age': np.random.randint(18, 70, 200)})
    y = (X['age'] > 40).astype(int).values.ravel()
    bt = BinningTransformer(strategy='tree', n_bins=4).fit(X, y)
    X_bin = bt.transform(X)
    assert X_bin['age'].nunique() <= 4


def test_optimal_strategy():
    X = pd.DataFrame({'age': np.random.randint(18, 70, 200)})
    y = (X['age'] > 40).astype(int).values.ravel()
    bt = BinningTransformer(strategy='optimal', n_bins=4).fit(X, y)
    X_bin = bt.transform(X)
    assert X_bin['age'].nunique() <= 4


def test_transform_unfitted_raises():
    bt = BinningTransformer()
    with pytest.raises(Exception):
        bt.transform(pd.DataFrame({'age': [1, 2, 3]}))


def test_fewer_bins_than_requested():
    X = pd.DataFrame({'age': [25, 25, 25, 30, 30]})
    bt = BinningTransformer(strategy='quantile', n_bins=5).fit(X)
    X_bin = bt.transform(X)
    assert X_bin['age'].nunique() <= 5


def test_variables_subset():
    X = pd.DataFrame({'a': range(100), 'b': range(100, 200)})
    bt = BinningTransformer(variables=['a']).fit(X)
    X_bin = bt.transform(X)
    assert 'a' in X_bin.columns
