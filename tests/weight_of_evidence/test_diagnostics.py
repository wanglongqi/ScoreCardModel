import numpy as np
import pandas as pd
import pytest
from ScoreCardModel.weight_of_evidence.diagnostics import (
    check_monotonicity,
    iv_by_bin,
    woe_chi_square,
    midpoint_correlation,
    bin_statistics,
)


def test_monotonicity_increasing():
    woe_map = {'bin1': -1.0, 'bin2': 0.0, 'bin3': 1.0}
    ordered_bins = ['bin1', 'bin2', 'bin3']
    result, strength = check_monotonicity(woe_map, ordered_bins)
    assert result == 'increasing'
    assert strength > 0.8


def test_monotonicity_decreasing():
    woe_map = {'bin1': 1.0, 'bin2': 0.0, 'bin3': -1.0}
    ordered_bins = ['bin1', 'bin2', 'bin3']
    result, strength = check_monotonicity(woe_map, ordered_bins)
    assert result == 'decreasing'
    assert strength > 0.8


def test_monotonicity_non_monotonic():
    woe_map = {'bin1': -1.0, 'bin2': 1.0, 'bin3': -0.5}
    ordered_bins = ['bin1', 'bin2', 'bin3']
    result, strength = check_monotonicity(woe_map, ordered_bins)
    assert result == 'non-monotonic'


def test_monotonicity_single_bin():
    woe_map = {'bin1': 0.5}
    ordered_bins = ['bin1']
    result, strength = check_monotonicity(woe_map, ordered_bins)
    assert result == 'single_bin'


def test_iv_by_bin():
    good = np.array([30, 70])
    bad = np.array([70, 30])
    ivs = iv_by_bin(good, bad, good_total=100, bad_total=100)
    assert len(ivs) == 2
    assert sum(ivs) > 0
    assert all(iv > 0 for iv in ivs)


def test_iv_by_bin_zero_division():
    good = np.array([0, 100])
    bad = np.array([100, 0])
    ivs = iv_by_bin(good, bad, good_total=100, bad_total=100)
    assert len(ivs) == 2
    assert not np.isnan(ivs).any()


def test_chi_square():
    X = pd.DataFrame({'bin': ['A', 'A', 'B', 'B']})
    y = pd.Series([1, 0, 1, 0])
    stat, pval = woe_chi_square(X['bin'], y)
    assert stat >= 0
    assert 0 <= pval <= 1


def test_midpoint_correlation_perfect():
    edges = [0, 1, 2, 3]
    woe = [-0.5, 0.0, 0.5]
    corr = midpoint_correlation(edges, woe)
    assert corr > 0.9


def test_bin_statistics_columns():
    X = pd.Series(['A', 'A', 'B', 'B', 'C'])
    y = pd.Series([1, 1, 0, 0, 1])
    stats = bin_statistics(X, y)
    expected_cols = ['bin', 'total', 'pop_pct', 'good', 'bad', 'event_rate', 'woe', 'iv']
    assert all(col in stats.columns for col in expected_cols)
    assert len(stats) == 3
