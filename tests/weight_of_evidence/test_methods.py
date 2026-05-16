import numpy as np
import pytest

from ScoreCardModel.weight_of_evidence.methods import (
    calculate_adjusted_woe,
    calculate_empirical_logit_woe,
    calculate_signed_woe,
    calculate_standard_woe,
    calculate_weighted_woe,
)


def test_standard_woe_same_dist():
    woe = calculate_standard_woe(np.array([50, 50]), np.array([50, 50]))
    assert woe[0] == pytest.approx(0.0, abs=1e-6)
    assert woe[1] == pytest.approx(0.0, abs=1e-6)


def test_standard_woe_positive():
    woe = calculate_standard_woe(np.array([80]), np.array([20]))
    assert woe[0] > 0


def test_standard_woe_negative():
    woe = calculate_standard_woe(np.array([20]), np.array([80]))
    assert woe[0] < 0


def test_adjusted_woe_no_inf():
    woe = calculate_adjusted_woe(
        np.array([0, 100]), np.array([100, 0]),
        good_total=100, bad_total=100,
    )
    assert not np.isinf(woe).any()
    assert not np.isnan(woe).any()


def test_adjusted_woe_matches_standard_when_no_zeros():
    g = np.array([30, 70])
    b = np.array([70, 30])
    std = calculate_standard_woe(g, b)
    adj = calculate_adjusted_woe(g, b, good_total=100, bad_total=100, smoothing=1e-10)
    assert adj == pytest.approx(std, abs=1e-4)


def test_empirical_logit_no_inf():
    woe = calculate_empirical_logit_woe(np.array([0, 100]), np.array([100, 0]))
    assert not np.isinf(woe).any()


def test_signed_woe_preserves_magnitude():
    g = np.array([80, 20])
    b = np.array([20, 80])
    woe = calculate_signed_woe(g, b)
    assert woe[0] > 0
    assert woe[1] < 0
    assert abs(woe[0]) == pytest.approx(abs(woe[1]), abs=1e-6)


def test_weighted_woe_no_nan():
    g = np.array([90, 10])
    b = np.array([10, 90])
    bt = np.array([100, 100])
    woe = calculate_weighted_woe(g, b, good_total=100, bad_total=100, bin_total=bt, n_total=200)
    assert not np.isnan(woe).any()
    assert not np.isinf(woe).any()


def test_weighted_woe_zero_weight_bin():
    g = np.array([50, 0])
    b = np.array([50, 0])
    bt = np.array([100, 0])
    woe = calculate_weighted_woe(g, b, good_total=50, bad_total=50, bin_total=bt, n_total=100)
    assert woe[1] == 0.0
