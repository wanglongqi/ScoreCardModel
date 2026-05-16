import numpy as np
import pytest

from ScoreCardModel.analytics.metrics import calculate_accuracy_ratio, calculate_ks, calculate_psi


def test_ks_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    ks = calculate_ks(y_true, y_prob)
    assert 0 < ks <= 1.0


def test_ks_no_separation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])
    ks = calculate_ks(y_true, y_prob)
    assert ks == 0.0


def test_ks_accepts_series():
    import pandas as pd
    y_true = pd.Series([0, 0, 1, 1])
    y_prob = pd.Series([0.1, 0.2, 0.8, 0.9])
    ks = calculate_ks(y_true, y_prob)
    assert 0 < ks <= 1.0


def test_psi_identical():
    data = np.random.normal(0, 1, 1000)
    psi = calculate_psi(data, data)
    assert psi == pytest.approx(0.0, abs=1e-2)


def test_psi_different():
    a = np.random.normal(0, 1, 1000)
    b = np.random.normal(2, 1, 1000)
    psi = calculate_psi(a, b)
    assert psi > 0.1


def test_psi_small_vs_large():
    small = np.array([1.0, 2.0, 3.0])
    large = np.array([10.0, 20.0, 30.0])
    psi = calculate_psi(small, large)
    assert psi >= 0


def test_accuracy_ratio_perfect():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    ar = calculate_accuracy_ratio(y_true, y_prob)
    assert ar > 0.9


def test_accuracy_ratio_random():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])
    ar = calculate_accuracy_ratio(y_true, y_prob)
    assert ar == pytest.approx(0.0, abs=1e-1)
