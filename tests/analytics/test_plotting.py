import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')

from ScoreCardModel.analytics.plotting import (
    plot_bin_stats,
    plot_calibration,
    plot_cap,
    plot_confusion_matrix,
    plot_cutoff_optimization,
    plot_double_lift,
    plot_event_rate_by_bin,
    plot_gain_lift,
    plot_iv_summary,
    plot_iv_summary_enhanced,
    plot_ks,
    plot_psi_drift,
    plot_roc,
    plot_score_distribution,
    plot_scorecard_heatmap,
    plot_scorecard_waterfall,
    plot_woe_pattern,
)


@pytest.fixture
def sample_scores():
    np.random.seed(42)
    n = 200
    y_true = np.random.binomial(1, 0.4, n)
    y_prob = np.clip(y_true * 0.8 + np.random.normal(0.5, 0.2, n), 0, 1)
    return y_true, y_prob


def test_plot_ks_returns_figure(sample_scores):
    y_true, y_prob = sample_scores
    fig = plot_ks(y_true, y_prob)
    assert fig is not None


def test_plot_roc_returns_figure(sample_scores):
    y_true, y_prob = sample_scores
    fig = plot_roc(y_true, y_prob)
    assert fig is not None


def test_plot_cap_returns_figure(sample_scores):
    y_true, y_prob = sample_scores
    fig = plot_cap(y_true, y_prob)
    assert fig is not None


def test_plot_gain_lift_returns_figure(sample_scores):
    y_true, y_prob = sample_scores
    fig = plot_gain_lift(y_true, y_prob)
    assert fig is not None


def test_plot_double_lift_returns_figure(sample_scores):
    y_true, y_prob = sample_scores
    y_prob2 = y_prob + np.random.normal(0, 0.05, len(y_prob))
    y_prob2 = np.clip(y_prob2, 0, 1)
    fig = plot_double_lift(y_true, y_prob, y_prob2)
    assert fig is not None


def test_plot_score_distribution(sample_scores):
    y_true, y_prob = sample_scores
    fig = plot_score_distribution(y_prob, y_true)
    assert fig is not None


def test_plot_calibration(sample_scores):
    y_true, y_prob = sample_scores
    fig = plot_calibration(y_true, y_prob)
    assert fig is not None


def test_plot_psi_drift(sample_scores):
    y_true, y_prob = sample_scores
    y_prob2 = y_prob + np.random.normal(0, 0.1, len(y_prob))
    fig = plot_psi_drift(y_prob, y_prob2)
    assert fig is not None


def test_plot_woe_pattern():
    woe_map = {'bin1': -0.5, 'bin2': 0.0, 'bin3': 0.8}
    ordered_bins = ['bin1', 'bin2', 'bin3']
    fig = plot_woe_pattern(woe_map, ordered_bins, counts=[50, 80, 70],
                           feature_name='age', iv=0.15)
    assert fig is not None


def test_plot_iv_summary_enhanced():
    iv_dict = {'age': 0.3, 'income': 0.2, 'education': 0.05}
    fig = plot_iv_summary_enhanced(iv_dict)
    assert fig is not None


def test_plot_iv_summary_enhanced_empty():
    fig = plot_iv_summary_enhanced({})
    assert fig is not None


def test_plot_event_rate_by_bin():
    df = pd.DataFrame({
        'bin': ['A', 'A', 'B', 'B', 'C', 'C'],
        'target': [1, 0, 1, 0, 0, 0],
    })
    fig = plot_event_rate_by_bin(df, 'bin')
    assert fig is not None


def test_plot_bin_stats_legacy():
    df = pd.DataFrame({
        'feat': ['A', 'A', 'B', 'B'],
        'target': [1, 0, 1, 0],
    })
    fig = plot_bin_stats(df, 'feat', 'target')
    assert fig is not None


def test_plot_iv_summary_legacy():
    fig = plot_iv_summary({'a': 0.1, 'b': 0.2})
    assert fig is not None


def test_plot_scorecard_waterfall():
    contributions = {'age': 15, 'income': -10, 'education': 5}
    fig = plot_scorecard_waterfall(contributions, base_points=600)
    assert fig is not None


def test_plot_scorecard_heatmap():
    card = pd.DataFrame({
        'Variable': ['age', 'age', 'income', 'income'],
        'Bin': ['10-20', '20-30', 'low', 'high'],
        'Points': [50, 30, -20, 40],
        'WOE': [0.5, 0.3, -0.2, 0.4],
    })
    fig = plot_scorecard_heatmap(card)
    assert fig is not None


def test_plot_scorecard_heatmap_empty():
    fig = plot_scorecard_heatmap(pd.DataFrame())
    assert fig is not None


def test_plot_cutoff_optimization(sample_scores):
    y_true, y_prob = sample_scores
    fig = plot_cutoff_optimization(y_true, y_prob)
    assert fig is not None


def test_plot_confusion_matrix(sample_scores):
    y_true, y_prob = sample_scores
    fig = plot_confusion_matrix(y_true, y_prob > 0.5)
    assert fig is not None


def test_all_plots_have_different_figures(sample_scores):
    y_true, y_prob = sample_scores
    figs = [
        plot_ks(y_true, y_prob),
        plot_roc(y_true, y_prob),
        plot_cap(y_true, y_prob),
    ]
    assert len({id(f) for f in figs}) == len(figs)
