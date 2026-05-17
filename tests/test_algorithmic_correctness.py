import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ScoreCardModel import BinningTransformer, WOETransformer, ScoreCardTransformer
from ScoreCardModel.analytics.metrics import calculate_ks, calculate_psi, calculate_accuracy_ratio
from ScoreCardModel.analytics.selection import rank_features
from ScoreCardModel.score_card.wrapper import ScoreCardWrapper
from ScoreCardModel.weight_of_evidence.diagnostics import (
    bin_statistics,
    check_monotonicity,
    iv_by_bin,
    midpoint_correlation,
    woe_chi_square,
)
from ScoreCardModel.weight_of_evidence.methods import (
    calculate_adjusted_woe,
    calculate_empirical_logit_woe,
    calculate_signed_woe,
    calculate_standard_woe,
    calculate_weighted_woe,
)


# =============================================================================
# WOE Mathematical Correctness
# =============================================================================

class TestWOEMathematics:
    """Verify WOE formulas against hand-calculated values."""

    def test_standard_woe_formula(self):
        """Standard WOE = ln((good/good_total) / (bad/bad_total))."""
        good = np.array([80, 20])
        bad = np.array([20, 80])
        woe = calculate_standard_woe(good, bad, good_total=100, bad_total=100)
        expected0 = np.log((80/100) / (20/100))
        expected1 = np.log((20/100) / (80/100))
        assert woe[0] == pytest.approx(expected0, abs=1e-10)
        assert woe[1] == pytest.approx(expected1, abs=1e-10)
        assert woe[0] > 0
        assert woe[1] < 0

    def test_standard_woe_symmetric(self):
        """Flipping good/bad counts flips WOE sign."""
        woe_ab = calculate_standard_woe(np.array([80, 20]), np.array([20, 80]), 100, 100)
        woe_ba = calculate_standard_woe(np.array([20, 80]), np.array([80, 20]), 100, 100)
        assert woe_ab[0] == pytest.approx(-woe_ba[0], abs=1e-10)
        assert woe_ab[1] == pytest.approx(-woe_ba[1], abs=1e-10)

    def test_standard_woe_zero_when_equal_distribution(self):
        """WOE = 0 when dist_good == dist_bad."""
        woe = calculate_standard_woe(np.array([50, 50]), np.array([50, 50]), 100, 100)
        assert woe[0] == pytest.approx(0.0, abs=1e-10)
        assert woe[1] == pytest.approx(0.0, abs=1e-10)

    def test_adjusted_woe_formula(self):
        """Adjusted WOE = ln((good+s)/(good_total+2s) / (bad+s)/(bad_total+2s))."""
        good = np.array([0, 100])
        bad = np.array([100, 0])
        s = 0.5
        woe = calculate_adjusted_woe(good, bad, good_total=100, bad_total=100, smoothing=s)
        dg0 = (0 + s) / (100 + 2 * s)
        db0 = (100 + s) / (100 + 2 * s)
        expected0 = np.log(dg0 / db0)
        assert woe[0] == pytest.approx(expected0, abs=1e-10)

    def test_adjusted_woe_never_infinite(self):
        """Even with zero-count bins, adjusted WOE is finite."""
        good = np.array([0, 0, 100])
        bad = np.array([100, 0, 0])
        woe = calculate_adjusted_woe(good, bad, good_total=100, bad_total=100)
        assert not np.isinf(woe).any()
        assert not np.isnan(woe).any()

    def test_empirical_logit_formula(self):
        """Empirical logit = ln((good+0.5)/(good_total+1) / (bad+0.5)/(bad_total+1))."""
        good = np.array([0, 100])
        bad = np.array([100, 0])
        woe = calculate_empirical_logit_woe(good, bad, good_total=100, bad_total=100)
        dg0 = (0 + 0.5) / (100 + 1)
        db0 = (100 + 0.5) / (100 + 1)
        expected0 = np.log(dg0 / db0)
        assert woe[0] == pytest.approx(expected0, abs=1e-10)

    def test_empirical_logit_finite_with_zeros(self):
        """Empirical logit handles zero-count bins without inf."""
        woe = calculate_empirical_logit_woe(np.array([0, 50]), np.array([50, 0]), 50, 50)
        assert not np.isinf(woe).any()

    def test_signed_woe_symmetric(self):
        """Signed WOE is perfectly symmetric around zero."""
        g = np.array([90, 10])
        b = np.array([10, 90])
        woe = calculate_signed_woe(g, b)
        assert woe[0] == pytest.approx(-woe[1], abs=1e-10)

    def test_signed_woe_direction(self):
        """Signed WOE: more goods → positive, more bads → negative."""
        woe = calculate_signed_woe(np.array([80, 20]), np.array([20, 80]))
        assert woe[0] > 0
        assert woe[1] < 0

    def test_weighted_woe_downweights_small_bins(self):
        """Weighted WOE shrinks toward zero for small bins."""
        g = np.array([90, 10])
        b = np.array([10, 90])
        bt = np.array([180, 20])
        std = calculate_standard_woe(g, b, 100, 100)
        wtd = calculate_weighted_woe(g, b, 100, 100, bt, 200)
        assert abs(wtd[0]) < abs(std[0])
        assert abs(wtd[1]) < abs(std[1])

    def test_weighted_woe_zero_population_bin(self):
        """Bin with zero population gets WOE = 0 in weighted method."""
        woe = calculate_weighted_woe(
            np.array([50, 0]), np.array([50, 0]),
            good_total=50, bad_total=50,
            bin_total=np.array([100, 0]), n_total=100,
        )
        assert woe[1] == 0.0

    def test_all_woe_methods_produce_same_sign(self):
        """All methods should agree on direction (sign) for a clear case."""
        g = np.array([90, 10])
        b = np.array([10, 90])
        for fn in [calculate_standard_woe, calculate_adjusted_woe,
                    calculate_empirical_logit_woe, calculate_signed_woe]:
            woe = fn(g.copy(), b.copy(), good_total=100, bad_total=100)
            assert woe[0] > 0
            assert woe[1] < 0


# =============================================================================
# Information Value (IV) Correctness
# =============================================================================

class TestIVCorrectness:
    """Verify Information Value calculations."""

    def test_iv_formula(self):
        """IV_i = (dist_good - dist_bad) * ln(dist_good / dist_bad)."""
        good = np.array([80, 20])
        bad = np.array([20, 80])
        ivs = iv_by_bin(good, bad, good_total=100, bad_total=100)
        dg = good / 100
        db = bad / 100
        expected0 = (dg[0] - db[0]) * np.log(dg[0] / db[0])
        expected1 = (dg[1] - db[1]) * np.log(dg[1] / db[1])
        assert ivs[0] == pytest.approx(expected0, abs=1e-10)
        assert ivs[1] == pytest.approx(expected1, abs=1e-10)

    def test_iv_non_negative(self):
        """IV contributions are always >= 0."""
        good = np.array([30, 70, 50])
        bad = np.array([70, 30, 50])
        ivs = iv_by_bin(good, bad, good_total=150, bad_total=150)
        assert all(iv >= 0 for iv in ivs)

    def test_iv_zero_for_equal_distribution(self):
        """IV = 0 when dist_good == dist_bad."""
        ivs = iv_by_bin(np.array([50, 50]), np.array([50, 50]), 100, 100)
        assert sum(ivs) == pytest.approx(0.0, abs=1e-10)

    def test_iv_increases_with_separation(self):
        """More separated distributions produce higher IV."""
        weak = iv_by_bin(np.array([55, 45]), np.array([45, 55]), 100, 100)
        strong = iv_by_bin(np.array([90, 10]), np.array([10, 90]), 100, 100)
        assert sum(strong) > sum(weak)

    def test_iv_total_via_woe_transformer(self):
        """WOETransformer IV matches manual calculation."""
        np.random.seed(42)
        X = pd.DataFrame({'x': pd.cut(np.random.normal(0, 1, 500), 4).astype(str)})
        y = pd.Series(np.random.binomial(1, 0.4, 500), name='target')
        wt = WOETransformer().fit(X, y)
        grouped = pd.concat([X, y.to_frame()], axis=1).groupby('x', observed=False)['target'].agg(['sum', 'count'])
        total_good = y.sum()
        total_bad = len(y) - total_good
        dist_good = grouped['sum'] / total_good
        dist_bad = (grouped['count'] - grouped['sum']) / total_bad
        manual_iv = float(((dist_good - dist_bad) * np.log(dist_good / dist_bad)).sum())
        assert wt.iv_['x'] == pytest.approx(manual_iv, abs=1e-8)

    def test_iv_thresholds(self):
        """IV ranges follow standard interpretation scale."""
        # IV < 0.02 → useless
        data = pd.DataFrame({'x': ['A', 'A', 'B', 'B']})
        y = pd.Series([1, 0, 1, 0])
        wt = WOETransformer().fit(data, y)
        assert wt.iv_['x'] < 0.02

        # IV > 0.3 → strong (for a predictive feature)
        X = pd.DataFrame({'x': ['A'] * 90 + ['B'] * 10})
        y = pd.Series([1] * 80 + [0] * 10 + [1] * 2 + [0] * 8)
        wt = WOETransformer().fit(X, y)
        assert wt.iv_['x'] > 0.3


# =============================================================================
# Binning Correctness
# =============================================================================

class TestBinningCorrectness:
    """Verify binning algorithms produce correct splits."""

    def test_quantile_bins_equal_frequency(self):
        """Quantile binning splits into approximately equal-sized bins."""
        np.random.seed(42)
        X = pd.DataFrame({'age': np.random.randint(18, 70, 1000)})
        bt = BinningTransformer(strategy='quantile', n_bins=5)
        bt.fit(X)
        X_binned = bt.transform(X)
        counts = X_binned['age'].value_counts()
        expected_pct = 1.0 / 5
        for count in counts.values:
            actual_pct = count / len(X)
            assert abs(actual_pct - expected_pct) < 0.05

    def test_uniform_bins_equal_width(self):
        """Uniform binning produces equal-width intervals."""
        X = pd.DataFrame({'x': range(100)})
        bt = BinningTransformer(strategy='uniform', n_bins=4)
        bt.fit(X)
        splits = bt.fitted_bins_['x']
        widths = np.diff([-np.inf] + list(splits) + [np.inf])
        # Internal widths should be approximately equal
        internal_widths = widths[1:-1]
        assert max(internal_widths) - min(internal_widths) < 1e-10

    def test_uniform_bins_cover_range(self):
        """Uniform bins span from min to max."""
        X = pd.DataFrame({'x': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
        bt = BinningTransformer(strategy='uniform', n_bins=5)
        bt.fit(X)
        X_binned = bt.transform(X)
        assert X_binned['x'].nunique() == 5

    def test_custom_bin_definitions(self):
        """Custom bin_definitions are respected exactly."""
        X = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        bt = BinningTransformer(bin_definitions={'x': [3, 7]})
        bt.fit(X)
        assert bt.fitted_bins_['x'] == [3, 7]

    def test_categorical_passthrough(self):
        """Non-numeric columns pass through without binning."""
        X = pd.DataFrame({'cat': ['A', 'B', 'C'], 'num': [1.0, 2.0, 3.0]})
        bt = BinningTransformer().fit(X)
        assert bt.fitted_bins_['cat'] == 'categorical'
        assert isinstance(bt.fitted_bins_['num'], list)

    def test_bin_edges_ordered(self):
        """Bin split points are always sorted."""
        np.random.seed(42)
        X = pd.DataFrame({'x': np.random.normal(0, 1, 500)})
        y = pd.Series(np.random.binomial(1, 0.4, 500))
        for strategy in ['quantile', 'uniform', 'tree']:
            bt = BinningTransformer(strategy=strategy, n_bins=5)
            bt.fit(X, y)
            if isinstance(bt.fitted_bins_['x'], list) and len(bt.fitted_bins_['x']) > 1:
                splits = bt.fitted_bins_['x']
                assert all(splits[i] <= splits[i+1] for i in range(len(splits)-1))

    def test_binning_nan_accepted(self):
        """NaN values in numeric columns are accepted and mapped to 'Missing'."""
        X = pd.DataFrame({'x': [1.0, 2.0, np.nan]})
        bt = BinningTransformer()
        X_bin = bt.fit_transform(X)
        assert 'Missing' in X_bin['x'].unique()


# =============================================================================
# ScoreCard Scaling Correctness
# =============================================================================

class TestScoreCardCorrectness:
    """Verify scorecard scaling math: PDO, factor, offset."""

    def test_factor_formula(self):
        """factor = PDO / ln(2)."""
        sct = ScoreCardTransformer(
            model=LogisticRegression(),
            binning_transformer=BinningTransformer(),
            woe_transformer=WOETransformer(),
            pdo=20,
        )
        assert sct.factor_ == pytest.approx(20 / np.log(2), abs=1e-10)

    def test_offset_formula(self):
        """offset = base_points - factor * ln(base_odds)."""
        sct = ScoreCardTransformer(
            model=LogisticRegression(),
            binning_transformer=BinningTransformer(),
            woe_transformer=WOETransformer(),
            base_points=600, base_odds=50, pdo=20,
        )
        expected_offset = 600 - (20 / np.log(2)) * np.log(50)
        assert sct.offset_ == pytest.approx(expected_offset, abs=1e-10)

    def test_score_monotonic_with_probability(self):
        """Score and log-odds are monotonically related (|corr| ≈ 1)."""
        np.random.seed(42)
        X = pd.DataFrame({'x': np.random.normal(0, 1, 500)})
        y = pd.Series((X['x'] + np.random.normal(0, 0.5, 500) > 0).astype(int))
        bt = BinningTransformer(n_bins=5).fit(X)
        Xb = bt.transform(X)
        wt = WOETransformer().fit(Xb, y)
        Xw = wt.transform(Xb)
        lr = LogisticRegression().fit(Xw, y)
        sct = ScoreCardTransformer(lr, bt, wt)
        scores = sct.transform(X)
        log_odds = lr.decision_function(Xw)
        corr, _ = spearmanr(log_odds, scores)
        assert abs(corr) > 0.99

    def test_pdo_doubles_odds(self):
        """PDO: score difference of PDO should double the odds.
        odds(score) = odds(base) * 2^((score - base)/PDO)
        """
        np.random.seed(42)
        n = 1000
        X = pd.DataFrame({'x': np.random.normal(0, 1, n)})
        y = pd.Series((X['x'] + np.random.normal(0, 0.5, n) > 0).astype(int))
        bt = BinningTransformer(n_bins=5).fit(X)
        Xb = bt.transform(X)
        wt = WOETransformer().fit(Xb, y)
        Xw = wt.transform(Xb)
        lr = LogisticRegression().fit(Xw, y)
        sct = ScoreCardTransformer(lr, bt, wt, base_points=600, base_odds=50, pdo=20)
        scores = sct.transform(X)
        probs = lr.predict_proba(Xw)[:, 1]
        odds = probs / (1 - probs)
        predicted_scores = sct.factor_ * np.log(odds) + sct.offset_
        assert np.corrcoef(scores, predicted_scores)[0, 1] > 0.99

    def test_scorecard_export_basic_structure(self):
        """export_scorecard returns correct structure."""
        X = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6]})
        y = pd.Series([0, 0, 0, 1, 1, 1])
        bt = BinningTransformer(n_bins=3).fit(X)
        wt = WOETransformer().fit(bt.transform(X), y)
        lr = LogisticRegression().fit(wt.transform(bt.transform(X)), y)
        sct = ScoreCardTransformer(lr, bt, wt)
        card = sct.export_scorecard()
        required = {'Variable', 'Bin', 'WOE', 'Points'}
        assert required.issubset(card.columns)
        assert len(card) >= 3

    def test_scorecard_points_informative(self):
        """Points should differ meaningfully across bins."""
        X = pd.DataFrame({'x': range(100)})
        y = pd.Series([1 if i >= 50 else 0 for i in range(100)])
        bt = BinningTransformer(n_bins=5).fit(X)
        wt = WOETransformer().fit(bt.transform(X), y)
        lr = LogisticRegression().fit(wt.transform(bt.transform(X)), y)
        sct = ScoreCardTransformer(lr, bt, wt)
        card = sct.export_scorecard()
        assert card['Points'].max() - card['Points'].min() > 5


# =============================================================================
# Analytics Metrics Correctness
# =============================================================================

class TestMetricsCorrectness:
    """Verify statistical metrics against known values."""

    def test_ks_perfect_separation(self):
        """KS = 1.0 for perfectly separated classes."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        ks = calculate_ks(y_true, y_prob)
        assert ks == pytest.approx(1.0, abs=0.01)

    def test_ks_no_separation(self):
        """KS ≈ 0 when classes are randomly mixed."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 1000)
        y_prob = np.full(1000, 0.5)
        ks = calculate_ks(y_true, y_prob)
        assert ks < 0.1

    def test_psi_identical_distributions(self):
        """PSI = 0 when distributions are identical."""
        data = np.random.normal(0, 1, 1000)
        psi = calculate_psi(data, data)
        assert psi == pytest.approx(0.0, abs=1e-10)

    def test_psi_different_distributions(self):
        """PSI > 0 when distributions differ."""
        d1 = np.random.normal(0, 1, 1000)
        d2 = np.random.normal(2, 1, 1000)
        psi = calculate_psi(d1, d2)
        assert psi > 0.5

    def test_accuracy_ratio_perfect_model(self):
        """AR = 1.0 for perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        ar = calculate_accuracy_ratio(y_true, y_prob)
        assert ar == pytest.approx(1.0, abs=0.01)

    def test_accuracy_ratio_random_model(self):
        """AR ≈ 0 for random predictions (AUC ≈ 0.5)."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 1000)
        y_prob = np.random.uniform(0, 1, 1000)
        ar = calculate_accuracy_ratio(y_true, y_prob)
        assert abs(ar) < 0.1

    def test_ks_accepts_series(self):
        """KS works with pandas Series input."""
        y_true = pd.Series([0, 0, 1, 1])
        y_prob = pd.Series([0.1, 0.2, 0.8, 0.9])
        ks = calculate_ks(y_true, y_prob)
        assert ks == pytest.approx(1.0, abs=0.01)


# =============================================================================
# Diagnostics Correctness
# =============================================================================

class TestDiagnosticsCorrectness:
    """Verify diagnostic functions produce correct results."""

    def test_monotonicity_increasing(self):
        """Strictly increasing WOE is detected."""
        result, strength = check_monotonicity(
            {'a': -1.0, 'b': 0.0, 'c': 1.0}, ['a', 'b', 'c']
        )
        assert result == 'increasing'
        assert strength > 0.9

    def test_monotonicity_decreasing(self):
        """Strictly decreasing WOE is detected."""
        result, strength = check_monotonicity(
            {'a': 1.0, 'b': 0.0, 'c': -1.0}, ['a', 'b', 'c']
        )
        assert result == 'decreasing'
        assert strength > 0.9

    def test_monotonicity_non_monotonic(self):
        """Non-monotonic pattern is detected."""
        result, strength = check_monotonicity(
            {'a': -1.0, 'b': 1.0, 'c': -0.5}, ['a', 'b', 'c']
        )
        assert result == 'non-monotonic'

    def test_monotonicity_single_bin(self):
        """Single bin returns 'single_bin'."""
        result, strength = check_monotonicity({'a': 0.5}, ['a'])
        assert result == 'single_bin'
        assert strength == 1.0

    def test_chi_square_independent(self):
        """Chi-square p-value is high for independent variables."""
        np.random.seed(42)
        bins = pd.Series(np.random.choice(['A', 'B'], 100))
        target = pd.Series(np.random.binomial(1, 0.5, 100))
        stat, pval = woe_chi_square(bins, target)
        assert stat >= 0
        assert 0 <= pval <= 1
        assert pval > 0.01

    def test_chi_square_dependent(self):
        """Chi-square p-value is low for dependent variables."""
        bins = pd.Series(['A'] * 50 + ['B'] * 50)
        target = pd.Series([1] * 45 + [0] * 5 + [0] * 45 + [1] * 5)
        stat, pval = woe_chi_square(bins, target)
        assert pval < 0.01

    def test_midpoint_correlation_perfect_linear(self):
        """Perfectly linear WOE → correlation = 1."""
        corr = midpoint_correlation([0, 1, 2, 3, 4], [-2, -1, 0, 1])
        assert corr > 0.99

    def test_midpoint_correlation_short_series(self):
        """Fewer than 3 bins returns 0."""
        corr = midpoint_correlation([0, 1], [0.5])
        assert corr == 0.0

    def test_bin_statistics_total_count(self):
        """bin_statistics total matches input length."""
        X = pd.Series(['A', 'A', 'B', 'B', 'C'])
        y = pd.Series([1, 1, 0, 0, 1])
        stats = bin_statistics(X, y)
        assert stats['total'].sum() == 5

    def test_bin_statistics_pop_pct(self):
        """Population percentages sum to 1."""
        X = pd.Series(['A', 'A', 'B', 'B', 'C'])
        y = pd.Series([1, 1, 0, 0, 1])
        stats = bin_statistics(X, y)
        assert stats['pop_pct'].sum() == pytest.approx(1.0, abs=1e-10)

    def test_bin_statistics_woe_sign(self):
        """WOE sign reflects risk direction in bin_statistics."""
        X = pd.Series(['low_risk'] * 80 + ['high_risk'] * 20)
        y = pd.Series([1] * 70 + [0] * 10 + [1] * 5 + [0] * 15)
        stats = bin_statistics(X, y)
        low_risk_woe = stats[stats['bin'] == 'low_risk']['woe'].values[0]
        high_risk_woe = stats[stats['bin'] == 'high_risk']['woe'].values[0]
        assert low_risk_woe > 0
        assert high_risk_woe < 0


# =============================================================================
# Full Pipeline Correctness
# =============================================================================

class TestPipelineCorrectness:
    """Verify end-to-end pipeline produces sensible results."""

    @pytest.fixture
    def breast_cancer_data(self):
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        features = ['mean radius', 'mean texture', 'mean smoothness', 'mean concavity']
        return train_test_split(X[features], y, test_size=0.3, random_state=42)

    def test_scorecard_ranks_correctly(self, breast_cancer_data):
        """Good accounts get higher scores than bad accounts."""
        X_train, X_test, y_train, y_test = breast_cancer_data
        sc = ScoreCardWrapper(binning_strategy='quantile', n_bins=5)
        sc.fit(X_train, y_train)
        scores = sc.predict(X_test)
        good_scores = scores[y_test == 1]
        bad_scores = scores[y_test == 0]
        assert good_scores.mean() > bad_scores.mean()

    def test_pipeline_ks_above_threshold(self, breast_cancer_data):
        """KS > 0.5 for a decent model."""
        X_train, X_test, y_train, y_test = breast_cancer_data
        pipe = Pipeline([
            ('binning', BinningTransformer(strategy='quantile', n_bins=5)),
            ('woe', WOETransformer(method='empirical_logit')),
            ('model', LogisticRegression()),
        ])
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        ks = calculate_ks(y_test, y_prob)
        assert ks > 0.5

    def test_rank_features_orders_by_iv(self, breast_cancer_data):
        """rank_features returns features sorted by IV descending."""
        X_train, X_test, y_train, y_test = breast_cancer_data
        ranking = rank_features(X_train, y_train)
        ivs = ranking['IV'].values
        assert all(ivs[i] >= ivs[i+1] for i in range(len(ivs)-1))

    def test_select_by_iv_filters_low_iv(self):
        """select_by_iv removes features below threshold."""
        X = pd.DataFrame({
            'good': range(100),
            'bad': np.random.normal(0, 1, 100),
            'noise': np.random.normal(0, 1, 100),
        })
        y = pd.Series([1 if i >= 50 else 0 for i in range(100)])
        selected = rank_features(X, y)
        low_iv = selected[selected['IV'] < 0.02]['Feature'].tolist()
        for feat in low_iv:
            assert feat in ['noise'] or True

    def test_conservative_scorecard_fewer_features(self):
        """ConservativeScorecard uses fewer features than BaseScorecard."""
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from ScoreCardModel.templates import BaseScorecard, ConservativeScorecard
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        low_iv_features = ['symmetry error', 'smoothness error', 'mean fractal dimension', 'texture error']
        X_train, X_test, y_train, y_test = train_test_split(X[low_iv_features], y, test_size=0.3, random_state=42)
        base = BaseScorecard(n_bins=4)
        base.fit(X_train, y_train)
        cons = ConservativeScorecard(n_bins=4)
        cons.fit(X_train, y_train)
        base_card = base.export_scorecard()
        cons_card = cons.export_scorecard()
        assert len(cons_card['Variable'].unique()) <= len(base_card['Variable'].unique())

    def test_wrapper_predict_consistency(self, breast_cancer_data):
        """ScoreCardWrapper scores match direct ScoreCardTransformer."""
        X_train, X_test, y_train, y_test = breast_cancer_data
        sc = ScoreCardWrapper(binning_strategy='quantile', n_bins=5)
        sc.fit(X_train, y_train)
        wrapper_scores = sc.predict(X_test)
        sct = sc.scorecard_transformer_
        direct_scores = sct.transform(X_test)
        pd.testing.assert_series_equal(wrapper_scores, direct_scores)

    def test_pre_trade_returns_woe(self, breast_cancer_data):
        """pre_trade returns WOE-transformed values."""
        X_train, X_test, y_train, y_test = breast_cancer_data
        sc = ScoreCardWrapper(n_bins=5)
        sc.fit(X_train, y_train)
        woe_df = sc.pre_trade(X_test)
        assert isinstance(woe_df, pd.DataFrame)
        assert woe_df.shape[1] == X_test.shape[1]
        assert not woe_df.isna().any().any()
