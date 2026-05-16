import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from ScoreCardModel.binning.transformers import BinningTransformer
from ScoreCardModel.weight_of_evidence.transformers import WOETransformer
from ScoreCardModel.analytics.reporting import generate_report


@pytest.fixture
def trained_pipeline():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        'age': np.random.randint(18, 70, n),
        'income': np.random.normal(50000, 15000, n),
    })
    y = pd.Series((X['age'] > 40).astype(int).values.ravel())
    pipe = Pipeline([
        ('binning', BinningTransformer(n_bins=4)),
        ('woe', WOETransformer()),
        ('model', LogisticRegression()),
    ])
    pipe.fit(X, y)
    return pipe, X, y


def test_generate_report_creates_file(trained_pipeline, tmp_path):
    pipe, X, y = trained_pipeline
    output = tmp_path / "report.html"
    result = generate_report(pipe, X, y, X, y, output_path=str(output))
    assert output.exists()
    assert output.stat().st_size > 500


def test_generate_report_returns_path(trained_pipeline, tmp_path):
    pipe, X, y = trained_pipeline
    output = tmp_path / "test_report.html"
    result = generate_report(pipe, X, y, X, y, output_path=str(output))
    assert result == str(output)


def test_generate_report_without_scorecard(tmp_path):
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({'x': np.random.normal(0, 1, n)})
    y = pd.Series((X['x'] > 0).astype(int).values.ravel())
    pipe = Pipeline([
        ('binning', BinningTransformer(n_bins=3)),
        ('woe', WOETransformer()),
        ('model', LogisticRegression()),
    ])
    pipe.fit(X, y)
    output = tmp_path / "simple_report.html"
    result = generate_report(pipe, X, y, X, y, output_path=str(output))
    assert output.exists()
