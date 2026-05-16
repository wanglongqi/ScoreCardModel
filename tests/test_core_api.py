import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ScoreCardModel.binning.transformers import BinningTransformer
from ScoreCardModel.score_card.transformers import ScoreCardTransformer
from ScoreCardModel.weight_of_evidence.transformers import WOETransformer


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        'age': np.random.randint(18, 70, n),
        'income': np.random.normal(50000, 15000, n),
        'education': np.random.choice(['HighSchool', 'Bachelor', 'Master', 'PhD'], n)
    })
    # Target: higher income and age -> lower risk (1 for good, 0 for bad)
    # Using a simple logic for synthetic target
    logit = (X['age'] - 40) / 20 + (X['income'] - 50000) / 10000 + \
            (X['education'].map({'HighSchool': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}))
    prob = 1 / (1 + np.exp(-logit))
    y = (prob > 0.5).astype(int)
    return X, y

def test_full_pipeline_compatibility(sample_data):
    """Test if the new transformers work within a standard sklearn Pipeline."""
    X, y = sample_data
    
    pipeline = Pipeline([
        ('binning', BinningTransformer(strategy='quantile', n_bins=5)),
        ('woe', WOETransformer()),
        ('model', LogisticRegression())
    ])
    
    pipeline.fit(X, y)
    probs = pipeline.predict_proba(X)
    
    assert probs.shape == (len(X), 2)
    assert pipeline.score(X, y) > 0.5

def test_scorecard_scaling(sample_data):
    """Test the ScoreCardTransformer for mapping WOE/Model to Points."""
    X, y = sample_data
    
    # Pre-calculated/fitted components for testing scaling
    bt = BinningTransformer(n_bins=5).fit(X)
    X_binned = bt.transform(X)
    
    wt = WOETransformer().fit(X_binned, y)
    X_woe = wt.transform(X_binned)
    
    # Ensure columns are in a stable order for LR
    feature_cols = sorted(X_woe.columns.tolist())
    X_woe = X_woe[feature_cols]
    
    lr = LogisticRegression().fit(X_woe, y)
    
    # ScoreCard Scaling: Base Odds 1:50 at 600 points, PDO 20
    sct = ScoreCardTransformer(
        model=lr, 
        binning_transformer=bt, 
        woe_transformer=wt,
        base_points=600,
        base_odds=50,
        pdo=20
    )
    
    scores = sct.transform(X)
    
    # Debug info
    from scipy.stats import spearmanr
    probs = lr.predict_proba(X_woe)[:, 1]
    correlation, _ = spearmanr(probs, scores)
    print(f"\nRank Correlation: {correlation}")
    print(f"Features: {feature_cols}")
    print(f"Coefs: {lr.coef_}")
    
    assert isinstance(scores, pd.Series)
    assert scores.mean() > 0 
    assert correlation > 0.99

def test_scorecard_table_export(sample_data):
    """Test if we can export a systematic scorecard table."""
    X, y = sample_data
    
    # Mock pipeline fitting
    bt = BinningTransformer(n_bins=3).fit(X)
    wt = WOETransformer().fit(bt.transform(X), y)
    lr = LogisticRegression().fit(wt.transform(bt.transform(X)), y)
    
    sct = ScoreCardTransformer(lr, bt, wt)
    card_df = sct.export_scorecard()
    
    assert isinstance(card_df, pd.DataFrame)
    assert all(col in card_df.columns for col in ['Variable', 'Bin', 'WOE', 'Points'])
    assert len(card_df) > 0
