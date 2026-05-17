
"""Compare all 5 WOE methods side-by-side on the Breast Cancer dataset."""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ScoreCardModel import BinningTransformer, WOETransformer
from ScoreCardModel.analytics.metrics import calculate_ks

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

methods = ['standard', 'adjusted', 'empirical_logit', 'signed', 'weighted']

for n_bins in [4, 5, 10]:
    results = []
    for method in methods:
        pipeline = Pipeline([
            ('binning', BinningTransformer(n_bins=n_bins)),
            ('woe', WOETransformer(method=method)),
            ('model', LogisticRegression(C=1.0, max_iter=5000))
        ])
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        ks = calculate_ks(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        iv_total = sum(pipeline.named_steps['woe'].iv_.values())
        results.append({
            'Method': method,
            'KS': round(ks, 4),
            'AUC': round(auc, 4),
            'Total IV': round(iv_total, 4),
        })

    print(f"## WOE Method Comparison ({n_bins} bins)\n")
    print(pd.DataFrame(results).to_markdown(index=False))
    print()
