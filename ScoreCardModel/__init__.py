"""
ScoreCardModel
==============

A professional and modern toolset for scorecard modeling, fully compatible 
with scikit-learn.

Core Components:
----------------
* binning: Discrete variable transformation (Quantile, Uniform, Optimal, Tree).
* weight_of_evidence: Vectorized WOE and Information Value (IV) calculation.
* score_card: Mapping of model probabilities to business-friendly scores.
* analytics: Professional metrics (KS, AUC, PSI) and systematic review reports.

Modern API Usage:
-----------------
The library provides two ways to build scorecards:

1. **Scikit-learn Pipeline (Recommended for Data Scientists):**
   ```python
   from sklearn.pipeline import Pipeline
   from sklearn.linear_model import LogisticRegression
   from ScoreCardModel.binning.transformers import BinningTransformer
   from ScoreCardModel.weight_of_evidence.transformers import WOETransformer
   
   pipeline = Pipeline([
       ('binning', BinningTransformer(strategy='optimal')),
       ('woe', WOETransformer()),
       ('model', LogisticRegression())
   ])
   pipeline.fit(X, y)
   ```

2. **ScoreCardWrapper (Recommended for Business Analysts):**
   ```python
   from ScoreCardModel.score_card.wrapper import ScoreCardWrapper
   
   sc = ScoreCardWrapper(binning_strategy='quantile', base_points=600)
   sc.fit(X, y)
   scores = sc.predict(X_new)
   ```

Professional Reports:
---------------------
```python
from ScoreCardModel.analytics.plotting import plot_ks, plot_bin_stats
plot_ks(y_test, y_prob)
```
"""

__version__ = "2.0.0"

from ScoreCardModel.binning.transformers import BinningTransformer
from ScoreCardModel.weight_of_evidence.transformers import WOETransformer
from ScoreCardModel.score_card.transformers import ScoreCardTransformer
from ScoreCardModel.score_card.wrapper import ScoreCardWrapper

__all__ = [
    "BinningTransformer",
    "WOETransformer",
    "ScoreCardTransformer",
    "ScoreCardWrapper",
]
