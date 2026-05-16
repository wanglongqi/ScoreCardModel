"""
ScoreCardModel
==============

A professional and modern toolset for scorecard modeling, fully compatible
with scikit-learn. Designed for credit risk analysts and data scientists.

Core Components:
----------------
* binning: Discrete variable transformation (Quantile, Uniform, Optimal, Tree).
* weight_of_evidence: 5 WOE methods, diagnostics, and IV calculation.
* score_card: PDO/Base-Odds scaling and business-friendly score mapping.
* analytics: Professional metrics, 16+ plot types, and automated HTML reports.
* templates: Pre-built scorecard configurations for common scenarios.

Modern API Usage:
-----------------

1. **ScoreCardWrapper (Recommended for Analysts):**
   ```python
   from ScoreCardModel import ScoreCardWrapper

   sc = ScoreCardWrapper(binning_strategy='quantile', base_points=600)
   sc.fit(X_train, y_train)
   scores = sc.predict(X_test)
   card = sc.export_scorecard()
   ```

2. **Scikit-learn Pipeline (Recommended for Data Scientists):**
   ```python
   from sklearn.pipeline import Pipeline
   from sklearn.linear_model import LogisticRegression
   from ScoreCardModel import BinningTransformer, WOETransformer

   pipeline = Pipeline([
       ('binning', BinningTransformer(strategy='optimal')),
       ('woe', WOETransformer(method='empirical_logit')),
       ('model', LogisticRegression())
   ])
   pipeline.fit(X, y)
   ```

3. **Automated Report:**
   ```python
   from ScoreCardModel.analytics.reporting import generate_report
   generate_report(pipeline, X_train, y_train, X_test, y_test)
   ```

4. **Scorecard Templates:**
   ```python
   from ScoreCardModel.templates import BaseScorecard
   sc = BaseScorecard()
   sc.fit(X_train, y_train)
   ```

For detailed WOE method selection and diagnostic guidance, see:
  `docs/woe_in_depth.md`
"""

__version__ = "2.0.0"

from ScoreCardModel.binning.transformers import BinningTransformer
from ScoreCardModel.score_card.transformers import ScoreCardTransformer
from ScoreCardModel.score_card.wrapper import ScoreCardWrapper
from ScoreCardModel.templates import BaseScorecard, ConservativeScorecard
from ScoreCardModel.weight_of_evidence.transformers import WOETransformer

__all__ = [
    "BinningTransformer",
    "WOETransformer",
    "ScoreCardTransformer",
    "ScoreCardWrapper",
    "BaseScorecard",
    "ConservativeScorecard",
]
