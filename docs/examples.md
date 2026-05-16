# Examples

## End-to-End Scorecard

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from ScoreCardModel.score_card.wrapper import ScoreCardWrapper

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

features = ['mean radius', 'mean texture', 'mean smoothness', 'mean concavity']
X = X[features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

sc = ScoreCardWrapper(binning_strategy='quantile', n_bins=5, base_points=600, pdo=20)
sc.fit(X_train, y_train)

scores = sc.predict(X_test)
print(scores.head())

scorecard_df = sc.export_scorecard()
print(scorecard_df.head(10))
```

## Pipeline with Custom Model

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from ScoreCardModel import BinningTransformer, WOETransformer, ScoreCardTransformer

pipeline = Pipeline([
    ('binning', BinningTransformer(strategy='optimal')),
    ('woe', WOETransformer(method='empirical_logit')),
    ('model', LogisticRegression())
])
pipeline.fit(X_train, y_train)

# Score new data
# ... build ScoreCardTransformer with the fitted components ...
```

## Generate Report

```python
from ScoreCardModel.analytics.reporting import generate_report

generate_report(pipeline, X_train, y_train, X_test, y_test,
                output_path="scorecard_report.html")
```

## Feature Selection

```python
from ScoreCardModel.analytics.selection import rank_features

ranking = rank_features(X_train, y_train)
print(ranking[ranking['IV'] > 0.02])
```
