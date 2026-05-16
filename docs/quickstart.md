# Quickstart

## Analyst Facade

```python
from ScoreCardModel import ScoreCardWrapper

sc = ScoreCardWrapper(binning_strategy='quantile', base_points=600, pdo=20)
sc.fit(X_train, y_train)

scores = sc.predict(X_test)
card = sc.export_scorecard()
print(card.head(10))
```

## Scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from ScoreCardModel import BinningTransformer, WOETransformer

pipeline = Pipeline([
    ('binning', BinningTransformer(strategy='tree', n_bins=5)),
    ('woe', WOETransformer(method='empirical_logit')),
    ('model', LogisticRegression())
])
pipeline.fit(X_train, y_train)
```

## Scorecard Templates

```python
from ScoreCardModel.templates import BaseScorecard

sc = BaseScorecard()
sc.fit(X_train, y_train)
scores = sc.predict(X_test)
```

## Automated Report

```python
from ScoreCardModel.analytics.reporting import generate_report

generate_report(pipeline, X_train, y_train, X_test, y_test,
                output_path="scorecard_report.html")
```

## Feature Analysis

```python
from ScoreCardModel.analytics.selection import rank_features

ranking = rank_features(X_train, y_train)
print(ranking[['Feature', 'IV', 'IV_Label', 'Monotonicity', 'Recommendation']])
```
