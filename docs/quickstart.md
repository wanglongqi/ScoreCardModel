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

## Feature Selection

```python
from ScoreCardModel.analytics.selection import rank_features, select_by_correlation

ranking = rank_features(X_train, y_train)
accept = ranking[ranking['Recommendation'] == 'Accept']['Feature'].tolist()
final = select_by_correlation(X_train[accept], max_corr=0.7)
pipeline.fit(X_train[final], y_train)
```

## Automated Report

```python
from ScoreCardModel.analytics.reporting import generate_report

generate_report(pipeline, X_train, y_train, X_test, y_test,
                output_path="scorecard_report.md")
```

Generates a Markdown report with embedded PNG plots, scorecard table, and executive summary — ready for regulatory review.

## Interactive What-If (Jupyter)

```python
# Requires: pip install scorecard-toolkit[interactive]
from ScoreCardModel.interactive import ScorecardWidget

widget = ScorecardWidget(pipeline, X_train)
widget.display()
```

Opens an interactive dashboard with sliders/dropdowns for each feature and a live waterfall chart that updates as you adjust values. See the [Interactive Notebook](examples/interactive_scorecard.ipynb) for a full walkthrough.

## Scorecard Table in Jupyter

```python
from ScoreCardModel.score_card.transformers import ScoreCardTransformer

lr, bt, wt = pipeline.named_steps['model'], pipeline.named_steps['binning'], pipeline.named_steps['woe']
sct = ScoreCardTransformer(lr, bt, wt)
sct  # renders as a styled HTML table in the notebook
```
