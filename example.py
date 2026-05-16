import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from ScoreCardModel import ScoreCardWrapper
from ScoreCardModel.analytics.metrics import calculate_ks

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

features = ['mean radius', 'mean texture', 'mean smoothness', 'mean concavity']
X = X[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

sc = ScoreCardWrapper(binning_strategy='quantile', n_bins=5, base_points=600, pdo=20)
sc.fit(X_train, y_train)

scores = sc.predict(X_test)
print("Sample Scores:")
print(scores.head())

card = sc.export_scorecard()
print("\nScorecard Table:")
print(card.head(10))

y_prob = sc.pipeline.predict_proba(X_test)[:, 1]
ks = calculate_ks(y_test, y_prob)
print(f"\nKS Statistic: {ks:.3f}")
