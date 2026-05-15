import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from ScoreCardModel.score_card.wrapper import ScoreCardWrapper
from ScoreCardModel.analytics.plotting import plot_ks, plot_roc, plot_bin_stats
from ScoreCardModel.analytics.metrics import calculate_ks

# 1. Load Data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target) # 0 is malignant, 1 is benign

# Select a few features for clarity
features = ['mean radius', 'mean texture', 'mean smoothness', 'mean concavity']
X = X[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("--- ScoreCard Modern Refactoring Demo ---")

# 2. Use the Analyst Facade (ScoreCardWrapper)
sc = ScoreCardWrapper(binning_strategy='quantile', n_bins=5, base_points=600, pdo=20)
sc.fit(X_train, y_train)

# 3. Predictions and Scoring
scores = sc.predict(X_test)
print(f"\nSample Scores (Higher is better/Benign):\n{scores.head()}")

# 4. Export the Systematic Scorecard
scorecard_df = sc.export_scorecard()
print(f"\nScorecard Table (Snippet):\n{scorecard_df.head(10)}")

# 5. Professional Analytics & Visualizations
print("\nGenerating Professional Review Reports...")

# Performance Metrics
y_prob = sc.pipeline.predict_proba(X_test)[:, 1]
ks_stat = calculate_ks(y_test, y_prob)
print(f"Model KS Statistic: {ks_stat:.3f}")

# Plotting (Note: In a non-interactive shell these might not show, 
# but they are ready for Notebooks/GUI)
# plot_ks(y_test, y_prob)
# plot_roc(y_test, y_prob)

# Bin Analysis for one feature
X_test_binned = sc.pipeline.named_steps['binning'].transform(X_test)
X_test_binned['target'] = y_test
# plot_bin_stats(X_test_binned, 'mean radius', 'target')

print("\nRefactoring Complete. Use `sc.pipeline` for standard sklearn workflows.")
