
import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from ScoreCardModel import BinningTransformer, WOETransformer, ScoreCardWrapper
from ScoreCardModel.analytics.reporting import generate_report
from ScoreCardModel.analytics.selection import rank_features, select_by_correlation
from ScoreCardModel.analytics.plotting import (
    plot_ks, plot_roc, plot_cap, plot_gain_lift,
    plot_score_distribution, plot_calibration,
    plot_woe_pattern, plot_iv_summary_enhanced,
    plot_scorecard_waterfall, plot_scorecard_heatmap,
    plot_cutoff_optimization, plot_confusion_matrix
)
from ScoreCardModel.score_card.transformers import ScoreCardTransformer

os.makedirs("docs/examples", exist_ok=True)
os.makedirs("docs/images", exist_ok=True)


def regenerate_breast_cancer():
    print("Regenerating Breast Cancer Report...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    pipeline = Pipeline([
        ('binning', BinningTransformer(n_bins=5)),
        ('woe', WOETransformer(method='empirical_logit')),
        ('model', LogisticRegression(C=1.0, max_iter=5000))
    ])
    pipeline.fit(X_train, y_train)

    generate_report(pipeline, X_train, y_train, X_test, y_test,
                    output_path="docs/examples/breast_cancer_report.md")


def regenerate_german_credit():
    print("Regenerating German Credit Report...")
    data = fetch_openml('credit-g', as_frame=True, parser='pandas', version=1)
    X, y = data.data, (data.target == 'good').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    ranking = rank_features(X_train, y_train, n_bins=4)
    final = ranking[ranking['Recommendation'] != 'Reject']['Feature'].tolist()

    pipeline = Pipeline([
        ('binning', BinningTransformer(n_bins=4)),
        ('woe', WOETransformer()),
        ('model', LogisticRegression(C=1.0, max_iter=5000))
    ])
    pipeline.fit(X_train[final], y_train)

    generate_report(pipeline, X_train[final], y_train, X_test[final], y_test,
                    output_path="docs/examples/german_credit_report.md")


def regenerate_taiwan_credit():
    print("Regenerating Taiwan Credit Report & Hero Images...")
    COLUMN_MAP = {
        'x1': 'LIMIT_BAL', 'x2': 'SEX', 'x3': 'EDUCATION', 'x4': 'MARRIAGE', 'x5': 'AGE',
        'x6': 'PAY_0', 'x7': 'PAY_2', 'x8': 'PAY_3', 'x9': 'PAY_4', 'x10': 'PAY_5', 'x11': 'PAY_6',
        'x12': 'BILL_AMT1', 'x13': 'BILL_AMT2', 'x14': 'BILL_AMT3', 'x15': 'BILL_AMT4',
        'x16': 'BILL_AMT5', 'x17': 'BILL_AMT6',
        'x18': 'PAY_AMT1', 'x19': 'PAY_AMT2', 'x20': 'PAY_AMT3', 'x21': 'PAY_AMT4',
        'x22': 'PAY_AMT5', 'x23': 'PAY_AMT6',
    }
    data = fetch_openml('default-of-credit-card-clients', as_frame=True, parser='pandas', version=1)
    X = data.data.rename(columns=COLUMN_MAP)
    y = (data.target == '0').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    final = ['PAY_3', 'LIMIT_BAL', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT6', 'PAY_AMT4', 'PAY_AMT5', 'PAY_0']

    pipeline = Pipeline([
        ('binning', BinningTransformer(n_bins=5)),
        ('woe', WOETransformer(method='empirical_logit')),
        ('model', LogisticRegression(C=1.0, max_iter=5000))
    ])
    pipeline.fit(X_train[final], y_train)

    generate_report(pipeline, X_train[final], y_train, X_test[final], y_test,
                    output_path="docs/examples/taiwan_credit_report.md")

    print("Generating Hero Images...")
    y_prob = pipeline.predict_proba(X_test[final])[:, 1]

    lr = pipeline.named_steps['model']
    bt = pipeline.named_steps['binning']
    wt = pipeline.named_steps['woe']
    sct = ScoreCardTransformer(lr, bt, wt)
    scores = sct.transform(X_test[final])
    card = sct.export_scorecard()

    plot_ks(y_test, y_prob).savefig("docs/images/ks_curve.png", dpi=150)
    plot_roc(y_test, y_prob).savefig("docs/images/roc_curve.png", dpi=150)
    plot_cap(y_test, y_prob).savefig("docs/images/cap_curve.png", dpi=150)
    plot_gain_lift(y_test, y_prob).savefig("docs/images/gain_lift.png", dpi=150)
    plot_score_distribution(scores, y_test).savefig("docs/images/score_distribution.png", dpi=150)
    plot_calibration(y_test, y_prob).savefig("docs/images/calibration.png", dpi=150)

    feat = 'PAY_0'
    woe_map = wt.woe_maps_[feat]
    plot_woe_pattern(woe_map, list(woe_map.keys()), feature_name=feat).savefig("docs/images/woe_pattern.png", dpi=150)

    plot_iv_summary_enhanced(wt.iv_).savefig("docs/images/iv_summary.png", dpi=150)
    plot_scorecard_heatmap(card).savefig("docs/images/scorecard_heatmap.png", dpi=150)
    plot_cutoff_optimization(y_test, y_prob).savefig("docs/images/cutoff_optimization.png", dpi=150)

    y_pred = (y_prob >= 0.5).astype(int)
    plot_confusion_matrix(y_test, y_pred).savefig("docs/images/confusion_matrix.png", dpi=150)


def regenerate_give_me_some_credit():
    print("Regenerating Give Me Some Credit Report...")
    data = fetch_openml('give-me-some-credit', as_frame=True, parser='pandas', version=1)
    X, y = data.data, (data.target == '0').astype(int)
    X['MonthlyIncome'] = X['MonthlyIncome'].fillna(X['MonthlyIncome'].median())
    X['NumberOfDependents'] = X['NumberOfDependents'].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    ranking = rank_features(X_train, y_train, n_bins=10)
    final = ranking[ranking['Recommendation'] != 'Reject']['Feature'].tolist()

    pipeline = Pipeline([
        ('binning', BinningTransformer(n_bins=10)),
        ('woe', WOETransformer()),
        ('model', LogisticRegression(C=1.0, max_iter=5000))
    ])
    pipeline.fit(X_train[final], y_train)

    generate_report(pipeline, X_train[final], y_train, X_test[final], y_test,
                    output_path="docs/examples/give_me_some_credit_report.md")


if __name__ == "__main__":
    regenerate_breast_cancer()
    regenerate_german_credit()
    regenerate_taiwan_credit()
    regenerate_give_me_some_credit()
    print("All examples regenerated successfully.")
