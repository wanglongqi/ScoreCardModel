import os
from typing import Optional

import matplotlib
import pandas as pd
from sklearn.pipeline import Pipeline

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from ScoreCardModel.analytics.metrics import calculate_ks
from ScoreCardModel.analytics.plotting import (
    plot_calibration,
    plot_cap,
    plot_cutoff_optimization,
    plot_iv_summary_enhanced,
    plot_ks,
    plot_roc,
    plot_score_distribution,
    plot_scorecard_heatmap,
)
from ScoreCardModel.analytics.selection import calculate_psi_report, rank_features
from ScoreCardModel.weight_of_evidence.diagnostics import bin_statistics


def export_to_excel(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    output_path: str = "scorecard_specification.xlsx",
) -> str:
    """Export a professional multi-sheet Excel specification for the model."""
    bt = pipeline.named_steps['binning']
    wt = pipeline.named_steps['woe']
    lr = pipeline.named_steps['model']

    from ScoreCardModel.score_card.transformers import ScoreCardTransformer
    sct = ScoreCardTransformer(lr, bt, wt)
    card = sct.export_scorecard()

    # 1. Feature Summary
    summary = rank_features(X_train, y_train, n_bins=bt.n_bins)
    if X_test is not None:
        psi_df = calculate_psi_report(X_train, X_test, n_bins=bt.n_bins)
        summary = summary.merge(psi_df[['Feature', 'PSI', 'Status']], on='Feature', how='left')

    # 2. Detailed Bin Stats
    bin_details = []
    X_bin = bt.transform(X_train)
    for col in X_train.columns:
        stats = bin_statistics(X_bin[col], y_train)
        stats.insert(0, 'Feature', col)
        bin_details.append(stats)
    full_bin_stats = pd.concat(bin_details, ignore_index=True)

    # Write to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        card.to_sheet = writer # type: ignore
        card.to_excel(writer, sheet_name='Scorecard', index=False)
        summary.to_excel(writer, sheet_name='Feature Summary', index=False)
        full_bin_stats.to_excel(writer, sheet_name='Bin Details', index=False)

        # Basic formatting could be added here using openpyxl objects if needed
        # but for now, raw export is a great start.

    return output_path


def generate_report(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str = "scorecard_report.md",
) -> str:
    """Generate a comprehensive report for a fitted scorecard pipeline.

    Creates a markdown file with embedded image references. Plot images are
    saved to a sibling directory named ``<output_stem>_plots/``.
    """
    report_dir = os.path.dirname(output_path) or '.'
    stem = os.path.splitext(os.path.basename(output_path))[0]
    plots_dir = os.path.join(report_dir, f"{stem}_plots")
    os.makedirs(plots_dir, exist_ok=True)

    def save_plot(fig, name):
        path = os.path.join(plots_dir, name)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return os.path.relpath(path, report_dir)

    y_prob_test = pipeline.predict_proba(X_test)[:, 1]
    ks = calculate_ks(y_test.values if hasattr(y_test, 'values') else y_test,
                       y_prob_test)

    from sklearn.metrics import roc_auc_score, roc_curve
    auc = float(roc_auc_score(y_test, y_prob_test))
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    accuracy_ratio = 2 * auc - 1

    sct = None
    try:
        lr = pipeline.named_steps.get('model')
        bt = pipeline.named_steps.get('binning')
        wt = pipeline.named_steps.get('woe')
        if lr is not None and bt is not None and wt is not None:
            from ScoreCardModel.score_card.transformers import ScoreCardTransformer
            sct = ScoreCardTransformer(lr, bt, wt)
            scores = sct.transform(X_test)
        else:
            scores = pd.Series(y_prob_test, index=X_test.index)
    except Exception as e:
        scores = pd.Series(y_prob_test, index=X_test.index)
        # scores assignment fallback is enough, sct remains None if failed

    n_features = 0
    bt = pipeline.named_steps.get('binning')
    if bt is not None and hasattr(bt, 'fitted_bins_'):
        n_features = len(bt.fitted_bins_)
        n_bins = pipeline.named_steps['binning'].n_bins

    lines = []
    lines.append("# Scorecard Model Report")
    lines.append("")

    # ── Summary ──
    lines.append("## Executive Summary")
    lines.append("")
    ks_qual = "very strong" if ks > 0.7 else "strong" if ks > 0.5 else "reasonable"

    wt = pipeline.named_steps.get('woe')
    woe_method = wt.method if wt is not None and hasattr(wt, 'method') else 'adjusted'

    lines.append(f"This report evaluates a credit scorecard model built on {n_features} features using "
                 f"Logistic Regression with **{woe_method}** WOE transformation. "
                 f"The model achieves a KS statistic of "
                 f"**{ks:.1%}** and an AUC of **{auc:.3f}** (Accuracy Ratio = {accuracy_ratio:.3f}), "
                 f"indicating {ks_qual} discriminatory power between good and bad accounts.")
    lines.append("")

    ks_guide = "very strong (excellent)" if ks > 0.7 else "strong (good)" if ks > 0.5 else "moderate (acceptable)"
    lines.append(
        f"**KS = {ks:.1%}** — the maximum separation between cumulative good and bad distributions. "
        f"This is considered {ks_guide} for credit scorecards."
    )
    lines.append("")
    lines.append(
        f"**AUC = {auc:.3f}** — the probability that the model ranks a randomly chosen good account "
        f"higher than a randomly chosen bad account. An AUC of 0.5 is random; values above 0.9 are excellent."
    )
    lines.append("")

    # ── Model Performance ──
    lines.append("## Model Performance")
    lines.append("")
    lines.append(
        "The four plots below assess the model's ability to separate good from bad accounts "
        "across the entire score range."
    )
    lines.append("")

    for title, desc, figfn in [
        ("Score Distribution: Good vs Bad",
         "Overlaid density of scores for good (blue) vs bad (red) accounts. "
         "Good separation means the two distributions have minimal overlap.",
         plot_score_distribution(scores.values, y_test.values)),
        ("KS Curve",
         "Cumulative proportion of goods and bads as we move from high-risk to low-risk scores. "
         "The KS statistic is the maximum vertical distance between the two curves.",
         plot_ks(y_test, y_prob_test)),
        ("ROC Curve",
         "Trade-off between True Positive Rate (sensitivity) and False Positive Rate (1 - specificity). "
         "The diagonal line represents a random model.",
         plot_roc(y_test, y_prob_test)),
        ("Cumulative Accuracy Profile (CAP)",
         "Cumulative goods captured as a function of the population fraction, ordered by risk score. "
         "The Accuracy Ratio (AR) measures how far the model is from random toward perfect.",
         plot_cap(y_test, y_prob_test)),
    ]:
        rel = save_plot(figfn, title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(":", "") + ".png")
        lines.append(f"### {title}")
        lines.append("")
        lines.append(desc)
        lines.append("")
        lines.append(f"![{title}]({rel})")
        lines.append("")

    # ── Feature Analysis ──
    lines.append("## Feature Analysis")
    lines.append("")
    lines.append(
        "Information Value (IV) measures each feature's predictive power. "
        "Industry-standard interpretation: <0.02 useless, 0.02–0.1 weak, 0.1–0.3 medium, "
        "0.3–0.5 strong, >0.5 suspicious (investigate for data leakage)."
    )
    lines.append("")

    wt = pipeline.named_steps.get('woe')
    bt = pipeline.named_steps.get('binning')
    if wt is not None and hasattr(wt, 'iv_') and wt.iv_:
        total_iv = sum(wt.iv_.values())
        lines.append(
            f"The model uses {len(wt.iv_)} features with a total IV of **{total_iv:.2f}**. "
            f"The chart below ranks features by individual IV contribution."
        )
        lines.append("")
        fig = plot_iv_summary_enhanced(wt.iv_)
        rel = save_plot(fig, "feature_iv_ranking.png")
        lines.append("### Feature IV Ranking")
        lines.append("")
        lines.append(f"![Feature IV Ranking]({rel})")
        lines.append("")

        try:
            ranking = rank_features(X_train, y_train, n_bins=bt.n_bins if bt else 5)
            top5 = ranking.head(10)[['Feature', 'IV', 'Monotonicity', 'Trend_Advice', 'Recommendation']]
            lines.append("### Top Features by IV")
            lines.append("")
            lines.append(
                "The table below ranks the top 10 features. Note the **Trend Advice**: features with "
                "'Strong Trend (Minor Violations)' are highly predictive but have minor non-monotonic bins. "
                "In many practical cases, keeping these features provides significant performance gains "
                "compared to enforcing strict monotonicity."
            )
            lines.append("")
            lines.append(top5.to_markdown(index=False))
            lines.append("")
        except Exception:
            pass

    # ── Scorecard ──
    lines.append("## Scorecard")
    lines.append("")
    lines.append(
        "The scorecard translates model log-odds into interpretable point values. "
        "Each feature is binned, and each bin is assigned a WOE (Weight of Evidence) and a Points value. "
        "Higher points indicate lower risk (more \"good\"-like). The total score for an applicant is the "
        "sum of points across all features plus a base offset."
    )
    lines.append("")

    if sct is not None:
        try:
            card = sct.export_scorecard()
            n_rows = len(card)
            lines.append(f"The table below shows the full scorecard ({n_rows} rows across {n_features} features).")
            lines.append("")
            lines.append(card.to_markdown(index=False))
            lines.append("")
            fig = plot_scorecard_heatmap(card)
            rel = save_plot(fig, "scorecard_points_heatmap.png")
            lines.append("### Scorecard Points Heatmap")
            lines.append("")
            lines.append(
                "The heatmap provides a bird's-eye view of the scorecard. Green cells = higher points (lower risk), "
                "red cells = lower points (higher risk). Consistent color gradients within each feature indicate "
                "good monotonicity."
            )
            lines.append("")
            lines.append(f"![Scorecard Points Heatmap]({rel})")
            lines.append("")
        except Exception as e:
            lines.append(f"Scorecard export failed: {e}")
            lines.append("")
    else:
        lines.append("Scorecard could not be generated (pipeline missing necessary steps).")
        lines.append("")

    # ── Calibration & Cutoff ──
    lines.append("## Calibration & Cutoff")
    lines.append("")

    try:
        base_rate = y_test.mean()
        lines.append(
            f"The base event rate in the test set is **{base_rate:.1%}**. "
            "The plots below assess probability calibration and help select an optimal decision threshold."
        )
    except Exception:
        pass
    lines.append("")

    for title, desc, figfn in [
        ("Calibration Curve",
         "Compares predicted probabilities against observed event rates. "
         "A well-calibrated model follows the diagonal. Points above the line mean "
         "the model underestimates risk; below the line means it overestimates.",
         plot_calibration(y_test, y_prob_test)),
        ("Cutoff Optimization",
         "Shows how approval rate, bad rate, and relative profit change with the score cutoff. "
         "The optimal cutoff balances the cost of false positives (approving a bad account) "
         "against false negatives (rejecting a good account).",
         plot_cutoff_optimization(y_test, y_prob_test)),
    ]:
        rel = save_plot(figfn, title.lower().replace(" ", "_") + ".png")
        lines.append(f"### {title}")
        lines.append("")
        lines.append(desc)
        lines.append("")
        lines.append(f"![{title}]({rel})")
        lines.append("")

    with open(output_path, 'w') as f:
        f.write("\n".join(lines))

    return output_path
