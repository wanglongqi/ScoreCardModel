import os
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

    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y_test, y_prob_test))

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
    except Exception:
        scores = pd.Series(y_prob_test, index=X_test.index)

    n_features = 0
    bt = pipeline.named_steps.get('binning')
    if bt is not None and hasattr(bt, 'fitted_bins_'):
        n_features = len(bt.fitted_bins_)

    lines = []
    lines.append("# Scorecard Model Report")
    lines.append("")
    lines.append("## Summary Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| KS Statistic | {ks:.3f} |")
    lines.append(f"| AUC | {auc:.3f} |")
    lines.append(f"| Features | {n_features} |")
    lines.append("")

    # Section 1: Model Performance
    lines.append("## Model Performance")
    lines.append("")
    for title, figfn in [
        ("Score Distribution: Good vs Bad", plot_score_distribution(scores.values, y_test.values)),
        ("KS Curve", plot_ks(y_test, y_prob_test)),
        ("ROC Curve", plot_roc(y_test, y_prob_test)),
        ("Cumulative Accuracy Profile (CAP)", plot_cap(y_test, y_prob_test)),
    ]:
        rel = save_plot(figfn, title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(":", "") + ".png")
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"![{title}]({rel})")
        lines.append("")

    # Section 2: Feature Analysis
    lines.append("## Feature Analysis")
    lines.append("")
    wt = pipeline.named_steps.get('woe')
    if wt is not None and hasattr(wt, 'iv_') and wt.iv_:
        fig = plot_iv_summary_enhanced(wt.iv_)
        rel = save_plot(fig, "feature_iv_ranking.png")
        lines.append("### Feature IV Ranking")
        lines.append("")
        lines.append(f"![Feature IV Ranking]({rel})")
        lines.append("")

    # Section 3: Scorecard
    lines.append("## Scorecard")
    lines.append("")
    try:
        card = sct.export_scorecard()
        lines.append(card.to_markdown(index=False))
        lines.append("")
        fig = plot_scorecard_heatmap(card)
        rel = save_plot(fig, "scorecard_points_heatmap.png")
        lines.append("### Scorecard Points Heatmap")
        lines.append("")
        lines.append(f"![Scorecard Points Heatmap]({rel})")
        lines.append("")
    except Exception:
        lines.append("Scorecard export failed.")
        lines.append("")

    # Section 4: Calibration & Cutoff
    lines.append("## Calibration & Cutoff")
    lines.append("")
    for title, figfn in [
        ("Calibration Curve", plot_calibration(y_test, y_prob_test)),
        ("Cutoff Optimization", plot_cutoff_optimization(y_test, y_prob_test)),
    ]:
        rel = save_plot(figfn, title.lower().replace(" ", "_") + ".png")
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"![{title}]({rel})")
        lines.append("")

    with open(output_path, 'w') as f:
        f.write("\n".join(lines))

    return output_path
