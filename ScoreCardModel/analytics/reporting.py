import matplotlib
import pandas as pd
from sklearn.pipeline import Pipeline

matplotlib.use('Agg')
import base64
from io import BytesIO

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


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Scorecard Model Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #34495e; margin-top: 30px; }}
.section {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0;
           box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
.card {{ margin: 15px 0; text-align: center; }}
.card img {{ max-width: 100%; height: auto; }}
.caption {{ color: #7f8c8d; font-size: 0.9em; margin-top: 5px; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #3498db; color: white; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.summary {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }}
.metric {{ background: #ecf0f1; padding: 15px; border-radius: 8px;
          flex: 1; min-width: 150px; text-align: center; }}
.metric .value {{ font-size: 1.5em; font-weight: bold; color: #2c3e50; }}
.metric .label {{ font-size: 0.8em; color: #7f8c8d; }}
</style>
</head>
<body>
<h1>Scorecard Model Report</h1>
<div class="summary">
<div class="metric"><div class="value">{ks:.3f}</div><div class="label">KS Statistic</div></div>
<div class="metric"><div class="value">{auc:.3f}</div><div class="label">AUC</div></div>
<div class="metric"><div class="value">{n_features}</div><div class="label">Features</div></div>
</div>
{body}
</body>
</html>"""


def generate_report(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str = "scorecard_report.html",
) -> str:
    """Generate a comprehensive HTML report for a fitted scorecard pipeline."""
    sections = []

    y_prob_test = pipeline.predict_proba(X_test)[:, 1]
    ks = calculate_ks(y_test.values if hasattr(y_test, 'values') else y_test,
                       y_prob_test)

    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y_test, y_prob_test))

    # Compute scores via ScoreCardTransformer
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

    # Page 1: Model Performance
    perf_html = ""
    perf_html += _img_card(_fig_to_base64(plot_score_distribution(scores.values, y_test.values)),
                            "Score Distribution: Good vs Bad")
    perf_html += _img_card(_fig_to_base64(plot_ks(y_test, y_prob_test)), "KS Curve")
    perf_html += _img_card(_fig_to_base64(plot_roc(y_test, y_prob_test)), "ROC Curve")
    perf_html += _img_card(_fig_to_base64(plot_cap(y_test, y_prob_test)),
                            "Cumulative Accuracy Profile")
    sections.append(_make_section("1. Model Performance", perf_html))

    # Page 2: Feature Analysis
    feat_html = ""
    wt = pipeline.named_steps.get('woe')
    if wt is not None and hasattr(wt, 'iv_') and wt.iv_:
        feat_html += _img_card(_fig_to_base64(plot_iv_summary_enhanced(wt.iv_)),
                                "Feature IV Ranking")
    sections.append(_make_section("2. Feature Analysis", feat_html))

    # Page 3: Scorecard
    card_html = ""
    try:
        card = sct.export_scorecard()
        card_html += f'<div style="overflow-x:auto;">{card.to_html(index=False)}</div>'
        card_html += _img_card(_fig_to_base64(plot_scorecard_heatmap(card)),
                                "Scorecard Points Heatmap")
    except Exception:
        card_html = "<p>Scorecard export failed.</p>"
    sections.append(_make_section("3. Scorecard", card_html))

    # Page 4: Calibration & Cutoff
    cal_html = _img_card(_fig_to_base64(plot_calibration(y_test, y_prob_test)),
                          "Calibration Curve")
    cal_html += _img_card(_fig_to_base64(plot_cutoff_optimization(y_test, y_prob_test)),
                           "Cutoff Optimization")
    sections.append(_make_section("4. Calibration & Cutoff", cal_html))

    n_features = 0
    bt = pipeline.named_steps.get('binning')
    if bt is not None and hasattr(bt, 'fitted_bins_'):
        n_features = len(bt.fitted_bins_)

    body = "\n".join(sections)
    html = HTML_TEMPLATE.format(ks=ks, auc=auc, n_features=n_features, body=body)

    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


def _make_section(title: str, content: str) -> str:
    return f'<div class="section"><h2>{title}</h2>{content}</div>'


def _img_card(img_b64: str, caption: str = "") -> str:
    img_tag = f'<img src="data:image/png;base64,{img_b64}">'
    cap = f'<p class="caption">{caption}</p>' if caption else ''
    return f'<div class="card">{img_tag}{cap}</div>'
