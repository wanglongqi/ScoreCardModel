from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import clear_output, display

from ScoreCardModel.score_card.transformers import ScoreCardTransformer


class ScorecardWidget:
    """Interactive what-if widget for scorecard models.

    Displays sliders/dropdowns for each model feature.  Adjusting a control
    re-computes the score and redraws a waterfall chart in real time.

    Requires ``ipywidgets`` and ``ipython`` (install via
    ``pip install scorecard-toolkit[interactive]``).

    Parameters
    ----------
    pipeline : sklearn.Pipeline
        A fitted pipeline with steps ``binning``, ``woe``, ``model``.
    X_train : pd.DataFrame
        Training data used to determine slider ranges and unique values.
    """

    def __init__(self, pipeline, X_train: pd.DataFrame):
        self.pipeline = pipeline
        self.bt = pipeline.named_steps["binning"]
        self.wt = pipeline.named_steps["woe"]
        self.model = pipeline.named_steps["model"]
        self.sct = ScoreCardTransformer(self.model, self.bt, self.wt)

        features = list(self.model.feature_names_in_)
        self.features = features
        self.X_train = X_train[features]

        self.controls: dict[str, widgets.Widget] = {}
        self._build_controls()

        self.output = widgets.Output()

    def _build_controls(self) -> None:
        for feat in self.features:
            col = self.X_train[feat]
            n_unique = col.nunique()
            if col.dtype in ("object", "category") or n_unique <= 12:
                options = sorted(col.dropna().unique().tolist())
                w = widgets.Dropdown(options=options, value=options[len(options) // 2], description=feat)
            elif col.dtype in ("int64",):
                lo, hi = int(col.min()), int(col.max())
                w = widgets.IntSlider(value=int(col.median()), min=lo, max=hi, description=feat, continuous_update=False)
            else:
                lo, hi = float(col.min()), float(col.max())
                w = widgets.FloatSlider(value=float(col.median()), min=lo, max=hi, description=feat, continuous_update=False)
            w.observe(self._on_change, "value")
            self.controls[feat] = w

    def _on_change(self, change: Any) -> None:
        with self.output:
            clear_output(wait=True)
            self._render()

    def _render(self) -> None:
        row = pd.DataFrame([{f: self.controls[f].value for f in self.features}])
        score = float(self.sct.transform(row).iloc[0])

        x_bin = self.bt.transform(row)
        x_woe = self.wt.transform(x_bin)

        base = self.sct.offset_ + self.sct.factor_ * float(self.model.intercept_[0])
        contribs: dict[str, float] = {}
        for i, feat in enumerate(self.features):
            d = self.sct.factor_ * self.model.coef_[0][i] * float(x_woe[feat].iloc[0])
            contribs[feat] = round(d, 2)

        display(widgets.HTML(f"<h2 style='margin:0 0 8px 0'>Score: <span style='color:#2c3e50'>{score:.0f}</span></h2>"))
        display(widgets.HTML(f"<p style='margin:0 0 12px 0;color:#666'>Base = {base:.0f} &nbsp;|&nbsp; Net adjustment = <b>{score - base:+.0f}</b></p>"))

        fig, ax = plt.subplots(figsize=(10, 4))
        labels = ["Base"] + list(contribs.keys())
        vals = [base] + list(contribs.values())
        running = np.cumsum(vals)

        for i in range(len(labels)):
            if i == 0:
                ax.bar(i, vals[i], color="gray", alpha=0.7)
            else:
                c = "#2ecc71" if vals[i] >= 0 else "#e74c3c"
                ax.bar(i, vals[i], bottom=running[i - 1], color=c, alpha=0.7)
            y_pos = running[i] + abs(max(vals)) * 0.02 if vals[i] >= 0 else running[i] - abs(max(vals)) * 0.06
            ax.text(i, y_pos, f"{vals[i]:+.0f}", ha="center", fontsize=9, fontweight="bold")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Points")
        ax.set_title("What-If Waterfall")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        display(fig)
        plt.close(fig)

    def display(self) -> widgets.HBox:
        """Render the widget.  Call this as the last expression in a notebook cell."""
        controls_box = widgets.VBox(list(self.controls.values()), layout=widgets.Layout(width="380px"))
        ui = widgets.HBox([controls_box, self.output])
        with self.output:
            self._render()
        return ui
