# ScoreCardModel Documentation

ScoreCardModel is a professional toolset for scorecard modeling, fully compatible with scikit-learn. Designed for credit risk analysts and data scientists building transparent, regulator-friendly scoring models.

## Getting Started

- [Installation](installation.md) — Install via pip or uv
- [Quickstart](quickstart.md) — Build your first scorecard in minutes

## User Guides

- [WOE In-Depth Guide](woe_in_depth.md) — WOE methods, diagnostics, IV interpretation
- [Best Practices](best_practices.md) — Credit risk scorecard development guidelines

## API Reference

| Module | Description |
|---|---|
| `ScoreCardModel.binning` | Binning strategies (quantile, uniform, optimal, tree) |
| `ScoreCardModel.weight_of_evidence` | WOE calculation (5 methods), diagnostics, IV |
| `ScoreCardModel.score_card` | Score mapping, PDO/Base-Odds scaling |
| `ScoreCardModel.analytics` | Metrics, 18+ plot types, automated Markdown reports, feature selection |
| `ScoreCardModel.interactive` | Jupyter what-if widget, rich scorecard display (`[interactive]` extra) |
| `ScoreCardModel.templates` | Pre-built scorecard configurations |

## Examples & Visualizations

- [Full Examples](examples.md) — End-to-end scorecard development with real output
- [Interactive Notebook](examples/interactive_scorecard.ipynb) — Live what-if widget in Jupyter
- [Visualizations](examples.md#visualizations) — KS, ROC, CAP, Gain/Lift, WOE patterns, scorecard heatmaps, and more
