# ScoreCardModel

<p align="center">
  <img src="images/scorecard_heatmap.png" width="800" alt="ScoreCardModel Banner">
</p>

<p align="center">
    <a href="https://pypi.org/project/scorecard-toolkit/"><img src="https://img.shields.io/pypi/v/scorecard-toolkit?color=blue" alt="PyPI version"></a>
    <a href="https://scorecardmodel.readthedocs.io/"><img src="https://img.shields.io/readthedocs/scorecardmodel?color=green" alt="Documentation Status"></a>
    <a href="https://github.com/wanglongqi/ScoreCardModel/blob/master/LICENSE"><img src="https://img.shields.io/github/license/wanglongqi/ScoreCardModel?color=yellow" alt="License"></a>
    <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
</p>

ScoreCardModel is a professional and modern toolset for scorecard modeling, fully compatible with **scikit-learn**. It is designed for credit risk analysts and data scientists who need to build transparent, regulator-friendly scoring models with ease.

## Key Features

- 🛠 **Scikit-Learn Compatible**: Use `BinningTransformer`, `WOETransformer`, and `ScoreCardTransformer` directly in your pipelines.
- 📊 **Rich Analytics**: 18+ plot types (KS, ROC, CAP, Lift, Calibration, PSI, etc.) for comprehensive model evaluation.
- 📝 **Automated Reporting**: Generate professional Markdown or Excel reports with one function call.
- 🔄 **5 WOE Methods**: Choose between standard, adjusted, empirical logit, signed, and weighted Weight of Evidence.
- 🎮 **Interactive Dashboard**: A Jupyter-based what-if widget for real-time scorecard testing.
- 🏢 **Industry Standard**: Built-in support for PDO (Points to Double Odds) and Base-Odds scaling.

## Documentation Sections

### 🚀 Getting Started
- [Installation](installation.md) — Install via pip or uv.
- [Quickstart](quickstart.md) — Build your first scorecard in minutes.

### 📖 User Guides
- [WOE In-Depth Guide](woe_in_depth.md) — Deep dive into WOE methods, diagnostics, and IV interpretation.
- [Best Practices](best_practices.md) — Guidelines for developing robust credit risk scorecards.

### 🛠 API Reference
- [Full API Reference](api.md) — Detailed documentation for all modules and classes.

### 🧪 Examples
- [Full Examples](examples.md) — End-to-end scorecard development with real-world datasets.
- [Interactive Notebook](examples/interactive_scorecard.ipynb) — Live what-if widget in action.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/wanglongqi/ScoreCardModel/blob/master/LICENSE) file for details.
