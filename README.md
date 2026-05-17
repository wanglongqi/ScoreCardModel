# ScoreCardModel

<p align="center">
  <img src="docs/images/scorecard_heatmap.png" width="800" alt="ScoreCardModel Banner">
</p>

<p align="center">
    <a href="https://pypi.org/project/scorecard-toolkit/"><img src="https://img.shields.io/pypi/v/scorecard-toolkit?color=blue" alt="PyPI version"></a>
    <a href="https://scorecardmodel.readthedocs.io/"><img src="https://img.shields.io/readthedocs/scorecardmodel?color=green" alt="Documentation Status"></a>
    <a href="https://github.com/wanglongqi/ScoreCardModel/blob/master/LICENSE"><img src="https://img.shields.io/github/license/wanglongqi/ScoreCardModel?color=yellow" alt="License"></a>
    <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
</p>

**ScoreCardModel** is a professional and modern toolset for scorecard modeling, fully compatible with **scikit-learn**. It is designed for credit risk analysts and data scientists who need to build transparent, regulator-friendly scoring models with ease.

## 🚀 Key Features

- 🛠 **Scikit-Learn Compatible**: `BinningTransformer`, `WOETransformer`, and `ScoreCardTransformer` work in any `Pipeline` or `GridSearchCV`.
- 📊 **Rich Analytics**: 18+ plot types (KS, ROC, CAP, Lift, Calibration, PSI, etc.) for comprehensive model evaluation.
- 📝 **Automated Reporting**: Generate professional Markdown or Excel reports with one function call.
- 🔄 **5 WOE Methods**: Standard, Adjusted (Laplace), Empirical Logit, Signed, and Weighted Weight of Evidence.
- 🎮 **Interactive Dashboard**: A Jupyter-based what-if widget for real-time scorecard testing.
- 🏢 **Industry Standard**: Built-in support for PDO (Points to Double Odds) and Base-Odds scaling.

## 📸 Visual Gallery

| KS Curve | ROC Curve | CAP Curve |
|---|---|---|
| ![KS](docs/images/ks_curve.png) | ![ROC](docs/images/roc_curve.png) | ![CAP](docs/images/cap_curve.png) |

| Score Distribution | Scorecard Waterfall | IV Summary |
|---|---|---|
| ![Score Dist](docs/images/score_distribution.png) | ![Waterfall](docs/images/scorecard_waterfall.png) | ![IV](docs/images/iv_summary.png) |

[See all 12+ visualizations →](https://scorecardmodel.readthedocs.io/en/latest/examples/#visualizations)

## 📦 Installation

```bash
pip install scorecard-toolkit
```

## ⚡ Quick Start

```python
from ScoreCardModel import ScoreCardWrapper

# Initialize and fit
sc = ScoreCardWrapper(binning_strategy='quantile', base_points=600, pdo=20)
sc.fit(X_train, y_train)

# Predict scores and export scorecard
scores = sc.predict(X_test)
card = sc.export_scorecard()
print(card.head(10))
```

## 📚 Documentation

Visit [scorecardmodel.readthedocs.io](https://scorecardmodel.readthedocs.io/) for the full documentation, including:

- [Installation Guide](https://scorecardmodel.readthedocs.io/en/latest/installation/)
- [Quickstart Guide](https://scorecardmodel.readthedocs.io/en/latest/quickstart/)
- [Full Examples & Visualizations](https://scorecardmodel.readthedocs.io/en/latest/examples/)
- [WOE In-Depth Guide](https://scorecardmodel.readthedocs.io/en/latest/woe_in_depth/)
- [API Reference](https://scorecardmodel.readthedocs.io/en/latest/api/)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
