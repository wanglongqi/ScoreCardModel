# Installation

## Prerequisites

- Python 3.8+
- pip or uv

## Install from source

```bash
git clone https://github.com/wanglongqi/ScoreCardModel.git
cd ScoreCardModel
pip install .
```

## Install with uv

```bash
git clone https://github.com/wanglongqi/ScoreCardModel.git
cd ScoreCardModel
uv pip install .
```

## Development install

```bash
pip install -e ".[dev,test]"
```

## Optional Extras

| Extra | Packages | Command |
|---|---|---|
| `interactive` | ipywidgets, ipython | `pip install scorecard-toolkit[interactive]` |
| `test` | pytest, pytest-cov | `pip install scorecard-toolkit[test]` |
| `dev` | ruff, mypy | `pip install scorecard-toolkit[dev]` |

## Dependencies

- numpy, scipy, pandas, scikit-learn
- optbinning (for optimal binning strategy)
- matplotlib, seaborn (for plotting)
