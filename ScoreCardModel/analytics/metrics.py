import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def calculate_ks(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate the Kolmogorov-Smirnov statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks = max(tpr - fpr)
    return float(ks)

def calculate_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Calculate the Population Stability Index (PSI)."""
    def scale_data(data: np.ndarray, bins: np.ndarray) -> np.ndarray:
        return np.histogram(data, bins=bins)[0] / len(data)

    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    bins = np.linspace(min_val, max_val, n_bins + 1)

    expected_per = scale_data(expected, bins)
    actual_per = scale_data(actual, bins)

    # Avoid zero division
    expected_per = np.where(expected_per == 0, 0.0001, expected_per)
    actual_per = np.where(actual_per == 0, 0.0001, actual_per)

    psi_val = np.sum((actual_per - expected_per) * np.log(actual_per / expected_per))
    return float(psi_val)

def calculate_accuracy_ratio(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate the Accuracy Ratio (AR) = 2 * AUC - 1."""
    auc = roc_auc_score(y_true, y_prob)
    return float(2 * auc - 1)
