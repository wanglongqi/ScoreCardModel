from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, confusion_matrix, roc_curve

from ScoreCardModel.analytics.metrics import calculate_psi
from ScoreCardModel.weight_of_evidence.diagnostics import check_monotonicity


def plot_ks(y_true: np.ndarray, y_prob: np.ndarray, title: str = "KS Curve") -> plt.Figure:
    """Plot the KS curve: cumulative goods vs cumulative bads across population percentiles."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks_idx = np.argmax(tpr - fpr)
    ks_stat = tpr[ks_idx] - fpr[ks_idx]

    x = np.linspace(0, 1, len(tpr))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, tpr, label='Good (Cumulative)', color='blue', lw=2)
    ax.plot(x, fpr, label='Bad (Cumulative)', color='red', lw=2)
    diff = tpr - fpr
    ax.plot(x, diff, label='KS Difference', color='green', lw=2, linestyle='--')
    ax.axvline(x[ks_idx], color='black', linestyle=':', alpha=0.5)
    ax.annotate(f'KS = {ks_stat:.3f}', xy=(x[ks_idx], ks_stat),
                fontsize=12, fontweight='bold',
                xytext=(x[ks_idx] + 0.05, ks_stat + 0.05),
                arrowprops={"arrowstyle": '->', "color": 'black'})

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Population Percentile')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray) -> plt.Figure:
    """Plot a professional ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.fill_between(fpr, tpr, alpha=0.1, color='darkorange')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    fig.tight_layout()
    return fig


def plot_cap(y_true: np.ndarray, y_prob: np.ndarray) -> plt.Figure:
    """Cumulative Accuracy Profile (CAP) / Lorenz curve with Accuracy Ratio."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    order = np.argsort(y_prob)[::-1]
    y_sorted = y_true[order]
    cum_goods = np.cumsum(y_sorted) / y_sorted.sum() if y_sorted.sum() > 0 else np.arange(n) / n
    pop_pct = np.arange(1, n + 1) / n

    total_goods = y_sorted.sum()
    perfect = np.minimum(pop_pct / (total_goods / n), 1.0)
    ar = (auc(pop_pct, cum_goods) - 0.5) / (auc(pop_pct, perfect) - 0.5) if total_goods > 0 else 0.0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(pop_pct, cum_goods, lw=2, label=f'Model (AR = {ar:.3f})')
    ax.plot(pop_pct, perfect, lw=2, linestyle='--', label='Perfect Model')
    ax.plot([0, 1], [0, 1], lw=2, linestyle=':', label='Random')
    ax.fill_between(pop_pct, cum_goods, alpha=0.05)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Population Proportion')
    ax.set_ylabel('Cumulative Goods')
    ax.set_title('Cumulative Accuracy Profile')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_gain_lift(y_true: np.ndarray, y_prob: np.ndarray) -> plt.Figure:
    """Gain and Lift chart (decile-based)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    order = np.argsort(y_prob)[::-1]
    y_sorted = y_true[order]
    base_rate = y_sorted.mean()
    decile_size = max(n // 10, 1)

    gains = []
    for i in range(10):
        start = i * decile_size
        end = (i + 1) * decile_size if i < 9 else n
        gains.append(y_sorted[start:end].sum() / y_sorted.sum() if y_sorted.sum() > 0 else 0)

    cum_gain = np.cumsum(gains) * 100
    lift = np.array([g / (base_rate / 10) if base_rate > 0 else 1.0 for g in gains])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(range(1, 11), cum_gain, alpha=0.3, color='blue', label='Cumulative Gain %')
    ax1.set_ylabel('Cumulative Gain %')
    ax1.set_ylim(0, 105)
    ax1.set_xlabel('Decile')

    ax2 = ax1.twinx()
    ax2.plot(range(1, 11), lift, 'ro-', lw=2, label='Lift')
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Lift')

    ax1.set_title('Gain / Lift Chart')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    fig.tight_layout()
    return fig


def plot_double_lift(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    labels: tuple[str, str] = ('Population A', 'Population B'),
) -> plt.Figure:
    """Double-lift chart comparing model performance across two populations."""
    y_true = np.asarray(y_true)
    y_prob_a = np.asarray(y_prob_a)
    y_prob_b = np.asarray(y_prob_b)

    def _lift(y_true_: np.ndarray, y_prob_: np.ndarray) -> list[float]:
        order = np.argsort(y_prob_)[::-1]
        y_sorted = y_true_[order]
        base_rate = y_sorted.mean()
        decile_size = max(len(y_true_) // 10, 1)
        lifts = []
        for i in range(10):
            start = i * decile_size
            end = (i + 1) * decile_size if i < 9 else len(y_true_)
            rate = y_sorted[start:end].mean()
            lifts.append(rate / base_rate if base_rate > 0 else 1.0)
        return lifts

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, 11), _lift(y_true, y_prob_a), 'b-o', label=labels[0])
    ax.plot(range(1, 11), _lift(y_true, y_prob_b), 'r-s', label=labels[1])
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Decile')
    ax.set_ylabel('Lift')
    ax.set_title('Double-Lift Chart')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_score_distribution(scores: np.ndarray, y_true: np.ndarray,
                            title: str = "Score Distribution by Population") -> plt.Figure:
    """Overlaid score distribution for good vs bad populations."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    fig, ax = plt.subplots(figsize=(10, 6))
    good_scores = scores[y_true == 1]
    bad_scores = scores[y_true == 0]
    ax.hist(good_scores, bins=50, alpha=0.5, color='blue',
            label=f'Good (n={len(good_scores)})', density=True)
    ax.hist(bad_scores, bins=50, alpha=0.5, color='red',
            label=f'Bad (n={len(bad_scores)})', density=True)
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray,
                     n_bins: int = 10) -> plt.Figure:
    """Calibration curve comparing predicted probabilities to observed event rates."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(prob_pred, prob_true, 'bo-', lw=2, label='Model')
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect Calibration')
    ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.1, color='blue')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Observed Event Rate')
    ax.set_title('Calibration Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_psi_drift(reference_scores: np.ndarray, current_scores: np.ndarray,
                   n_bins: int = 10, feature_name: str = "Score") -> plt.Figure:
    """PSI drift monitoring plot with bin-level contribution."""
    psi = calculate_psi(reference_scores, current_scores, n_bins)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8),
                             gridspec_kw={'height_ratios': [3, 1]})
    axes[0].hist(reference_scores, bins=n_bins, alpha=0.5, color='blue',
                 label='Reference', density=True)
    axes[0].hist(current_scores, bins=n_bins, alpha=0.5, color='red',
                 label='Current', density=True)
    axes[0].set_title(f'Score Distribution Drift (PSI = {psi:.4f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    min_v = min(reference_scores.min(), current_scores.min())
    max_v = max(reference_scores.max(), current_scores.max())
    bins = np.linspace(min_v, max_v, n_bins + 1)
    ref_hist, _ = np.histogram(reference_scores, bins=bins)
    cur_hist, _ = np.histogram(current_scores, bins=bins)
    ref_pct = ref_hist / len(reference_scores)
    cur_pct = cur_hist / len(current_scores)
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)
    contributions = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    colors = ['red' if c > 0 else 'green' for c in contributions]
    axes[1].bar(range(n_bins), contributions, color=colors, alpha=0.7)
    axes[1].set_ylabel('PSI Contribution')
    axes[1].set_xlabel('Bin')
    axes[1].axhline(0, color='black', lw=0.5)
    axes[1].set_title('PSI Contribution by Bin')

    fig.tight_layout()
    return fig


def plot_woe_pattern(woe_map: dict[str, float], ordered_bins: list[str],
                     counts: Optional[list[int]] = None,
                     feature_name: str = "", iv: float = 0.0) -> plt.Figure:
    """WOE characteristic plot with monotonicity annotation and population %."""
    valid_bins = [b for b in ordered_bins if b in woe_map]
    woe_values = [woe_map[b] for b in valid_bins]

    mono, strength = check_monotonicity(woe_map, ordered_bins)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    colors = ['#2ecc71' if w >= 0 else '#e74c3c' for w in woe_values]
    bars = ax1.bar(range(len(valid_bins)), woe_values, color=colors, alpha=0.7)
    ax1.axhline(0, color='black', lw=1)
    ax1.set_ylabel('WOE')
    ax1.set_xticks(range(len(valid_bins)))
    ax1.set_xticklabels(valid_bins, rotation=45, ha='right')

    if counts:
        total = sum(counts)
        for _i, (bar, cnt) in enumerate(zip(bars, counts)):
            pct = cnt / total * 100
            y_pos = bar.get_height() + (0.02 if bar.get_height() >= 0 else -0.08)
            ax1.text(bar.get_x() + bar.get_width() / 2, y_pos,
                     f'{pct:.0f}%', ha='center', fontsize=9)

    title = feature_name if feature_name else 'WOE Pattern'
    if iv > 0:
        title += f' (IV={iv:.3f})'
    title += f' [{mono}]'
    ax1.set_title(title)
    fig.tight_layout()
    return fig


def plot_iv_summary_enhanced(iv_dict: dict[str, float]) -> plt.Figure:
    """Enhanced IV ranking bar chart with color-coded predictive power categories."""
    if not iv_dict:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, 'No IV data available', ha='center', va='center')
        return fig

    items = sorted(iv_dict.items(), key=lambda x: x[1])
    features, ivs = zip(*items)

    colors = []
    for iv in ivs:
        if iv < 0.02:
            colors.append('gray')
        elif iv < 0.1:
            colors.append('orange')
        elif iv < 0.3:
            colors.append('green')
        elif iv < 0.5:
            colors.append('blue')
        else:
            colors.append('red')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(features)), ivs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Information Value (IV)')
    ax.set_title('Feature Predictive Power (IV Ranking)')
    for i, v in enumerate(ivs):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

    ax.axvline(0.02, color='gray', linestyle=':', alpha=0.5, label='0.02 (useless)')
    ax.axvline(0.1, color='orange', linestyle=':', alpha=0.5, label='0.1 (weak)')
    ax.axvline(0.3, color='green', linestyle=':', alpha=0.5, label='0.3 (medium)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_event_rate_by_bin(df: pd.DataFrame, bin_col: str,
                           event_col: str = 'target') -> plt.Figure:
    """Event rate and population distribution by bin (dual-axis)."""
    stats = df.groupby(bin_col)[event_col].agg(['count', 'mean']).reset_index()
    stats.columns = [bin_col, 'Count', 'Event Rate']

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(stats[bin_col], stats['Count'], alpha=0.3, color='gray', label='Population')
    ax1.set_ylabel('Population Count')
    ax1.tick_params(axis='x', rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(stats[bin_col], stats['Event Rate'], 'bo-', lw=2, label='Event Rate')
    ax2.set_ylabel('Event Rate')
    ax2.set_ylim(0, 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.set_title(f'Event Rate by {bin_col}')
    fig.tight_layout()
    return fig


def plot_bin_stats(df: pd.DataFrame, feature: str, target: str) -> plt.Figure:
    """Legacy wrapper: plot bin-level event rate and population distribution."""
    return plot_event_rate_by_bin(df, bin_col=feature, event_col=target)


def plot_iv_summary(iv_dict: dict) -> plt.Figure:
    """Legacy wrapper: plot IV summary."""
    return plot_iv_summary_enhanced(iv_dict)


def plot_scorecard_waterfall(contributions: dict[str, float],
                             base_points: float = 600,
                             final_score: Optional[float] = None) -> plt.Figure:
    """Waterfall chart showing how each feature contributes to the final score."""
    labels = ['Base Points'] + list(contributions.keys())
    values = [base_points] + list(contributions.values())

    fig, ax = plt.subplots(figsize=(12, 6))
    colors_list = ['gray'] + ['#2ecc71' if v >= 0 else '#e74c3c' for v in values[1:]]

    running = np.cumsum(values)
    for i in range(len(labels)):
        if i == 0:
            ax.bar(i, values[i], color='gray', alpha=0.7)
            values[i]
        else:
            ax.bar(i, values[i], bottom=running[i - 1],
                   color=colors_list[i], alpha=0.7)
            ax.plot([i - 1, i], [running[i - 1], running[i - 1]], 'k-', lw=0.5)

        ax.text(i, running[i] + abs(max(values)) * 0.02,
                f'{values[i]:+.0f}', ha='center', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Points')
    ax.set_title('Scorecard Points Contribution')
    ax.axhline(base_points, color='gray', linestyle=':', alpha=0.5)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    return fig


def plot_scorecard_heatmap(scorecard_df: pd.DataFrame) -> plt.Figure:
    """Heatmap of scorecard points by feature and bin."""
    if scorecard_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'Empty scorecard', ha='center', va='center')
        return fig

    pivot = scorecard_df.pivot_table(
        index='Variable', columns='Bin', values='Points', aggfunc='first'
    )

    n_rows, n_cols = pivot.shape
    fig, ax = plt.subplots(
        figsize=(max(8, n_cols * 1.5), max(4, n_rows * 0.6)))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot.index)

    for i in range(n_rows):
        for j in range(n_cols):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = 'black' if abs(val - np.nanmean(pivot.values)) < np.nanstd(pivot.values) * 0.5 else 'white'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                        fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label='Points')
    ax.set_title('Scorecard Points Heatmap')
    fig.tight_layout()
    return fig


def plot_cutoff_optimization(y_true: np.ndarray, y_prob: np.ndarray,
                             cost_fp: float = 1.0, cost_fn: float = 1.0) -> plt.Figure:
    """Cutoff optimization: approval rate, bad rate, and profit vs threshold."""
    thresholds = np.linspace(0, 1, 100)
    approval_rates = []
    bad_rates = []
    profits = []

    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        ((pred == 0) & (y_true == 0)).sum()
        approval_rates.append((tp + fp) / len(y_true))
        bad_rates.append(fp / (tp + fp) if (tp + fp) > 0 else 0)
        profits.append(tp * 1.0 - fp * cost_fp - fn * cost_fn)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(thresholds, approval_rates, 'b-', lw=2, label='Approval Rate')
    ax1.plot(thresholds, bad_rates, 'r-', lw=2, label='Bad Rate')
    ax1.set_xlabel('Score Threshold')
    ax1.set_ylabel('Rate')
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, profits, 'g--', lw=2, label='Profit (scaled)')
    ax2.set_ylabel('Relative Profit')
    ax2.legend(loc='upper right')

    fig.suptitle('Cutoff Optimization')
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray) -> plt.Figure:
    """Confusion matrix heatmap with counts and percentages."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum() * 100 if cm.sum() > 0 else cm
    labels = [['TN (Bad Correct)', 'FP (Bad Wrong)'],
              ['FN (Good Wrong)', 'TP (Good Correct)']]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{labels[i][j]}\n{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)',
                    ha='center', va='center', fontsize=11)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Predicted Bad (0)', 'Predicted Good (1)'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Actual Bad (0)', 'Actual Good (1)'])
    plt.colorbar(im, ax=ax)
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    return fig
