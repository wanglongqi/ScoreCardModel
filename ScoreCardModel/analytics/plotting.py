import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import auc, roc_curve

def plot_ks(y_true: np.ndarray, y_prob: np.ndarray, title: str = "KS Curve"):
    """Plot the KS curve with professional styling."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks_idx = np.argmax(tpr - fpr)
    ks_stat = tpr[ks_idx] - fpr[ks_idx]
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    plt.plot(thresholds, tpr, label='True Positive Rate (Good)', color='blue')
    plt.plot(thresholds, fpr, label='False Positive Rate (Bad)', color='red')
    plt.plot(thresholds, tpr - fpr, label='KS Curve', color='green', linestyle='--')
    
    # Highlight KS
    plt.axvline(thresholds[ks_idx], color='black', linestyle=':')
    plt.text(thresholds[ks_idx], ks_stat, f'KS={ks_stat:.3f}', fontsize=12)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Threshold / Probability')
    plt.ylabel('Rate')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_roc(y_true: np.ndarray, y_prob: np.ndarray):
    """Plot a professional ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plot_bin_stats(df: pd.DataFrame, feature: str, target: str):
    """Plot bin-level event rate and population distribution."""
    # This assumes df is already binned
    stats = df.groupby(feature)[target].agg(['count', 'mean']).reset_index()
    stats.columns = [feature, 'Count', 'Event Rate']
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    sns.barplot(data=stats, x=feature, y='Count', ax=ax1, alpha=0.3, color='grey')
    ax1.set_ylabel('Population Count')
    
    ax2 = ax1.twinx()
    sns.pointplot(data=stats, x=feature, y='Event Rate', ax=ax2, color='blue', markers='o')
    ax2.set_ylabel('Event Rate (Good %)')
    
    plt.title(f'Bin Analysis: {feature}')
    plt.show()

def plot_iv_summary(iv_dict: dict):
    """Plot Information Value summary for all features."""
    iv_df = pd.DataFrame(list(iv_dict.items()), columns=['Feature', 'IV'])
    iv_df = iv_df.sort_values(by='IV', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=iv_df, x='IV', y='Feature', palette='viridis')
    plt.title('Feature Predictive Power (Information Value)')
    plt.xlabel('Information Value')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

def plot_score_distribution(scores: pd.Series, y_true: np.ndarray):
    """Plot the distribution of scores for good vs bad populations."""
    plt.figure(figsize=(10, 6))
    sns.histplot(x=scores, hue=y_true, bins=50, kde=True, element="step", palette="coolwarm")
    plt.title('Score Distribution by Population (Good vs Bad)')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend(title='Population', labels=['Bad (0)', 'Good (1)'])
    plt.show()
