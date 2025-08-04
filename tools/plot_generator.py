import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def save_plot(fig, path):
    """Save the figure to the specified path and close it."""
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_confusion_matrix_plot(
    y_true, y_pred, classes=None, title="Confusion Matrix", path=None
):
    """Generate and save a confusion matrix heatmap."""
    if classes is None:
        classes = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )
    ax.set_title(title)
    if path:
        save_plot(fig, path)


def generate_roc_curve_plot(
    y_true_bin, y_prob, title="ROC Curve", path=None
):
    """Generate and save an ROC curve plot. Assumes y_true_bin is binary 0/1."""
    fpr, tpr, _ = roc_curve(y_true_bin, y_prob)
    roc_auc = roc_auc_score(y_true_bin, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    if path:
        save_plot(fig, path)


def generate_pr_curve_plot(
    y_true_bin, y_prob, title="Precision-Recall Curve", path=None
):
    """Generate and save a Precision-Recall curve plot. Assumes y_true_bin is binary 0/1."""
    precision, recall, _ = precision_recall_curve(y_true_bin, y_prob)
    ap = average_precision_score(y_true_bin, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.step(recall, precision, where="post", label=f"AP = {ap:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    if path:
        save_plot(fig, path)


def generate_calibration_plot(
    y_true_bin, y_prob, n_bins=10, title="Calibration Plot", path=None
):
    """Generate and save a calibration plot. Assumes y_true_bin is binary 0/1."""
    prob_true, prob_pred = calibration_curve(
        y_true_bin, y_prob, n_bins=n_bins, strategy="uniform"
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Probability")
    ax.set_title(title)
    ax.legend()
    if path:
        save_plot(fig, path)


def generate_per_class_metrics_plot(
    y_true,
    y_pred,
    metrics=["precision", "recall", "f1-score"],
    title="Per-Class Metrics",
    path=None,
):
    """Generate and save a bar plot of per-class metrics."""
    report = classification_report(y_true, y_pred, output_dict=True)
    classes = [
        cls
        for cls in report
        if cls not in ["accuracy", "macro avg", "micro avg", "weighted avg"]
    ]
    df = pd.DataFrame(report).T.loc[classes, metrics]
    fig, ax = plt.subplots(figsize=(8, 5))
    df.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    if path:
        save_plot(fig, path)


def generate_scatter_plot(
    y_true, y_pred, title="Predicted vs Actual", path=None
):
    """Generate and save a scatter plot of predicted vs actual values."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    if path:
        save_plot(fig, path)


def generate_residual_plot(
    y_true, y_pred, title="Residual Plot", path=None
):
    """Generate and save a residual plot."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title(title)
    if path:
        save_plot(fig, path)


def generate_residual_histogram(
    y_true, y_pred, bins=30, title="Residual Histogram", path=None
):
    """Generate and save a histogram of residuals."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(residuals, bins=bins)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    if path:
        save_plot(fig, path)


def generate_metric_comparison_bar(
    metrics_scores,
    phases=["train", "val", "test"],
    title="Metric Comparison Across Phases",
    path=None,
):
    """Generate and save a bar plot comparing metrics across phases."""
    df = pd.DataFrame(metrics_scores, index=phases).T
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(df.max().max(), 1) * 1.1)
    ax.legend(title="Phase")
    if path:
        save_plot(fig, path)
