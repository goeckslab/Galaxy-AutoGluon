import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
import plotly.express as px
import plotly.graph_objects as go


def save_plot(fig, path):
    """Save the Plotly figure to the specified path."""
    if path:
        fig.write_image(path)


def generate_confusion_matrix_plot(
    y_true, y_pred, classes=None, title="Confusion Matrix", path=None
):
    """Generate and save a confusion matrix heatmap using Plotly."""
    if classes is None:
        classes = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            colorscale="Blues",
            text=cm.astype(str),
            texttemplate="%{text}",
            textfont={"size": 12},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted label",
        yaxis_title="True label",
    )
    save_plot(fig, path)


def generate_roc_curve_plot(
    y_true_bin, y_prob, title="ROC Curve", path=None
):
    """Generate and save an ROC curve plot using Plotly. Assumes y_true_bin is binary 0/1."""
    fpr, tpr, _ = roc_curve(y_true_bin, y_prob)
    roc_auc = roc_auc_score(y_true_bin, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.2f}"))
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash"),
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    save_plot(fig, path)


def generate_pr_curve_plot(
    y_true_bin, y_prob, title="Precision-Recall Curve", path=None
):
    """Generate and save a Precision-Recall curve plot using Plotly. Assumes y_true_bin is binary 0/1."""
    precision, recall, _ = precision_recall_curve(y_true_bin, y_prob)
    ap = average_precision_score(y_true_bin, y_prob)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            line_shape="hv",
            name=f"AP = {ap:.2f}",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Recall",
        yaxis_title="Precision",
    )
    save_plot(fig, path)


def generate_calibration_plot(
    y_true_bin, y_prob, n_bins=10, title="Calibration Plot", path=None
):
    """Generate and save a calibration plot using Plotly. Assumes y_true_bin is binary 0/1."""
    prob_true, prob_pred = calibration_curve(
        y_true_bin, y_prob, n_bins=n_bins, strategy="uniform"
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name="Model")
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash"),
            name="Perfect",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Probability",
        yaxis_title="Observed Probability",
    )
    save_plot(fig, path)


def generate_per_class_metrics_plot(
    y_true,
    y_pred,
    metrics=["precision", "recall", "f1-score"],
    title="Per-Class Metrics",
    path=None,
):
    """Generate and save a bar plot of per-class metrics using Plotly."""
    report = classification_report(y_true, y_pred, output_dict=True)
    classes = [
        cls
        for cls in report
        if cls not in ["accuracy", "macro avg", "micro avg", "weighted avg"]
    ]
    df = pd.DataFrame(report).T.loc[classes, metrics].reset_index().rename(
        columns={"index": "Class"}
    )
    df_long = df.melt(id_vars="Class", var_name="Metric", value_name="Score")
    fig = px.bar(
        df_long,
        x="Class",
        y="Score",
        color="Metric",
        barmode="group",
        title=title,
    )
    fig.update_yaxes(range=[0, 1])
    save_plot(fig, path)


def generate_scatter_plot(
    y_true, y_pred, title="Predicted vs Actual", path=None
):
    """Generate and save a scatter plot of predicted vs actual values using Plotly."""
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    fig = px.scatter(
        x=y_true,
        y=y_pred,
        opacity=0.5,
        labels={"x": "Actual", "y": "Predicted"},
        title=title,
    )
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(dash="dash"),
            name="Perfect",
        )
    )
    save_plot(fig, path)


def generate_residual_plot(
    y_true, y_pred, title="Residual Plot", path=None
):
    """Generate and save a residual plot using Plotly."""
    residuals = y_true - y_pred
    fig = px.scatter(
        x=y_pred,
        y=residuals,
        opacity=0.5,
        labels={"x": "Predicted", "y": "Residual (Actual - Predicted)"},
        title=title,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    save_plot(fig, path)


def generate_residual_histogram(
    y_true, y_pred, bins=30, title="Residual Histogram", path=None
):
    """Generate and save a histogram of residuals using Plotly."""
    residuals = y_true - y_pred
    fig = px.histogram(
        x=residuals,
        nbins=bins,
        labels={"x": "Residual"},
        title=title,
    )
    fig.update_layout(yaxis_title="Frequency")
    save_plot(fig, path)


def generate_metric_comparison_bar(
    metrics_scores,
    phases=["train", "val", "test"],
    title="Metric Comparison Across Phases",
    path=None,
):
    """Generate and save a bar plot comparing metrics across phases using Plotly."""
    df = pd.DataFrame(metrics_scores, index=phases).T.reset_index().rename(
        columns={"index": "Metric"}
    )
    df_long = df.melt(id_vars="Metric", var_name="Phase", value_name="Score")
    fig = px.bar(
        df_long,
        x="Metric",
        y="Score",
        color="Phase",
        barmode="group",
        title=title,
    )
    max_score = df_long["Score"].max()
    fig.update_yaxes(range=[0, max(max_score, 1) * 1.1])
    save_plot(fig, path)


# Additional SHAP plots (using matplotlib since SHAP natively uses it)
import shap
import matplotlib.pyplot as plt


def generate_shap_summary_plot(
    shap_values, features, title="SHAP Summary Plot", path=None
):
    """Generate and save a SHAP summary plot."""
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, features, show=False)
    plt.title(title)
    if path:
        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)


def generate_shap_force_plot(
    explainer, instance, title="SHAP Force Plot", path=None
):
    """Generate and save a SHAP force plot for a single instance."""
    shap_values = explainer(instance)
    fig = plt.figure(figsize=(10, 4))
    shap.plots.force(shap_values[0], show=False)
    plt.title(title)
    if path:
        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)


def generate_shap_waterfall_plot(
    explainer, instance, title="SHAP Waterfall Plot", path=None
):
    """Generate and save a SHAP waterfall plot for a single instance."""
    shap_values = explainer(instance)
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title(title)
    if path:
        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)
