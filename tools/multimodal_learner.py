import argparse
import base64
import json
import os
import random
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from autogluon.multimodal import MultiModalPredictor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from utils import get_metrics_help_modal

# Suppress warnings
warnings.filterwarnings("ignore")

def path_expander(path, base_folder):
    """Expand relative image paths into absolute paths."""
    path = str(path).lstrip("/")
    return os.path.abspath(os.path.join(base_folder, path))

def encode_image_to_base64(img_path):
    """Encode an image file to base64 string for HTML embedding."""
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def format_config_table_html(config, train_data, image_column, label_column):
    """Format configuration table with multimodal-specific details."""
    display_keys = [
        "foundation_model",
        "fine_tuned",
        "modalities",
        "fusion_method",
        "task_type",
        "learning_rate",
        "batch_size",
        "num_epochs",
        "random_seed",
        "time_limit"
    ]
    rows = []
    
    # Determine modalities
    modalities = ["Image"] if image_column in train_data.columns else []
    if any(col != label_column and col != image_column and train_data[col].dtype == 'object' for col in train_data.columns):
        modalities.append("Text")
    if any(col != label_column and col != image_column and train_data[col].dtype in ['int64', 'float64', 'category'] for col in train_data.columns):
        modalities.append("Tabular")
    
    config_info = {
        "foundation_model": config.get("model_name", "Pre-trained model selected by AutoGluon (e.g., timm, transformers, CLIP)"),
        "fine_tuned": "Yes, on provided training data",
        "modalities": ", ".join(modalities) if modalities else "N/A",
        "fusion_method": "Automatically selected by AutoGluon (early, late, or hybrid)",
        "task_type": config.get("problem_type", "N/A"),
        "learning_rate": config.get("learning_rate", "Auto-selected by AutoGluon"),
        "batch_size": config.get("batch_size", "Auto-selected by AutoGluon"),
        "num_epochs": config.get("num_epochs", "Auto-selected by AutoGluon"),
        "random_seed": config.get("random_seed", "N/A"),
        "time_limit": config.get("time_limit", "None")
    }
    
    for key in display_keys:
        val = config_info.get(key, "N/A")
        if isinstance(val, float) and key == "learning_rate":
            val = f"{val:.6f}"
        if val is not None:
            rows.append(
                f"<tr>"
                f"<td style='padding: 6px 12px; border: 1px solid #ccc; text-align: left;'>"
                f"{key.replace('_', ' ').title()}</td>"
                f"<td style='padding: 6px 12px; border: 1px solid #ccc; text-align: center;'>"
                f"{val}</td>"
                f"</tr>"
            )
    split_info = "Train/Validation split: 80/20 (stratified); Test set provided separately."
    rows.append(
        f"<tr>"
        f"<td style='padding: 6px 12px; border: 1px solid #ccc; text-align: left;'>"
        f"Data Split</td>"
        f"<td style='padding: 6px 12px; border: 1px solid #ccc; text-align: center;'>"
        f"{split_info}</td>"
        f"</tr>"
    )
    return (
        "<h2 style='text-align: center;'>Training Setup</h2>"
        "<div style='display: flex; justify-content: center;'>"
        "<table style='border-collapse: collapse; width: 60%; table-layout: auto;'>"
        "<thead><tr>"
        "<th style='padding: 10px; border: 1px solid #ccc; text-align: left;'>"
        "Parameter</th>"
        "<th style='padding: 10px; border: 1px solid #ccc; text-align: center;'>"
        "Value</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div><br>"
        "<p style='text-align: center; font-size: 0.9em;'>"
        "Model trained using AutoGluon. "
        "See <a href='https://auto.gluon.ai/stable/api/autogluon.multimodal.MultiModalPredictor.html' target='_blank'>"
        "AutoGluon documentation</a> for details.</p><hr>"
    )

def generate_table_row(cells, styles):
    """Helper function to generate an HTML table row."""
    return (
        "<tr>"
        + "".join(f"<td style='{styles}'>{cell}</td>" for cell in cells)
        + "</tr>"
    )

def format_stats_table_html(train_scores, val_scores, test_scores):
    """Format a combined HTML table for training, validation, and test metrics."""
    metrics = set(train_scores.keys()) & set(val_scores.keys()) & set(test_scores.keys())
    rows = []
    for metric in sorted(metrics):
        t = train_scores.get(metric)
        v = val_scores.get(metric)
        te = test_scores.get(metric)
        if all(isinstance(x, (int, float, np.integer, np.floating)) for x in [t, v, te]):
            display_name = metric.replace('_', ' ').title()
            rows.append([display_name, f"{t:.4f}", f"{v:.4f}", f"{te:.4f}"])
    if not rows:
        return "<table><tr><td>No metric values found.</td></tr></table>"
    html = (
        "<h2 style='text-align: center;'>Model Performance Summary</h2>"
        "<div style='display: flex; justify-content: center;'>"
        "<table style='border-collapse: collapse; table-layout: auto;'>"
        "<thead><tr>"
        "<th style='padding: 10px; border: 1px solid #ccc; text-align: left; white-space: nowrap;'>Metric</th>"
        "<th style='padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap;'>Train</th>"
        "<th style='padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap;'>Validation</th>"
        "<th style='padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap;'>Test</th>"
        "</tr></thead><tbody>"
    )
    for row in rows:
        html += generate_table_row(
            row,
            "padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap;"
        )
    html += "</tbody></table></div><br>"
    return html

def format_train_val_stats_table_html(train_scores, val_scores):
    """Format HTML table for training and validation metrics."""
    metrics = set(train_scores.keys()) & set(val_scores.keys())
    rows = []
    for metric in sorted(metrics):
        t = train_scores.get(metric)
        v = val_scores.get(metric)
        if all(isinstance(x, (int, float, np.integer, np.floating)) for x in [t, v]):
            display_name = metric.replace('_', ' ').title()
            rows.append([display_name, f"{t:.4f}", f"{v:.4f}"])
    if not rows:
        return "<table><tr><td>No metric values found for Train/Validation.</td></tr></table>"
    html = (
        "<h2 style='text-align: center;'>Train/Validation Performance Summary</h2>"
        "<div style='display: flex; justify-content: center;'>"
        "<table style='border-collapse: collapse; table-layout: auto;'>"
        "<thead><tr>"
        "<th style='padding: 10px; border: 1px solid #ccc; text-align: left; white-space: nowrap;'>Metric</th>"
        "<th style='padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap;'>Train</th>"
        "<th style='padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap;'>Validation</th>"
        "</tr></thead><tbody>"
    )
    for row in rows:
        html += generate_table_row(
            row,
            "padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap;"
        )
    html += "</tbody></table></div><br>"
    return html

def format_test_stats_table_html(test_scores):
    """Format HTML table for test metrics."""
    rows = []
    for key in sorted(test_scores.keys()):
        value = test_scores[key]
        if isinstance(value, (int, float, np.integer, np.floating)):
            display_name = key.replace('_', ' ').title()
            rows.append([display_name, f"{value:.4f}"])
    if not rows:
        return "<table><tr><td>No test metric values found.</td></tr></table>"
    html = (
        "<h2 style='text-align: center;'>Test Performance Summary</h2>"
        "<div style='display: flex; justify-content: center;'>"
        "<table style='border-collapse: collapse; table-layout: auto;'>"
        "<thead><tr>"
        "<th style='padding: 10px; border: 1px solid #ccc; text-align: left; white-space: nowrap;'>Metric</th>"
        "<th style='padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap;'>Test</th>"
        "</tr></thead><tbody>"
    )
    for row in rows:
        html += generate_table_row(
            row,
            "padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap;"
        )
    html += "</tbody></table></div><br>"
    return html

def generate_plots(y_true, y_pred, y_prob, phase, problem_type, html_dir, classes, data=None, predictor=None):
    """Generate plots for a specific phase (train, val, test)."""
    os.makedirs(html_dir, exist_ok=True)
    plot_files = []

    # Confusion Matrix (classification)
    if problem_type in ["binary", "multiclass"]:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{phase} Confusion Matrix')
        cm_path = os.path.join(html_dir, f'{phase.lower()}_confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        plot_files.append((f'{phase} Confusion Matrix', cm_path))

        # Prediction Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(y_pred, bins=len(classes), kde=True)
        plt.title(f'{phase} Prediction Distribution')
        plt.xlabel('Predicted Classes')
        plt.ylabel('Frequency')
        dist_path = os.path.join(html_dir, f'{phase.lower()}_prediction_distribution.png')
        plt.savefig(dist_path)
        plt.close()
        plot_files.append((f'{phase} Prediction Distribution', dist_path))

        # ROC and Precision-Recall Curves (for validation and test phases)
        if phase in ["Validation", "Test"] and y_prob is not None:
            if problem_type == "binary":
                # Convert y_true to binary if needed
                if not np.issubdtype(y_true.dtype, np.number):
                    y_true_bin = pd.factorize(y_true)[0]
                else:
                    y_true_bin = y_true
                y_score = y_prob.iloc[:, 1] if hasattr(y_prob, "columns") else y_prob[:, 1]

                # ROC Curve
                fpr, tpr, _ = roc_curve(y_true_bin, y_score)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{phase} ROC Curve')
                plt.legend(loc="lower right")
                roc_path = os.path.join(html_dir, f'{phase.lower()}_roc_curve.png')
                plt.savefig(roc_path)
                plt.close()
                plot_files.append((f'{phase} ROC Curve', roc_path))

                # Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
                pr_auc = auc(recall, precision)
                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'{phase} Precision-Recall Curve')
                plt.legend(loc="lower left")
                pr_path = os.path.join(html_dir, f'{phase.lower()}_precision_recall_curve.png')
                plt.savefig(pr_path)
                plt.close()
                plot_files.append((f'{phase} Precision-Recall Curve', pr_path))

            elif problem_type == "multiclass":
                # ROC Curves (One-vs-Rest)
                y_true_bin = pd.get_dummies(y_true).values
                plt.figure(figsize=(10, 6))
                for i, class_name in enumerate(classes):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob.iloc[:, i] if hasattr(y_prob, "columns") else y_prob[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f'{class_name} (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{phase} ROC Curves (One-vs-Rest)')
                plt.legend(loc="lower right")
                roc_path = os.path.join(html_dir, f'{phase.lower()}_roc_curves.png')
                plt.savefig(roc_path)
                plt.close()
                plot_files.append((f'{phase} ROC Curves (One-vs-Rest)', roc_path))

                # Precision-Recall Curves (One-vs-Rest)
                plt.figure(figsize=(10, 6))
                for i, class_name in enumerate(classes):
                    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob.iloc[:, i] if hasattr(y_prob, "columns") else y_prob[:, i])
                    pr_auc = auc(recall, precision)
                    plt.plot(recall, precision, label=f'{class_name} (area = {pr_auc:.2f})')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'{phase} Precision-Recall Curves (One-vs-Rest)')
                plt.legend(loc="lower left")
                pr_path = os.path.join(html_dir, f'{phase.lower()}_precision_recall_curves.png')
                plt.savefig(pr_path)
                plt.close()
                plot_files.append((f'{phase} Precision-Recall Curves (One-vs-Rest)', pr_path))

    elif problem_type == "regression":
        # Residual Plot
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(f'{phase} Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        res_path = os.path.join(html_dir, f'{phase.lower()}_residuals.png')
        plt.savefig(res_path)
        plt.close()
        plot_files.append((f'{phase} Residual Plot', res_path))

        # Prediction Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(y_pred, bins=50, kde=True)
        plt.title(f'{phase} Prediction Distribution')
        plt.xlabel('Predicted Values')
        plt.ylabel('Frequency')
        dist_path = os.path.join(html_dir, f'{phase.lower()}_prediction_distribution.png')
        plt.savefig(dist_path)
        plt.close()
        plot_files.append((f'{phase} Prediction Distribution', dist_path))

    # Feature Importance (for test phase, if tabular data is available)
    if phase == "Test" and data is not None and predictor is not None:
        try:
            feature_importance = predictor.feature_importance(data)
            if not feature_importance.empty:
                plt.figure(figsize=(10, 6))
                sns.barplot(x='importance', y=feature_importance.index, data=feature_importance)
                plt.title(f'{phase} Feature Importance')
                plt.xlabel('Importance Score')
                plt.ylabel('Features')
                fi_path = os.path.join(html_dir, f'{phase.lower()}_feature_importance.png')
                plt.savefig(fi_path)
                plt.close()
                plot_files.append((f'{phase} Feature Importance', fi_path))
        except Exception as e:
            print(f"Warning: Could not generate feature importance plot: {e}")

    return plot_files

def render_img_section(title, plot_files):
    """Render HTML section for plots."""
    if not plot_files:
        return f"<h2 style='text-align: center;'>{title}</h2><p><em>No plots found.</em></p>"
    section_html = f"<h2 style='text-align: center;'>{title}</h2><div>"
    for plot_title, img_path in plot_files:
        b64 = encode_image_to_base64(img_path)
        section_html += (
            f'<div class="plot" style="margin-bottom:20px;text-align:center;">'
            f"<h3>{plot_title}</h3>"
            f'<img src="data:image/png;base64,{b64}" '
            f'style="max-width:90%;max-height:600px;border:1px solid #ddd;" />'
            f"</div>"
        )
    section_html += "</div>"
    return section_html

def generate_html_report(config, train_scores, val_scores, test_scores, problem_type, html_dir, classes, train_data, val_data, test_data, image_column, label_column, predictor):
    """Generate tabbed HTML report with config, metrics, and plots."""
    config_html = format_config_table_html(config, train_data, image_column, label_column)
    metrics_html = format_stats_table_html(train_scores, val_scores, test_scores)
    train_val_html = format_train_val_stats_table_html(train_scores, val_scores)
    test_html = format_test_stats_table_html(test_scores)
    
    # Generate plots for each phase
    train_plots = generate_plots(train_scores.get('y_true'), train_scores.get('y_pred'), train_scores.get('y_prob'), "Train", problem_type, html_dir, classes)
    val_plots = generate_plots(val_scores.get('y_true'), val_scores.get('y_pred'), val_scores.get('y_prob'), "Validation", problem_type, html_dir, classes)
    test_plots = generate_plots(test_scores.get('y_true'), test_scores.get('y_pred'), test_scores.get('y_prob'), "Test", problem_type, html_dir, classes, test_data, predictor)
    
    # Note about learning curves
    learning_curve_note = (
        "<p style='text-align: center; font-size: 0.9em;'>"
        "Note: Learning curves are not available as AutoGluon does not provide epoch-wise training history by default."
        "</p>"
    ) if problem_type in ["binary", "multiclass"] else ""
    
    button_html = """
    <button class="help-modal-btn" id="openMetricsHelp">Model Evaluation Metrics — Help Guide</button>
    <br><br>
    <style>
    .help-modal-btn {
        background-color: #17623b;
        color: #fff;
        border: none;
        border-radius: 24px;
        padding: 10px 28px;
        font-size: 1.1rem;
        font-weight: bold;
        letter-spacing: 0.03em;
        cursor: pointer;
        transition: background 0.2s, box-shadow 0.2s;
        box-shadow: 0 2px 8px rgba(23,98,59,0.07);
    }
    .help-modal-btn:hover, .help-modal-btn:focus {
        background-color: #21895e;
        outline: none;
        box-shadow: 0 4px 16px rgba(23,98,59,0.14);
    }
    </style>
    """
    tab1_content = button_html + config_html + metrics_html
    tab2_content = button_html + train_val_html + learning_curve_note + render_img_section("Training & Validation Visualizations", train_plots + val_plots)
    tab3_content = button_html + test_html + render_img_section("Test Visualizations", test_plots)
    
    tabbed_html = build_tabbed_html(tab1_content, tab2_content, tab3_content)
    modal_html = get_metrics_help_modal()
    
    html = f"""
    <html>
    <head>
    <title>Image Classification Results</title>
    </head>
    <body>
    <h1 style='text-align: center;'>Image Classification Results</h1>
    {tabbed_html}
    {modal_html}
    </body>
    </html>
    """
    return html

def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate AutoGluon MultiModal on an image dataset"
    )
    parser.add_argument(
        "--train_csv",
        required=True,
        help="Path to training metadata CSV (must include an image-ID column and a label column).",
    )
    parser.add_argument(
        "--test_csv",
        required=True,
        help="Path to test metadata CSV (must include the same image-ID column).",
    )
    parser.add_argument(
        "--label_column",
        default="dx",
        help="Name of the target/label column in your CSVs (default: dx).",
    )
    parser.add_argument(
        "--image_column",
        default="image_id",
        help="Name of the column that holds the image filename/ID (default: image_id).",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=None,
        help="Optional: wall-clock time (in seconds) to limit training. If omitted, no time limit is passed.",
    )
    parser.add_argument(
        "--output_json",
        default="results.json",
        help="Output JSON file where metrics + config will be saved.",
    )
    parser.add_argument(
        "--output_html",
        default="report.html",
        help="Output HTML file where the report will be saved.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Read CSVs
    try:
        train_data = pd.read_csv(args.train_csv)
    except Exception as e:
        print(f"ERROR reading train CSV at {args.train_csv}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        test_data = pd.read_csv(args.test_csv)
    except Exception as e:
        print(f"ERROR reading test CSV at {args.test_csv}: {e}", file=sys.stderr)
        sys.exit(1)

    # Verify image_column
    if args.image_column not in train_data.columns:
        print(
            f"ERROR: The specified image_column '{args.image_column}' does not exist in train CSV.",
            file=sys.stderr,
        )
        print(
            f"Available columns in {args.train_csv}:\n  " + ", ".join(train_data.columns),
            file=sys.stderr,
        )
        sys.exit(1)

    if args.image_column not in test_data.columns:
        print(
            f"ERROR: The specified image_column '{args.image_column}' does not exist in test CSV.",
            file=sys.stderr,
        )
        print(
            f"Available columns in {args.test_csv}:\n  " + ", ".join(test_data.columns),
            file=sys.stderr,
        )
        sys.exit(1)

    # Split train_data
    train, val = train_test_split(
        train_data,
        test_size=0.2,
        random_state=args.random_seed,
        stratify=train_data[args.label_column] if args.label_column in train_data.columns else None
    )

    # Expand image paths
    base_folder = os.path.dirname(os.path.abspath(args.train_csv))
    train[args.image_column] = train[args.image_column].apply(
        lambda x: path_expander(str(x), base_folder)
    )
    val[args.image_column] = val[args.image_column].apply(
        lambda x: path_expander(str(x), base_folder)
    )
    test_data[args.image_column] = test_data[args.image_column].apply(
        lambda x: path_expander(str(x), base_folder)
    )

    # Create predictor
    predictor = MultiModalPredictor(label=args.label_column)

    # Fit model
    if args.time_limit is not None:
        predictor.fit(train, time_limit=args.time_limit)
    else:
        predictor.fit(train)

    # Define metrics
    problem_type = predictor.problem_type.lower() if predictor.problem_type else "unknown"
    if problem_type == "binary":
        base_metrics = [
            "accuracy", "balanced_accuracy", "f1", "f1_macro", "f1_micro", "f1_weighted",
            "precision", "precision_macro", "precision_micro", "precision_weighted",
            "recall", "recall_macro", "recall_micro", "recall_weighted", "log_loss",
            "roc_auc", "average_precision", "mcc", "quadratic_kappa"
        ]
    elif problem_type == "multiclass":
        base_metrics = [
            "accuracy", "balanced_accuracy", "f1_macro", "f1_micro", "f1_weighted",
            "precision_macro", "precision_micro", "precision_weighted",
            "recall_macro", "recall_micro", "recall_weighted", "log_loss",
            "roc_auc_ovr", "roc_auc_ovo", "mcc", "quadratic_kappa"
        ]
    else:
        base_metrics = ["root_mean_squared_error", "mean_absolute_error", "r2"]

    # Evaluate
    train_scores = predictor.evaluate(train, metrics=base_metrics)
    val_scores = predictor.evaluate(val, metrics=base_metrics)
    test_scores = predictor.evaluate(test_data, metrics=base_metrics)

    # Get predictions and probabilities
    train_y_true = train[args.label_column]
    train_y_pred = predictor.predict(train)
    train_y_prob = predictor.predict_proba(train) if problem_type in ["binary", "multiclass"] else None
    val_y_true = val[args.label_column]
    val_y_pred = predictor.predict(val)
    val_y_prob = predictor.predict_proba(val) if problem_type in ["binary", "multiclass"] else None
    test_y_true = test_data[args.label_column]
    test_y_pred = predictor.predict(test_data)
    test_y_prob = predictor.predict_proba(test_data) if problem_type in ["binary", "multiclass"] else None
    classes = sorted(np.unique(train_y_true)) if problem_type in ["binary", "multiclass"] else []

    # Store predictions in scores for plot generation
    train_scores['y_true'] = train_y_true
    train_scores['y_pred'] = train_y_pred
    train_scores['y_prob'] = train_y_prob
    val_scores['y_true'] = val_y_true
    val_scores['y_pred'] = val_y_pred
    val_scores['y_prob'] = val_y_prob
    test_scores['y_true'] = test_y_true
    test_scores['y_pred'] = test_y_pred
    test_scores['y_prob'] = test_y_prob

    # Extract config
    config = predictor._learner._config if hasattr(predictor._learner, '_config') else {}
    config['random_seed'] = args.random_seed
    config['time_limit'] = args.time_limit
    config['problem_type'] = problem_type

    # Write JSON
    output = {
        "train_metrics": train_scores,
        "validation_metrics": val_scores,
        "test_metrics": test_scores,
        "config": config,
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2, default=lambda o: str(o))
    print(f"Saved results + config to '{args.output_json}'")

    # Write HTML
    html_dir = os.path.dirname(args.output_html) or "."
    html_report = generate_html_report(
        config, train_scores, val_scores, test_scores, problem_type, html_dir, classes,
        train_data, val, test_data, args.image_column, args.label_column, predictor
    )
    with open(args.output_html, "w") as f:
        f.write(html_report)
    print(f"Saved HTML report to '{args.output_html}'")

if __name__ == "__main__":
    main()
