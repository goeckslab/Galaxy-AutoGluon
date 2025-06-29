import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import random
import torch
from autogluon.multimodal import MultiModalPredictor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from utils import get_html_template, get_html_closing, encode_image_to_base64, get_metrics_help_modal
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def path_expander(path, base_folder):
    """Expand relative image paths into absolute paths."""
    path = str(path).lstrip("/")
    return os.path.abspath(os.path.join(base_folder, path))


def format_config_table_html(config: dict) -> str:
    """Format configuration table for AutoGluon."""
    display_keys = [
        "model_name", "time_limit", "random_seed", "problem_type",
        "image_missing_strategy", "text_normalize", "document_missing_strategy",
        "label_preprocessing", "optimizer_type", "learner_class",
        "label_column", "eval_metric", "validation_metric", "pretrained"
    ]
    rows = []
    for key in display_keys:
        val = config.get(key, "N/A")
        if val is not None:
            rows.append(
                f"<tr>"
                f"<th>{key.replace('_', ' ').title()}</th>"
                f"<td>{val}</td>"
                f"</tr>"
            )
    split_info = "Train/Validation split: 80/20 (stratified); Test set provided separately."
    rows.append(
        f"<tr>"
        f"<th>Data Split</th>"
        f"<td>{split_info}</td>"
        f"</tr>"
    )
    return (
        "<h2>Training Setup</h2>"
        "<table>"
        "<tr><th>Parameter</th><th>Value</th></tr>" + "".join(rows) + "</table>"
    )


def format_stats_table_html(train_scores, val_scores, test_scores):
    """Format a combined HTML table for training, validation, and test metrics."""
    metrics = set(train_scores.keys()) & set(val_scores.keys()) & set(test_scores.keys())
    rows = []
    for metric in sorted(metrics):
        t = train_scores.get(metric)
        v = val_scores.get(metric)
        te = test_scores.get(metric)
        if all(isinstance(x, (int, float)) for x in [t, v, te]):
            display_name = metric.replace('_', ' ').title()
            rows.append([display_name, f"{t:.4f}", f"{v:.4f}", f"{te:.4f}"])
    if not rows:
        return "<table><tr><td>No metric values found.</td></tr></table>"
    html = (
        "<h2>Model Performance Summary</h2>"
        "<table>"
        "<tr><th>Metric</th><th>Train</th><th>Validation</th><th>Test</th></tr>"
    )
    for row in rows:
        html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    html += "</table>"
    return html

def format_train_val_stats_table_html(train_scores, val_scores):
    """Format HTML table for training and validation metrics."""
    metrics = set(train_scores.keys()) & set(val_scores.keys())
    rows = []
    for metric in sorted(metrics):
        t = train_scores.get(metric)
        v = val_scores.get(metric)
        if all(isinstance(x, (int, float)) for x in [t, v]):
            display_name = metric.replace('_', ' ').title()
            rows.append([display_name, f"{t:.4f}", f"{v:.4f}"])
    if not rows:
        return "<table><tr><td>No metric values found for Train/Validation.</td></tr></table>"
    html = (
        "<h2>Train/Validation Performance Summary</h2>"
        "<table>"
        "<tr><th>Metric</th><th>Train</th><th>Validation</th></tr>"
    )
    for row in rows:
        html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    html += "</table>"
    return html

def format_test_stats_table_html(test_scores):
    """Format HTML table for test metrics."""
    rows = []
    for key in sorted(test_scores.keys()):
        value = test_scores[key]
        if isinstance(value, (int, float)):
            display_name = key.replace('_', ' ').title()
            rows.append([display_name, f"{value:.4f}"])
    if not rows:
        return "<table><tr><td>No test metric values found.</td></tr></table>"
    html = (
        "<h2>Test Performance Summary</h2>"
        "<table>"
        "<tr><th>Metric</th><th>Test</th></tr>"
    )
    for row in rows:
        html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    html += "</table>"
    return html


def build_tabbed_html(metrics_html, train_val_html, test_html):
    """Build tabbed HTML structure with help button inside each tab."""
    help_button_html = """
    <button class="help-modal-btn" id="openMetricsHelp">Model Evaluation Metrics — Help Guide</button>
    <style>
    .help-modal-btn {
        background-color: #17623b;
        color: #fff;
        border: none;
        border-radius: 24px;
        padding: 8px 20px; /* Adjusted padding for better fit */
        font-size: 1rem; /* Slightly smaller for layout compatibility */
        font-weight: bold;
        letter-spacing: 0.03em;
        cursor: pointer;
        transition: background 0.2s, box-shadow 0.2s;
        box-shadow: 0 2px 8px rgba(23,98,59,0.07);
        margin-bottom: 20px; /* Space below button */
        display: block; /* Ensure it takes its own line */
        margin-left: auto; /* Center or adjust as needed */
        margin-right: auto;
    }
    .help-modal-btn:hover, .help-modal-btn:focus {
        background-color: #21895e;
        outline: none;
        box-shadow: 0 4px 16px rgba(23,98,59,0.14);
    }
    </style>
    """
    # Add button to each tab's content
    metrics_html = help_button_html + metrics_html
    train_val_html = help_button_html + train_val_html
    test_html = help_button_html + test_html

    tabbed_html = f"""
<style>
.tabs {{
  display: flex;
  border-bottom: 2px solid #ccc;
  margin-bottom: 1rem;
}}
.tab {{
  padding: 10px 20px;
  cursor: pointer;
  border: 1px solid #ccc;
  border-bottom: none;
  background: #f9f9f9;
  margin-right: 5px;
  border-top-left-radius: 8px;
  border-top-right-radius: 8px;
}}
.tab.active {{
  background: white;
  font-weight: bold;
}}
.tab-content {{
  display: none;
  padding: 20px;
  border: 1px solid #ccc;
  border-top: none;
}}
.tab-content.active {{
  display: block;
}}
</style>
<div class="tabs">
  <div class="tab active" onclick="showTab('metrics')">Config & Results Summary</div>
  <div class="tab" onclick="showTab('trainval')">Train/Validation Results</div>
  <div class="tab" onclick="showTab('test')">Test Results</div>
</div>
<div id="metrics" class="tab-content active">
  {metrics_html}
</div>
<div id="trainval" class="tab-content">
  {train_val_html}
</div>
<div id="test" class="tab-content">
  {test_html}
</div>
<script>
function showTab(id) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.querySelector(`.tab[onclick*="${{id}}"]`).classList.add('active');
}}
</script>
"""
    tabbed_html += get_metrics_help_modal()
    return tabbed_html

def generate_confusion_matrix_plot(y_true, y_pred, classes, phase, temp_dir):
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{phase} Confusion Matrix')
    plot_path = os.path.join(temp_dir, f'{phase.lower()}_confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def generate_roc_curve_plot(y_true, y_prob, classes, phase, temp_dir):
    """Generate and save ROC curve plot for binary classification."""
    if len(classes) != 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{phase} ROC Curve')
    plt.legend(loc="lower right")
    plot_path = os.path.join(temp_dir, f'{phase.lower()}_roc_curve.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def render_img_section(title, plot_files):
    """Render HTML section for plots."""
    if not plot_files:
        return f"<h2>{title}</h2><p><em>No plots found.</em></p>"
    section_html = f"<h2>{title}</h2>"
    for plot_title, img_path in plot_files:
        b64 = encode_image_to_base64(img_path)
        section_html += (
            f'<div class="plot">'
            f"<h3>{plot_title}</h3>"
            f'<img src="data:image/png;base64,{b64}" />'
            f"</div>"
        )
    return section_html

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate an AutoGluon MultiModal model")
    parser.add_argument("--train_csv", required=True, help="Path to training metadata CSV")
    parser.add_argument("--test_csv", required=True, help="Path to test metadata CSV")
    parser.add_argument("--label_column", default="label", help="Name of the target/label column")
    parser.add_argument("--image_column", default="image_path", help="Name of the image column")
    parser.add_argument("--time_limit", type=int, default=None, help="Training time limit in seconds")
    parser.add_argument("--output_json", default="results.json", help="Output JSON file")
    parser.add_argument("--output_html", default="report.html", help="Output HTML report file")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Load data
    try:
        train_data = pd.read_csv(args.train_csv)
        test_data = pd.read_csv(args.test_csv)
    except Exception as e:
        logger.error(f"Error reading CSV files: {e}")
        sys.exit(1)

    # Verify columns
    for df, name in [(train_data, "train"), (test_data, "test")]:
        if args.image_column not in df.columns or args.label_column not in df.columns:
            logger.error(f"Missing columns in {name} CSV: {df.columns}")
            sys.exit(1)

    # Split train data
    train, val = train_test_split(
        train_data,
        test_size=0.2,
        random_state=args.random_seed,
        stratify=train_data[args.label_column]
    )

    # Expand image paths
    base_folder = os.path.dirname(os.path.abspath(args.train_csv))
    for df in [train, val, test_data]:
        df[args.image_column] = df[args.image_column].apply(lambda x: path_expander(str(x), base_folder))

    # Train model
    save_path = "AutogluonModels/custom_model"
    predictor = MultiModalPredictor(label=args.label_column, path=save_path)
    if args.time_limit:
        predictor.fit(train, time_limit=args.time_limit)
    else:
        predictor.fit(train)

    # Evaluate
    problem_type = predictor.problem_type.lower()
    metrics = {
        "binary": ["accuracy", "roc_auc", "precision", "recall", "f1"],
        "multiclass": ["accuracy", "f1_macro", "precision_macro", "recall_macro"],
        "regression": ["root_mean_squared_error", "mean_absolute_error", "r2"]
    }.get(problem_type, ["root_mean_squared_error"])

    train_scores = predictor.evaluate(train, metrics=metrics)
    val_scores = predictor.evaluate(val, metrics=metrics)
    test_scores = predictor.evaluate(test_data, metrics=metrics)

    # Predictions for plots
    test_y_true = test_data[args.label_column]
    test_y_pred = predictor.predict(test_data)
    test_y_prob = predictor.predict_proba(test_data) if problem_type in ["binary", "multiclass"] else None
    classes = sorted(np.unique(test_y_true)) if problem_type in ["binary", "multiclass"] else []

    config_path = os.path.join(save_path, "config.yaml")
    try:
        with open(config_path, "r") as f:
            config_yaml = yaml.safe_load(f)
        model_name = config_yaml["model"]["names"][0]
        config_data = {
            "Model Names": model_name,
            "Image Missing Value Strategy": config_yaml["data"]["image"]["missing_value_strategy"],
            "Text Normalize": config_yaml["data"]["text"]["normalize_text"],
            "Document Missing Value Strategy": config_yaml["data"]["document"]["missing_value_strategy"],
            "Label Numerical Preprocessing": config_yaml["data"]["label"]["numerical_preprocessing"],
            "Optimizer Type": config_yaml["optim"]["optim_type"]
        }
    except (FileNotFoundError, KeyError) as e:
        logger.warning(f"Error reading config.yaml: {e}. Using defaults.")
        config_data = {}

    # Read assets.json
    assets_path = os.path.join(save_path, "assets.json")
    try:
        with open(assets_path, "r") as f:
            assets = json.load(f)
            assets_data = {
                "Learner Class": assets["learner_class"],
                "Label Column": assets["label_column"],
                "Problem Type": assets["problem_type"],
                "Evaluation Metric": assets["eval_metric_name"],
                "Validation Metric": assets["validation_metric_name"],
                "Pretrained": assets["pretrained"]
            }
    except (FileNotFoundError, KeyError) as e:
        logger.warning(f"Error reading assets.json: {e}. Using defaults.")
        assets_data = {}

    # Combine config data
    config = {
        "model_name": config_data.get("Model Names", "AutoGluon MultiModal"),
        "time_limit": args.time_limit,
        "random_seed": args.random_seed,
        "problem_type": assets_data.get("Problem Type", problem_type),
        "image_missing_strategy": config_data.get("Image Missing Value Strategy", "N/A"),
        "text_normalize": config_data.get("Text Normalize", "N/A"),
        "document_missing_strategy": config_data.get("Document Missing Value Strategy", "N/A"),
        "label_preprocessing": config_data.get("Label Numerical Preprocessing", "N/A"),
        "optimizer_type": config_data.get("Optimizer Type", "N/A"),
        "learner_class": assets_data.get("Learner Class", "N/A"),
        "label_column": assets_data.get("Label Column", args.label_column),
        "eval_metric": assets_data.get("Evaluation Metric", "N/A"),
        "validation_metric": assets_data.get("Validation Metric", "N/A"),
        "pretrained": assets_data.get("Pretrained", "N/A")
    }

    # Save results
    output = {"train_metrics": train_scores, "val_metrics": val_scores, "test_metrics": test_scores, "config": config}
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Generate HTML report
    with tempfile.TemporaryDirectory() as temp_dir:
        html_dir = os.path.dirname(args.output_html) or "."
        os.makedirs(html_dir, exist_ok=True)

        config_html = format_config_table_html(config)
        metrics_html = format_stats_table_html(train_scores, val_scores, test_scores)
        train_val_html = format_train_val_stats_table_html(train_scores, val_scores)
        test_html = format_test_stats_table_html(test_scores)

        # Generate plots
        plot_files = []
        if problem_type in ["binary", "multiclass"]:
            cm_plot = generate_confusion_matrix_plot(test_y_true, test_y_pred, classes, "Test", temp_dir)
            plot_files.append(("Test Confusion Matrix", cm_plot))
            if problem_type == "binary":
                roc_plot = generate_roc_curve_plot(test_y_true, test_y_prob, classes, "Test", temp_dir)
                if roc_plot:
                    plot_files.append(("Test ROC Curve", roc_plot))

        test_html += render_img_section("Test Visualizations", plot_files)
        tabbed_html = build_tabbed_html(config_html + metrics_html, train_val_html, test_html)

        html = get_html_template() + "<h1>AutoGluon Model Report</h1>" + tabbed_html + get_html_closing()
        with open(args.output_html, "w") as f:
            f.write(html)

    logger.info(f"Report saved to {args.output_html}")

if __name__ == "__main__":
    main()
