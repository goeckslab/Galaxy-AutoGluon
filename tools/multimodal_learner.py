import argparse
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
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Suppress warnings
warnings.filterwarnings("ignore")


def path_expander(path, base_folder):
    """Expand relative image paths into absolute paths."""
    path = str(path).lstrip("/")
    return os.path.abspath(os.path.join(base_folder, path))


def generate_config_table_html(config):
    rows = []
    for key, val in config.items():
        if isinstance(val, (int, float, str, bool)):
            rows.append(
                f"<tr><td style='padding: 6px 12px; border: 1px solid #ccc; text-align: left;'>{key}</td>"
                f"<td style='padding: 6px 12px; border: 1px solid #ccc; text-align: center;'>{val}</td></tr>"
            )
    return f"""
    <h2 style='text-align: center;'>Training Configuration</h2>
    <div style='display: flex; justify-content: center;'>
    <table style='border-collapse: collapse; width: 60%; table-layout: auto;'>
        <thead><tr><th style='padding: 10px; border: 1px solid #ccc; text-align: left;'>Parameter</th>
        <th style='padding: 10px; border: 1px solid #ccc; text-align: center;'>Value</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
    </table></div><br>
    <p style='text-align: center; font-size: 0.9em;'>Model trained using AutoGluon. 
    For more details, see the <a href='https://auto.gluon.ai/stable/api/autogluon.multimodal.MultiModalPredictor.html'>official documentation</a>.</p><hr>
    """


def generate_metrics_table_html(train_scores, val_scores, test_scores):
    # Use only metrics present in all three sets and are numeric
    metrics = set(train_scores.keys()) & set(val_scores.keys()) & set(test_scores.keys())
    rows = []
    for metric in sorted(metrics):
        train_val = train_scores.get(metric, "N/A")
        val_val = val_scores.get(metric, "N/A")
        test_val = test_scores.get(metric, "N/A")
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in [train_val, val_val, test_val]):
            rows.append(
                f"<tr><td style='padding: 10px; border: 1px solid #ccc; text-align: left; white-space: nowrap;'>{metric.replace('_', ' ').title()}</td>"
                f"<td style='padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap;'>{train_val:.4f}</td>"
                f"<td style='padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap;'>{val_val:.4f}</td>"
                f"<td style='padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap;'>{test_val:.4f}</td></tr>"
            )
    return f"""
    <h2 style='text-align: center;'>Model Performance Summary</h2>
    <div style='display: flex; justify-content: center;'>
    <table style='border-collapse: collapse; table-layout: auto;'>
        <thead><tr><th style='padding: 10px; border: 1px solid #ccc; text-align: left; white-space: nowrap;'>Metric</th>
        <th style='padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap;'>Train</th>
        <th style='padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap;'>Validation</th>
        <th style='padding: 10px; border: 1px solid #ccc; text-align: center; white-space: nowrap;'>Test</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
    </table></div><br>
    """


def get_metrics_help_modal():
    modal_html = """
<div id="metricsHelpModal" class="modal">
  <div class="modal-content">
    <span class="close">×</span>
    <h2>Model Evaluation Metrics — Help Guide</h2>
    <div class="metrics-guide">
      <h3>1) General Metrics</h3>
      <p><strong>Loss:</strong> Measures the difference between predicted and actual values. Lower is better. Often used for optimization during training.</p>
      <p><strong>Accuracy:</strong> Proportion of correct predictions among all predictions. Simple but can be misleading for imbalanced datasets.</p>
      <p><strong>Micro Accuracy:</strong> Calculates accuracy by summing up all individual true positives and true negatives across all classes, making it suitable for multiclass or multilabel problems.</p>
      <p><strong>Token Accuracy:</strong> Measures how often the predicted tokens (e.g., in sequences) match the true tokens. Useful in sequence prediction tasks like NLP.</p>
      <h3>2) Precision, Recall & Specificity</h3>
      <p><strong>Precision:</strong> Out of all positive predictions, how many were correct. Precision = TP / (TP + FP). Helps when false positives are costly.</p>
      <p><strong>Recall (Sensitivity):</strong> Out of all actual positives, how many were predicted correctly. Recall = TP / (TP + FN). Important when missing positives is risky.</p>
      <p><strong>Specificity:</strong> True negative rate. Measures how well the model identifies negatives. Specificity = TN / (TN + FP). Useful in medical testing to avoid false alarms.</p>
      <h3>3) Macro, Micro, and Weighted Averages</h3>
      <p><strong>Macro Precision / Recall / F1:</strong> Averages the metric across all classes, treating each class equally, regardless of class frequency. Best when class sizes are balanced.</p>
      <p><strong>Micro Precision / Recall / F1:</strong> Aggregates TP, FP, FN across all classes before computing the metric. Gives a global view and is ideal for class-imbalanced problems.</p>
      <p><strong>Weighted Precision / Recall / F1:</strong> Averages each metric across classes, weighted by the number of true instances per class. Balances importance of classes based on frequency.</p>
      <h3>4) Average Precision (PR-AUC Variants)</h3>
      <p><strong>Average Precision Macro:</strong> Precision-Recall AUC averaged across all classes equally. Useful for balanced multi-class problems.</p>
      <p><strong>Average Precision Micro:</strong> Global Precision-Recall AUC using all instances. Best for imbalanced data or multi-label classification.</p>
      <p><strong>Average Precision Samples:</strong> Precision-Recall AUC averaged across individual samples (not classes). Ideal for multi-label problems where each sample can belong to multiple classes.</p>
      <h3>5) ROC-AUC Variants</h3>
      <p><strong>ROC-AUC:</strong> Measures model's ability to distinguish between classes. AUC = 1 is perfect; 0.5 is random guessing. Use for binary classification.</p>
      <p><strong>Macro ROC-AUC:</strong> Averages the AUC across all classes equally. Suitable when classes are balanced and of equal importance.</p>
      <p><strong>Micro ROC-AUC:</strong> Computes AUC from aggregated predictions across all classes. Useful in multiclass or multilabel settings with imbalance.</p>
      <h3>6) Ranking Metrics</h3>
      <p><strong>Hits at K:</strong> Measures whether the true label is among the top-K predictions. Common in recommendation systems and retrieval tasks.</p>
      <h3>7) Confusion Matrix Stats (Per Class)</h3>
      <p><strong>True Positives / Negatives (TP / TN):</strong> Correct predictions for positives and negatives respectively.</p>
      <p><strong>False Positives / Negatives (FP / FN):</strong> Incorrect predictions — false alarms and missed detections.</p>
      <h3>8) Other Useful Metrics</h3>
      <p><strong>Cohen's Kappa:</strong> Measures agreement between predicted and actual values adjusted for chance. Useful for multiclass classification with imbalanced labels.</p>
      <p><strong>Matthews Correlation Coefficient (MCC):</strong> Balanced measure of prediction quality that takes into account TP, TN, FP, and FN. Particularly effective for imbalanced datasets.</p>
      <h3>9) Metric Recommendations</h3>
      <ul>
        <li>Use <strong>Accuracy + F1</strong> for balanced data.</li>
        <li>Use <strong>Precision, Recall, ROC-AUC</strong> for imbalanced datasets.</li>
        <li>Use <strong>Average Precision Micro</strong> for multilabel or class-imbalanced problems.</li>
        <li>Use <strong>Macro scores</strong> when all classes should be treated equally.</li>
        <li>Use <strong>Weighted scores</strong> when class imbalance should be accounted for without ignoring small classes.</li>
        <li>Use <strong>Confusion Matrix stats</strong> to analyze class-wise performance.</li>
        <li>Use <strong>Hits at K</strong> for recommendation or ranking-based tasks.</li>
      </ul>
    </div>
  </div>
</div>
"""
    modal_css = """
<style>
.modal {
  display: none;
  position: fixed;
  z-index: 1;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  overflow: auto;
  background-color: rgba(0,0,0,0.4);
}
.modal-content {
  background-color: #fefefe;
  margin: 15% auto;
  padding: 20px;
  border: 1px solid #888;
  width: 80%;
  max-width: 800px;
}
.close {
  color: #aaa;
  float: right;
  font-size: 28px;
  font-weight: bold;
}
.close:hover,
.close:focus {
  color: black;
  text-decoration: none;
  cursor: pointer;
}
.metrics-guide h3 {
  margin-top: 20px;
}
.metrics-guide p {
  margin: 5px 0;
}
.metrics-guide ul {
  margin: 10px 0;
  padding-left: 20px;
}
</style>
"""
    modal_js = """
<script>
document.addEventListener("DOMContentLoaded", function() {
  var modal = document.getElementById("metricsHelpModal");
  var openBtn = document.getElementById("openMetricsHelp");
  var span = document.getElementsByClassName("close")[0];
  if (openBtn && modal) {
    openBtn.onclick = function() {
      modal.style.display = "block";
    };
  }
  if (span && modal) {
    span.onclick = function() {
      modal.style.display = "none";
    };
  }
  window.onclick = function(event) {
    if (event.target == modal) {
      modal.style.display = "none";
    }
  }
});
</script>
"""
    return modal_css + modal_html + modal_js


def generate_html_report(config, train_scores, val_scores, test_scores, problem_type, html_dir):
    config_html = generate_config_table_html(config)
    metrics_html = generate_metrics_table_html(train_scores, val_scores, test_scores)
    plots_html = ""
    if problem_type in ["binary", "multiclass"]:
        plots_html += (
            '<h2 style="text-align: center;">Visualizations</h2>'
            '<div style="text-align: center;">'
            '<img src="confusion_matrix.png" alt="Confusion Matrix" '
            'style="max-width:90%; max-height:600px; border:1px solid #ddd; margin-bottom:20px;">'
            '</div>'
        )
        if problem_type == "binary":
            plots_html += (
                '<div style="text-align: center;">'
                '<img src="roc_curve.png" alt="ROC Curve" '
                'style="max-width:90%; max-height:600px; border:1px solid #ddd; margin-bottom:20px;">'
                '</div>'
            )
    elif problem_type == "regression":
        plots_html += (
            '<h2 style="text-align: center;">Visualizations</h2>'
            '<div style="text-align: center;">'
            '<img src="true_vs_pred.png" alt="True vs Predicted" '
            'style="max-width:90%; max-height:600px; border:1px solid #ddd; margin-bottom:20px;">'
            '</div>'
        )
    modal_html = get_metrics_help_modal()
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
    html = f"""
    <html>
    <head>
    <title>Model Report</title>
    </head>
    <body>
    <h1 style='text-align: center;'>Model Report</h1>
    {button_html}
    {config_html}
    {metrics_html}
    {plots_html}
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
        stratify=train_data[args.label_column],
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
    problem_type = predictor.problem_type.lower()
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

    # Get predictions and generate plots
    y_true = test_data[args.label_column]
    y_pred = predictor.predict(test_data)
    html_dir = os.path.dirname(args.output_html) or "."
    os.makedirs(html_dir, exist_ok=True)

    if problem_type in ["binary", "multiclass"]:
        y_prob = predictor.predict_proba(test_data)
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(html_dir, 'confusion_matrix.png'))
        plt.close()
        if problem_type == "binary":
            # ROC Curve
            # y_prob is a DataFrame for autogluon, get positive class prob
            if hasattr(y_prob, "values"):
                y_prob_np = y_prob.values
            else:
                y_prob_np = np.array(y_prob)
            # Try to get positive class for binary
            try:
                if hasattr(y_prob, "columns"):
                    # Try to get the column representing positive class
                    if len(y_prob.columns) == 2:
                        pos_class = y_prob.columns[1]
                        y_score = y_prob[pos_class]
                    else:
                        y_score = y_prob.iloc[:, 1]
                else:
                    y_score = y_prob_np[:, 1]
            except Exception:
                y_score = y_prob_np[:, 1] if y_prob_np.shape[1] > 1 else y_prob_np[:, 0]
            # Ensure y_true is binary (0/1 or class labels), convert if needed
            if not np.issubdtype(y_true.dtype, np.number):
                y_true_bin = pd.factorize(y_true)[0]
            else:
                y_true_bin = y_true
            fpr, tpr, _ = roc_curve(y_true_bin, y_score)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(html_dir, 'roc_curve.png'))
            plt.close()
    elif problem_type == "regression":
        # Scatter Plot
        plt.figure()
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('True vs Predicted')
        plt.savefig(os.path.join(html_dir, 'true_vs_pred.png'))
        plt.close()

    # Extract config
    config = predictor._learner._config

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
    html_report = generate_html_report(
        config, train_scores, val_scores, test_scores, problem_type, html_dir
    )
    with open(args.output_html, "w") as f:
        f.write(html_report)
    print(f"Saved HTML report to '{args.output_html}'")


if __name__ == "__main__":
    main()
