import argparse
import json
import logging
import os
import sys
import tempfile
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import torch
from autogluon.multimodal import MultiModalPredictor
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

from feature_help_modal import get_metrics_help_modal
from utils import (
    build_tabbed_html,
    encode_image_to_base64,
    get_html_closing,
    get_html_template,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def path_expander(p: str, base_folder: str) -> str:
    """Make relative paths absolute against base_folder."""
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(base_folder, p))


def main():
    parser = argparse.ArgumentParser(description="Train & report an AutoGluon Multimodal model")
    parser.add_argument(
        "--input_csv_train",
        dest="train_csv",
        required=True,
        help="Galaxy input train CSV",
    )
    parser.add_argument(
        "--input_csv_test",
        dest="test_csv",
        required=True,
        help="Galaxy input test CSV",
    )
    parser.add_argument(
        "--target_column",
        dest="label_column",
        required=True,
        help="Name of the target column in the CSVs",
    )
    parser.add_argument(
        "--output_csv",
        dest="output_csv",
        required=True,
        help="Galaxy output metrics CSV",
    )
    parser.add_argument(
        "--image_column",
        dest="image_column",
        default=None,
        help="Name of the image-path column (optional, required if using images)",
    )
    parser.add_argument(
        "--images_zip",
        dest="images_zip",
        default=None,
        help="Optional ZIP file containing images",
    )
    parser.add_argument(
        "--time_limit",
        dest="time_limit",
        type=int,
        default=None,
        help="Time limit for training in seconds (optional)",
    )
    parser.add_argument(
        "--random_seed",
        dest="random_seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output_json",
        dest="output_json",
        default="results.json",
        help="Write full JSON metrics here (not required by Galaxy)",
    )
    parser.add_argument(
        "--output_html",
        dest="output_html",
        default="report.html",
        help="Write interactive HTML report here",
    )
    parser.add_argument(
        "--image_folder",
        dest="image_folder",
        default=None,
        help="Folder where images were unpacked (defaults to current dir if no ZIP)",
    )
    args = parser.parse_args()

    # Set seeds for reproducibility
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Handle images ZIP if provided
    base_folder = args.image_folder or os.getcwd()
    if args.images_zip and os.path.isfile(args.images_zip):
        if not args.image_column:
            logger.warning("Images ZIP provided but no image_column specified; ignoring ZIP.")
        else:
            extract_dir = tempfile.mkdtemp()
            try:
                with zipfile.ZipFile(args.images_zip, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
                base_folder = extract_dir
                logger.info(f"Extracted images ZIP to {extract_dir}")
            except Exception as e:
                logger.error(f"Failed to extract images ZIP: {e}")
                sys.exit(1)

    # Load CSVs
    try:
        df_train_full = pd.read_csv(args.train_csv)
        df_test = pd.read_csv(args.test_csv)
    except Exception as e:
        logger.error(f"Failed to read input CSVs: {e}")
        sys.exit(1)

    # Handle numeric column selections from Galaxy (e.g., '6' as index)
    def map_column_arg(col_arg, df, arg_name):
        if col_arg and col_arg.isdigit():
            col_idx = int(col_arg) - 1  # Galaxy columns start at 1
            if col_idx < 0 or col_idx >= len(df.columns):
                logger.error(f"Invalid {arg_name} index {col_arg} (out of range 1-{len(df.columns)})")
                sys.exit(1)
            mapped_col = df.columns[col_idx]
            logger.info(f"Mapped {arg_name} index {col_arg} to column name '{mapped_col}'")
            return mapped_col
        return col_arg

    args.label_column = map_column_arg(args.label_column, df_train_full, "target_column")
    args.image_column = map_column_arg(args.image_column, df_train_full, "image_column")

    # Verify columns
    for df, name in [(df_train_full, "train"), (df_test, "test")]:
        if args.label_column not in df.columns:
            logger.error(f"Missing target column '{args.label_column}' in {name} CSV")
            sys.exit(1)
        if args.image_column and args.image_column not in df.columns:
            logger.error(f"Missing image column '{args.image_column}' in {name} CSV")
            sys.exit(1)

    # Split train → train/val (80/20 stratified)
    stratify_col = df_train_full[args.label_column] if df_train_full[args.label_column].nunique() > 1 else None
    df_train, df_val = train_test_split(
        df_train_full,
        test_size=0.2,
        random_state=args.random_seed,
        stratify=stratify_col,
    )
    logger.info(f"Split: {len(df_train)} train / {len(df_val)} val / {len(df_test)} test")

    # Expand image paths if using images
    if args.image_column:
        for df in (df_train, df_val, df_test):
            df[args.image_column] = df[args.image_column].apply(lambda p: path_expander(str(p), base_folder))

    # Prepare column_types if images
    column_types = {args.image_column: "image_path"} if args.image_column else None

    # Train
    logger.info("Starting AutoGluon MultiModal training...")
    predictor = MultiModalPredictor(
        label=args.label_column,
        path="AutogluonModels",
    )
    predictor.fit(
        train_data=df_train,
        tuning_data=df_val,
        time_limit=args.time_limit,
        column_types=column_types,
    )

    # Evaluate
    problem_type = predictor.problem_type
    if problem_type == "regression":
        kind = "regression"
    elif problem_type == "binary":
        kind = "binary"
    elif problem_type == "multiclass":
        kind = "multiclass"
    else:
        kind = "other"
    metrics_map = {
        "binary": ["accuracy", "roc_auc", "precision", "recall", "f1"],
        "multiclass": ["accuracy", "f1_macro", "precision_macro", "recall_macro"],
        "regression": ["root_mean_squared_error", "mean_absolute_error", "r2"],
    }
    metrics = metrics_map.get(kind, ["accuracy"])
    train_scores = predictor.evaluate(df_train, metrics=metrics)
    val_scores = predictor.evaluate(df_val, metrics=metrics)
    test_scores = predictor.evaluate(df_test, metrics=metrics)
    logger.info("Train metrics: %s", train_scores)
    logger.info("Val metrics: %s", val_scores)
    logger.info("Test metrics: %s", test_scores)

    # Write Galaxy CSV
    df_out = pd.DataFrame(
        [
            {"phase": "train", **train_scores},
            {"phase": "validation", **val_scores},
            {"phase": "test", **test_scores},
        ]
    )
    df_out.to_csv(args.output_csv, index=False)
    logger.info(f"Wrote metrics CSV → {args.output_csv}")

    # (optional) write JSON
    with open(args.output_json, "w") as f:
        json.dump(
            {
                "train": train_scores,
                "val": val_scores,
                "test": test_scores,
                "fit_summary": predictor.fit_summary(),
            },
            f,
            indent=2,
            default=str,
        )
    logger.info(f"Wrote full JSON → {args.output_json}")

    # Build HTML report
    # 1) config table
    config = {
        "Label Column": args.label_column,
        "Image Column": args.image_column or "N/A (no images used)",
        "Time Limit (s)": args.time_limit or "unbounded",
        "Random Seed": args.random_seed,
        "Problem Type": problem_type,
        "AutoGluon Path": os.path.abspath("AutogluonModels"),
    }
    cfg_rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in config.items())
    summary_html = (
        "<h2>Configuration</h2>"
        "<table>"
        "<tr><th>Parameter</th><th>Value</th></tr>"
        + cfg_rows
        + "</table>"
        "<h2>Performance Summary</h2>"
        "<table class='table sortable'>"
        "<tr><th>Metric</th><th>Train</th><th>Val</th><th>Test</th></tr>"
        + "".join(
            f"<tr>"
            f"<td>{m.replace('_',' ').title()}</td>"
            f"<td>{train_scores[m]:.4f}</td>"
            f"<td>{val_scores[m]:.4f}</td>"
            f"<td>{test_scores[m]:.4f}</td>"
            f"</tr>"
            for m in sorted(set(train_scores) & set(val_scores) & set(test_scores))
            if isinstance(train_scores[m], (int, float))
        )
        + "</table>"
    )

    # 2) test-only details + plots
    tmpdir = tempfile.mkdtemp()
    y_true = df_test[args.label_column]
    y_pred = predictor.predict(df_test)
    plots = []
    if kind in ["binary", "multiclass"]:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=sorted(y_true.unique()),
            yticklabels=sorted(y_true.unique()),
            cmap="Blues",
        )
        plt.title("Test Confusion Matrix")
        cm_path = os.path.join(tmpdir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        plots.append(("Confusion Matrix", cm_path))

        if kind == "binary":
            classes = predictor.class_labels
            positive_class = classes[1]  # Assume second class is positive
            y_prob = predictor.predict_proba(df_test)[positive_class]
            fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=positive_class)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], "k--")
            plt.title("Test ROC Curve")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.legend(loc="lower right")
            roc_path = os.path.join(tmpdir, "roc_curve.png")
            plt.savefig(roc_path)
            plt.close()
            plots.append(("ROC Curve", roc_path))
    elif kind == "regression":
        plt.figure(figsize=(6, 5))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot(
            [min(y_true), max(y_true)],
            [min(y_true), max(y_true)],
            "r--",
            label="Perfect Fit",
        )
        plt.title("Predicted vs Actual (Test)")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.legend()
        scatter_path = os.path.join(tmpdir, "scatter_plot.png")
        plt.savefig(scatter_path)
        plt.close()
        plots.append(("Predicted vs Actual", scatter_path))

    test_html = (
        "<h2>Test Metrics Detail</h2>"
        + "".join(
            f"<p><strong>{k.replace('_',' ').title()}:</strong> {v:.4f}</p>"
            for k, v in test_scores.items()
            if isinstance(v, (int, float))
        )
        + "<h2>Visualizations</h2>"
        + "".join(
            f"<div class='plot'><h3>{title}</h3>"
            f"<img src='data:image/png;base64,{encode_image_to_base64(path)}'/></div>"
            for title, path in plots
        )
    )

    # 3) feature tab placeholder
    feature_html = "<p><em>Feature importance is not available for multimodal models.</em></p>"

    # Assemble tabs + help modal (placed outside tabs for proper display)
    full_html = (
        get_html_template()
        + "<h1>AutoGluon Multimodal Learner Report</h1>"
        + build_tabbed_html(
            summary_html,
            test_html,
            feature_html,
            explainer_html=None,  # No explainer plots; help modal separate
        )
        + get_metrics_help_modal()
        + get_html_closing()
    )

    with open(args.output_html, "w") as f:
        f.write(full_html)
    logger.info(f"Wrote HTML report → {args.output_html}")


if __name__ == "__main__":
    main()
