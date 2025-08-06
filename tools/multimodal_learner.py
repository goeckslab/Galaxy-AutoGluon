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
from autogluon.tabular import TabularPredictor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    log_loss,
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from feature_help_modal import get_metrics_help_modal
from plot_generator import (
    generate_calibration_plot,
    generate_confusion_matrix_plot,
    generate_metric_comparison_bar,
    generate_per_class_metrics_plot,
    generate_pr_curve_plot,
    generate_residual_histogram,
    generate_residual_plot,
    generate_roc_curve_plot,
    generate_scatter_plot,
    generate_shap_summary_plot,
    generate_shap_force_plot,
    generate_shap_waterfall_plot,
)
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


def compute_metrics(y_true, y_pred, y_prob, kind, metrics):
    """Compute specified metrics given y_true, y_pred, y_prob."""
    scores = {}
    for m in metrics:
        if m == "accuracy":
            scores[m] = accuracy_score(y_true, y_pred)
        elif m == "balanced_accuracy":
            scores[m] = balanced_accuracy_score(y_true, y_pred)
        elif m == "f1":
            scores[m] = f1_score(y_true, y_pred, average="binary" if kind == "binary" else "macro")
        elif m == "f1_macro":
            scores[m] = f1_score(y_true, y_pred, average="macro")
        elif m == "f1_micro":
            scores[m] = f1_score(y_true, y_pred, average="micro")
        elif m == "f1_weighted":
            scores[m] = f1_score(y_true, y_pred, average="weighted")
        elif m == "precision":
            scores[m] = precision_score(y_true, y_pred, average="binary" if kind == "binary" else "macro")
        elif m == "precision_macro":
            scores[m] = precision_score(y_true, y_pred, average="macro")
        elif m == "precision_micro":
            scores[m] = precision_score(y_true, y_pred, average="micro")
        elif m == "precision_weighted":
            scores[m] = precision_score(y_true, y_pred, average="weighted")
        elif m == "recall":
            scores[m] = recall_score(y_true, y_pred, average="binary" if kind == "binary" else "macro")
        elif m == "recall_macro":
            scores[m] = recall_score(y_true, y_pred, average="macro")
        elif m == "recall_micro":
            scores[m] = recall_score(y_true, y_pred, average="micro")
        elif m == "recall_weighted":
            scores[m] = recall_score(y_true, y_pred, average="weighted")
        elif m == "roc_auc":
            if kind == "binary":
                scores[m] = roc_auc_score(y_true, y_prob[:, 1])
        elif m == "roc_auc_ovo_macro":
            scores[m] = roc_auc_score(y_true, y_prob, multi_class="ovo", average="macro")
        elif m == "roc_auc_ovr_macro":
            scores[m] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        elif m == "average_precision":
            scores[m] = average_precision_score(y_true, y_prob[:, 1])
        elif m == "mcc":
            scores[m] = matthews_corrcoef(y_true, y_pred)
        elif m == "log_loss":
            scores[m] = log_loss(y_true, y_prob)
        elif m == "root_mean_squared_error":
            scores[m] = mean_squared_error(y_true, y_pred, squared=False)
        elif m == "mean_squared_error":
            scores[m] = mean_squared_error(y_true, y_pred)
        elif m == "mean_absolute_error":
            scores[m] = mean_absolute_error(y_true, y_pred)
        elif m == "median_absolute_error":
            scores[m] = median_absolute_error(y_true, y_pred)
        elif m == "mean_absolute_percentage_error":
            scores[m] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.nan
        elif m == "r2":
            scores[m] = r2_score(y_true, y_pred)
    return scores


def infer_problem_type(predictor, df):
    if isinstance(predictor, TabularPredictor):
        return predictor.problem_type
    else:
        label_col = predictor.label
        if pd.api.types.is_numeric_dtype(df[label_col]):
            # Regression or binary based on values
            unique_vals = df[label_col].dropna().unique()
            if len(unique_vals) == 2:
                return "binary"
            else:
                return "regression"
        else:
            return "multiclass"


def main():
    parser = argparse.ArgumentParser(description="Train & report an AutoGluon model")
    parser.add_argument(
        "--input_csv_train",
        dest="train_csv",
        required=True,
        help="Galaxy input train CSV (or full CSV if no test provided)",
    )
    parser.add_argument(
        "--input_csv_test",
        dest="test_csv",
        default=None,
        help="Galaxy input test CSV (optional; if not provided, split from train_csv)",
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

    # Handle numeric column selections from Galaxy (e.g., '6' as index)
    def map_column_arg(col_arg, df, arg_name):
        if col_arg and col_arg.isdigit():
            col_idx = int(col_arg) - 1  # Galaxy columns start at 1
            if col_idx < 0 or col_idx >= len(df.columns):
                logger.error(
                    f"Invalid {arg_name} index {col_arg} (out of range 1-{len(df.columns)})"
                )
                sys.exit(1)
            mapped_col = df.columns[col_idx]
            logger.info(
                f"Mapped {arg_name} index {col_arg} to column name '{mapped_col}'"
            )
            return mapped_col
        return col_arg

    # Load CSVs and handle splitting
    try:
        if args.test_csv:
            df_train_full = pd.read_csv(args.train_csv)
            df_test = pd.read_csv(args.test_csv)
            # Map columns before accessing them
            args.label_column = map_column_arg(args.label_column, df_train_full, "target_column")
            args.image_column = map_column_arg(args.image_column, df_train_full, "image_column")
            stratify_col = (
                df_train_full[args.label_column]
                if df_train_full[args.label_column].nunique() > 1
                else None
            )
            df_train, df_val = train_test_split(
                df_train_full,
                test_size=0.2,
                random_state=args.random_seed,
                stratify=stratify_col,
            )
        else:
            df_full = pd.read_csv(args.train_csv)
            # Map columns before accessing them
            args.label_column = map_column_arg(args.label_column, df_full, "target_column")
            args.image_column = map_column_arg(args.image_column, df_full, "image_column")
            stratify_col = (
                df_full[args.label_column] if df_full[args.label_column].nunique() > 1 else None
            )
            df_train_temp, df_test = train_test_split(
                df_full,
                test_size=0.2,
                random_state=args.random_seed,
                stratify=stratify_col,
            )
            stratify_col = (
                df_train_temp[args.label_column]
                if df_train_temp[args.label_column].nunique() > 1
                else None
            )
            df_train, df_val = train_test_split(
                df_train_temp,
                test_size=0.125,
                random_state=args.random_seed,
                stratify=stratify_col,
            )
            df_train_full = df_train_temp  # For consistency in later code if needed
    except Exception as e:
        logger.error(f"Failed to read input CSVs: {e}")
        sys.exit(1)

    # Verify columns
    for df, name in [(df_train_full, "train"), (df_test, "test")]:
        if args.label_column not in df.columns:
            logger.error(f"Missing target column '{args.label_column}' in {name} CSV")
            sys.exit(1)
        if args.image_column and args.image_column not in df.columns:
            logger.error(f"Missing image column '{args.image_column}' in {name} CSV")
            sys.exit(1)
    logger.info(
        f"Split: {len(df_train)} train / {len(df_val)} val / {len(df_test)} test"
    )

    # Expand image paths if using images
    if args.image_column:
        for df in (df_train, df_val, df_test):
            df[args.image_column] = df[args.image_column].apply(
                lambda p: path_expander(str(p), base_folder)
            )

    # Prepare column_types if images
    column_types = {args.image_column: "image_path"} if args.image_column else None

    # Train
    if args.image_column:
        logger.info("Starting AutoGluon MultiModal training...")
        predictor = MultiModalPredictor(
            label=args.label_column,
            path=None,
        )
        predictor.fit(
            train_data=df_train,
            tuning_data=df_val,
            time_limit=args.time_limit,
            column_types=column_types,
        )
    else:
        logger.info("Starting AutoGluon Tabular training...")
        predictor = TabularPredictor(
            label=args.label_column,
            path=None,
        )
        predictor.fit(
            train_data=df_train,
            tuning_data=df_val,
            time_limit=args.time_limit,
        )

    # Evaluate
    kind = infer_problem_type(predictor, df_train_full)
    metrics_map = {
        "binary": ["accuracy", "f1", "roc_auc", "log_loss"],
        "multiclass": ["accuracy", "log_loss"],
        "regression": ["root_mean_squared_error", "r2", "mean_absolute_error"]
    }
    metrics = metrics_map.get(kind, ["accuracy"])

    def evaluate_phase(df):
        y_true = df[args.label_column]
        y_pred = predictor.predict(df)
        y_prob = predictor.predict_proba(df).values if kind != "regression" else None
        return compute_metrics(y_true, y_pred, y_prob, kind, metrics)

    train_scores = evaluate_phase(df_train)
    val_scores = evaluate_phase(df_val)
    test_scores = evaluate_phase(df_test)
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
        "Problem Type": kind,
        "AutoGluon Path": os.path.abspath(predictor.path),
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

    # Prepare for plots
    tmpdir = tempfile.mkdtemp()
    y_true = df_test[args.label_column]
    y_pred = predictor.predict(df_test)
    # Collect scores for metric comparison plot
    common_metrics = set(train_scores) & set(val_scores) & set(test_scores)
    scores_for_plot = {
        m: [train_scores[m], val_scores[m], test_scores[m]]
        for m in common_metrics
        if isinstance(train_scores[m], (int, float))
    }
    metric_bar_path = os.path.join(tmpdir, "metric_comparison.png")
    generate_metric_comparison_bar(scores_for_plot, path=metric_bar_path)
    summary_plots = [("Metric Comparison", metric_bar_path)]
    # Add visualizations to summary_html
    summary_html += (
        "<h2>Visualizations</h2>"
        + "".join(
            f"<div class='plot'><h3>{title}</h3>"
            f"<img src='data:image/png;base64,{encode_image_to_base64(path)}'/></div>"
            for title, path in summary_plots
        )
    )

    # 2) test-only details + plots
    test_plots = []
    if kind in ["binary", "multiclass"]:
        classes = predictor.class_labels
        cm_path = os.path.join(tmpdir, "confusion_matrix.png")
        generate_confusion_matrix_plot(
            y_true, y_pred, classes=classes, title="Test Confusion Matrix", path=cm_path
        )
        test_plots.append(("Confusion Matrix", cm_path))
        per_class_path = os.path.join(tmpdir, "per_class_metrics.png")
        generate_per_class_metrics_plot(
            y_true, y_pred, title="Test Per-Class Metrics", path=per_class_path
        )
        test_plots.append(("Per-Class Metrics", per_class_path))
        if kind == "binary":
            positive_class = classes[1]
            y_prob = predictor.predict_proba(df_test)[positive_class].values
            y_true_bin = (y_true == positive_class).astype(int)
            roc_path = os.path.join(tmpdir, "roc_curve.png")
            generate_roc_curve_plot(
                y_true_bin, y_prob, title="Test ROC Curve", path=roc_path
            )
            test_plots.append(("ROC Curve", roc_path))
            pr_path = os.path.join(tmpdir, "pr_curve.png")
            generate_pr_curve_plot(
                y_true_bin, y_prob, title="Test Precision-Recall Curve", path=pr_path
            )
            test_plots.append(("Precision-Recall Curve", pr_path))
            cal_path = os.path.join(tmpdir, "calibration_plot.png")
            generate_calibration_plot(
                y_true_bin, y_prob, title="Test Calibration Plot", path=cal_path
            )
            test_plots.append(("Calibration Plot", cal_path))
    elif kind == "regression":
        scatter_path = os.path.join(tmpdir, "scatter_plot.png")
        generate_scatter_plot(
            y_true, y_pred, title="Test Predicted vs Actual", path=scatter_path
        )
        test_plots.append(("Predicted vs Actual", scatter_path))
        residual_path = os.path.join(tmpdir, "residual_plot.png")
        generate_residual_plot(
            y_true, y_pred, title="Test Residual Plot", path=residual_path
        )
        test_plots.append(("Residual Plot", residual_path))
        hist_path = os.path.join(tmpdir, "residual_histogram.png")
        generate_residual_histogram(
            y_true, y_pred, title="Test Residual Histogram", path=hist_path
        )
        test_plots.append(("Residual Histogram", hist_path))
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
            for title, path in test_plots
        )
    )

    # 3) feature tab
    feature_html = "<p><em>Feature importance is not available for this model.</em></p>"
    if isinstance(predictor, TabularPredictor):
        try:
            fi = predictor.feature_importance(
                df_test,
                subsample_size=min(1000, len(df_test)),
                num_shuffle_sets=10,
                silent=True,
            )
            plt.figure(figsize=(8, max(4, len(fi) * 0.4)))
            sns.barplot(x=fi["importance"].values, y=fi.index, xerr=fi["stddev"].values)
            plt.title("Feature Importance (Permutation)")
            plt.xlabel("Importance")
            plt.ylabel("Features")
            fi_path = os.path.join(tmpdir, "feature_importance.png")
            plt.savefig(fi_path, bbox_inches="tight")
            plt.close()
            feature_html = (
                "<h2>Feature Importance</h2>"
                "<div class='plot'>"
                f"<img src='data:image/png;base64,{encode_image_to_base64(fi_path)}'/>"
                "</div>"
            )
        except Exception as e:
            logger.warning(f"Feature importance failed: {e}")
    else:
        feature_plots = []
        try:
            import shap

            if args.image_column:
                raise ValueError("SHAP not supported for models with image data")
            df_shap = df_test.drop(columns=[args.label_column]).sample(
                min(50, len(df_test)), random_state=args.random_seed
            )

            def model_func(df):
                if kind == "regression":
                    return predictor.predict(df).values
                else:
                    return predictor.predict_proba(df).values

            explainer = shap.Explainer(model_func, df_shap)
            shap_values = explainer(df_shap)
            summary_path = os.path.join(tmpdir, "shap_summary.png")
            generate_shap_summary_plot(shap_values, df_shap, path=summary_path)
            feature_plots.append(("SHAP Summary", summary_path))
            instance = df_shap.iloc[0:1]
            force_path = os.path.join(tmpdir, "shap_force.png")
            generate_shap_force_plot(explainer, instance, path=force_path)
            feature_plots.append(("SHAP Force Plot (Sample)", force_path))
            waterfall_path = os.path.join(tmpdir, "shap_waterfall.png")
            generate_shap_waterfall_plot(explainer, instance, path=waterfall_path)
            feature_plots.append(("SHAP Waterfall Plot (Sample)", waterfall_path))
            feature_html = (
                "<h2>Feature Importance via SHAP</h2>"
                + "".join(
                    f"<div class='plot'><h3>{title}</h3>"
                    f"<img src='data:image/png;base64,{encode_image_to_base64(path)}'/></div>"
                    for title, path in feature_plots
                )
            )
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")

    # Assemble tabs + help modal (placed outside tabs for proper display)
    full_html = (
        get_html_template()
        + "<h1>AutoGluon Learner Report</h1>"
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
