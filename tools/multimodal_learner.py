"""
multimodal_learner.py

Galaxy‐compatible wrapper that trains an AutoGluon MultiModal model,
splits train→val, evaluates on train/val/test, writes out a metrics CSV
and an HTML report with tabs, buttons, tables, and plots.
"""

import argparse
import logging
import os
import sys
import tempfile
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import pandas as pd
import numpy as np
import random
import torch

from autogluon.multimodal import MultiModalPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc

from utils import (
    get_html_template,
    get_html_closing,
    build_tabbed_html,
    encode_image_to_base64,
)
from feature_help_modal import get_feature_metrics_help_modal

# setup logging
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
        "--input_csv_train", dest="train_csv", required=True,
        help="Galaxy input train CSV"
    )
    parser.add_argument(
        "--input_csv_test", dest="test_csv", required=True,
        help="Galaxy input test CSV"
    )
    parser.add_argument(
        "--target_column", dest="label_column", required=True,
        help="Name of the target column in the CSVs"
    )
    parser.add_argument(
        "--output_csv", dest="output_csv", required=True,
        help="Galaxy output metrics CSV"
    )
    parser.add_argument(
        "--image_column", dest="image_column", default="filename",
        help="Name of the image‐path column (default: filename)"
    )
    parser.add_argument(
        "--time_limit", dest="time_limit", type=int, default=None,
        help="Time limit for training in seconds (optional)"
    )
    parser.add_argument(
        "--random_seed", dest="random_seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output_json", dest="output_json", default="results.json",
        help="Write full JSON metrics here (not required by Galaxy)"
    )
    parser.add_argument(
        "--output_html", dest="output_html", default="report.html",
        help="Write interactive HTML report here"
    )
    parser.add_argument(
        "--image_folder", dest="image_folder", default=None,
        help="Folder where images were unpacked (defaults to current dir)"
    )
    args = parser.parse_args()

    # --- set seeds
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # --- load CSVs
    try:
        df_train_full = pd.read_csv(args.train_csv)
        df_test      = pd.read_csv(args.test_csv)
    except Exception as e:
        logger.error(f"Failed to read input CSVs: {e}")
        sys.exit(1)

    # --- verify columns
    for df, name in [(df_train_full, "train"), (df_test, "test")]:
        if args.label_column not in df.columns:
            logger.error(f"Missing target column '{args.label_column}' in {name} CSV")
            sys.exit(1)
        if args.image_column not in df.columns:
            logger.error(f"Missing image column '{args.image_column}' in {name} CSV")
            sys.exit(1)

    # --- split train → train/val (80/20 stratified)
    df_train, df_val = train_test_split(
        df_train_full,
        test_size=0.2,
        random_state=args.random_seed,
        stratify=df_train_full[args.label_column],
    )
    logger.info(f"Split: {len(df_train)} train / {len(df_val)} val / {len(df_test)} test")

    # --- expand image paths
    base_folder = args.image_folder or os.getcwd()
    for df in (df_train, df_val, df_test):
        df[args.image_column] = df[args.image_column].apply(
            lambda p: path_expander(str(p), base_folder)
        )

    # --- train
    logger.info("Starting AutoGluon MultiModal training...")
    predictor = MultiModalPredictor(
        label=args.label_column,
        path="AutogluonModels",
        seed=args.random_seed,
    )
    predictor.fit(
        train_data=df_train,
        tuning_data=df_val,
        time_limit=args.time_limit,
        image_column=args.image_column,
    )

    # --- evaluate
    problem = predictor.problem_type.lower()
    metrics_map = {
        "binary":    ["accuracy", "roc_auc", "precision", "recall", "f1"],
        "multiclass": ["accuracy", "f1_macro", "precision_macro", "recall_macro"],
        "regression": ["root_mean_squared_error", "mean_absolute_error", "r2"],
    }
    metrics = metrics_map.get(problem, ["accuracy"])
    train_scores = predictor.evaluate(df_train, metrics=metrics)
    val_scores   = predictor.evaluate(df_val,   metrics=metrics)
    test_scores  = predictor.evaluate(df_test,  metrics=metrics)
    logger.info("Train metrics: %s", train_scores)
    logger.info("Val   metrics: %s", val_scores)
    logger.info("Test  metrics: %s", test_scores)

    # --- write Galaxy CSV
    df_out = pd.DataFrame([
        {"phase": "train",      **train_scores},
        {"phase": "validation", **val_scores},
        {"phase": "test",       **test_scores},
    ])
    df_out.to_csv(args.output_csv, index=False)
    logger.info(f"Wrote metrics CSV → {args.output_csv}")

    # --- (optional) write JSON
    with open(args.output_json, "w") as f:
        json.dump({
            "train": train_scores,
            "val":   val_scores,
            "test":  test_scores,
            "fit_summary": predictor.fit_summary(),
        }, f, indent=2, default=str)
    logger.info(f"Wrote full JSON → {args.output_json}")

    # --- build HTML report
    # 1) config table
    config = {
        "Label Column":      args.label_column,
        "Image Column":      args.image_column,
        "Time Limit (s)":    args.time_limit or "unbounded",
        "Random Seed":       args.random_seed,
        "Problem Type":      problem,
        "AutoGluon Path":    os.path.abspath("AutogluonModels"),
    }
    cfg_rows = "".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>"
        for k, v in config.items()
    )
    summary_html = (
        "<h2>Configuration</h2>"
        "<table>" 
        "<tr><th>Parameter</th><th>Value</th></tr>"
        + cfg_rows +
        "</table>"
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

    # 2) test‐only details + plots
    # confusion matrix
    tmpdir = tempfile.mkdtemp()
    y_true = df_test[args.label_column]
    y_pred = predictor.predict(df_test)
    plots = []
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=sorted(y_true.unique()),
                yticklabels=sorted(y_true.unique()),
                cmap="Blues")
    plt.title("Test Confusion Matrix")
    cm_path = os.path.join(tmpdir, "confusion_matrix.png")
    plt.savefig(cm_path); plt.close()
    plots.append(("Confusion Matrix", cm_path))

    if problem == "binary":
        y_prob = predictor.predict_proba(df_test)
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
        plt.plot([0,1], [0,1], "k--")
        plt.title("Test ROC Curve")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(loc="lower right")
        roc_path = os.path.join(tmpdir, "roc_curve.png")
        plt.savefig(roc_path); plt.close()
        plots.append(("ROC Curve", roc_path))

    test_html = (
        "<h2>Test Metrics Detail</h2>"
        + "".join(f"<p><strong>{k.replace('_',' ').title()}:</strong> {v:.4f}</p>"
                  for k, v in test_scores.items()
                  if isinstance(v, (int, float)))
        + "<h2>Visualizations</h2>"
        + "".join(
            f"<div class='plot'><h3>{title}</h3>"
            f"<img src='data:image/png;base64,{encode_image_to_base64(path)}'/></div>"
            for title, path in plots
        )
    )

    # 3) feature tab placeholder
    feature_html = "<p><em>Feature importance is not available for multimodal models.</em></p>"

    # assemble tabs + help modal
    full_html = (
        get_html_template()
        + "<h1>AutoGluon Multimodal Learner Report</h1>"
        + build_tabbed_html(
            summary_html,
            test_html,
            feature_html,
            explainer_html=get_feature_metrics_help_modal()
        )
        + get_html_closing()
    )

    with open(args.output_html, "w") as f:
        f.write(full_html)
    logger.info(f"Wrote HTML report → {args.output_html}")


if __name__ == "__main__":
    main()
