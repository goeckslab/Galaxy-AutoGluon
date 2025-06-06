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

# Suppress warnings
warnings.filterwarnings("ignore")


def path_expander(path, base_folder):
    """Expand relative image paths into absolute paths."""
    path = path.lstrip("/")
    return os.path.abspath(os.path.join(base_folder, path))


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate AutoGluon MultiModal \
                     on an image dataset"
    )
    parser.add_argument(
        "--train_csv",
        required=True,
        help="Path to training metadata CSV (must include an image-ID column \
              and a label column).",
    )
    parser.add_argument(
        "--test_csv",
        required=True,
        help="Path to test metadata CSV (must include the same \
              image-ID column).",
    )
    parser.add_argument(
        "--label_column",
        default="dx",
        help="Name of the target/label column in your CSVs (default: dx).",
    )
    parser.add_argument(
        "--image_column",
        default="image_id",
        help="Name of the column that holds the image filename/ID \
              (default: image_id).",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=None,
        help="Optional: wall-clock time (in seconds) to limit training. \
              If omitted, no time limit is passed.",
    )
    parser.add_argument(
        "--output_json",
        default="results.json",
        help="Output JSON file where metrics + config will be saved.",
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

    # 1) Read CSVs (do NOT use index_col=0 so all columns remain)
    try:
        train_data = pd.read_csv(args.train_csv)
    except Exception as e:
        print(
            f"ERROR reading train CSV at {args.train_csv}: {e}",
            file=sys.stderr
        )
        sys.exit(1)

    try:
        test_data = pd.read_csv(args.test_csv)
    except Exception as e:
        print(
            f"ERROR reading test CSV at {args.test_csv}: {e}",
            file=sys.stderr
        )
        sys.exit(1)

    # 2) Verify that image_column actually exists
    if args.image_column not in train_data.columns:
        print(
            f"ERROR: The specified image_column '{args.image_column}' \
            does not exist in train CSV.",
            file=sys.stderr,
        )
        print(
            f"Available columns in {args.train_csv}:\n  "
            + ", ".join(train_data.columns),
            file=sys.stderr,
        )
        sys.exit(1)

    if args.image_column not in test_data.columns:
        print(
            f"ERROR: The specified image_column '{args.image_column}' \
            does not exist in test CSV.",
            file=sys.stderr,
        )
        print(
            f"Available columns in {args.test_csv}:\n  " +
            ", ".join(test_data.columns),
            file=sys.stderr,
        )
        sys.exit(1)

    # 3) Split train_data into train and validation sets
    train, val = train_test_split(
        train_data,
        test_size=0.2,
        random_state=args.random_seed,
        stratify=train_data[args.label_column],
    )

    # 4) Expand relative image paths into absolute paths
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

    # 5) Create the predictor
    predictor = MultiModalPredictor(label=args.label_column)

    # 6) Fit with or without time_limit
    if args.time_limit is not None:
        predictor.fit(train, time_limit=args.time_limit)
    else:
        predictor.fit(train)

    # 7) Define comprehensive metrics based on problem type
    problem_type = predictor.problem_type.lower()
    if problem_type == "binary":
        base_metrics = [
            "accuracy",
            "balanced_accuracy",
            "f1",
            "f1_macro",
            "f1_micro",
            "f1_weighted",
            "precision",
            "precision_macro",
            "precision_micro",
            "precision_weighted",
            "recall",
            "recall_macro",
            "recall_micro",
            "recall_weighted",
            "log_loss",
            "roc_auc",
            "average_precision",
            "mcc",
            "quadratic_kappa",
        ]
    elif problem_type == "multiclass":
        base_metrics = [
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "f1_micro",
            "f1_weighted",
            "precision_macro",
            "precision_micro",
            "precision_weighted",
            "recall_macro",
            "recall_micro",
            "recall_weighted",
            "log_loss",
            "roc_auc_ovr",
            "roc_auc_ovo",
            "mcc",
            "quadratic_kappa",
        ]
    else:  # For regression or other types
        base_metrics = ["root_mean_squared_error", "mean_absolute_error", "r2"]

    # 8) Evaluate on training, validation, and test sets
    train_scores = predictor.evaluate(train, metrics=base_metrics)
    val_scores = predictor.evaluate(val, metrics=base_metrics)
    test_scores = predictor.evaluate(test_data, metrics=base_metrics)

    # 9) Extract the learner’s internal config (all hyperparameters)
    config = predictor._learner._config

    # 10) Write everything out as JSON
    output = {
        "train_metrics": train_scores,
        "validation_metrics": val_scores,
        "test_metrics": test_scores,
        "config": config,
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2, default=lambda o: str(o))
    print(f"Saved results + config to '{args.output_json}'")


if __name__ == "__main__":
    main()
