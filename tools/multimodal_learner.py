import pandas as pd
from autogluon.multimodal import MultiModalPredictor
import argparse
from sklearn.metrics import (
    accuracy_score, log_loss, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a MultiModalPredictor and evaluate metrics")
    parser.add_argument('--input_csv_train', required=True, help="Path to the training CSV file")
    parser.add_argument('--input_csv_test', required=True, help="Path to the test CSV file")
    parser.add_argument('--target_column', required=True, help="Name of the target column")
    parser.add_argument('--output_csv', required=True, help="Path to save the metrics CSV")
    args = parser.parse_args()

    # Load datasets
    print("Loading datasets...")
    train_df = pd.read_csv(args.input_csv_train)
    test_df = pd.read_csv(args.input_csv_test)

    # Initialize and train the model
    print("Initializing MultiModalPredictor...")
    predictor = MultiModalPredictor(label=args.target_column)
    print("Training the model...")
    predictor.fit(train_data=train_df)
    print("Training completed.")

    # Get the problem type
    problem_type = predictor.problem_type
    print(f"Detected problem type: {problem_type}")

    # Extract true labels
    train_true = train_df[args.target_column]
    test_true = test_df[args.target_column]

    # Handle classification tasks
    if problem_type in ['binary', 'multiclass']:
        # Predict labels
        train_pred_labels = predictor.predict(train_df)
        test_pred_labels = predictor.predict(test_df)

        # Predict probabilities
        if problem_type == 'binary':
            train_pred_probs = predictor.predict_proba(train_df)[:, 1]  # Probability of positive class
            test_pred_probs = predictor.predict_proba(test_df)[:, 1]
        else:  # multiclass
            train_pred_probs = predictor.predict_proba(train_df)  # Full probability array
            test_pred_probs = predictor.predict_proba(test_df)

        # Common classification metrics
        train_accuracy = accuracy_score(train_true, train_pred_labels)
        test_accuracy = accuracy_score(test_true, test_pred_labels)
        train_log_loss = log_loss(train_true, train_pred_probs)
        test_log_loss = log_loss(test_true, test_pred_probs)

        if problem_type == 'binary':
            # Binary-specific metrics
            train_precision = precision_score(train_true, train_pred_labels)
            test_precision = precision_score(test_true, test_pred_labels)
            train_recall = recall_score(train_true, train_pred_labels)
            test_recall = recall_score(test_true, test_pred_labels)
            train_f1 = f1_score(train_true, train_pred_labels)
            test_f1 = f1_score(test_true, test_pred_labels)
            train_roc_auc = roc_auc_score(train_true, train_pred_probs)
            test_roc_auc = roc_auc_score(test_true, test_pred_probs)
            train_specificity = recall_score(train_true, train_pred_labels, pos_label=0)
            test_specificity = recall_score(test_true, test_pred_labels, pos_label=0)
        else:  # multiclass
            # Multi-class metrics with macro averaging
            train_precision = precision_score(train_true, train_pred_labels, average='macro')
            test_precision = precision_score(test_true, test_pred_labels, average='macro')
            train_recall = recall_score(train_true, train_pred_labels, average='macro')
            test_recall = recall_score(test_true, test_pred_labels, average='macro')
            train_f1 = f1_score(train_true, train_pred_labels, average='macro')
            test_f1 = f1_score(test_true, test_pred_labels, average='macro')
            # Skip ROC-AUC and specificity for simplicity
            train_roc_auc = None
            test_roc_auc = None
            train_specificity = None
            test_specificity = None

        # Compile classification metrics
        metrics_df = pd.DataFrame({
            'metrics': ['accuracy', 'loss', 'precision', 'recall', 'f1', 'roc_auc', 'specificity'],
            'train': [train_accuracy, train_log_loss, train_precision, train_recall, train_f1, train_roc_auc, train_specificity],
            'test': [test_accuracy, test_log_loss, test_precision, test_recall, test_f1, test_roc_auc, test_specificity]
        })

    # Handle regression tasks
    elif problem_type == 'regression':
        # Predict continuous values
        train_pred = predictor.predict(train_df)
        test_pred = predictor.predict(test_df)

        # Regression metrics
        train_mse = mean_squared_error(train_true, train_pred)
        test_mse = mean_squared_error(test_true, test_pred)
        train_mae = mean_absolute_error(train_true, train_pred)
        test_mae = mean_absolute_error(test_true, test_pred)
        train_r2 = r2_score(train_true, train_pred)
        test_r2 = r2_score(test_true, test_pred)

        # Compile regression metrics
        metrics_df = pd.DataFrame({
            'metrics': ['mse', 'mae', 'r2'],
            'train': [train_mse, train_mae, train_r2],
            'test': [test_mse, test_mae, test_r2]
        })

    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    # Save metrics to CSV
    metrics_df.to_csv(args.output_csv, index=False)
    print(f"Metrics saved to {args.output_csv}")


if __name__ == "__MAIN__":
    main()
