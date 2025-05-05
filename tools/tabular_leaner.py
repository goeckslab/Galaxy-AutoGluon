import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run AutoGluon on tabular data")
    parser.add_argument("--input", required=True, help="Path to input dataset")
    parser.add_argument("--target", required=True, help="Name of the target column")
    args = parser.parse_args()

    # Load the dataset
    df = pd.read_csv(args.input)

    # Step 1: Split into train_val (80%) and test (20%)
    df_train_val, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # Step 2: Split train_val into train (70% of total) and val (10% of total)
    df_train, df_val = train_test_split(df_train_val, test_size=0.125, random_state=42)

    # Train a model using AutoGluon with tuning_data
    predictor = TabularPredictor(label=args.target).fit(train_data=df_train, tuning_data=df_val)

    # Evaluate on test set
    performance = predictor.evaluate(df_test)

    # Write performance metrics to a file
    with open("performance.txt", "w") as f:
        for key, value in performance.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main()
