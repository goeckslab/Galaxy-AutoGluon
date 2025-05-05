from autogluon.vision import ImageClassification as task
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os
from sklearn.metrics import accuracy_score

def get_image_df(data_dir):
    image_list = []
    label_list = []
    for class_folder in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_folder)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(class_path, file)
                    image_list.append(image_path)
                    label_list.append(class_folder)
    df = pd.DataFrame({'image': image_list, 'label': label_list})
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True, help="Path to training data folder")
    args = parser.parse_args()

    # Step 1: Create DataFrame from folder
    df = get_image_df(args.train_data)

    # Step 2: Split into train_val (80%) and test (20%)
    df_train_val, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # Step 3: Split train_val into train (70% of total) and val (10% of total)
    df_train, df_val = train_test_split(df_train_val, test_size=0.125, random_state=42)

    # Step 4: Train the model with tuning_data
    classifier = task.fit(train_data=df_train, tuning_data=df_val, time_limits=60*5, verbose=False)

    # Step 5: Predict on test set
    predictions = classifier.predict(df_test['image'])

    # Step 6: Compute accuracy
    true_labels = df_test['label']
    accuracy = accuracy_score(true_labels, predictions)

    # Step 7: Write to performance.txt
    with open("performance.txt", "w") as f:
        f.write(f"Test Accuracy: {accuracy}\n")

if __name__ == "__main__":
    main()
