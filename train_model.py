import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
import argparse
import os


def train_model(data_path, model_out):
    # Load dataset
    print(f"[INFO] Loading dataset from {data_path} ...")
    df = pd.read_csv(data_path)

    # Features and labels
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    # Train logistic regression
    print("[INFO] Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\n[RESULTS] Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save model + scaler
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, model_out)
    print(f"[INFO] Model saved to {model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--model_out", type=str, required=True, help="Path to save model")
    args = parser.parse_args()

    train_model(args.data_path, args.model_out)
