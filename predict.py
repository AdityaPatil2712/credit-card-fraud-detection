import joblib
import pandas as pd


def load_model(path="models/model.joblib"):
    """
    Load trained model and scaler from joblib file.
    """
    data = joblib.load(path)
    return data["model"], data["scaler"]


def predict_single(sample: dict, model, scaler):
    """
    Predict fraud probability for a single transaction.
    sample: dict with keys = all feature names (Time, V1..V28, Amount)
    """
    # Training feature order (must match training pipeline exactly!)
    feature_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    # Create DataFrame in correct order
    df = pd.DataFrame([sample], columns=feature_order)

    # Scale ALL features exactly like training
    df_scaled = scaler.transform(df)

    # Prediction
    proba = model.predict_proba(df_scaled)[:, 1]
    pred = model.predict(df_scaled)
    return int(pred[0]), float(proba[0])


if __name__ == "__main__":
    # Load model + scaler
    model, scaler = load_model()

    # Example sample transaction
    sample = {
        "Time": 1000,
        "Amount": 149.62,
    }

    # Fill in dummy values for V1..V28 (required!)
    for i in range(1, 29):
        sample[f"V{i}"] = 0.0

    # Run prediction
    pred, prob = predict_single(sample, model, scaler)
    print(f"[PREDICTION] Class={pred}, Fraud probability={prob:.4f}")
