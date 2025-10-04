import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame, test_size=0.2, random_state=42):
    # Separate features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Scale 'Time' and 'Amount'
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, scaler
    