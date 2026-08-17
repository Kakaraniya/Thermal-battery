import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/battery_sensor_data.csv"
MODEL_PATH = "models/model.joblib"
SCALER_PATH = "models/scaler.joblib"
METRICS_PATH = "models/metrics.json"


def train_model():
    data = pd.read_csv(DATA_PATH)

    features = [
        "temperature",
        "voltage",
        "current",
        "coolant_flow_rate"
    ]

    X = data[features]
    y = data["thermal_status"]

    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    with open(METRICS_PATH, "w") as file:
        json.dump({"accuracy": accuracy}, file, indent=4)

    print(f"Model accuracy: {accuracy:.4f}")
    print("Model saved successfully.")


if __name__ == "__main__":
    train_model()