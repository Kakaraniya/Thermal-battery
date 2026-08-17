import json

import joblib


MODEL_PATH = "models/model.joblib"
METRICS_PATH = "models/metrics.json"

MIN_ACCURACY = 0.80


model = joblib.load(MODEL_PATH)

with open(METRICS_PATH) as file:
    metrics = json.load(file)

accuracy = metrics["accuracy"]

if accuracy < MIN_ACCURACY:
    raise ValueError(
        f"Model accuracy {accuracy:.2f} is below threshold."
    )

if not hasattr(model, "predict"):
    raise ValueError("Invalid model.")

print("Model validation passed.")
print(f"Accuracy: {accuracy:.2f}")