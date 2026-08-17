import joblib
import numpy as np


def test_model_exists():
    model = joblib.load("models/model.joblib")
    assert model is not None


def test_model_can_predict():
    model = joblib.load("models/model.joblib")
    scaler = joblib.load("models/scaler.joblib")

    sample = np.array([
        [40, 3.7, 3.0, 4.0]
    ])

    sample = scaler.transform(sample)

    prediction = model.predict(sample)

    assert prediction[0] in [0, 1]