import joblib
import numpy as np
from flask import Flask, render_template, request


app = Flask(__name__, template_folder="../templates")

model = joblib.load("models/model.joblib")
scaler = joblib.load("models/scaler.joblib")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        temperature = float(request.form["temperature"])
        voltage = float(request.form["voltage"])
        current = float(request.form["current"])
        coolant_flow = float(request.form["coolant_flow"])

        data = np.array([[
            temperature,
            voltage,
            current,
            coolant_flow
        ]])

        data = scaler.transform(data)
        result = model.predict(data)[0]

        prediction = (
            "High Thermal Condition"
            if result == 1
            else "Normal Thermal Condition"
        )

    return render_template("index.html", prediction=prediction)


@app.route("/health")
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)