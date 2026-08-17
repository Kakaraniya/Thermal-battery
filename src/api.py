import joblib
from fastapi import FastAPI
from pydantic import BaseModel


model = joblib.load("models/model.joblib")
scaler = joblib.load("models/scaler.joblib")

app = FastAPI()


class BatteryData(BaseModel):
    temperature: float
    voltage: float
    current: float
    coolant_flow_rate: float


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: BatteryData):
    values = [[
        data.temperature,
        data.voltage,
        data.current,
        data.coolant_flow_rate
    ]]

    values = scaler.transform(values)

    prediction = model.predict(values)

    return {
        "thermal_status": int(prediction[0])
    }