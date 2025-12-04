#!/usr/bin/env python3
"""
api_fastapi.py
Simple FastAPI server that serves the TFET predictor.
Run:
    uvicorn api_fastapi:app --reload --port 8000
POST /predict_curve with JSON:
{
  "Lov":4.0, "tn":2.0, "tox":2.5, "WK":4.25,
  "VG_start":0.0, "VG_end":0.6, "points":100
}
Response:
  {"VG":[...],"ID":[...]}
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np
import uvicorn

from tfet_feature_engineer import register_pickle_aliases

register_pickle_aliases("main", "__main__")

app = FastAPI(title="Line-TFET ML Predictor API (simple)")

# Load latest artifacts (train_pipeline.py saves tfet_latest_*)
MODEL_PATH = "saved/tfet_latest_rf.pkl"
SCALER_PATH = "saved/tfet_latest_scaler.pkl"
FE_PATH = "saved/tfet_latest_fe.pkl"

# lazy load
_model = None
_scaler = None
_fe = None

def load_artifacts():
    global _model, _scaler, _fe
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
    if _fe is None:
        _fe = joblib.load(FE_PATH)
    return _model, _scaler, _fe

class DeviceInput(BaseModel):
    Lov: float
    tn: float
    tox: float
    WK: float
    VG_start: float = 0.0
    VG_end: float = 0.6
    points: int = 100

@app.get("/")
def root():
    return {"msg": "Line-TFET Predictor API. POST /predict_curve"}

@app.post("/predict_curve")
def predict_curve(inp: DeviceInput):
    model, scaler, fe = load_artifacts()
    VG_vals = np.linspace(inp.VG_start, inp.VG_end, inp.points)
    data = np.column_stack([
        np.full(inp.points, inp.Lov),
        np.full(inp.points, inp.tn),
        np.full(inp.points, inp.tox),
        np.full(inp.points, inp.WK),
        VG_vals
    ])
    data_fe = fe.transform(data)
    data_scaled = scaler.transform(data_fe)
    ylog_pred = model.predict(data_scaled)
    y_pred = np.exp(ylog_pred)
    return {"VG": VG_vals.tolist(), "ID": y_pred.tolist()}

# run as `python api_fastapi.py` optionally
if __name__ == "__main__":
    print("Starting API (uvicorn)...")
    uvicorn.run("api_fastapi:app", host="0.0.0.0", port=8000, reload=True)
