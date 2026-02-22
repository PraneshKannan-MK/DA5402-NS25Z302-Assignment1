import os
import socket
import joblib
import pandas as pd
from datetime import datetime
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.utils import load_config


app = FastAPI()

# Load config
config = load_config()

model_path = config["deployment"]["model_path"]
port = config["deployment"]["port"]

# Load model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

model = joblib.load(model_path)


# Request schema
class PredictionRequest(BaseModel):
    features: List[float]


# Log deployment when API starts (better practice)
@app.on_event("startup")
def log_deployment():
    log_file = "deployment_log.csv"

    timestamp = datetime.now()
    hostname = socket.gethostname()
    model_version = config["data"]["current_version"]

    with open(log_file, "a") as f:
        f.write(f"{timestamp},{model_version},{port},{hostname}\n")

    print("Deployment logged.")


# Prediction endpoint
@app.post("/predict")
def predict(payload: PredictionRequest):

    features = payload.features

    # Validate feature length
    expected_features = model.n_features_in_

    if len(features) != expected_features:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_features} features but got {len(features)}"
        )

    # Convert to dataframe
    df = pd.DataFrame([features])

    try:
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0].max()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }