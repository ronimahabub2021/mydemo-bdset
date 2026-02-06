from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

# ==============================
# FastAPI app
# ==============================
app = FastAPI(title="Heart Disease Predictor API")

# ==============================
# Enable CORS
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # সব frontend থেকে access করতে
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# ML Model Path
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "heart_model.joblib")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("❌ heart_model.joblib file not found")

# Load model
model = joblib.load(MODEL_PATH)

# ==============================
# Input Schema
# ==============================
class HeartInput(BaseModel):
    age: int = Field(..., example=50)
    sex: int = Field(..., example=1)
    cp: int = Field(..., example=3)
    trestbps: int = Field(..., example=130)
    chol: int = Field(..., example=250)
    fbs: int = Field(..., example=0)
    restecg: int = Field(..., example=1)
    thalach: int = Field(..., example=150)
    exang: int = Field(..., example=0)
    oldpeak: float = Field(..., example=2.3)
    slope: int = Field(..., example=2)
    ca: int = Field(..., example=0)
    thal: int = Field(..., example=2)

# ==============================
# Root endpoint
# ==============================
@app.get("/")
def root():
    return {"message": "Heart Disease Predictor API running"}

# ==============================
# Predict endpoint
# ==============================
@app.post("/predict")
def predict(data: HeartInput):
    try:
        values = np.array([[ 
            data.age, data.sex, data.cp, data.trestbps, data.chol, 
            data.fbs, data.restecg, data.thalach, data.exang, 
            data.oldpeak, data.slope, data.ca, data.thal 
        ]])
        prediction = model.predict(values)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
