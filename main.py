"""
FastAPI service for Bhoomi AI's crop recommendation feature.

Endpoints:
  GET  /health         -> liveness check
  POST /predict-crop    -> { N, P, K, temperature, humidity, ph, rainfall } -> top-3 crop suggestions

Deploy this on Render / Railway / Fly.io (all have free tiers) and point
the Flutter app's CropService at the deployed URL.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(title="Bhoomi AI - Crop Recommendation API")

# Allow the Flutter app (any origin) to call this API.
# Tighten this to your app's actual domain once you have one.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("crop_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Rough sane ranges the training data actually covers — inputs way
# outside this are a sign of a typo or a sensor glitch, not a valid
# prediction, so we flag rather than silently guess.
VALID_RANGES = {
    "N": (0, 150),
    "P": (0, 150),
    "K": (0, 210),
    "temperature": (0, 55),
    "humidity": (0, 100),
    "ph": (2, 11),
    "rainfall": (0, 320),
}


class CropRequest(BaseModel):
    N: float = Field(..., description="Nitrogen content in soil (kg/ha)")
    P: float = Field(..., description="Phosphorus content in soil (kg/ha)")
    K: float = Field(..., description="Potassium content in soil (kg/ha)")
    temperature: float = Field(..., description="Avg temperature in Celsius")
    humidity: float = Field(..., description="Avg relative humidity in %")
    ph: float = Field(..., description="Soil pH")
    rainfall: float = Field(..., description="Avg rainfall in mm")


class CropSuggestion(BaseModel):
    crop: str
    confidence: float


class CropResponse(BaseModel):
    suggestions: list[CropSuggestion]
    warnings: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict-crop", response_model=CropResponse)
def predict_crop(req: CropRequest):
    data = req.model_dump()
    warnings = []
    for field, (lo, hi) in VALID_RANGES.items():
        val = data[field]
        if val < lo or val > hi:
            warnings.append(
                f"{field}={val} is outside the typical range ({lo}-{hi}); "
                "double check this value before trusting the result."
            )

    x = np.array([[data[f] for f in FEATURES]])
    try:
        probs = model.predict_proba(x)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    top3_idx = np.argsort(probs)[::-1][:3]
    suggestions = [
        CropSuggestion(
            crop=label_encoder.inverse_transform([i])[0],
            confidence=round(float(probs[i]), 4),
        )
        for i in top3_idx
    ]

    return CropResponse(suggestions=suggestions, warnings=warnings)
