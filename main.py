# ═══════════════════════════════════════════════════════════════════════════
#  DriveSense Backend  |  FastAPI  |  Python 3.10+
#  POST /predict  ← ESP32 sends a 28-sample window of 6-axis IMU data
#  Returns: class_id, class_name, confidence, all_probs
#
#  Place driver_behaviour_model.pkl and scaler.pkl in the same directory.
#  Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# ═══════════════════════════════════════════════════════════════════════════

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

# ─── CONFIG ──────────────────────────────────────────────────────────────────

MODEL_PATH  = os.getenv("MODEL_PATH",  "driver_behaviour_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")

WINDOW_SIZE = 28   # must match notebook
SENSOR_COLS = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]

CLASS_NAMES = {
    1: "Sudden Acceleration",
    2: "Sudden Right Turn",
    3: "Sudden Left Turn",
    4: "Sudden Brake",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("drivesense")

# ─── MODEL STATE ─────────────────────────────────────────────────────────────

class ModelState:
    model       = None
    scaler      = None
    is_linear   = False   # True → apply StandardScaler before predict

state = ModelState()

# ─── LIFESPAN ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Load on startup ──────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Run the notebook to generate driver_behaviour_model.pkl first."
        )
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler file not found: {SCALER_PATH}\n"
            "Run the notebook to generate scaler.pkl first."
        )

    state.model  = joblib.load(MODEL_PATH)
    state.scaler = joblib.load(SCALER_PATH)

    # Detect if it's a linear model that needs scaling
    model_type = type(state.model).__name__
    state.is_linear = model_type in ("LogisticRegression", "SVC", "LinearSVC")

    log.info(f"Model loaded  : {MODEL_PATH}  ({model_type})")
    log.info(f"Scaler loaded : {SCALER_PATH}")
    log.info(f"Scaling needed: {state.is_linear}")
    log.info("DriveSense backend ready.")
    yield
    # cleanup (none needed)

# ─── APP ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DriveSense",
    description="Driving behaviour detection via IMU window classification",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── SCHEMAS ─────────────────────────────────────────────────────────────────

class SensorReading(BaseModel):
    AccX:  float = Field(..., description="Acceleration X  (m/s²)")
    AccY:  float = Field(..., description="Acceleration Y  (m/s²)")
    AccZ:  float = Field(..., description="Acceleration Z  (m/s²)")
    GyroX: float = Field(..., description="Gyroscope X     (°/s)")
    GyroY: float = Field(..., description="Gyroscope Y     (°/s)")
    GyroZ: float = Field(..., description="Gyroscope Z     (°/s)")


class PredictRequest(BaseModel):
    window: list[SensorReading] = Field(
        ...,
        min_length=WINDOW_SIZE,
        max_length=WINDOW_SIZE,
        description=f"Exactly {WINDOW_SIZE} consecutive sensor readings",
    )

    @model_validator(mode="after")
    def check_window_size(self):
        if len(self.window) != WINDOW_SIZE:
            raise ValueError(
                f"window must contain exactly {WINDOW_SIZE} samples, "
                f"got {len(self.window)}"
            )
        return self


class PredictResponse(BaseModel):
    class_id:   int
    class_name: str
    confidence: float   # percent, 0–100
    all_probs:  dict[str, float]

# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────
# Exact replica of the notebook's extract_window_features() function.
# DO NOT change the feature order — it must match the training feature order.

def extract_window_features(window_df: pd.DataFrame) -> np.ndarray:
    """
    Given a DataFrame with columns matching SENSOR_COLS and exactly WINDOW_SIZE
    rows, extract the same 66 statistical features the notebook uses.
    Returns a (1, 66) numpy array ready for model.predict().
    """
    feats: dict[str, float] = {}

    for col in SENSOR_COLS:
        vals = window_df[col].values.astype(np.float64)

        feats[f"{col}_mean"]   = float(np.mean(vals))
        feats[f"{col}_std"]    = float(np.std(vals))
        feats[f"{col}_min"]    = float(np.min(vals))
        feats[f"{col}_max"]    = float(np.max(vals))
        feats[f"{col}_range"]  = float(np.max(vals) - np.min(vals))
        feats[f"{col}_median"] = float(np.median(vals))
        feats[f"{col}_iqr"]    = float(np.percentile(vals, 75) - np.percentile(vals, 25))
        feats[f"{col}_rms"]    = float(np.sqrt(np.mean(vals ** 2)))
        feats[f"{col}_energy"] = float(np.sum(vals ** 2))
        feats[f"{col}_skew"]   = float(pd.Series(vals).skew())
        feats[f"{col}_kurt"]   = float(pd.Series(vals).kurtosis())

    return np.array(list(feats.values()), dtype=np.float64).reshape(1, -1)

# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
async def root():
    return {
        "status":  "ok",
        "service": "DriveSense",
        "model":   type(state.model).__name__ if state.model else "not loaded",
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify a driving-behaviour window",
)
async def predict(req: PredictRequest):
    if state.model is None:
        raise HTTPException(503, "Model not loaded")

    # ── Build DataFrame ───────────────────────────────────────────────────
    rows = [
        {
            "AccX":  r.AccX,  "AccY":  r.AccY,  "AccZ":  r.AccZ,
            "GyroX": r.GyroX, "GyroY": r.GyroY, "GyroZ": r.GyroZ,
        }
        for r in req.window
    ]
    window_df = pd.DataFrame(rows)

    # ── Extract features ──────────────────────────────────────────────────
    X_live = extract_window_features(window_df)

    if state.is_linear:
        X_live = state.scaler.transform(X_live)

    # ── Predict ───────────────────────────────────────────────────────────
    pred_enc  = int(state.model.predict(X_live)[0])          # 0-indexed
    proba     = state.model.predict_proba(X_live)[0]         # shape (4,)
    class_id  = pred_enc + 1                                  # back to 1-indexed
    confidence = float(proba[pred_enc]) * 100.0

    all_probs = {
        CLASS_NAMES[i + 1]: round(float(p) * 100, 2)
        for i, p in enumerate(proba)
    }

    result = PredictResponse(
        class_id=class_id,
        class_name=CLASS_NAMES[class_id],
        confidence=round(confidence, 2),
        all_probs=all_probs,
    )

    log.info(
        f"Prediction → [{class_id}] {CLASS_NAMES[class_id]}  "
        f"({confidence:.1f}%)"
    )
    return result


@app.post(
    "/predict/raw",
    response_model=PredictResponse,
    summary="Classify using a single raw reading (repeats it into a window — less accurate)",
)
async def predict_raw(reading: SensorReading):
    """
    Convenience endpoint: takes one reading and inflates it to a WINDOW_SIZE
    window. Useful for quick testing. Production should use /predict.
    """
    fake_window_req = PredictRequest(window=[reading] * WINDOW_SIZE)
    return await predict(fake_window_req)