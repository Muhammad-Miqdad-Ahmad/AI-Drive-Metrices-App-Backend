# ═══════════════════════════════════════════════════════════════════════════
#  DriveSense Backend  |  FastAPI  |  Python 3.10+
#  POST /predict  ← ESP32 sends a 28-sample window of 6-axis IMU data
#  Returns: class_id, class_name, confidence, all_probs
#
#  Place driver_behaviour_model.pkl, scaler.pkl, and model_meta.json
#  in the same directory.
#  Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# ═══════════════════════════════════════════════════════════════════════════

import json
import logging
import os
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

# ─── CONFIG ──────────────────────────────────────────────────────────────────

MODEL_PATH  = os.getenv("MODEL_PATH",  "driver_behaviour_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")
META_PATH   = os.getenv("META_PATH",   "model_meta.json")

WINDOW_SIZE = 28   # must match notebook

# !! CRITICAL: column order MUST match the notebook's FEATURES list !!
# Notebook: FEATURES = ['GyroX', 'GyroY', 'GyroZ', 'AccX', 'AccY', 'AccZ']
SENSOR_COLS = ["GyroX", "GyroY", "GyroZ", "AccX", "AccY", "AccZ"]

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
    model     = None
    scaler    = None
    is_xgb    = False   # True → model was trained on 0-indexed labels (0-3)

state = ModelState()

# ─── LIFESPAN ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    for path in (MODEL_PATH, SCALER_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run the notebook to generate the artefacts first."
            )

    state.model  = joblib.load(MODEL_PATH)
    state.scaler = joblib.load(SCALER_PATH)

    # Detect XGBoost via model_meta.json (written by notebook cell 6).
    # Fall back to isinstance check if the file is absent.
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
        state.is_xgb = meta.get("best_name") == "XGBoost"
    else:
        try:
            from xgboost import XGBClassifier # type: ignore
            state.is_xgb = isinstance(state.model, XGBClassifier)
        except ImportError:
            state.is_xgb = False

    model_type = type(state.model).__name__
    log.info(f"Model loaded  : {MODEL_PATH}  ({model_type})")
    log.info(f"Scaler loaded : {SCALER_PATH}")
    log.info(f"XGBoost model : {state.is_xgb}  (0-indexed labels → +1 shift)")
    log.info("DriveSense backend ready.")
    yield

# ─── APP ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DriveSense",
    description="Driving behaviour detection via IMU window classification",
    version="1.1.0",
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
# Exact replica of the notebook's window_features() function.
#
# Per-column features (9 each × 6 columns = 54 total):
#   mean, std, min, max, range, rms, zero_crossing_rate, p25, p75
#
# Column order mirrors the notebook's FEATURES list:
#   ['GyroX', 'GyroY', 'GyroZ', 'AccX', 'AccY', 'AccZ']
#
# DO NOT change either the feature set or column order — both must match
# exactly what the model was trained on.

def _zero_crossing_rate(x: np.ndarray) -> float:
    """Fraction of consecutive sign-changes in a 1-D signal."""
    return float(np.mean(np.diff(np.sign(x)) != 0))


def extract_window_features(window_df: pd.DataFrame) -> np.ndarray:
    """
    Given a DataFrame with columns matching SENSOR_COLS and exactly WINDOW_SIZE
    rows, extract the same 54 statistical features the notebook uses.
    Returns a (1, 54) numpy array ready for scaler.transform() → model.predict().
    """
    feats: list[float] = []

    for col in SENSOR_COLS:          # order is critical
        s = window_df[col].values.astype(np.float64)

        feats.append(float(s.mean()))
        feats.append(float(s.std()))
        feats.append(float(s.min()))
        feats.append(float(s.max()))
        feats.append(float(s.max() - s.min()))           # range
        feats.append(float(np.sqrt(np.mean(s ** 2))))    # RMS
        feats.append(_zero_crossing_rate(s))             # ZCR
        feats.append(float(np.percentile(s, 25)))        # p25
        feats.append(float(np.percentile(s, 75)))        # p75

    return np.array(feats, dtype=np.float64).reshape(1, -1)  # (1, 54)

# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
async def root():
    return {
        "status":  "ok",
        "service": "DriveSense",
        "model":   type(state.model).__name__ if state.model else "not loaded",
        "is_xgb":  state.is_xgb,
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
            "GyroX": r.GyroX, "GyroY": r.GyroY, "GyroZ": r.GyroZ,
            "AccX":  r.AccX,  "AccY":  r.AccY,  "AccZ":  r.AccZ,
        }
        for r in req.window
    ]
    window_df = pd.DataFrame(rows)

    # ── Extract features & scale ──────────────────────────────────────────
    # The scaler is ALWAYS applied: the model was fitted on scaled data
    # regardless of model type (notebook always calls scaler.transform).
    X_live = extract_window_features(window_df)       # (1, 54)
    X_live = state.scaler.transform(X_live)

    # ── Predict ───────────────────────────────────────────────────────────
    raw_pred  = int(state.model.predict(X_live)[0])
    proba     = state.model.predict_proba(X_live)[0]  # shape (4,)

    # XGBoost was trained on 0-indexed labels (0-3) → shift back to 1-4.
    # All other models were trained on 1-4 → raw_pred IS already the class id.
    if state.is_xgb:
        class_id   = raw_pred + 1
        proba_idx  = raw_pred           # 0-based index into proba array
    else:
        class_id   = raw_pred
        proba_idx  = raw_pred - 1       # 1-based → 0-based for proba lookup

    confidence = float(proba[proba_idx]) * 100.0

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