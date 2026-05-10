# ═══════════════════════════════════════════════════════════════════════════
#  DriveSense Backend  |  FastAPI  |  Python 3.10+
#  POST /predict  ← ESP32 sends a 28-sample window of 6-axis IMU data
#  Returns: class_id, class_name, confidence, is_harsh, all_probs
#
#  Required files in same directory:
#    rf_model.pkl
#    scaler.pkl
#    label_encoder.pkl
#
#  Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
#
#  Changes in v4.1 vs v4.0:
#   1. GYRO_Y_CORRECTION added (+1.527 deg/s)
#      merge_datasets.py added +1.527 to GyroY in your self-collected data
#      (labels 0 and 1) so it aligned with the original dataset's GyroY
#      baseline (~4.13 deg/s). The same correction must be applied at
#      inference time so the model sees the same GyroY range it was trained on.
#      Without this: firmware sends GyroY 2.67, model trained on 4.13 → gap
#      of 1.44 deg/s → Idle/Normal confidence stuck at ~44%.
#      With this: 2.67 + 1.527 = 4.20 → matches training → ~80%+ confidence.
#   2. /health and / endpoints now report gyro_y_correction value.
#   3. All other logic unchanged from v4.0.
# ═══════════════════════════════════════════════════════════════════════════

import os
import logging
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

# ─── CONFIG ──────────────────────────────────────────────────────────────────

MODEL_PATH   = os.getenv("MODEL_PATH",   "rf_model.pkl")
SCALER_PATH  = os.getenv("SCALER_PATH",  "scaler.pkl")
ENCODER_PATH = os.getenv("ENCODER_PATH", "label_encoder.pkl")

WINDOW_SIZE = 28   # must match notebook WINDOW_SIZE

# Column order — must exactly match notebook FEATURES list
SENSOR_COLS = ["GyroX", "GyroY", "GyroZ", "AccX", "AccY", "AccZ"]

# ── CLASS_NAMES must match combined_dataset.csv label scheme exactly ──────
#   0 = Idle State          <- your self-collected data, label 0
#   1 = Normal Driving      <- your self-collected data, label 1
#   2 = Sudden Acceleration <- dataset.csv original label 1, remapped to 2
#   3 = Sudden Right Turn   <- dataset.csv original label 2, remapped to 3
#   4 = Sudden Left Turn    <- dataset.csv original label 3, remapped to 4
#   5 = Sudden Brake        <- dataset.csv original label 4, remapped to 5
CLASS_NAMES = {
    0: "Idle State",
    1: "Normal Driving",
    2: "Sudden Acceleration",
    3: "Sudden Right Turn",
    4: "Sudden Left Turn",
    5: "Sudden Brake",
}

# Labels 0 and 1 are both non-harsh; 2-5 are harsh events
HARSH_CLASSES  = {2, 3, 4, 5}
NORMAL_CLASSES = {0, 1}

# When a harsh class is predicted but its confidence is below this threshold,
# the response falls back to Normal Driving (label 1).
# Set to 0.0 to disable (always trust model). Recommended: 0.35 after validation.
HARSH_THRESHOLD = float(os.getenv("HARSH_THRESHOLD", "0.0"))

# ── Gyro corrections ─────────────────────────────────────────────────────────
#
# GYRO_X_BIAS (-2.58 deg/s)
#   Static DC idle offset on GyroX measured from your sensor's idle.txt.
#   Subtracted from every incoming GyroX value before feature extraction.
#   Raw firmware GyroX ~ -2.75 deg/s at rest
#   After correction  ~  -0.17 deg/s  -> matches training data baseline
#   Override: export GYRO_X_BIAS=-2.58
#
# GYRO_Y_CORRECTION (+1.527 deg/s)
#   merge_datasets.py added +1.527 to GyroY in your self-collected data
#   so labels 0 and 1 aligned with the original dataset's GyroY baseline.
#   The model was trained on the corrected values, so inference must apply
#   the same addition to every incoming GyroY value.
#   Raw firmware GyroY ~ 2.67 deg/s at rest
#   After correction  ~  4.20 deg/s  -> matches training data baseline (4.13)
#   Override: export GYRO_Y_CORRECTION=1.527
#
GYRO_X_BIAS       = float(os.getenv("GYRO_X_BIAS",       "-2.58"))
GYRO_Y_CORRECTION = float(os.getenv("GYRO_Y_CORRECTION", "1.527"))

# Number of statistical features per axis — must match notebook window_features()
# Notebook computes: mean, std, min, max, Q1, Q3, RMS, range  ->  8 stats
N_STATS    = 8
N_SENSORS  = len(SENSOR_COLS)       # 6
N_FEATURES = N_STATS * N_SENSORS    # 48  <- must match model's n_features_in_

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("drivesense")

# ─── MODEL STATE ─────────────────────────────────────────────────────────────

class ModelState:
    model         = None
    scaler        = None
    label_encoder = None

state = ModelState()

# ─── LIFESPAN ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Check all pkl files exist
    required = [
        (MODEL_PATH,   "Model  (rf_model.pkl)"),
        (SCALER_PATH,  "Scaler (scaler.pkl)"),
        (ENCODER_PATH, "LabelEncoder (label_encoder.pkl)"),
    ]
    for path, label in required:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} not found at: {path}\n"
                "Run driver_behaviour_detection.ipynb top-to-bottom "
                "to generate all .pkl files, then copy them here."
            )

    state.model         = joblib.load(MODEL_PATH)
    state.scaler        = joblib.load(SCALER_PATH)
    state.label_encoder = joblib.load(ENCODER_PATH)

    model_type      = type(state.model).__name__
    encoded_classes = list(state.model.classes_)
    original_labels = [
        int(state.label_encoder.inverse_transform([c])[0])
        for c in encoded_classes
    ]

    # Sanity-check 1: must have all 6 classes
    expected_labels = set(CLASS_NAMES.keys())   # {0,1,2,3,4,5}
    missing = expected_labels - set(original_labels)
    if missing:
        missing_names = [CLASS_NAMES[m] for m in sorted(missing)]
        raise ValueError(
            f"Loaded model is missing classes: {missing} ({missing_names}).\n"
            "Make sure you trained on combined_dataset.csv which has 6 classes (0-5).\n"
            "Re-run driver_behaviour_detection.ipynb with the correct dataset."
        )

    # Sanity-check 2: feature count must be 48
    model_features = getattr(state.model, "n_features_in_", None)
    if model_features is not None and model_features != N_FEATURES:
        raise ValueError(
            f"Feature count mismatch:\n"
            f"  Backend produces : {N_FEATURES} features  "
            f"({N_STATS} stats x {N_SENSORS} axes)\n"
            f"  Model expects    : {model_features} features\n"
            f"Ensure extract_window_features() uses the same 8 stats as "
            f"the notebook window_features():\n"
            f"  mean, std, min, max, Q1, Q3, RMS, range"
        )

    log.info(f"Model            : {MODEL_PATH}  ({model_type})")
    log.info(f"Classes          : {original_labels}  -> {[CLASS_NAMES[l] for l in original_labels]}")
    log.info(f"Features         : {N_FEATURES}  ({N_STATS} stats x {N_SENSORS} axes)")
    log.info(f"Harsh classes    : {sorted(HARSH_CLASSES)}")
    log.info(f"Normal classes   : {sorted(NORMAL_CLASSES)}")
    log.info(f"Harsh threshold  : {HARSH_THRESHOLD}  "
             f"({'disabled' if HARSH_THRESHOLD == 0 else f'active at {HARSH_THRESHOLD:.0%}'})")
    log.info(f"Gyro X bias      : {GYRO_X_BIAS} deg/s  (subtracted from raw GyroX)")
    log.info(f"Gyro Y correction: +{GYRO_Y_CORRECTION} deg/s  (added to raw GyroY)")
    log.info("DriveSense v4.1 backend ready.")
    yield

# ─── APP ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DriveSense",
    description="Driving behaviour detection via IMU window classification",
    version="4.1.0",
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
    GyroX: float = Field(..., description="Gyroscope X  (deg/s)  — raw, no bias removed")
    GyroY: float = Field(..., description="Gyroscope Y  (deg/s)  — raw, no correction applied")
    GyroZ: float = Field(..., description="Gyroscope Z  (deg/s)")
    AccX:  float = Field(..., description="Acceleration X  (g)")
    AccY:  float = Field(..., description="Acceleration Y  (g)")
    AccZ:  float = Field(..., description="Acceleration Z  (g)")


class PredictRequest(BaseModel):
    window: list[SensorReading] = Field(
        ...,
        min_length=WINDOW_SIZE,
        max_length=WINDOW_SIZE,
        description=f"Exactly {WINDOW_SIZE} consecutive sensor readings at 2 Hz",
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
    # Stage 1: harsh vs normal
    is_harsh:    bool  = Field(..., description="True if any harsh event detected")
    normal_prob: float = Field(..., description="Combined probability of Idle + Normal Driving (0-100)")
    harsh_prob:  float = Field(..., description="Combined probability of any harsh event (0-100)")

    # Stage 2: which specific event
    class_id:    int   = Field(..., description="0=Idle, 1=Normal, 2=Accel, 3=RightTurn, 4=LeftTurn, 5=Brake")
    class_name:  str   = Field(..., description="Human-readable class label")
    confidence:  float = Field(..., description="Confidence of predicted class (0-100)")

    # Full breakdown
    all_probs:   dict[str, float] = Field(..., description="Per-class probabilities (0-100)")

# ─── FEATURE EXTRACTION ──────────────────────────────────────────────────────

def extract_window_features(window_df: pd.DataFrame) -> np.ndarray:
    """
    Compute 8 statistical features per axis — exactly matching
    the notebook's window_features() function.

    Stats (in order):
        1. mean
        2. std
        3. min
        4. max
        5. Q1  (25th percentile)
        6. Q3  (75th percentile)
        7. RMS (sqrt of mean of squares)
        8. range (max - min)

    8 stats x 6 axes = 48 features total.
    Axis order follows SENSOR_COLS = [GyroX, GyroY, GyroZ, AccX, AccY, AccZ].
    """
    feats = []
    for col in SENSOR_COLS:
        s = window_df[col].values.astype(np.float64)
        feats.extend([
            float(np.mean(s)),
            float(np.std(s)),
            float(np.min(s)),
            float(np.max(s)),
            float(np.percentile(s, 25)),            # Q1
            float(np.percentile(s, 75)),            # Q3
            float(np.sqrt(np.mean(s ** 2))),        # RMS
            float(np.max(s) - np.min(s)),           # range
        ])
    return np.array(feats, dtype=np.float64).reshape(1, -1)


# ─── PREDICTION PIPELINE ─────────────────────────────────────────────────────

def run_prediction(window_df: pd.DataFrame) -> PredictResponse:
    """
    Two-stage prediction pipeline.

    Gyro corrections applied before feature extraction:
      GyroX: subtract GYRO_X_BIAS      (-2.58 deg/s DC idle offset from sensor)
      GyroY: add    GYRO_Y_CORRECTION  (+1.527 deg/s — mirrors the offset applied
             by merge_datasets.py to training data labels 0 and 1)

    Stage 1 — Harsh vs Normal:
        Compute normal_prob = P(Idle) + P(Normal Driving).
        If predicted class is 0 or 1, return normal immediately.
        If HARSH_THRESHOLD > 0 and harsh confidence is too low,
        fall back to Normal Driving (label 1).

    Stage 2 — Which event:
        Reached only when Stage 1 confirms a harsh event.
        Returns specific label (2-5) with name and confidence.
    """
    window_df = window_df.copy()

    # ── Gyro corrections ─────────────────────────────────────────────────────
    #
    # GyroX: remove the sensor's static DC idle offset.
    #   Raw ~ -2.75 deg/s  -->  corrected ~ -0.17 deg/s  (training baseline)
    window_df["GyroX"] = window_df["GyroX"] - GYRO_X_BIAS

    # GyroY: apply the same offset that merge_datasets.py baked into training data.
    #   Raw ~ +2.67 deg/s  -->  corrected ~ +4.20 deg/s  (training baseline ~4.13)
    window_df["GyroY"] = window_df["GyroY"] + GYRO_Y_CORRECTION

    # ── Guard: reject zero-variance windows ──────────────────────────────────
    if window_df[SENSOR_COLS].std().max() < 1e-6:
        raise HTTPException(
            status_code=422,
            detail=(
                "Window has zero variance — all rows appear identical. "
                f"Send {WINDOW_SIZE} real consecutive sensor readings."
            ),
        )

    # ── Feature extraction and scaling ───────────────────────────────────────
    X_live   = extract_window_features(window_df)
    X_scaled = state.scaler.transform(X_live)

    # ── Model output ─────────────────────────────────────────────────────────
    encoded_classes = list(state.model.classes_)
    proba           = state.model.predict_proba(X_scaled)[0]
    pred_enc        = int(state.model.predict(X_scaled)[0])
    class_id        = int(state.label_encoder.inverse_transform([pred_enc])[0])

    # Build per-class probability dict  { "Idle State": 12.5, ... }
    all_probs: dict[str, float] = {}
    for i, enc in enumerate(encoded_classes):
        orig_lbl = int(state.label_encoder.inverse_transform([enc])[0])
        all_probs[CLASS_NAMES[orig_lbl]] = round(float(proba[i]) * 100, 2)

    # Stage 1: combined normal vs harsh probability
    normal_prob = round(
        all_probs.get("Idle State", 0.0) + all_probs.get("Normal Driving", 0.0),
        2,
    )
    harsh_prob = round(100.0 - normal_prob, 2)

    # Stage 1 threshold fallback
    if class_id in HARSH_CLASSES and HARSH_THRESHOLD > 0.0:
        pred_confidence = all_probs[CLASS_NAMES[class_id]]
        if pred_confidence < HARSH_THRESHOLD * 100:
            log.info(
                f"Threshold fallback: [{class_id}] {CLASS_NAMES[class_id]} "
                f"({pred_confidence:.1f}%) < threshold ({HARSH_THRESHOLD:.0%}) "
                f"-> falling back to Normal Driving"
            )
            class_id = 1   # Normal Driving

    class_name = CLASS_NAMES[class_id]
    is_harsh   = class_id in HARSH_CLASSES
    confidence = all_probs[class_name]

    log.info(
        f"[Stage 1] is_harsh={is_harsh}  "
        f"normal={normal_prob:.1f}%  harsh={harsh_prob:.1f}%"
    )
    if is_harsh:
        log.info(f"[Stage 2] [{class_id}] {class_name}  ({confidence:.1f}%)")

    return PredictResponse(
        is_harsh    = is_harsh,
        normal_prob = normal_prob,
        harsh_prob  = harsh_prob,
        class_id    = class_id,
        class_name  = class_name,
        confidence  = confidence,
        all_probs   = all_probs,
    )

# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.get("/", summary="Basic status check")
async def root():
    model_classes = []
    if state.model and state.label_encoder:
        for enc in state.model.classes_:
            orig = int(state.label_encoder.inverse_transform([enc])[0])
            model_classes.append(f"{orig}: {CLASS_NAMES[orig]}")
    return {
        "status":            "ok",
        "service":           "DriveSense",
        "version":           "4.1.0",
        "model":             type(state.model).__name__ if state.model else "not loaded",
        "model_classes":     model_classes,
        "harsh_classes":     sorted(HARSH_CLASSES),
        "harsh_threshold":   HARSH_THRESHOLD,
        "gyro_x_bias":       GYRO_X_BIAS,
        "gyro_y_correction": GYRO_Y_CORRECTION,
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify a 28-sample driving-behaviour window",
)
async def predict(req: PredictRequest):
    if state.model is None:
        raise HTTPException(503, "Model not loaded")

    rows = [
        {
            "GyroX": r.GyroX, "GyroY": r.GyroY, "GyroZ": r.GyroZ,
            "AccX":  r.AccX,  "AccY":  r.AccY,  "AccZ":  r.AccZ,
        }
        for r in req.window
    ]
    window_df = pd.DataFrame(rows, columns=SENSOR_COLS)
    return run_prediction(window_df)


@app.get("/health", summary="Detailed model and config health check")
async def health():
    if state.model is None:
        raise HTTPException(503, "Model not loaded")

    encoded   = list(state.model.classes_)
    original  = [int(state.label_encoder.inverse_transform([c])[0]) for c in encoded]
    class_map = {o: CLASS_NAMES[o] for o in original}

    warnings = []
    if len(class_map) < 6:
        warnings.append(
            f"Model only has {len(class_map)} classes. "
            "Expected 6 (0=Idle, 1=Normal, 2=Accel, 3=RightTurn, 4=LeftTurn, 5=Brake). "
            "Re-train with combined_dataset.csv."
        )

    return {
        "model_type":    type(state.model).__name__,
        "n_features":    getattr(state.model, "n_features_in_", "unknown"),
        "class_map":     class_map,
        "n_classes":     len(class_map),
        "harsh_classes": sorted(HARSH_CLASSES),
        "config": {
            "window_size":       WINDOW_SIZE,
            "sensor_cols":       SENSOR_COLS,
            "n_stats":           N_STATS,
            "n_features":        N_FEATURES,
            "gyro_x_bias":       GYRO_X_BIAS,
            "gyro_y_correction": GYRO_Y_CORRECTION,
            "harsh_threshold":   HARSH_THRESHOLD,
        },
        "warnings": warnings,
    }


@app.get("/features", summary="Show all 48 feature names in model-input order")
async def features():
    """
    Returns the 48 feature names in the exact order the model expects.
    Use this to debug feature mismatches between notebook and backend.
    """
    stat_names = ["mean", "std", "min", "max", "Q1", "Q3", "RMS", "range"]
    feat_names = [f"{col}_{stat}" for col in SENSOR_COLS for stat in stat_names]
    return {
        "n_features":   len(feat_names),
        "n_stats":      N_STATS,
        "sensor_order": SENSOR_COLS,
        "stat_order":   stat_names,
        "features":     feat_names,
    }