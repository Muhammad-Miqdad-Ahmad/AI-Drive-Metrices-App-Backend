# ═══════════════════════════════════════════════════════════════════════════
#  DriveSense Backend  |  FastAPI  |  Python 3.10+
#  UART listener receives a 28-sample window of 6-axis IMU data from ESP32
#  Returns: class_id, class_name, confidence, all_probs
#
#  Place driver_behaviour_model.pkl, scaler.pkl, and model_meta.json
#  in the same directory.
#
#  Run:
#      uvicorn main:app --host 0.0.0.0 --port 8000 --reload
#
#  UART notes:
#      - Set SERIAL_PORT env var if auto-detection picks the wrong port.
#      - ESP32 sends lines prefixed with "WINDOW:" followed by JSON.
#      - Python replies with one JSON line and a trailing newline.
# ═══════════════════════════════════════════════════════════════════════════

import os
import time
import json
import logging
import threading
from contextlib import asynccontextmanager

import serial
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

# ─── CONFIG ──────────────────────────────────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "driver_behaviour_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")
META_PATH = os.getenv("META_PATH", "model_meta.json")

WINDOW_SIZE = 28   # must match notebook

# UART config
SERIAL_PORT = os.getenv("SERIAL_PORT", "").strip()  # e.g. COM3 or /dev/ttyUSB0
SERIAL_BAUD = int(os.getenv("SERIAL_BAUD", "115200"))
SERIAL_TIMEOUT = float(os.getenv("SERIAL_TIMEOUT", "1.0"))  # seconds
SERIAL_AUTODETECT = os.getenv("SERIAL_AUTODETECT", "1").strip() != "0"

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
    model = None
    scaler = None
    is_xgb = False   # True → model was trained on 0-indexed labels (0-3)

    serial_conn = None
    serial_thread = None
    serial_stop = None
    serial_lock = None

state = ModelState()

# ─── LIFESPAN ────────────────────────────────────────────────────────────────

def _pick_serial_port() -> str:
    """
    Pick a UART port. SERIAL_PORT env var wins.
    Otherwise try to auto-detect a single candidate.
    """
    if SERIAL_PORT:
        return SERIAL_PORT

    if not SERIAL_AUTODETECT:
        raise RuntimeError(
            "SERIAL_PORT is not set and SERIAL_AUTODETECT=0. "
            "Set SERIAL_PORT to your ESP32 port."
        )

    try:
        from serial.tools import list_ports
    except Exception as exc:
        raise RuntimeError(
            "pyserial is installed, but serial.tools.list_ports could not be imported."
        ) from exc

    ports = list(list_ports.comports())
    if not ports:
        raise RuntimeError("No serial ports found. Set SERIAL_PORT explicitly.")

    # Prefer common USB/UART device names; otherwise fall back to the first port.
    preferred = []
    for p in ports:
        dev = (p.device or "").lower()
        desc = (p.description or "").lower()
        hwid = (p.hwid or "").lower()
        if any(tag in dev for tag in ("ttyusb", "ttyacm", "com")) or any(
            tag in desc for tag in ("usb", "serial", "uart", "acm")
        ) or "usb" in hwid:
            preferred.append(p.device)

    if len(preferred) == 1:
        return preferred[0]
    if len(ports) == 1:
        return ports[0].device

    # Ambiguous: choose the first one, but log the list so it is obvious.
    log.warning(
        "Multiple serial ports detected. Auto-selecting the first one: %s",
        ports[0].device,
    )
    for p in ports:
        log.info("Available port: %s | %s | %s", p.device, p.description, p.hwid)
    return ports[0].device


def _serial_write_line(conn: serial.Serial, payload: dict) -> None:
    line = json.dumps(payload, separators=(",", ":")) + "\n"
    conn.write(line.encode("utf-8"))
    conn.flush()


def _predict_from_window_rows(rows: list[dict]) -> dict:
    if len(rows) != WINDOW_SIZE:
        raise ValueError(f"window must contain exactly {WINDOW_SIZE} samples, got {len(rows)}")

    window_df = pd.DataFrame(rows)

    # ── Extract features & scale ──────────────────────────────────────────
    # The scaler is ALWAYS applied: the model was fitted on scaled data
    # regardless of model type (notebook always calls scaler.transform).
    X_live = extract_window_features(window_df)       # (1, 54)
    X_live = state.scaler.transform(X_live)

    # ── Predict ───────────────────────────────────────────────────────────
    raw_pred = int(state.model.predict(X_live)[0])
    proba = state.model.predict_proba(X_live)[0]  # shape (4,)

    # XGBoost was trained on 0-indexed labels (0-3) → shift back to 1-4.
    # All other models were trained on 1-4 → raw_pred IS already the class id.
    if state.is_xgb:
        class_id = raw_pred + 1
        proba_idx = raw_pred           # 0-based index into proba array
    else:
        class_id = raw_pred
        proba_idx = raw_pred - 1       # 1-based → 0-based for proba lookup

    confidence = float(proba[proba_idx]) * 100.0

    all_probs = {
        CLASS_NAMES[i + 1]: round(float(p) * 100, 2)
        for i, p in enumerate(proba)
    }

    result = {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "confidence": round(confidence, 2),
        "all_probs": all_probs,
    }

    log.info(
        f"Prediction → [{class_id}] {CLASS_NAMES[class_id]}  "
        f"({confidence:.1f}%)"
    )
    return result


def _uart_loop() -> None:
    conn = state.serial_conn
    assert conn is not None

    log.info("UART listener started on %s @ %d", conn.port, conn.baudrate)

    while not state.serial_stop.is_set():
        try:
            raw = conn.readline()
            if not raw:
                continue

            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            # Ignore debug chatter from the ESP32 unless it is an actual window.
            if not line.startswith("WINDOW:"):
                continue

            payload = line[len("WINDOW:"):].strip()
            if not payload:
                log.warning("Received WINDOW prefix with empty payload.")
                continue

            try:
                doc = json.loads(payload)
            except json.JSONDecodeError as exc:
                log.exception("Invalid JSON received over UART: %s", exc)
                continue

            rows = doc.get("window")
            if not isinstance(rows, list):
                log.warning("UART payload missing 'window' list.")
                continue

            try:
                result = _predict_from_window_rows(rows)
            except Exception as exc:
                log.exception("Prediction failed: %s", exc)
                # Keep the line-oriented protocol alive; send an explicit error JSON.
                err = {
                    "class_id": -1,
                    "class_name": "Error",
                    "confidence": 0.0,
                    "all_probs": {},
                    "error": str(exc),
                }
                _serial_write_line(conn, err)
                continue

            _serial_write_line(conn, result)

        except (serial.SerialException, OSError) as exc:
            if state.serial_stop.is_set():
                break
            log.exception("UART error: %s", exc)
            time.sleep(1.0)
        except Exception as exc:
            log.exception("Unexpected UART loop error: %s", exc)
            time.sleep(0.2)

    log.info("UART listener stopped.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    for path in (MODEL_PATH, SCALER_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run the notebook to generate the artefacts first."
            )

    state.model = joblib.load(MODEL_PATH)
    state.scaler = joblib.load(SCALER_PATH)

    # Detect XGBoost via model_meta.json (written by notebook cell 6).
    # Fall back to isinstance check if the file is absent.
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
        state.is_xgb = meta.get("best_name") == "XGBoost"
    else:
        try:
            from xgboost import XGBClassifier  # type: ignore
            state.is_xgb = isinstance(state.model, XGBClassifier)
        except ImportError:
            state.is_xgb = False

    model_type = type(state.model).__name__
    log.info(f"Model loaded  : {MODEL_PATH}  ({model_type})")
    log.info(f"Scaler loaded : {SCALER_PATH}")
    log.info(f"XGBoost model : {state.is_xgb}  (0-indexed labels → +1 shift)")

    state.serial_lock = threading.Lock()
    state.serial_stop = threading.Event()

    try:
        port = _pick_serial_port()
        state.serial_conn = serial.Serial(
            port=port,
            baudrate=SERIAL_BAUD,
            timeout=SERIAL_TIMEOUT,
            write_timeout=SERIAL_TIMEOUT,
        )
        # Give the ESP32 a moment after opening the port.
        time.sleep(2.0)
        log.info("Serial port opened: %s @ %d", port, SERIAL_BAUD)

        state.serial_thread = threading.Thread(target=_uart_loop, daemon=True)
        state.serial_thread.start()
        log.info("DriveSense backend ready.")
    except Exception as exc:
        log.exception("Failed to start UART listener: %s", exc)
        log.info("DriveSense backend is still up, but UART input is disabled.")

    try:
        yield
    finally:
        if state.serial_stop is not None:
            state.serial_stop.set()

        if state.serial_conn is not None:
            try:
                state.serial_conn.close()
            except Exception:
                pass
            state.serial_conn = None

        if state.serial_thread is not None and state.serial_thread.is_alive():
            state.serial_thread.join(timeout=2.0)


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
    AccX: float = Field(..., description="Acceleration X  (m/s²)")
    AccY: float = Field(..., description="Acceleration Y  (m/s²)")
    AccZ: float = Field(..., description="Acceleration Z  (m/s²)")
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
    class_id: int
    class_name: str
    confidence: float   # percent, 0–100
    all_probs: dict[str, float]

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
        "status": "ok",
        "service": "DriveSense",
        "model": type(state.model).__name__ if state.model else "not loaded",
        "is_xgb": state.is_xgb,
        "uart_enabled": state.serial_conn is not None,
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify a driving-behaviour window",
)
async def predict(req: PredictRequest):
    if state.model is None:
        raise HTTPException(503, "Model not loaded")

    rows = [
        {
            "GyroX": r.GyroX, "GyroY": r.GyroY, "GyroZ": r.GyroZ,
            "AccX": r.AccX, "AccY": r.AccY, "AccZ": r.AccZ,
        }
        for r in req.window
    ]

    result = _predict_from_window_rows(rows)
    return PredictResponse(**result)


@app.post(
    "/predict/raw",
    response_model=PredictResponse,
    summary="Classify using a single raw reading (repeats it into a window — less accurate)",
)
async def predict_raw(reading: SensorReading):
    """
    Convenience endpoint: takes one reading and inflates it to a WINDOW_SIZE
    window. Useful for quick testing. Production should use UART.
    """
    fake_window_req = PredictRequest(window=[reading] * WINDOW_SIZE)
    return await predict(fake_window_req)
