# ═══════════════════════════════════════════════════════════════════════════
#  DriveSense Backend  |  FastAPI + UART  |  Python 3.10+
#
#  Receives 28-sample IMU windows from ESP32 over USB-Serial (UART).
#  Classifies driving behaviour with a RandomForest trained in the notebook.
#
#  Artefacts required (same directory):
#      rf_model.pkl        ← notebook cell 8
#      scaler.pkl          ← notebook cell 5
#      label_encoder.pkl   ← notebook cell 5
#
#  Run:
#      pip install fastapi uvicorn pyserial joblib numpy pandas scikit-learn
#      uvicorn main:app --host 0.0.0.0 --port 8000 --reload
#
#  UART:
#      Set SERIAL_PORT env var if auto-detection picks the wrong port.
#      e.g.  SERIAL_PORT=COM3 uvicorn main:app ...
#            SERIAL_PORT=/dev/ttyUSB0 uvicorn main:app ...
#
#  Protocol:
#      ESP32 → PC : "WINDOW:<json>\n"
#      PC → ESP32 : "<json>\n"
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

MODEL_PATH   = os.getenv("MODEL_PATH",   "rf_model.pkl")
SCALER_PATH  = os.getenv("SCALER_PATH",  "scaler.pkl")
LE_PATH      = os.getenv("LE_PATH",      "label_encoder.pkl")

WINDOW_SIZE = 28   # must match notebook

# UART config
SERIAL_PORT      = os.getenv("SERIAL_PORT", "").strip()
SERIAL_BAUD      = int(os.getenv("SERIAL_BAUD", "115200"))
SERIAL_TIMEOUT   = float(os.getenv("SERIAL_TIMEOUT", "1.0"))
SERIAL_AUTODETECT = os.getenv("SERIAL_AUTODETECT", "1").strip() != "0"

# !! Column order MUST match notebook's FEATURES list !!
# FEATURES = ['GyroX', 'GyroY', 'GyroZ', 'AccX', 'AccY', 'AccZ']
SENSOR_COLS = ["GyroX", "GyroY", "GyroZ", "AccX", "AccY", "AccZ"]

# 6-class system from the notebook
CLASS_NAMES = {
    0: "Idle State",
    1: "Normal Driving",
    2: "Sudden Acceleration",
    3: "Sudden Right Turn",
    4: "Sudden Left Turn",
    5: "Sudden Brake",
}

HARSH_CLASSES = {2, 3, 4, 5}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("drivesense")

# ─── MODEL STATE ─────────────────────────────────────────────────────────────

class ModelState:
    model         = None
    scaler        = None
    le            = None   # LabelEncoder — maps encoded index ↔ original class id
    serial_conn   = None
    serial_thread = None
    serial_stop   = None
    serial_lock   = None

state = ModelState()

# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────
# Exact replica of the notebook's window_features() — 8 stats × 6 axes = 48.
#
#   Per column (in SENSOR_COLS order): mean, std, min, max, Q1, Q3, RMS, range
#
# DO NOT add or reorder features — the scaler and model were fitted on this
# exact 48-element vector.

def extract_window_features(window_df: pd.DataFrame) -> np.ndarray:
    """
    Returns a (1, 48) numpy array ready for scaler.transform() → model.predict().
    """
    feats: list[float] = []

    for col in SENSOR_COLS:           # order is critical
        s = window_df[col].values.astype(np.float64)

        feats.append(float(s.mean()))
        feats.append(float(s.std()))
        feats.append(float(s.min()))
        feats.append(float(s.max()))
        feats.append(float(np.percentile(s, 25)))        # Q1
        feats.append(float(np.percentile(s, 75)))        # Q3
        feats.append(float(np.sqrt(np.mean(s ** 2))))    # RMS
        feats.append(float(s.max() - s.min()))           # range

    return np.array(feats, dtype=np.float64).reshape(1, -1)   # (1, 48)


# ─── PREDICTION (shared by UART loop and HTTP endpoint) ──────────────────────

def _predict_from_rows(rows: list[dict]) -> dict:
    if len(rows) != WINDOW_SIZE:
        raise ValueError(
            f"window must contain exactly {WINDOW_SIZE} samples, got {len(rows)}"
        )

    window_df = pd.DataFrame(rows)
    X_live    = extract_window_features(window_df)   # (1, 48)
    X_live    = state.scaler.transform(X_live)

    raw_pred  = int(state.model.predict(X_live)[0])  # encoded label (0-5)
    proba     = state.model.predict_proba(X_live)[0] # shape (6,)

    # Decode back to original class id via the notebook's LabelEncoder
    class_id  = int(state.le.inverse_transform([raw_pred])[0])
    confidence = float(proba[raw_pred]) * 100.0

    all_probs = {
        CLASS_NAMES[int(state.le.inverse_transform([enc])[0])]: round(float(p) * 100, 2)
        for enc, p in enumerate(proba)
    }

    result = {
        "class_id":   class_id,
        "class_name": CLASS_NAMES[class_id],
        "confidence": round(confidence, 2),
        "all_probs":  all_probs,
        "is_harsh":   class_id in HARSH_CLASSES,
    }

    log.info(
        "Prediction → [%d] %s  (%.1f%%)  harsh=%s",
        class_id, CLASS_NAMES[class_id], confidence, class_id in HARSH_CLASSES,
    )

    # Pretty-print to server console (mirrors ESP32 serial output style)
    print("┌──────────────────────────────────────┐")
    print(f"│  Event      : {CLASS_NAMES[class_id]:<22} │")
    print(f"│  Class ID   : {class_id:<22} │")
    print(f"│  Confidence : {confidence:<21.1f}% │")
    print(f"│  Harsh      : {'YES' if class_id in HARSH_CLASSES else 'NO':<22} │")
    print("└──────────────────────────────────────┘")

    return result


# ─── UART HELPERS ────────────────────────────────────────────────────────────

def _pick_serial_port() -> str:
    if SERIAL_PORT:
        return SERIAL_PORT

    if not SERIAL_AUTODETECT:
        raise RuntimeError(
            "SERIAL_PORT is not set and SERIAL_AUTODETECT=0. "
            "Set SERIAL_PORT to your ESP32 port (e.g. COM3 or /dev/ttyUSB0)."
        )

    try:
        from serial.tools import list_ports
    except Exception as exc:
        raise RuntimeError("Could not import serial.tools.list_ports.") from exc

    ports = list(list_ports.comports())
    if not ports:
        raise RuntimeError("No serial ports found. Set SERIAL_PORT explicitly.")

    preferred = [
        p.device for p in ports
        if any(tag in (p.device or "").lower()   for tag in ("ttyusb", "ttyacm", "com"))
        or any(tag in (p.description or "").lower() for tag in ("usb", "serial", "uart", "acm"))
        or "usb" in (p.hwid or "").lower()
    ]

    if len(preferred) == 1:
        return preferred[0]
    if len(ports) == 1:
        return ports[0].device

    log.warning("Multiple ports found — auto-selecting %s.", ports[0].device)
    for p in ports:
        log.info("  Available: %s | %s | %s", p.device, p.description, p.hwid)
    return ports[0].device


def _write_line(conn: serial.Serial, payload: dict) -> None:
    line = json.dumps(payload, separators=(",", ":")) + "\n"
    conn.write(line.encode("utf-8"))
    conn.flush()


def _uart_loop() -> None:
    conn = state.serial_conn
    log.info("UART listener started on %s @ %d baud", conn.port, conn.baudrate)

    while not state.serial_stop.is_set():
        try:
            raw = conn.readline()
            if not raw:
                continue

            line = raw.decode("utf-8", errors="ignore").strip()
            if not line or not line.startswith("WINDOW:"):
                # Pass-through: ESP32 debug lines just go to the console
                if line:
                    print("[ESP32]", line)
                continue

            payload = line[len("WINDOW:"):].strip()
            if not payload:
                log.warning("Received empty WINDOW payload.")
                continue

            try:
                doc = json.loads(payload)
            except json.JSONDecodeError as exc:
                log.error("Bad JSON from UART: %s", exc)
                continue

            rows = doc.get("window")
            if not isinstance(rows, list):
                log.warning("UART payload missing 'window' list.")
                continue

            try:
                result = _predict_from_rows(rows)
            except Exception as exc:
                log.exception("Prediction failed: %s", exc)
                _write_line(conn, {
                    "class_id": -1, "class_name": "Error",
                    "confidence": 0.0, "all_probs": {}, "error": str(exc),
                })
                continue

            _write_line(conn, result)

        except (serial.SerialException, OSError) as exc:
            if state.serial_stop.is_set():
                break
            log.error("UART error: %s — retrying in 1 s", exc)
            time.sleep(1.0)
        except Exception as exc:
            log.exception("Unexpected UART loop error: %s", exc)
            time.sleep(0.2)

    log.info("UART listener stopped.")


# ─── LIFESPAN ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Verify all three artefacts exist before loading
    missing = [p for p in (MODEL_PATH, SCALER_PATH, LE_PATH) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing artefact(s): {missing}\n"
            "Run the notebook (cells 5-8) to generate them first."
        )

    state.model  = joblib.load(MODEL_PATH)
    state.scaler = joblib.load(SCALER_PATH)
    state.le     = joblib.load(LE_PATH)

    log.info("Model   loaded : %s  (%s)", MODEL_PATH,  type(state.model).__name__)
    log.info("Scaler  loaded : %s", SCALER_PATH)
    log.info("Encoder loaded : %s  classes=%s", LE_PATH, list(state.le.classes_))

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
        time.sleep(2.0)   # let ESP32 boot after DTR toggle
        log.info("Serial opened  : %s @ %d", port, SERIAL_BAUD)

        state.serial_thread = threading.Thread(target=_uart_loop, daemon=True)
        state.serial_thread.start()
    except Exception as exc:
        log.error("UART unavailable: %s — HTTP endpoint still works.", exc)

    log.info("DriveSense backend ready.")
    try:
        yield
    finally:
        if state.serial_stop:
            state.serial_stop.set()
        if state.serial_conn:
            try:
                state.serial_conn.close()
            except Exception:
                pass
        if state.serial_thread and state.serial_thread.is_alive():
            state.serial_thread.join(timeout=2.0)


# ─── APP ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DriveSense",
    description="Driving behaviour detection via IMU window classification (UART + HTTP)",
    version="2.0.0",
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
    AccX:  float = Field(..., description="Acceleration X (g)")
    AccY:  float = Field(..., description="Acceleration Y (g)")
    AccZ:  float = Field(..., description="Acceleration Z (g)")
    GyroX: float = Field(..., description="Gyroscope X (°/s)")
    GyroY: float = Field(..., description="Gyroscope Y (°/s)")
    GyroZ: float = Field(..., description="Gyroscope Z (°/s)")


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
    confidence: float
    all_probs:  dict[str, float]
    is_harsh:   bool


# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
async def root():
    return {
        "status":       "ok",
        "service":      "DriveSense",
        "model":        type(state.model).__name__ if state.model else "not loaded",
        "classes":      CLASS_NAMES,
        "uart_enabled": state.serial_conn is not None and state.serial_conn.is_open,
    }


@app.post("/predict", response_model=PredictResponse, summary="Classify a driving-behaviour window")
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
    return PredictResponse(**_predict_from_rows(rows))


@app.post(
    "/predict/raw",
    response_model=PredictResponse,
    summary="Classify using one reading repeated into a window (testing only)",
)
async def predict_raw(reading: SensorReading):
    fake_req = PredictRequest(window=[reading] * WINDOW_SIZE)
    return await predict(fake_req)