from __future__ import annotations

import base64
import io
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from digit_modeling import DEFAULT_MNIST_LIMIT, build_classifier, load_mnist_subset

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "digit_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
MNIST_CACHE_PATH = MODEL_DIR / "mnist_subset.joblib"
FEEDBACK_FILE = BASE_DIR / "feedback_data.csv"

FEEDBACK_COLUMNS = [
    "timestamp",
    "predicted_digit",
    "correct_digit",
    "image_b64",
    "confidence",
]
MAX_FEEDBACK_SAMPLES = 2_500
FEEDBACK_WEIGHT = 4
BLANK_THRESHOLD = 24

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

model = None
scaler = None
model_load_error: str | None = None
model_lock = threading.RLock()
retraining_lock = threading.Lock()


def json_error(message: str, status_code: int, **details: Any):
    payload = {"error": message}
    payload.update(details)
    return jsonify(payload), status_code


def ensure_feedback_store() -> None:
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    if FEEDBACK_FILE.exists():
        return
    pd.DataFrame(columns=FEEDBACK_COLUMNS).to_csv(FEEDBACK_FILE, index=False)


def read_feedback_frame() -> pd.DataFrame:
    ensure_feedback_store()

    try:
        feedback_frame = pd.read_csv(FEEDBACK_FILE)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)

    if set(FEEDBACK_COLUMNS).issubset(feedback_frame.columns):
        return feedback_frame[FEEDBACK_COLUMNS]

    legacy_frame = pd.read_csv(FEEDBACK_FILE, names=FEEDBACK_COLUMNS, header=None)
    legacy_frame = legacy_frame[legacy_frame["timestamp"] != "timestamp"]
    return legacy_frame


def append_feedback_row(feedback_row: dict[str, Any]) -> None:
    ensure_feedback_store()
    header_needed = FEEDBACK_FILE.stat().st_size == 0
    pd.DataFrame([feedback_row]).to_csv(
        FEEDBACK_FILE,
        mode="a",
        header=header_needed,
        index=False,
    )


def validate_digit(value: Any, field_name: str) -> int:
    try:
        digit = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer between 0 and 9.") from exc

    if digit < 0 or digit > 9:
        raise ValueError(f"{field_name} must be an integer between 0 and 9.")

    return digit


def validate_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be a number between 0 and 1.") from exc

    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be a number between 0 and 1.")

    return confidence


def decode_data_url_to_image(image_data: Any) -> Image.Image:
    if not isinstance(image_data, str) or not image_data.strip():
        raise ValueError("No image data provided.")

    encoded_image = image_data
    if image_data.startswith("data:image"):
        _, encoded_image = image_data.split(",", 1)

    try:
        image_bytes = base64.b64decode(encoded_image, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Image data is not valid base64.") from exc

    try:
        image = Image.open(io.BytesIO(image_bytes))
        return ImageOps.grayscale(image)
    except OSError as exc:
        raise ValueError("Image payload could not be decoded.") from exc


def extract_digit_vector(image_data: Any) -> np.ndarray:
    image = decode_data_url_to_image(image_data)
    image = ImageOps.fit(image, (280, 280), method=Image.Resampling.LANCZOS)

    pixel_array = np.asarray(image, dtype=np.float32)
    inverted = 255.0 - pixel_array
    ink_mask = inverted > BLANK_THRESHOLD

    if not np.any(ink_mask):
        raise ValueError("Draw a digit before asking for a prediction.")

    rows, cols = np.where(ink_mask)
    top, bottom = rows.min(), rows.max()
    left, right = cols.min(), cols.max()

    cropped = Image.fromarray(
        inverted[top : bottom + 1, left : right + 1].astype(np.uint8),
        mode="L",
    )
    resized = ImageOps.contain(cropped, (20, 20), method=Image.Resampling.LANCZOS)

    centered = Image.new("L", (28, 28), color=0)
    offset = ((28 - resized.width) // 2, (28 - resized.height) // 2)
    centered.paste(resized, offset)

    return np.asarray(centered, dtype=np.float32).reshape(1, -1) / 255.0


def load_model_assets(force_reload: bool = False) -> bool:
    global model, scaler, model_load_error

    with model_lock:
        if not force_reload and model is not None and scaler is not None:
            return True

        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            model_load_error = None
            return True
        except FileNotFoundError:
            model = None
            scaler = None
            model_load_error = (
                "Model artifacts are missing. Run train_model.py to generate them."
            )
            return False
        except Exception as exc:
            model = None
            scaler = None
            model_load_error = f"Model assets could not be loaded: {exc}"
            return False


def build_prediction_payload(pixel_vector: np.ndarray) -> dict[str, Any]:
    if not load_model_assets():
        raise RuntimeError(model_load_error or "Model is not available.")

    with model_lock:
        scaled_vector = scaler.transform(pixel_vector)
        probabilities = model.predict_proba(scaled_vector)[0]
        classes = np.asarray(model.classes_, dtype=int)

    ranked = sorted(
        zip(classes.tolist(), probabilities.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    predicted_digit, confidence = ranked[0]

    return {
        "digit": int(predicted_digit),
        "confidence": float(confidence),
        "probabilities": {
            str(int(digit)): float(probability)
            for digit, probability in zip(classes, probabilities)
        },
        "top_predictions": [
            {"digit": int(digit), "confidence": float(probability)}
            for digit, probability in ranked[:3]
        ],
    }


def load_feedback_training_data(
    limit: int = MAX_FEEDBACK_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, int]:
    feedback_frame = read_feedback_frame().tail(limit)
    training_vectors: list[np.ndarray] = []
    training_labels: list[int] = []
    skipped_rows = 0

    for row in feedback_frame.itertuples(index=False):
        try:
            digit = validate_digit(row.correct_digit, "correct_digit")
            vector = extract_digit_vector(str(row.image_b64))[0]
        except ValueError:
            skipped_rows += 1
            continue

        training_vectors.append(vector)
        training_labels.append(digit)

    if not training_vectors:
        return np.empty((0, 784), dtype=np.float32), np.empty((0,), dtype=np.int32), skipped_rows

    return (
        np.asarray(training_vectors, dtype=np.float32),
        np.asarray(training_labels, dtype=np.int32),
        skipped_rows,
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}

    try:
        pixel_vector = extract_digit_vector(payload.get("image"))
        return jsonify(build_prediction_payload(pixel_vector))
    except ValueError as exc:
        return json_error(str(exc), 400)
    except RuntimeError as exc:
        return json_error(str(exc), 503)
    except Exception as exc:
        return json_error("Prediction failed.", 500, details=str(exc))


@app.route("/feedback_stats", methods=["GET"])
def feedback_stats():
    feedback_frame = read_feedback_frame()

    if feedback_frame.empty:
        return jsonify(
            {
                "total_feedback": 0,
                "wrong_predictions": 0,
                "agreement_rate": 0.0,
                "average_confidence": 0.0,
            }
        )

    predicted = pd.to_numeric(feedback_frame["predicted_digit"], errors="coerce")
    corrected = pd.to_numeric(feedback_frame["correct_digit"], errors="coerce")
    confidence = pd.to_numeric(feedback_frame["confidence"], errors="coerce")
    valid_rows = predicted.notna() & corrected.notna()

    total_feedback = int(valid_rows.sum())
    wrong_predictions = int(
        (predicted[valid_rows].astype(int) != corrected[valid_rows].astype(int)).sum()
    )
    agreement_rate = ((total_feedback - wrong_predictions) / total_feedback) if total_feedback else 0.0
    average_confidence = (
        float(confidence[valid_rows].fillna(0).mean()) if total_feedback else 0.0
    )

    return jsonify(
        {
            "total_feedback": total_feedback,
            "wrong_predictions": wrong_predictions,
            "agreement_rate": round(agreement_rate, 4),
            "average_confidence": round(average_confidence, 4),
        }
    )


@app.route("/feedback", methods=["POST"])
def feedback():
    payload = request.get_json(silent=True) or {}

    try:
        image_data = payload.get("image")
        predicted_digit = validate_digit(payload.get("predicted_digit"), "predicted_digit")
        correct_digit = validate_digit(payload.get("correct_digit"), "correct_digit")
        confidence = validate_confidence(payload.get("confidence", 0))
        extract_digit_vector(image_data)
    except ValueError as exc:
        return json_error(str(exc), 400)

    feedback_row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "predicted_digit": predicted_digit,
        "correct_digit": correct_digit,
        "image_b64": image_data,
        "confidence": confidence,
    }

    append_feedback_row(feedback_row)
    total_entries = len(read_feedback_frame())

    return jsonify(
        {
            "status": "saved",
            "total_entries": total_entries,
            "was_correction": predicted_digit != correct_digit,
        }
    )


@app.route("/retrain", methods=["POST"])
def retrain():
    global model, scaler, model_load_error

    if not retraining_lock.acquire(blocking=False):
        return json_error("A retraining job is already running.", 409)

    try:
        X_mnist, y_mnist = load_mnist_subset(
            limit=DEFAULT_MNIST_LIMIT,
            cache_path=MNIST_CACHE_PATH,
        )
        feedback_X, feedback_y, skipped_rows = load_feedback_training_data()

        if len(feedback_y):
            weighted_feedback_X = np.repeat(feedback_X, FEEDBACK_WEIGHT, axis=0)
            weighted_feedback_y = np.repeat(feedback_y, FEEDBACK_WEIGHT)
            X = np.vstack([X_mnist, weighted_feedback_X])
            y = np.concatenate([y_mnist, weighted_feedback_y])
        else:
            X, y = X_mnist, y_mnist

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        new_scaler = StandardScaler()
        X_train_scaled = new_scaler.fit_transform(X_train)
        X_test_scaled = new_scaler.transform(X_test)

        new_model = build_classifier(random_state=42)
        new_model.fit(X_train_scaled, y_train)
        test_accuracy = float(new_model.score(X_test_scaled, y_test))

        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(new_model, MODEL_PATH)
        joblib.dump(new_scaler, SCALER_PATH)

        with model_lock:
            model = new_model
            scaler = new_scaler
            model_load_error = None

        return jsonify(
            {
                "status": "success",
                "test_accuracy": round(test_accuracy, 4),
                "samples_used": int(len(X)),
                "feedback_used": int(len(feedback_y)),
                "skipped_feedback": skipped_rows,
            }
        )
    except Exception as exc:
        return json_error("Retraining failed.", 500, details=str(exc))
    finally:
        retraining_lock.release()


@app.route("/health", methods=["GET"])
def health():
    feedback_count = len(read_feedback_frame())
    model_ready = load_model_assets()

    return jsonify(
        {
            "status": "healthy" if model_ready else "degraded",
            "model_loaded": model_ready,
            "retraining": retraining_lock.locked(),
            "feedback_count": feedback_count,
            "model_error": model_load_error,
        }
    )


load_model_assets()


if __name__ == "__main__":
    app.run(
        debug=os.getenv("FLASK_DEBUG", "1") == "1",
        host="127.0.0.1",
        port=5000,
    )
