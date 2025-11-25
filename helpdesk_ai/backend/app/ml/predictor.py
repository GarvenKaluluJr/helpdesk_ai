# helpdesk_ai/backend/app/ml/predictor.py

from pathlib import Path
from typing import Optional, Tuple

import joblib

MODEL_PATH = Path(__file__).resolve().parent / "models" / "ticket_category_model.joblib"

_pipeline = None


def _load_model():
    global _pipeline
    if MODEL_PATH.exists():
        _pipeline = joblib.load(MODEL_PATH)
        print(f"[ML] Loaded ticket classifier from {MODEL_PATH}")
    else:
        _pipeline = None
        print("[ML] No classifier model found yet.")


_load_model()


def predict_category(subject: str, body: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Returns (predicted_category, confidence) or (None, None) if model not available.
    Confidence is the max predicted probability in [0,1].
    """
    if _pipeline is None:
        return None, None

    text = f"{subject or ''} {body or ''}".strip()
    if not text:
        return None, None

    proba = _pipeline.predict_proba([text])[0]
    classes = _pipeline.classes_

    # index of max probability
    best_idx = max(range(len(proba)), key=lambda i: proba[i])
    label = classes[best_idx]
    confidence = float(proba[best_idx])

    return label, confidence
