# helpdesk_ai/backend/app/ml/predictor.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import joblib

# Where the trained model is saved
MODEL_PATH = (
    Path(__file__).resolve().parent / "models" / "ticket_category_model.joblib"
)

_classifier = None
PREDICTOR_LOADED: bool = False  # <- this is what main.py wants to import


def _load_model() -> None:
    """Load the scikit-learn pipeline from disk, if it exists."""
    global _classifier, PREDICTOR_LOADED

    if MODEL_PATH.exists():
        _classifier = joblib.load(MODEL_PATH)
        PREDICTOR_LOADED = True
        print(f"[ML] Loaded ticket classifier from {MODEL_PATH}")
    else:
        _classifier = None
        PREDICTOR_LOADED = False
        print(f"[ML] No classifier model found at {MODEL_PATH}")


# Load at import time
_load_model()


def predict_category(text: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Return (predicted_label, confidence) or (None, None) if model not available.
    Confidence is taken from predict_proba if available.
    """
    if not PREDICTOR_LOADED or _classifier is None:
        return None, None

    try:
        if hasattr(_classifier, "predict_proba"):
            probs = _classifier.predict_proba([text])[0]
            idx = probs.argmax()
            label = _classifier.classes_[idx]
            return str(label), float(probs[idx])
        else:
            # fallback: no probabilities
            label = _classifier.predict([text])[0]
            return str(label), None
    except Exception as exc:
        print(f"[ML] prediction error: {exc}")
        return None, None


def compute_priority(text: str, category: Optional[str]) -> str:
    """
    Very simple rules-based priority:
      - High: urgent / cannot login / system down / deadline etc.
      - Medium: technical/account/finance/admin without urgent flags.
      - Low: everything else.
    """
    t = (text or "").lower()
    cat = (category or "").lower()

    urgent_keywords = ["urgent", "asap", "immediately", "right away"]
    critical_phrases = [
        "cannot login",
        "can't login",
        "cant login",
        "system down",
        "not working",
        "exam",
        "deadline today",
        "deadline tomorrow",
        "payment failed",
    ]

    if any(k in t for k in urgent_keywords) or any(p in t for p in critical_phrases):
        return "High"

    if cat in {
        "technical",
        "tech",
        "account",
        "access",
        "finance",
        "financy",
        "billing",
        "administration",
        "admissions",
    }:
        return "Medium"

    return "Low"


def route_to_queue(category: Optional[str]) -> str:
    """
    Map ticket category to department queue.
    Falls back to 'General' if we don't recognize it.
    """
    if not category:
        return "General"

    cat = category.strip().lower()

    if cat in {"technical", "tech", "account", "access"}:
        return "IT"

    if cat in {"finance", "financy", "billing"}:
        return "Finance"

    if cat in {"administration", "admissions"}:
        return "Admissions"

    return "General"
