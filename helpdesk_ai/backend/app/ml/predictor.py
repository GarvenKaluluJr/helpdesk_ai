# helpdesk_ai/backend/app/ml/predictor.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Any

import joblib

# Where the trained model is saved
MODEL_PATH = (
    Path(__file__).resolve().parent / "models" / "ticket_category_model.joblib"
)

# This will hold either:
#  - a sklearn Pipeline (TF-IDF + LogisticRegression), OR/AND
#  - a small wrapper around the old {"vectorizer": ..., "classifier": ...} dict.
_classifier: Any = None
PREDICTOR_LOADED: bool = False  # main.py imports this


class _DictModelWrapper:
    """
    Backwards-compatibility wrapper for old saved models in the form:
      {"vectorizer": tfidf, "classifier": clf}
    so that predict_category() keeps working after Phase 8
    """

    def __init__(self, vectorizer, classifier):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.classes_ = getattr(classifier, "classes_", None)

    def _transform(self, texts):
        return self.vectorizer.transform(texts)

    def predict(self, texts):
        X = self._transform(texts)
        return self.classifier.predict(X)

    def predict_proba(self, texts):
        if hasattr(self.classifier, "predict_proba"):
            X = self._transform(texts)
            return self.classifier.predict_proba(X)
        raise AttributeError("Underlying classifier has no predict_proba")


def _load_model() -> None:
    """Load the model from disk (pipeline or old dict)."""
    global _classifier, PREDICTOR_LOADED

    if not MODEL_PATH.exists():
        _classifier = None
        PREDICTOR_LOADED = False
        print(f"[ML] No classifier model found at {MODEL_PATH}")
        return

    try:
        obj = joblib.load(MODEL_PATH)

# Preferred format: sklearn Pipeline
        if hasattr(obj, "predict"):
            _classifier = obj

# Old format from earlier phases: {"vectorizer": ..., "classifier": ...}
        elif isinstance(obj, dict) and "vectorizer" in obj and "classifier" in obj:
            _classifier = _DictModelWrapper(
                vectorizer=obj["vectorizer"],
                classifier=obj["classifier"],
            )
        else:
            print(f"[ML] Unrecognized model object type: {type(obj)!r}")
            _classifier = None

        PREDICTOR_LOADED = _classifier is not None
        if PREDICTOR_LOADED:
            print(f"[ML] Loaded ticket classifier from {MODEL_PATH}")
        else:
            print(f"[ML] Failed to load valid classifier from {MODEL_PATH}")
    except Exception as exc:
        _classifier = None
        PREDICTOR_LOADED = False
        print(f"[ML] Error loading model from {MODEL_PATH}: {exc}")


# Load at import time
_load_model()


def predict_category(text: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Return (predicted_label, confidence) or (None, None) if model not available.

    Confidence is the highest predicted probability if the underlying model
    supports predict_proba; otherwise confidence is None.
    """
    if not PREDICTOR_LOADED or _classifier is None:
        return None, None

    try:
# Use probabilities
        if hasattr(_classifier, "predict_proba"):
            probs = _classifier.predict_proba([text])[0]
# sklearn classifiers expose `classes_`
            classes = getattr(_classifier, "classes_", None)
            if classes is None:
# Fallback: just take argmax but no class labels
                idx = probs.argmax()
                return None, float(probs[idx])
            idx = probs.argmax()
            label = classes[idx]
            return str(label), float(probs[idx])
# Fallback: simple predict without probabilities
        label = _classifier.predict([text])[0]
        return str(label), None

    except Exception as exc:
        print(f"[ML] prediction error: {exc}")
        return None, None


def compute_priority(text: str, category: Optional[str]) -> str:
    """
    Very simple rules-based priority:

      - High: urgent / cannot login / system down / exam / deadline / payment failed.
      - Medium: technical/account/finance/admin without urgent flags.
      - Low: everything else.

    This is intentionally simple for the project the ML part is only for category.
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

# IT / technical queues
    if cat in {"technical", "tech", "account", "access"}:
        return "IT"

# Finance / billing
    if cat in {"finance", "financy", "billing"}:
        return "Finance"

# Admissions / admin
    if cat in {"administration", "admissions"}:
        return "Admissions"

# Default
    return "General"
