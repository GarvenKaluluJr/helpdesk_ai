# helpdesk_ai/backend/app/ml/predictor.py
from pathlib import Path
from typing import Optional, Tuple

import joblib

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "ticket_category_model.joblib"


class TicketCategoryPredictor:
    def __init__(self) -> None:
        self.pipeline = None
        self._load()

    def _load(self) -> None:
        if MODEL_PATH.exists():
            self.pipeline = joblib.load(MODEL_PATH)
            print(f"[ML] Loaded ticket classifier from {MODEL_PATH}")
        else:
            print("[ML] No classifier model found yet.")
            self.pipeline = None

    def predict(self, text: str) -> Tuple[Optional[str], Optional[float]]:
        if not self.pipeline:
            return None, None
        probs = self.pipeline.predict_proba([text])[0]
        idx = probs.argmax()
        label = self.pipeline.classes_[idx]
        confidence = float(probs[idx])
        return str(label), confidence


predictor = TicketCategoryPredictor()


def compute_priority(text: str, predicted_category: Optional[str]) -> str:
    """
    Simple rule-based priority:

    High:
      - text contains "urgent", "cannot log", "exam", "deadline today", "system down", etc.
    Medium:
      - non-urgent but category is technical/account/finance-like.
    Low:
      - everything else.
    """
    txt = (text or "").lower()

    urgent_keywords = [
        "urgent",
        "asap",
        "immediately",
        "cannot log",
        "can't log",
        "cant log",
        "login problem",
        "log in problem",
        "system down",
        "server down",
        "not working",
        "exam",
        "examination",
        "deadline today",
        "due today",
    ]
    if any(k in txt for k in urgent_keywords):
        return "high"

    high_importance_categories = {
        "Account",
        "Technical",
        "Financy",   # your current spelling in data
        "Finance",
        "IT",
    }
    if predicted_category in high_importance_categories:
        return "medium"

    return "low"
