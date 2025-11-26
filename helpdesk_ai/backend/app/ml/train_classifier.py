# helpdesk_ai/backend/app/ml/train_classifier.py

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict, Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ..db import SessionLocal
from ..models.training_sample import TrainingSample
from ..models.training_run import TrainingRun

# Where the trained model lives
MODEL_PATH = Path(__file__).resolve().parent / "models" / "ticket_category_model.joblib"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Baseline keyword classifier
# -----------------------------

def baseline_predict(text: str) -> str:
    """
    Very simple keyword-based baseline for categories.
    This is ONLY for metrics comparison, not used in production.
    """
    t = (text or "").lower()

    # Account / access
    if any(k in t for k in ["password", "login", "log in", "account", "sign in"]):
        return "Account"

    # Finance / billing
    if any(k in t for k in ["fee", "tuition", "payment", "invoice", "billing", "money"]):
        return "Financy"

    # Administration / admissions
    if any(k in t for k in ["admission", "admissions", "enrol", "enroll", "application"]):
        return "Administration"

    # Technical-ish words
    if any(k in t for k in ["server", "system", "error", "bug", "lms", "portal"]):
        return "Technical"

    # Fallback
    return "General"


# -----------------------------
# Training data from DB
# -----------------------------

def _load_training_data() -> Tuple[List[str], List[str]]:
    """
    Read labelled samples from training_samples table.
    Returns (texts, labels) where each text is "subject\\nbody".
    """
    db = SessionLocal()
    try:
        samples = db.query(TrainingSample).all()
        texts = [f"{s.subject or ''}\n{s.body or ''}" for s in samples]
        labels = [s.true_category for s in samples]
        return texts, labels
    finally:
        db.close()


def train_and_save_from_db() -> Dict[str, Any] | None:
    """
    Phase 8 main entrypoint.

    1. Load labelled tickets from training_samples.
    2. Train TF-IDF + LogisticRegression inside a single sklearn Pipeline.
    3. Evaluate against either:
         - a stratified train/test split (if we have >=2 per class), or
         - the whole dataset (train & test on same data) for tiny datasets.
    4. Compare with a keyword baseline (baseline_predict).
    5. Save the trained pipeline to MODEL_PATH.
    6. Store metrics in training_runs table.

    Returns metrics dict (also stored in DB), or None if no samples.
    """
    texts, labels = _load_training_data()
    if not texts:
        print("[TRAIN] No training samples found in training_samples table.")
        return None

    label_counts = Counter(labels)
    classes = sorted(label_counts.keys())
    print(f"[TRAIN] Loaded {len(texts)} samples, {len(classes)} classes: {classes}")
    print(f"[TRAIN] Label counts: {label_counts}")

    # Decide whether we can afford a real train/test split
    min_count = min(label_counts.values())
    use_train_test = min_count >= 2 and len(texts) >= 10

    if use_train_test:
        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=0.3,
            stratify=labels,
            random_state=42,
        )
        print(f"[TRAIN] Using train/test split: {len(X_train)} train, {len(X_test)} test")
    else:
        # Very small dataset – we train and evaluate on all samples.
        X_train, y_train = texts, labels
        X_test, y_test = texts, labels
        print("[TRAIN] Dataset is small / imbalanced → "
              "training & evaluating on ALL samples (no split).")

    # -------------------------------------------------
    #  TF-IDF + Logistic Regression pipeline
    # -------------------------------------------------
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=5000,
                    min_df=1,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=300,
                    n_jobs=1,
                    multi_class="auto",
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # Train ML model
    pipeline.fit(X_train, y_train)

    # ML predictions
    y_pred_ml = pipeline.predict(X_test)

    acc_ml = float(accuracy_score(y_test, y_pred_ml))
    macro_f1_ml = float(f1_score(y_test, y_pred_ml, average="macro"))
    report_ml = classification_report(y_test, y_pred_ml, output_dict=True)

    print(f"[TRAIN] ML accuracy={acc_ml:.3f}, macro F1={macro_f1_ml:.3f}")

    # -------------------------------------------------
    #  Baseline predictions
    # -------------------------------------------------
    y_pred_base = [baseline_predict(t) for t in X_test]
    acc_base = float(accuracy_score(y_test, y_pred_base))
    macro_f1_base = float(f1_score(y_test, y_pred_base, average="macro"))
    report_base = classification_report(y_test, y_pred_base, output_dict=True)

    print(f"[TRAIN] Baseline accuracy={acc_base:.3f}, macro F1={macro_f1_base:.3f}")

    # -------------------------------------------------
    #  Save model for production predictions
    # -------------------------------------------------
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[TRAIN] Saved model pipeline to {MODEL_PATH}")

    # -------------------------------------------------
    #  Save metrics in DB
    # -------------------------------------------------
    db = SessionLocal()
    try:
        metrics = {
            "ml": {
                "accuracy": acc_ml,
                "macro_f1": macro_f1_ml,
                "report": report_ml,
            },
            "baseline": {
                "accuracy": acc_base,
                "macro_f1": macro_f1_base,
                "report": report_base,
            },
        }
        run = TrainingRun(
            accuracy_ml=acc_ml,
            macro_f1_ml=macro_f1_ml,
            accuracy_baseline=acc_base,
            macro_f1_baseline=macro_f1_base,
        )
        run.set_report(metrics)
        db.add(run)
        db.commit()
        print(f"[TRAIN] Stored TrainingRun id={run.id} at {run.run_at}")
    finally:
        db.close()

    return metrics


if __name__ == "__main__":
    # CLI usage: python -m helpdesk_ai.backend.app.ml.train_classifier
    train_and_save_from_db()
