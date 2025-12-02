# helpdesk_ai/backend/app/ml/train_classifier.py
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from ..db import SessionLocal
from ..models.training_sample import TrainingSample
from ..models.training_run import TrainingRun

# Where the trained model lives
MODEL_PATH = Path(__file__).resolve().parent / "models" / "ticket_category_model.joblib"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


# Baseline keyword classifier

def baseline_predict(text: str) -> str:
    """
    Very simple keyword-based baseline for categories.
    This is ONLY for metrics comparison, not used in production.
    """
    t = text.lower()

    if any(k in t for k in ["password", "login", "log in", "account", "sign in"]):
        return "Account"

    if any(k in t for k in ["fee", "tuition", "payment", "invoice", "billing", "money"]):
        return "Financy"

    if any(k in t for k in ["admission", "admissions", "enrol", "enroll", "application"]):
        return "Administration"

    return "General"



# Load labelled data from DB

def _load_training_data() -> Tuple[List[str], List[str]]:
    db = SessionLocal()
    try:
        samples = db.query(TrainingSample).all()
        texts = [f"{s.subject or ''}\n{s.body or ''}" for s in samples]
        labels = [s.true_category for s in samples]
        return texts, labels
    finally:
        db.close()


def train_and_save_from_db() -> dict | None:
    """
    Phase 8 main entrypoint.

    - Reads labelled samples from training_samples table
    - Trains TF-IDF + LogisticRegression
    - Compares to baseline keyword rules
    - Saves model to disk
    - Stores metrics in training_runs table
    """
    texts, labels = _load_training_data()
    if not texts:
        print("[TRAIN] No training samples found in training_samples table.")
        return None

    label_counts = Counter(labels)
    classes = sorted(label_counts.keys())
    print(f"[TRAIN] Loaded {len(texts)} samples, {len(classes)} classes: {classes}")
    print(f"[TRAIN] Label counts: {label_counts}")

# NEW: only do a train/test split when the dataset is big enough 
    MIN_SAMPLES_PER_CLASS_FOR_SPLIT = 5
    MIN_TOTAL_SAMPLES_FOR_SPLIT = 50

    min_count = min(label_counts.values())
    use_train_test = (
        min_count >= MIN_SAMPLES_PER_CLASS_FOR_SPLIT
        and len(texts) >= MIN_TOTAL_SAMPLES_FOR_SPLIT
    )

    if use_train_test:
        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=0.3,
            stratify=labels,
            random_state=42,
        )
        print(f"[TRAIN] Using train/test split: {len(X_train)} train, {len(X_test)} test")
        eval_mode = "train/test split"
    else:
# For small datasets (your current case), train and evaluate on ALL samples.
        X_train, y_train = texts, labels
        X_test, y_test = texts, labels
        eval_mode = "full dataset (no hold-out)"
        print(
            "[TRAIN] Dataset is small → training & evaluating on ALL samples "
            "(no separate test set)."
        )

# TF-IDF + Logistic Regression model
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=1,
    )
    X_train_vec = vectorizer.fit_transform(X_train)

    clf = LogisticRegression(
        max_iter=1000,
        multi_class="auto",
        n_jobs=None,
    )
    clf.fit(X_train_vec, y_train)

# ML predictions
    X_test_vec = vectorizer.transform(X_test)
    y_pred_ml = clf.predict(X_test_vec)

    acc_ml = float(accuracy_score(y_test, y_pred_ml))
    macro_f1_ml = float(f1_score(y_test, y_pred_ml, average="macro"))
    report_ml = classification_report(y_test, y_pred_ml, output_dict=True)

    print(
        f"[TRAIN] [{eval_mode}] ML accuracy={acc_ml:.3f}, "
        f"macro F1={macro_f1_ml:.3f}"
    )

# Baseline predictions
    y_pred_base = [baseline_predict(t) for t in X_test]
    acc_base = float(accuracy_score(y_test, y_pred_base))
    macro_f1_base = float(f1_score(y_test, y_pred_base, average="macro"))
    report_base = classification_report(y_test, y_pred_base, output_dict=True)

    print(
        f"[TRAIN] [{eval_mode}] Baseline accuracy={acc_base:.3f}, "
        f"macro F1={macro_f1_base:.3f}"
    )

# Save model
    joblib.dump(
        {"vectorizer": vectorizer, "classifier": clf},
        MODEL_PATH,
    )
    print(f"[TRAIN] Saved model to {MODEL_PATH}")

# Save metrics in DB
    db = SessionLocal()
    try:
        metrics = {
            "eval_mode": eval_mode,
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
