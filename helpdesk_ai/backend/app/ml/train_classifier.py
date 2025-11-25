# helpdesk_ai/backend/app/ml/train_classifier.py

from collections import Counter
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ..db import SessionLocal
from ..models.ticket import Ticket

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "ticket_category_model.joblib"


def load_labeled_tickets():
    db = SessionLocal()
    try:
        rows = (
            db.query(Ticket)
            .filter(Ticket.category_final.isnot(None))
            .all()
        )
        texts, labels = [], []
        for t in rows:
            text = f"{t.subject or ''} {t.body or ''}".strip()
            if not text:
                continue
            texts.append(text)
            labels.append(t.category_final)
    finally:
        db.close()
    return texts, labels


def build_pipeline():
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=5000,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    n_jobs=1,
                    multi_class="auto",
                ),
            ),
        ]
    )


def train_and_save():
    texts, labels = load_labeled_tickets()

    if not texts:
        print("No labeled tickets (category_final) found. Set category_final on some tickets first.")
        return

    label_counts = Counter(labels)
    classes = sorted(label_counts.keys())
    print(f"Training on {len(texts)} samples, {len(classes)} classes: {classes}")
    print(f"Label counts: {label_counts}")

    pipeline = build_pipeline()

    # If any class has < 2 samples, skip stratified split
    if min(label_counts.values()) < 2:
        print("Some classes have < 2 samples; training on ALL data (no train/test split).")
        pipeline.fit(texts, labels)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {acc:.4f}\n")
        print("Classification report:\n")
        print(classification_report(y_test, y_pred))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save()
