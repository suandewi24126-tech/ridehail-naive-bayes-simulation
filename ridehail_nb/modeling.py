"""Multinomial Naive Bayes training and evaluation."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .preprocessing import clean_text

CLASS_NAMES = {0: "Objektif", 1: "Subjektif"}


def build_pipeline(alpha: float = 1.0) -> Pipeline:
    """Build a reproducible TF-IDF + MultinomialNB pipeline."""
    if alpha <= 0:
        raise ValueError("Nilai alpha harus lebih besar dari 0.")

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=15000,
                    sublinear_tf=True,
                ),
            ),
            ("nb", MultinomialNB(alpha=float(alpha))),
        ]
    )


def calculate_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def fit_and_evaluate(
    df: pd.DataFrame,
    alpha: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Split, train, and evaluate without fitting TF-IDF on test data."""
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size sebaiknya berada pada rentang 0.1 sampai 0.5.")

    x_train, x_test, y_train, y_test = train_test_split(
        df["review"].astype(str),
        df["label"].astype(int),
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    model = build_pipeline(alpha=alpha)
    start_train = perf_counter()
    model.fit(x_train, y_train)
    training_seconds = perf_counter() - start_train

    start_predict = perf_counter()
    predictions = model.predict(x_test)
    prediction_seconds = perf_counter() - start_predict

    metrics = calculate_metrics(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Objektif", "Subjektif"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "model": model,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "classification_report": report,
        "y_test": y_test,
        "predictions": predictions,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "training_seconds": float(training_seconds),
        "prediction_seconds": float(prediction_seconds),
        "alpha": float(alpha),
        "random_state": int(random_state),
    }


def predict_review(model: Pipeline, review: str) -> dict[str, Any]:
    if not review or not review.strip():
        raise ValueError("Teks ulasan tidak boleh kosong.")
    prediction = int(model.predict([review])[0])
    probabilities = model.predict_proba([review])[0]
    classes = list(model.named_steps["nb"].classes_)
    probability_by_class = {int(cls): float(prob) for cls, prob in zip(classes, probabilities)}
    return {
        "label": prediction,
        "label_name": CLASS_NAMES[prediction],
        "confidence": probability_by_class.get(prediction, 0.0),
        "probability_objective": probability_by_class.get(0, 0.0),
        "probability_subjective": probability_by_class.get(1, 0.0),
    }
