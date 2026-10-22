"""Local, paired downstream classification benchmark for anonymized text."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline


def fit_classifier(train_texts: list[str], train_labels: list[str]):
    """Fit a classifier on training data only; reuse it unchanged for both test inputs."""
    if not train_texts or len(train_texts) != len(train_labels):
        raise ValueError("Training texts and labels must be nonempty and aligned.")
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True),
        LogisticRegression(C=4, max_iter=1000, random_state=20261022),
    )
    return model.fit(train_texts, train_labels)


def _accuracy_delta_ci(original_ok: np.ndarray, anonymized_ok: np.ndarray, seed: int) -> list[float]:
    """Percentile interval from paired resampling of document indices."""
    difference = anonymized_ok.astype(np.int8) - original_ok.astype(np.int8)
    rng = np.random.default_rng(seed)
    draws = np.empty(2000)
    batch_size = max(1, min(128, 1_000_000 // len(difference)))
    for start in range(0, len(draws), batch_size):
        end = min(start + batch_size, len(draws))
        indices = rng.integers(len(difference), size=(end - start, len(difference)))
        draws[start:end] = difference[indices].mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def _summary(labels: np.ndarray, original: np.ndarray, anonymized: np.ndarray,
             classes: np.ndarray, seed: int) -> dict:
    count = len(labels)
    if not count:
        return {"count": 0, "original_accuracy": None, "anonymized_accuracy": None,
                "accuracy_delta": None, "original_macro_f1": None, "anonymized_macro_f1": None,
                "macro_f1_delta": None, "original_correct_to_wrong": 0, "original_wrong_to_correct": 0,
                "original_correct_retention": None, "accuracy_delta_ci_95": None}
    original_ok = original == labels
    anonymized_ok = anonymized == labels
    original_accuracy = float(original_ok.mean())
    anonymized_accuracy = float(anonymized_ok.mean())
    original_f1 = float(f1_score(labels, original, labels=classes, average="macro", zero_division=0))
    anonymized_f1 = float(f1_score(labels, anonymized, labels=classes, average="macro", zero_division=0))
    original_correct = int(original_ok.sum())
    return {
        "count": count,
        "original_accuracy": original_accuracy,
        "anonymized_accuracy": anonymized_accuracy,
        "accuracy_delta": anonymized_accuracy - original_accuracy,
        "original_macro_f1": original_f1,
        "anonymized_macro_f1": anonymized_f1,
        "macro_f1_delta": anonymized_f1 - original_f1,
        "original_correct_to_wrong": int((original_ok & ~anonymized_ok).sum()),
        "original_wrong_to_correct": int((~original_ok & anonymized_ok).sum()),
        "original_correct_retention": float((original_ok & anonymized_ok).sum() / original_correct)
        if original_correct else None,
        "accuracy_delta_ci_95": _accuracy_delta_ci(original_ok, anonymized_ok, seed),
    }


def _class_accuracy(labels: np.ndarray, original: np.ndarray, anonymized: np.ndarray,
                    selected: np.ndarray) -> dict:
    count = int(selected.sum())
    if not count:
        return {"support": 0, "original_accuracy": None, "anonymized_accuracy": None,
                "accuracy_delta": None}
    original_accuracy = float((original[selected] == labels[selected]).mean())
    anonymized_accuracy = float((anonymized[selected] == labels[selected]).mean())
    return {"support": count, "original_accuracy": original_accuracy,
            "anonymized_accuracy": anonymized_accuracy,
            "accuracy_delta": anonymized_accuracy - original_accuracy}


def evaluate_classifier(model, originals: list[str], outputs: list[str],
                        labels: list[str], seed: int = 20261022,
                        available: list[bool] | None = None) -> dict:
    """Score paired text; unavailable outputs abstain and remain in the denominator."""
    if not originals or len(originals) != len(outputs) or len(originals) != len(labels):
        raise ValueError("Originals, outputs, and labels must be nonempty and aligned.")
    if any(not isinstance(text, str) for text in originals + outputs):
        raise ValueError("Original and anonymized inputs must be strings.")
    if available is not None and (type(available) is not list or len(available) != len(originals)
                                  or any(type(value) is not bool for value in available)):
        raise ValueError("Availability must be a boolean list aligned with the inputs.")
    availability = np.asarray(available if available is not None else [True] * len(originals))
    classes = np.asarray(model.classes_)
    truth = np.asarray(labels)
    if not np.isin(truth, classes).all():
        raise ValueError("Evaluation labels must be among the classifier's training classes.")
    original_predictions = np.asarray(model.predict(originals))
    abstention = "__CONTEXTVEIL_ABSTAIN__"
    while abstention in classes:
        abstention += "_"
    anonymized_predictions = np.full(len(outputs), abstention, dtype=object)
    if availability.any():
        predictions = np.asarray(model.predict([text for text, ok in zip(outputs, availability) if ok]))
        if len(predictions) != int(availability.sum()):
            raise ValueError("The classifier returned the wrong number of predictions.")
        anonymized_predictions[availability] = predictions
    if len(original_predictions) != len(truth):
        raise ValueError("The classifier returned the wrong number of predictions.")
    changed = np.fromiter((ok and before != after for before, after, ok in
                           zip(originals, outputs, availability)),
                          dtype=bool, count=len(originals))
    by_class = []
    for index, label in enumerate(classes):
        selected = truth == label
        by_class.append({
            "class_index": index,
            "overall": _class_accuracy(truth, original_predictions, anonymized_predictions, selected),
            "changed": _class_accuracy(truth, original_predictions, anonymized_predictions,
                                       selected & changed),
        })
    return {
        "schema_version": 1,
        "overall": _summary(truth, original_predictions, anonymized_predictions, classes, seed),
        "changed": {**_summary(truth[changed], original_predictions[changed],
                                anonymized_predictions[changed], classes, seed),
                    "rate": float(changed.mean())},
        "unchanged_count": int((availability & ~changed).sum()),
        "failed_output_count": int((~availability).sum()),
        "abstention_count": int((~availability).sum()),
        "by_class": by_class,
    }
