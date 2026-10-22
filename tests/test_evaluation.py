"""Metric arithmetic and fail-closed gold fixture validation."""

import json
from difflib import SequenceMatcher
from pathlib import Path

import pytest

from contextveil.evaluation import evaluate, evaluate_utility
from contextveil.models import ContextVeilError, Entity


class FixedDetector:
    name = "fixed"
    contextual = False

    def __init__(self, entities_by_text):
        self.entities_by_text = entities_by_text

    def detect(self, text):
        return self.entities_by_text[text]


def test_exact_scores_and_character_coverage_are_distinct():
    text = "ABCDEFG"
    cases = [{
        "id": "case-1",
        "text": text,
        "entities": [
            {"start": 1, "end": 3, "label": "PERSON"},
            {"start": 4, "end": 6, "label": "EMAIL"},
        ],
    }]
    detector = FixedDetector({text: [Entity(1, 3, "PERSON"), Entity(4, 7, "ORG")]})
    report = evaluate(cases, detector)
    assert report["overall"]["exact"] == {
        "gold_entities": 2,
        "predicted_entities": 2,
        "true_positives": 1,
        "false_positives": 1,
        "false_negatives": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    assert report["overall"]["gold_character_coverage"] == {"covered": 4, "total": 4, "rate": 1.0}
    assert report["by_type"]["EMAIL"]["exact"]["precision"] is None
    assert report["by_type"]["ORG"]["exact"]["recall"] is None
    assert report["by_type"]["EMAIL"]["gold_character_coverage"]["rate"] == 1.0
    assert "case-1" not in json.dumps(report)
    assert text not in json.dumps(report)


def test_resolved_outer_span_counts_as_coverage_but_not_exact_match():
    text = "ABCDE"
    report = evaluate(
        [{"text": text, "entities": [{"start": 1, "end": 3, "label": "PERSON"}]}],
        FixedDetector({text: [Entity(0, 5, "ORG"), Entity(1, 3, "PERSON")]}),
    )
    assert report["overall"]["exact"]["true_positives"] == 0
    assert report["overall"]["exact"]["false_positives"] == 1
    assert report["overall"]["gold_character_coverage"]["rate"] == 1.0


def test_crossing_detector_spans_fail_closed():
    text = "ABCDE"
    with pytest.raises(ContextVeilError, match="partially overlap"):
        evaluate(
            [{"text": text, "entities": []}],
            FixedDetector({text: [Entity(0, 3, "PERSON"), Entity(2, 5, "ORG")]}),
        )


def test_empty_denominators_are_null_not_perfect_scores():
    report = evaluate([{"text": "nothing", "entities": []}], FixedDetector({"nothing": []}))
    assert report["overall"]["exact"]["precision"] is None
    assert report["overall"]["exact"]["recall"] is None
    assert report["overall"]["exact"]["f1"] is None
    assert report["overall"]["gold_character_coverage"]["rate"] is None
    assert report["by_type"] == {}


@pytest.mark.parametrize("case", [
    {"text": "private", "entities": [{"start": 0, "end": 8, "label": "PERSON"}]},
    {"text": "private", "entities": [{"start": True, "end": 3, "label": "PERSON"}]},
    {"text": "private", "entities": [{"start": 0, "end": 3, "label": "person"}]},
    {"text": "private", "entities": [{"start": 0, "end": 3, "label": "PERSON"},
                                     {"start": 2, "end": 4, "label": "ORG"}]},
    {"text": "private", "entities": ["bad"]},
    {"text": "private", "entities": [], "id": "private@example.org"},
])
def test_malformed_gold_cases_fail_without_echoing_content(case):
    with pytest.raises(ContextVeilError) as error:
        evaluate([case], FixedDetector({}))
    assert "private" not in str(error.value)


def test_detector_failure_is_safe():
    class BrokenDetector:
        def detect(self, text):
            raise ValueError("private data")

    with pytest.raises(ContextVeilError, match="failed during evaluation") as error:
        evaluate([{"text": "private data", "entities": []}], BrokenDetector())
    assert "private data" not in str(error.value)


def test_synthetic_fixture_is_valid_and_has_negative_case():
    fixture = Path(__file__).resolve().parents[1] / "examples" / "evaluation.jsonl"
    cases = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines()]
    detector = FixedDetector({case["text"]: [] for case in cases})
    report = evaluate(cases, detector)
    assert report["cases"] >= 12
    assert any(not case["entities"] for case in cases)
    assert report["overall"]["exact"]["recall"] == 0.0
    assert "Synthetic fixture scores are not real-world privacy validation." in report["limits"]


def test_paired_utility_detects_negation_loss_and_correct_answer_retention():
    original = "The request is approved."
    changed = "The request is not approved."
    assert SequenceMatcher(None, original, changed).ratio() > 0.8

    def decision(text):
        return "rejected" if "not approved" in text else "approved"

    cases = [
        {"original": original, "anonymized": changed, "expected": "approved"},
        {"original": "Payment approved for Mara Chen.",
         "anonymized": "Payment approved for [[PERSON_1]].", "expected": "approved"},
    ]
    report = evaluate_utility(cases, decision)
    assert report["cases"] == 2
    assert report["original_correct"] == 2
    assert report["anonymized_correct"] == 1
    assert report["correct_original_retained"] == 1
    assert report["original_accuracy"] == 1.0
    assert report["anonymized_accuracy"] == 0.5
    assert report["accuracy_delta"] == -0.5
    assert report["correct_original_retained_rate"] == 0.5
    assert report["case_results"] == [
        {"index": 1, "original_correct": True, "anonymized_correct": False},
        {"index": 2, "original_correct": True, "anonymized_correct": True},
    ]
    assert "Mara Chen" not in json.dumps(report)
    assert "approved" not in json.dumps(report)
    assert "generic semantic-preservation guarantee" in report["limits"][1]


def test_utility_empty_and_no_original_correct_use_null_rates():
    empty = evaluate_utility([], lambda text: None)
    assert empty["original_accuracy"] is None
    assert empty["anonymized_accuracy"] is None
    assert empty["accuracy_delta"] is None
    assert empty["correct_original_retained_rate"] is None

    wrong = evaluate_utility([{"original": "A", "anonymized": "B", "expected": 3}], lambda text: 4)
    assert wrong["original_accuracy"] == 0.0
    assert wrong["correct_original_retained_rate"] is None


@pytest.mark.parametrize("case", [
    {"original": "private", "anonymized": "masked"},
    {"original": "private", "anonymized": "masked", "expected": float("nan")},
    {"original": "private", "anonymized": 5, "expected": "x"},
    {"original": "private", "anonymized": "masked", "expected": "x", "id": "private@example.org"},
])
def test_utility_rejects_malformed_cases_without_echoing_content(case):
    with pytest.raises(ContextVeilError) as error:
        evaluate_utility([case], lambda text: "x")
    assert "private" not in str(error.value)


def test_utility_task_failures_and_invalid_answers_are_safe():
    case = {"original": "private", "anonymized": "masked", "expected": "x"}

    def broken(text):
        raise ValueError("private input")

    with pytest.raises(ContextVeilError, match="task failed") as error:
        evaluate_utility([case], broken)
    assert "private" not in str(error.value)
    with pytest.raises(ContextVeilError, match="JSON value"):
        evaluate_utility([case], lambda text: object())
