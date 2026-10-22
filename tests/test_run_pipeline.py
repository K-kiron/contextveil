"""Pipeline replay keeps real residual detection and release gates intact."""

import json

import pytest

from benchmarks.run_pipeline import ReplayDetector, run_pipeline, select_neural_predictions
from contextveil.models import ContextVeilError, Entity, ValidationError


class ResidualDetector:
    name = "synthetic"
    contextual = True

    def __init__(self, *, find_remnant=True):
        self.find_remnant = find_remnant
        self.batches = []

    def detect(self, text):
        raise AssertionError("Residual input must use the batch detector")

    def detect_many(self, texts):
        self.batches.append(list(texts))
        return [[Entity(text.index("remnant"), text.index("remnant") + 7, "ID")]
                if self.find_remnant and "remnant" in text else [] for text in texts]


def test_real_residual_candidate_requires_review():
    detector = ResidualDetector()
    report = run_pipeline(["Alice and remnant"], [[Entity(0, 5, "PERSON")]], detector)

    assert report["success"] == [True]
    assert report["auto_release"] == [False]
    assert report["reports"][0]["status"] == "review_required"
    assert report["reports"][0]["residual_candidates"] == 1
    assert report["residual_inputs"] == 1
    assert len(detector.batches) == 1 and len(detector.batches[0]) == 1
    assert "Alice" not in report["outputs"][0]
    assert "Alice" not in json.dumps(report["reports"])
    assert report["errors"] == {}


def test_sequential_detector_really_scans_residual_input():
    class SequentialDetector:
        name = "sequential"
        contextual = True

        def __init__(self):
            self.calls = []

        def detect(self, text):
            self.calls.append(text)
            start = text.find("remnant")
            return [Entity(start, start + 7, "ID")] if start >= 0 else []

    detector = SequentialDetector()
    report = run_pipeline(["Alice and remnant"], [[Entity(0, 5, "PERSON")]], detector)

    assert report["success"] == [True]
    assert report["auto_release"] == [False]
    assert report["reports"][0]["residual_candidates"] == 1
    assert len(detector.calls) == 1 and detector.calls[0].startswith("     ")


def test_each_document_gets_a_fresh_linkage_scope():
    detector = ResidualDetector(find_remnant=False)
    report = run_pipeline(["Alice wrote.", "Alice wrote."],
                          [[Entity(0, 5, "PERSON")], [Entity(0, 5, "PERSON")]], detector)

    assert report["success"] == [True, True]
    assert report["auto_release"] == [True, True]
    assert report["outputs"][0] != report["outputs"][1]
    assert report["residual_inputs"] == 1
    assert len(detector.batches) == 1 and len(detector.batches[0]) == 1


def test_original_candidate_conflict_is_counted_as_failure():
    detector = ResidualDetector()
    report = run_pipeline(["Alice"], [[Entity(0, 3, "PERSON"), Entity(2, 5, "ORG")]], detector)

    assert report["outputs"] == [""]
    assert report["success"] == [False]
    assert report["auto_release"] == [False]
    assert report["errors"] == {"candidate_partial_overlap": 1}
    assert report["preparation_failures"] == {"candidate_partial_overlap": 1}
    assert report["residual_inputs"] == 0
    assert detector.batches == []


def test_strict_replay_and_conflicting_original_cache_fail_closed():
    replay = ReplayDetector({}, True, prepare=False)
    with pytest.raises(ValidationError, match="uncached detector input"):
        replay.detect("private residual")
    with pytest.raises(ContextVeilError, match="conflicting cached predictions"):
        run_pipeline(["Alice", "Alice"], [[Entity(0, 5, "PERSON")], []], ResidualDetector())


def test_neural_selection_uses_threshold_and_regex_rules():
    text = "a@b.co Bob"
    selected = select_neural_predictions([text], [[
        Entity(7, 10, "PERSON", 0.8, "gliner"),
        Entity(0, 1, "ID", 0.2, "gliner"),
    ]], 0.5)

    assert [(entity.start, entity.end, entity.label) for entity in selected[0]] == [
        (0, 6, "EMAIL"), (7, 10, "PERSON")]


def test_neural_selection_includes_valid_cued_dob_and_application_id():
    text = "DOB: 1990-01-02; application no. AB123; DOB: 1990-02-31; application no. ABC"
    selected = select_neural_predictions([text], [[]], 0.5)[0]

    assert [(text[entity.start:entity.end], entity.label) for entity in selected] == [
        ("1990-01-02", "DATE"), ("AB123", "ID")]
