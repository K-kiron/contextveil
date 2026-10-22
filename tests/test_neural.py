"""Coverage and overlap checks independent of optional neural model assets."""

import re
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from contextveil.detectors import RegexDetector
from contextveil.models import ContextVeilError, Entity
from contextveil.neural import ContextualRulesDetector, GlinerDetector, merge_predictions


def fake_detector():
    def words(text):
        return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\w+|\S", text)]

    def infer(texts, labels, **kwargs):
        return [[{"start": m.start(), "end": m.end(), "label": "person", "score": 0.9}
                 for m in re.finditer("Alice", text)] for text in texts]

    detector = GlinerDetector.__new__(GlinerDetector)
    detector._model = SimpleNamespace(data_processor=SimpleNamespace(words_splitter=words), inference=infer)
    detector._tokenizer = lambda tokens, **kwargs: {"input_ids": [0] * (sum(len(t) for t in tokens) + 2)}
    detector._torch = SimpleNamespace(inference_mode=nullcontext)
    detector._prompt = ["person"]
    detector._rules = RegexDetector()
    detector.threshold = 0.5
    detector.batch_size = 8
    return detector


def test_every_word_is_scanned_and_tail_offsets_are_preserved():
    text = "🌿 " + "word " * 1100 + "Alice"
    detector = fake_detector()
    windows = detector._windows(text)
    assert len(windows) > 2
    covered = set()
    for offset, chunk in windows:
        assert text[offset:offset + len(chunk)] == chunk
        covered.update(range(offset, offset + len(chunk)))
        words = [w[0] for w in detector._model.data_processor.words_splitter(chunk)]
        assert len(detector._tokenizer(detector._prompt + words)["input_ids"]) <= 512
    assert all(i in covered for i, char in enumerate(text) if not char.isspace())
    entities = detector.detect(text)
    assert any(text[e.start:e.end] == "Alice" and e.start == text.index("Alice") for e in entities)


def test_empty_inputs_and_oversized_single_word_do_not_silently_truncate():
    detector = fake_detector()
    assert detector.detect_many(["", " \n\t"]) == [[], []]
    with pytest.raises(ContextVeilError, match="single input token"):
        detector.detect("a" * 600)


def test_union_covers_crossing_edges_and_prefers_outer_or_regex_type():
    result = merge_predictions([Entity(2, 8, "PERSON", 0.9, "gliner"),
                                Entity(5, 12, "ADDRESS", 0.8, "gliner")])
    assert [(e.start, e.end, e.label) for e in result] == [(2, 12, "SENSITIVE")]
    same = merge_predictions([Entity(0, 7, "PERSON", 0.99, "gliner"),
                              Entity(0, 7, "EMAIL", 1, "regex"), Entity(2, 4, "ID")])
    assert [(e.start, e.end, e.label) for e in same] == [(0, 7, "EMAIL")]


def test_threshold_does_not_disable_verified_rules():
    detector = fake_detector()
    text = "Alice a@example.com"
    selected = detector.select(text, [Entity(0, 5, "PERSON", 0.4, "gliner")])
    assert [(text[e.start:e.end], e.label) for e in selected] == [("a@example.com", "EMAIL")]


def test_contextual_rules_capture_dates_and_identifiers_without_generic_numbers():
    text = ("DOB: 1992-02-29; MRN: M123456; customer_id=809132; application no. 1748/92; "
            "born on 2021-02-30; stock 809132; date 1992-02-29; MRN: unknown.")
    found = ContextualRulesDetector().detect(text)
    values = {(text[e.start:e.end], e.label) for e in found}
    assert values == {("1992-02-29", "DATE"), ("M123456", "ID"), ("809132", "ID"), ("1748/92", "ID")}
    assert len([e for e in found if e.label == "DATE"]) == 1


def test_context_cues_work_inside_markdown_json_and_parentheses():
    text = '**Date of Birth:** 1992-02-29; "customer_id": "AB12345"; application (no. 1748/92)'
    values = {(text[e.start:e.end], e.label) for e in ContextualRulesDetector().detect(text)}
    assert values == {("1992-02-29", "DATE"), ("AB12345", "ID"), ("1748/92", "ID")}


def test_invalid_model_output_is_rejected_without_echoing_text():
    detector = fake_detector()
    detector._model.inference = lambda *a, **k: [[{"start": 0, "end": 999, "label": "person", "score": .9}]]
    with pytest.raises(ContextVeilError, match="invalid entity") as error:
        detector.detect("private")
    assert "private" not in str(error.value)


@pytest.mark.parametrize("threshold", [0, 1, float("nan"), "x"])
def test_invalid_threshold_and_missing_assets_fail_before_model_loading(tmp_path, threshold):
    with pytest.raises(ContextVeilError, match="Invalid GLiNER"):
        GlinerDetector(tmp_path, threshold=threshold)
    with pytest.raises(ContextVeilError, match="complete local"):
        GlinerDetector(tmp_path)
