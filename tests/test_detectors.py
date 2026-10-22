"""Behavior checks for local candidate detection and exact text offsets."""

import importlib.util

import pytest

from contextveil.detectors import CompositeDetector, GlossaryDetector, RegexDetector, SpacyDetector
from contextveil.models import ContextVeilError


def test_regex_detects_structured_values_and_validates_checksums():
    text = (
        "Write to a@example.org; call (416) 555-0182. "
        "SSN 123-45-6789; card 4111 1111 1111 1111. "
        "Host 192.0.2.4 or 2001:db8::1. See https://example.org/page. "
        "Ignore 4111 1111 1111 1112, 999.1.1.1, and 000-12-3456."
    )
    found = RegexDetector().detect(text)
    values = {(entity.label, text[entity.start : entity.end]) for entity in found}
    assert values == {
        ("EMAIL", "a@example.org"),
        ("PHONE", "(416) 555-0182"),
        ("SSN", "123-45-6789"),
        ("CREDIT_CARD", "4111 1111 1111 1111"),
        ("IP_ADDRESS", "192.0.2.4"),
        ("IP_ADDRESS", "2001:db8::1"),
        ("URL", "https://example.org/page"),
    }


def test_regex_requires_context_for_ids_and_secrets_and_preserves_utility_numbers():
    text = "Usage 123456 kWh, price $123.45 on 2026-10-22; Account ID: AB-1234; API key = sk-abcdefgh1234."
    found = RegexDetector().detect(text)
    assert [(entity.label, text[entity.start : entity.end]) for entity in found] == [
        ("ID", "AB-1234"),
        ("SECRET", "sk-abcdefgh1234"),
    ]


def test_phone_extension_is_included_in_the_same_redaction_span():
    text = "Call 416-555-0182 ext. 123; backup +1 647-555-0199x42."
    found = [entity for entity in RegexDetector().detect(text) if entity.label == "PHONE"]
    assert [text[entity.start : entity.end] for entity in found] == [
        "416-555-0182 ext. 123",
        "+1 647-555-0199x42",
    ]


def test_glossary_unicode_offsets_repeats_and_explicit_alias_linkage():
    text = "🌿 Montréal met MTL. Montréalais stayed; MTL returned."
    detector = GlossaryDetector(
        [
            {"value": "Montréal", "label": "LOCATION", "canonical": "Montréal"},
            {"value": "MTL", "label": "LOCATION", "canonical": "Montréal"},
        ]
    )
    found = detector.detect(text)
    assert [text[entity.start : entity.end] for entity in found] == ["Montréal", "MTL", "MTL"]
    assert {entity.canonical for entity in found} == {"Montréal"}
    assert found[0].start == 2  # Python Unicode codepoint offset after the emoji and space.


def test_glossary_handles_punctuation_and_rejects_invalid_entries_without_echoing_them():
    assert ["C++", "C++"] == [
        "C++ toolkit C++"[entity.start : entity.end]
        for entity in GlossaryDetector([{"value": "C++", "label": "ORG"}]).detect("C++ toolkit C++")
    ]
    with pytest.raises(ContextVeilError, match="nonempty string") as error:
        GlossaryDetector([{"value": "private data", "label": ""}])
    assert "private data" not in str(error.value)


@pytest.mark.parametrize("entries", [None, "private", ["private"], [{"value": "private", "label": "lowercase"}]])
def test_glossary_malformed_configuration_has_safe_errors(entries):
    with pytest.raises(ContextVeilError) as error:
        GlossaryDetector(entries)
    assert "private" not in str(error.value)


def test_glossary_rejects_conflicts_and_bounds_config():
    base = {"value": "Acme", "label": "ORG", "canonical": "Acme Corp"}
    for conflicting in (
        {"value": "ACME", "label": "PERSON", "canonical": "Acme Corp"},
        {"value": "acme", "label": "ORG", "canonical": "Other Corp"},
    ):
        with pytest.raises(ContextVeilError, match="conflicting"):
            GlossaryDetector([base, conflicting])
    assert len(GlossaryDetector([base, base]).detect("Acme")) == 1
    with pytest.raises(ContextVeilError, match="too many"):
        GlossaryDetector([base] * 1001)
    with pytest.raises(ContextVeilError, match="too long"):
        GlossaryDetector([{"value": "a" * 257, "label": "ORG"}])


def test_composite_accumulates_candidates_including_duplicates():
    child = GlossaryDetector([{"value": "Acme", "label": "ORG"}])
    combined = CompositeDetector([child, child])
    assert combined.contextual is False
    assert len(combined.detect("Acme")) == 2


def test_spacy_missing_model_is_actionable_and_safe():
    if importlib.util.find_spec("spacy") is None:
        with pytest.raises(ContextVeilError, match="Install spaCy"):
            SpacyDetector()
    else:
        with pytest.raises(ContextVeilError, match="Install the requested spaCy model") as error:
            SpacyDetector("model_that_is_not_installed")
        assert "model_that_is_not_installed" not in str(error.value)


def test_spacy_requires_ner_and_reports_processing_errors_without_input(monkeypatch):
    spacy = pytest.importorskip("spacy")
    monkeypatch.setattr(spacy.util, "is_package", lambda model: True)
    monkeypatch.setattr(spacy, "load", lambda model: spacy.blank("en"))
    with pytest.raises(ContextVeilError, match="enabled NER"):
        SpacyDetector()

    class BrokenNlp:
        max_length = 100

        def __call__(self, text):
            raise ValueError("private input")

    detector = SpacyDetector.__new__(SpacyDetector)
    detector._nlp = BrokenNlp()
    with pytest.raises(ContextVeilError, match="maximum document length") as length_error:
        detector.detect("private input" * 20)
    assert "private input" not in str(length_error.value)
    with pytest.raises(ContextVeilError, match="could not process") as processing_error:
        detector.detect("private input")
    assert "private input" not in str(processing_error.value)


def test_spacy_installed_model_smoke():
    spacy = pytest.importorskip("spacy")
    if not spacy.util.is_package("en_core_web_sm"):
        pytest.skip("en_core_web_sm is not installed")
    detector = SpacyDetector()
    text = "Ada Lovelace visited London on 10 October 1843."
    found = detector.detect(text)
    assert detector.contextual is True
    assert {(entity.label, text[entity.start : entity.end]) for entity in found} >= {
        ("PERSON", "Ada Lovelace"),
        ("LOCATION", "London"),
    }
    assert all(text[entity.start : entity.end] for entity in found)
    assert all(entity.label in {"PERSON", "ORG", "LOCATION"} for entity in found)
    assert not any(entity.label in {"DATE", "MONEY", "QUANTITY"} for entity in found)
