"""Regressions found by an independent adversarial integration review."""

import json
import re
from decimal import Decimal

import pytest

from contextveil import Anonymizer, Entity, GlossaryDetector, Policy, ValidationError, load_text
from contextveil.evaluation import evaluate
from contextveil.policy import field_key


class NoCandidates:
    name = "none"
    contextual = True

    def detect(self, text):
        return []


@pytest.mark.parametrize("key, normalized", [("customerId", "customer_id"), ("APIKey", "api_key"),
                                             ("dateOfBirth", "date_of_birth"), ("Date Of Birth", "date_of_birth")])
def test_policy_normalizes_common_field_spellings(key, normalized):
    assert field_key(key) == normalized


def test_camel_case_identifiers_are_selected_without_ner():
    source = '{"customerId":"A12345","dateOfBirth":"1990-05-06","apiKey":"abcdefgh123456"}'
    result = Anonymizer(NoCandidates()).anonymize(load_text(source, "json"))
    assert all(value.startswith("[[") for value in json.loads(result.text).values())


def test_field_selection_preserves_alias_identity_and_whitespace():
    detector = GlossaryDetector([
        {"value": "Robert Smith", "label": "PERSON", "canonical": "person-1"},
        {"value": "Bob", "label": "PERSON", "canonical": "person-1"},
    ])
    result = Anonymizer(detector, Policy(require_contextual=False)).anonymize(
        load_text('{"patient_name":"  Robert Smith  ","note":"Bob"}', "json"))
    values = json.loads(result.text)
    assert values["patient_name"] == "  " + values["note"] + "  "
    with pytest.raises(ValidationError, match="whole-value"):
        Anonymizer(detector).anonymize(load_text('{"patient_name":"Robert Smith Jr.","note":"Bob"}', "json"))


def test_literal_value_cannot_accidentally_join_canonical_group():
    class Detector:
        name = "synthetic"
        contextual = True

        def detect(self, text):
            return [Entity(m.start(), m.end(), "PERSON", canonical="Bob" if m.group() == "Alice" else None)
                    for m in re.finditer("Alice|Bob", text)]

    result = Anonymizer(Detector()).anonymize_text("Alice owes Bob $5.")
    assert len(set(re.findall(r"\[\[PERSON_\d+_[0-9a-f]+\]\]", result.text))) == 2


def test_sensitive_numeric_id_does_not_remove_unrelated_quantity():
    result = Anonymizer(NoCandidates()).anonymize(load_text('{"patient_id":12,"dose":"12 mg"}', "json"))
    assert json.loads(result.text)["dose"] == "12 mg"
    assert result.report["schema_changes"]["scalar_types"] == 1
    assert result.report["status"] == "review_required"


@pytest.mark.parametrize("format", ["json", "jsonl", "yaml"])
def test_pipeline_retains_precision_and_sensitive_replacement(format):
    source = ('{"amount":0.12345678901234567890,"tiny":1e-400,"name":"Alice"}'
              if format != "yaml" else "amount: 0.12345678901234567890\ntiny: 1e-400\nname: Alice\n")
    result = Anonymizer(NoCandidates()).anonymize(load_text(source, format))
    parsed = load_text(result.text, format)
    values = {segment.field_name: segment.text for segment in parsed.segments if segment.kind == "value"}
    assert Decimal(values["amount"]) == Decimal("0.12345678901234567890")
    assert Decimal(values["tiny"]) == Decimal("1e-400")
    assert values["name"].startswith("[[PERSON_")
    assert result.report["status"] == "checks_passed"


def test_sensitive_key_change_requires_review():
    engine = Anonymizer(GlossaryDetector([{"value": "Alice", "label": "PERSON"}]), Policy(require_contextual=False))
    result = engine.anonymize(load_text('{"Alice": 5}', "json"))
    assert result.report["schema_changes"]["keys_or_headers"] == 1
    assert result.report["status"] == "review_required"


def test_f1_is_zero_when_all_gold_entities_are_missed():
    metrics = evaluate([{"text": "Alice", "entities": [{"start": 0, "end": 5, "label": "PERSON"}]}],
                       NoCandidates())["overall"]["exact"]
    assert metrics["precision"] is None and metrics["recall"] == 0 and metrics["f1"] == 0
