"""Behavioral release checks for the anonymization pipeline."""

import json
import re

import pytest

from contextveil.detectors import GlossaryDetector, RegexDetector
from contextveil.engine import Anonymizer, UtilityCheck, resolve_entities
from contextveil.formats import load_text
from contextveil.models import Entity, ValidationError
from contextveil.policy import Policy


class NoCandidates:
    name = "none"
    contextual = False

    def detect(self, text):
        return []


class FixedCandidates:
    name = "fixed"
    contextual = True

    def __init__(self, candidates):
        self.candidates = candidates

    def detect(self, text):
        return list(self.candidates) if "Alice" in text else []


def _token(text, label="PERSON"):
    tokens = set(re.findall(r"\[\[" + label + r"_\d+_[0-9a-f]+\]\]", text))
    assert len(tokens) == 1
    return tokens.pop()


def test_repeated_mentions_and_explicit_aliases_share_one_token():
    glossary = GlossaryDetector([
        {"value": "Alice Nguyen", "label": "PERSON", "canonical": "person-1"},
        {"value": "Dr. Nguyen", "label": "PERSON", "canonical": "person-1"},
    ])
    source = "Alice Nguyen approved the plan. Dr. Nguyen called Alice Nguyen."
    result = Anonymizer(detector=glossary, policy=Policy(require_contextual=False)).anonymize_text(source)
    token = _token(result.text)
    assert result.text == f"{token} approved the plan. {token} called {token}."
    assert result.report["entities_replaced"] == 3


def test_unselected_negation_amounts_units_and_order_stay_exact():
    source = "Alice did not owe $50.00; 5 mg was given before 2 mg. Alice agreed."

    class AliceDetector:
        name = "synthetic"
        contextual = True

        def detect(self, text):
            return [Entity(match.start(), match.end(), "PERSON")
                    for match in re.finditer(r"\bAlice\b", text)]

    result = Anonymizer(detector=AliceDetector()).anonymize_text(source)
    token = _token(result.text)
    assert result.text.replace(token, "Alice") == source
    assert "did not owe $50.00; 5 mg was given before 2 mg" in result.text
    assert result.report["status"] == "checks_passed"


def test_json_sensitive_numeric_field_changes_type_only_for_that_value():
    source = '{"name":"Alice","age":42,"account_number":12345678,"active":true,"note":"Alice"}'
    result = Anonymizer(detector=NoCandidates(), policy=Policy(require_contextual=False)).anonymize(
        load_text(source, "json"))
    data = json.loads(result.text)
    assert data["name"] == data["note"] == _token(result.text)
    assert data["account_number"].startswith("[[ID_")
    assert data["age"] == 42 and type(data["age"]) is int
    assert data["active"] is True
    assert list(data) == ["name", "age", "account_number", "active", "note"]
    assert result.report["checks"]["structured_roundtrip"] is True


def test_batch_propagates_explicit_field_identity_to_other_file():
    documents = [load_text('{"name":"Alice"}', "json"), load_text("Alice reviewed the draft.")]
    results = Anonymizer(detector=NoCandidates(), policy=Policy(require_contextual=False)).anonymize_many(documents)
    token = json.loads(results[0].text)["name"]
    assert token == _token(results[1].text)
    assert results[1].text == f"{token} reviewed the draft."


def test_outer_entity_wins_nested_candidate_but_crossing_spans_fail():
    source = "Alice Smith replied."
    nested = FixedCandidates([Entity(0, 11, "PERSON"), Entity(0, 5, "PERSON")])
    result = Anonymizer(detector=nested).anonymize_text(source)
    assert result.text == _token(result.text) + " replied."
    with pytest.raises(ValidationError, match="partially overlap"):
        resolve_entities(source, [Entity(0, 8, "PERSON"), Entity(6, 11, "PERSON")])


@pytest.mark.parametrize("candidate", [
    Entity(-1, 5, "PERSON"),
    Entity(0, 99, "PERSON"),
    Entity(True, 5, "PERSON"),
    Entity(0, 5, "person"),
    Entity(0, 5, "PERSON", score=float("nan")),
])
def test_invalid_detector_spans_fail_closed(candidate):
    with pytest.raises(ValidationError, match="invalid entity"):
        Anonymizer(detector=FixedCandidates([candidate])).anonymize_text("Alice replied.")


def test_report_contains_no_source_pii_or_field_hint():
    result = Anonymizer(detector=RegexDetector(), policy=Policy(require_contextual=False)).anonymize(
        load_text('{"Alice":"alice@example.test","amount":50}', "json"))
    serialized = json.dumps(result.report)
    assert "Alice" not in serialized
    assert "alice@example.test" not in serialized
    assert "amount" not in serialized
    assert result.report["entity_counts"]["EMAIL"] == 1


def test_regex_only_requires_review_when_contextual_detection_is_required():
    result = Anonymizer(detector=RegexDetector()).anonymize_text("Contact alice@example.test.")
    assert "alice@example.test" not in result.text
    assert result.report["status"] == "review_required"
    assert any("contextual" in reason for reason in result.report["review_reasons"])


def test_failing_utility_invariant_rejects_output():
    check = UtilityCheck("preserve-negation", lambda original, output: "not" not in output)
    with pytest.raises(ValidationError, match="utility invariant"):
        Anonymizer(detector=NoCandidates(), validators=[check]).anonymize_text("Do not release.")


def test_postscan_residual_candidate_requires_review():
    class ResidualDetector:
        name = "synthetic"
        contextual = True

        def __init__(self):
            self.calls = 0

        def detect(self, text):
            self.calls += 1
            if self.calls == 1:
                return [Entity(0, 5, "PERSON")]
            start = text.index("remnant")
            return [Entity(start, start + len("remnant"), "ID")]

    result = Anonymizer(detector=ResidualDetector()).anonymize_text("Alice and remnant")
    assert result.report["residual_candidates"] == 1
    assert result.report["status"] == "review_required"
    assert any("residual" in reason.lower() for reason in result.report["review_reasons"])
