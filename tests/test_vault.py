"""Authenticated recovery must bind the exact normalized output."""

import json

import pytest

from contextveil.detectors import GlossaryDetector
from contextveil.engine import Anonymizer
from contextveil.formats import load_text
from contextveil.models import ContextVeilError
from contextveil.policy import Policy
from contextveil.vault import restore, seal

PASSPHRASE = "correct horse battery staple"
DOCUMENT_ID = "a" * 32


def test_vault_round_trip_randomness_and_no_plaintext():
    original = "Alice Nguyen denied the $50.00 charge."
    output = "[[PERSON_1_scope]] denied the $50.00 charge."
    first = seal(original, output, PASSPHRASE, document_id=DOCUMENT_ID)
    second = seal(original, output, PASSPHRASE, document_id=DOCUMENT_ID)
    assert first != second
    assert original.encode("utf-8") not in first
    assert output.encode("utf-8") not in first
    assert restore(output, first, PASSPHRASE, document_id=DOCUMENT_ID) == original
    assert restore(output, second, PASSPHRASE, document_id=DOCUMENT_ID) == original


@pytest.mark.parametrize("variant", ["wrong_passphrase", "changed_output", "changed_vault"])
def test_vault_rejects_authentication_failures(variant):
    original = "Alice said no."
    output = "[[PERSON_1_scope]] said no."
    vault = seal(original, output, PASSPHRASE, document_id=DOCUMENT_ID)
    if variant == "wrong_passphrase":
        passphrase = "another lengthy private passphrase"
    else:
        passphrase = PASSPHRASE
    if variant == "changed_output":
        output += " "
    if variant == "changed_vault":
        changed = bytearray(vault)
        changed[-1] ^= 1
        vault = bytes(changed)
    with pytest.raises(ContextVeilError, match="authentication failed|output has changed"):
        restore(output, vault, passphrase, document_id=DOCUMENT_ID)


def test_vault_rejects_short_passphrase_and_invalid_header():
    with pytest.raises(ContextVeilError, match="at least 16"):
        seal("Alice", "[PERSON]", "short", document_id=DOCUMENT_ID)
    with pytest.raises(ContextVeilError, match="Invalid recovery vault"):
        restore("[PERSON]", b"not a vault", PASSPHRASE, document_id=DOCUMENT_ID)


def test_engine_recovery_restores_alias_spellings_and_normalized_source():
    glossary = GlossaryDetector([
        {"value": "Alice Nguyen", "label": "PERSON", "canonical": "person-1"},
        {"value": "Dr. Nguyen", "label": "PERSON", "canonical": "person-1"},
    ])
    document = load_text(' { "note" : "Alice Nguyen met Dr. Nguyen" } ', "json")
    normalized = document.render([segment.text for segment in document.segments])
    result = Anonymizer(detector=glossary, policy=Policy(require_contextual=False)).anonymize(
        document, passphrase=PASSPHRASE)
    assert result.recovery is not None
    assert "Alice Nguyen" not in result.text and "Dr. Nguyen" not in result.text
    assert len(set(json.loads(result.text)["note"].split(" met "))) == 1
    assert restore(result.text, result.recovery, PASSPHRASE, document_id=result.report["document_id"]) == normalized


def test_identical_outputs_cannot_use_another_documents_vault():
    detector = GlossaryDetector([
        {"value": "Alice", "label": "PERSON", "canonical": "person-1"},
        {"value": "Bob", "label": "PERSON", "canonical": "person-1"},
    ])
    first, second = Anonymizer(detector, Policy(require_contextual=False)).anonymize_many(
        [load_text("Alice"), load_text("Bob")], passphrase=PASSPHRASE)
    assert first.text == second.text
    with pytest.raises(ContextVeilError, match="authentication failed"):
        restore(first.text, second.recovery, PASSPHRASE, document_id=first.report["document_id"])
