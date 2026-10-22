"""Privacy arithmetic for overlapping gold, partial masks, and TAB decisions."""

import json

import pytest

from benchmarks.privacy_metrics import score_predictions
from contextveil.models import ContextVeilError, Entity


def test_binary_characters_and_exact_spans_expose_partial_leakage():
    cases = [{"text": "AB CD EF X", "uid": "private-case", "entities": [
        {"start": 0, "end": 5, "label": "PERSON", "source_label": "name"},
        {"start": 6, "end": 8, "label": "ID", "source_label": "customer_id"},
    ]}]
    predictions = [[Entity(0, 2, "PERSON"), Entity(3, 5, "PERSON"),
                    Entity(6, 7, "ORG"), Entity(9, 10, "ORG")]]

    report = score_predictions(cases, predictions)

    assert report["overall"]["characters"] == {
        "gold_characters": 6, "predicted_characters": 6,
        "true_positives": 5, "false_positives": 1, "false_negatives": 1,
        "precision": 5 / 6, "recall": 5 / 6, "f1": 5 / 6,
    }
    assert report["overall"]["exact"]["true_positives"] == 0
    assert report["overall"]["exact"]["false_negatives"] == 2
    assert report["overall"]["mentions"] == {"covered": 0, "total": 2, "recall": 0.0}
    assert report["by_source_label"]["name"]["sensitive_characters"]["recall"] == 1.0
    assert report["by_source_label"]["name"]["mentions"]["recall"] == 0.0
    assert report["by_source_label"]["customer_id"]["sensitive_characters"]["recall"] == 0.5
    assert report["by_type"]["ID"]["exact"]["false_negatives"] == 1
    assert report["by_type"]["ORG"]["exact"]["false_positives"] == 2
    assert "private-case" not in json.dumps(report)
    assert "AB CD EF X" not in json.dumps(report)


def test_nested_and_duplicate_gold_use_character_union_and_unique_mentions():
    case = {"text": "Alice Bob", "entities": [
        {"start": 0, "end": 9, "label": "PERSON", "source_label": "name"},
        {"start": 0, "end": 5, "label": "PERSON", "source_label": "name"},
        {"start": 0, "end": 5, "label": "PERSON", "source_label": "name"},
    ]}
    report = score_predictions([case], [[Entity(0, 5, "PERSON"), Entity(0, 5, "PERSON")]])

    assert report["overall"]["characters"]["gold_characters"] == 8
    assert report["overall"]["characters"]["predicted_characters"] == 5
    assert report["overall"]["characters"]["recall"] == 5 / 8
    assert report["overall"]["exact"]["gold_entities"] == 2
    assert report["overall"]["exact"]["predicted_entities"] == 1
    assert report["overall"]["exact"]["true_positives"] == 1
    assert report["overall"]["mentions"] == {"covered": 1, "total": 2, "recall": 0.5}
    assert report["by_source_label"]["name"]["sensitive_characters"]["total"] == 8


def test_tab_annotation_units_and_whole_entity_protection():
    text = "Alice Bob"
    first = {"text": text, "uid": "case-secret-1", "entities": [
        {"start": 0, "end": 5, "label": "PERSON", "source_label": "PERSON",
         "identifier_type": "DIRECT", "entity_id": "person-secret"},
        {"start": 6, "end": 9, "label": "PERSON", "source_label": "PERSON",
         "identifier_type": "DIRECT", "entity_id": "person-secret"},
    ], "no_mask": []}
    second = {"text": text, "uid": "case-secret-2", "entities": [
        {"start": 0, "end": 5, "label": "PERSON", "source_label": "PERSON",
         "identifier_type": "QUASI", "entity_id": "other-secret"},
    ], "no_mask": [{"start": 6, "end": 9, "label": "PERSON", "source_label": "PERSON",
                    "identifier_type": "NO_MASK", "entity_id": "public-id"}]}

    report = score_predictions([first, second], [[Entity(0, 5, "PERSON")],
                                                 [Entity(0, 9, "PERSON")]])

    assert report["cases"] == 2
    assert report["overall"]["entity_protection"]["DIRECT"] == {
        "covered": 0, "total": 1, "recall": 0.0,
    }
    assert report["overall"]["entity_protection"]["QUASI"] == {
        "covered": 1, "total": 1, "recall": 1.0,
    }
    assert report["overall"]["no_mask"] == {
        "false_redactions": 1, "total": 1, "rate": 1.0,
        "annotation_conflicts": 0, "unambiguous_total": 1,
        "unambiguous_false_redactions": 1, "unambiguous_rate": 1.0,
    }
    assert report["case_results"][0]["characters"]["recall"] == 5 / 8
    assert report["case_results"][1]["characters"]["false_positives"] == 3
    encoded = json.dumps(report)
    assert "case-secret" not in encoded and "person-secret" not in encoded


def test_empty_denominators_and_no_mask_only_case():
    empty = score_predictions([{"text": "X", "entities": []}], [[]])
    assert empty["overall"]["characters"]["precision"] is None
    assert empty["overall"]["characters"]["recall"] is None
    assert empty["overall"]["characters"]["f1"] is None
    assert empty["overall"]["mentions"]["recall"] is None
    assert empty["overall"]["entity_protection"]["DIRECT"]["recall"] is None
    assert empty["overall"]["no_mask"]["rate"] is None

    negative = score_predictions([{"text": "XY", "entities": [], "no_mask": [
        {"start": 0, "end": 1, "label": "PERSON", "identifier_type": "NO_MASK"}
    ]}], [[Entity(0, 1, "PERSON")]])
    assert negative["overall"]["characters"]["precision"] == 0.0
    assert negative["overall"]["characters"]["recall"] is None
    assert negative["overall"]["characters"]["f1"] == 0.0
    assert negative["overall"]["no_mask"]["false_redactions"] == 1


def test_no_mask_overlap_with_sensitive_gold_is_reported_separately():
    report = score_predictions([{"text": "AB CD", "entities": [
        {"start": 0, "end": 2, "label": "PERSON"},
    ], "no_mask": [
        {"start": 0, "end": 2, "label": "PERSON", "identifier_type": "NO_MASK"},
        {"start": 3, "end": 5, "label": "ORG", "identifier_type": "NO_MASK"},
    ]}], [[Entity(0, 2, "PERSON")]])

    assert report["overall"]["exact"]["true_positives"] == 1
    assert report["overall"]["no_mask"] == {
        "false_redactions": 1, "total": 2, "rate": 0.5,
        "annotation_conflicts": 1, "unambiguous_total": 1,
        "unambiguous_false_redactions": 0, "unambiguous_rate": 0.0,
    }


@pytest.mark.parametrize("cases,predictions", [
    ([{"text": "private", "entities": [{"start": True, "end": 3, "label": "PERSON"}]}], [[]]),
    ([{"text": "private", "entities": [{"start": 0, "end": 8, "label": "PERSON"}]}], [[]]),
    ([{"text": "private", "entities": [{"start": 0, "end": 3, "label": "person"}]}], [[]]),
    ([{"text": "private", "entities": [{"start": 0, "end": 3, "label": "PERSON",
                                       "identifier_type": [], "entity_id": "secret"}]}], [[]]),
    ([{"text": "private", "entities": []}], [[Entity(-1, 3, "PERSON")]]),
    ([{"text": "private", "entities": []}], [[Entity(0, 3, "person")]]),
    ([{"text": "private", "entities": []}], []),
    ([{"text": "private", "uid": "private-id", "entities": []},
      {"text": "private", "uid": "private-id", "entities": []}], [[], []]),
])
def test_invalid_inputs_fail_with_safe_errors(cases, predictions):
    with pytest.raises(ContextVeilError) as error:
        score_predictions(cases, predictions)
    assert "private" not in str(error.value)
