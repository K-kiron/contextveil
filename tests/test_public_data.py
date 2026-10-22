"""Synthetic schema and offset checks for the public corpus adapters."""

import json

import pytest

from benchmarks.public_data import load_gretel, load_tab


def _gretel_file(tmp_path, *, text="Ann met Ann.", entities=None):
    parquet = pytest.importorskip("pyarrow.parquet")
    arrow = pytest.importorskip("pyarrow")
    if entities is None:
        entities = [{"entity": "Ann", "types": ["first_name"]}]
    path = tmp_path / "sample.parquet"
    parquet.write_table(arrow.Table.from_pylist([{
        "uid": "sample-1", "domain": "legal", "document_type": "note",
        "text": text, "entities": repr(entities),
    }]), path)
    return path


def _tab_file(tmp_path, mentions):
    path = tmp_path / "sample.json"
    path.write_text(json.dumps([{
        "doc_id": "case-1", "text": "Ann Inc.", "dataset_type": "dev",
        "annotations": {"reviewer-a": {"entity_mentions": mentions},
                        "reviewer-b": {"entity_mentions": []}},
    }]), encoding="utf-8")
    return path


def _mention(start, end, value, category, decision, entity_id="entity-1"):
    return {"start_offset": start, "end_offset": end, "span_text": value,
            "entity_type": category, "identifier_type": decision, "entity_id": entity_id}


def test_gretel_expands_all_occurrences_and_audits_duplicates(tmp_path):
    path = _gretel_file(tmp_path, entities=[
        {"entity": "Ann", "types": ["first_name"]},
        {"entity": "Ann", "types": ["first_name"]},
    ])
    cases, audit = load_gretel(path)
    assert [(e["start"], e["end"], e["label"], e["source_label"])
            for e in cases[0]["entities"]] == [
                (0, 3, "PERSON", "first_name"), (8, 11, "PERSON", "first_name")]
    assert audit["ambiguous_entities"] == 2
    assert audit["duplicate_spans"] == 2
    assert audit["gold_spans"] == 2


def test_gretel_audits_exact_unbounded_fallback_and_rejects_absent_gold(tmp_path):
    cases, audit = load_gretel(_gretel_file(
        tmp_path, text="Annabelle", entities=[{"entity": "Ann", "types": ["name"]}]))
    assert cases[0]["entities"][0]["start"] == 0
    assert audit["unbounded_fallback_entities"] == 1
    with pytest.raises(ValueError, match="Unresolved Gretel gold"):
        load_gretel(_gretel_file(
            tmp_path, text="Nothing", entities=[{"entity": "Absent", "types": ["name"]}]))


def test_gretel_rejects_unknown_type_and_malformed_entities(tmp_path):
    with pytest.raises(ValueError, match="unsupported=1"):
        load_gretel(_gretel_file(
            tmp_path, entities=[{"entity": "Ann", "types": ["not_a_type"]}]))
    path = _gretel_file(tmp_path)
    parquet = pytest.importorskip("pyarrow.parquet")
    arrow = pytest.importorskip("pyarrow")
    parquet.write_table(arrow.Table.from_pylist([{
        "uid": "sample-1", "domain": "legal", "document_type": "note",
        "text": "Ann", "entities": "not valid literal",
    }]), path)
    with pytest.raises(ValueError, match="Malformed Gretel entities"):
        load_gretel(path)


def test_tab_keeps_annotator_layers_masking_decisions_and_overlap(tmp_path):
    mentions = [
        _mention(0, 3, "Ann", "PERSON", "DIRECT"),
        _mention(1, 3, "nn", "MISC", "QUASI", "entity-2"),
        _mention(4, 7, "Inc", "ORG", "NO_MASK", "entity-3"),
    ]
    cases, audit = load_tab(_tab_file(tmp_path, mentions))
    assert len(cases) == 2
    assert cases[0]["uid"] == "case-1:0"
    assert [e["label"] for e in cases[0]["entities"]] == ["PERSON", "SENSITIVE"]
    assert [e["identifier_type"] for e in cases[0]["entities"]] == ["DIRECT", "QUASI"]
    assert cases[0]["no_mask"][0]["label"] == "ORG"
    assert cases[1]["entities"] == []
    assert audit["nested_spans"] == 1
    assert audit["gold_spans"] == 2
    assert audit["no_mask_spans"] == 1


def test_tab_rejects_mismatched_offset_and_unknown_category(tmp_path):
    with pytest.raises(ValueError, match="Invalid TAB offset or span"):
        load_tab(_tab_file(tmp_path, [_mention(0, 3, "Bob", "PERSON", "DIRECT")]))
    with pytest.raises(ValueError, match="unsupported=1"):
        load_tab(_tab_file(tmp_path, [_mention(0, 3, "Ann", "UNKNOWN", "DIRECT")]))
