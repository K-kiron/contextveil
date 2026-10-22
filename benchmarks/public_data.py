"""Read pinned public PII corpora for local, split-preserving benchmarks.

Gretel gives entity strings rather than offsets. Every exact, word-bounded
occurrence is gold. If none exists, use literal substrings and audit the
fallback; wholly absent strings are fatal because their location is unknown.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path

GRETEL_LABELS = {
    "name": "PERSON", "first_name": "PERSON", "last_name": "PERSON",
    "company_name": "ORG", "city": "LOCATION", "country": "LOCATION",
    "state": "LOCATION", "coordinate": "LOCATION", "postcode": "ADDRESS",
    "address": "ADDRESS", "street_address": "ADDRESS",
    "date_of_birth": "DATE", "date": "DATE", "time": "DATE", "date_time": "DATE",
    "email": "EMAIL", "phone_number": "PHONE", "ipv4": "IP_ADDRESS",
    "ipv6": "IP_ADDRESS", "credit_card_number": "CREDIT_CARD", "ssn": "SSN",
    "url": "URL", "api_key": "SECRET", "password": "SECRET",
    "cvv": "SECRET", "pin": "SECRET", "medical_record_number": "ID",
    "customer_id": "ID", "employee_id": "ID", "license_plate": "ID",
    "user_name": "ID", "device_identifier": "ID", "bank_routing_number": "ID",
    "unique_identifier": "ID", "account_number": "ID",
    "certificate_license_number": "ID", "biometric_identifier": "ID",
    "vehicle_identifier": "ID", "health_plan_beneficiary_number": "ID",
    "national_id": "ID", "tax_id": "ID", "swift_bic": "ID",
}

TAB_LABELS = {
    "PERSON": "PERSON", "ORG": "ORG", "LOC": "LOCATION", "CODE": "ID",
    "DATETIME": "DATE", "DEM": "SENSITIVE", "MISC": "SENSITIVE",
    "QUANTITY": "QUANTITY",
}


def _overlap_count(entities: list[dict]) -> int:
    ordered = sorted(entities, key=lambda item: (item["start"], item["end"]))
    return sum(left["start"] < right["end"] and right["start"] < left["end"]
               for i, left in enumerate(ordered) for right in ordered[i + 1:])


def _add_unique(entities: list[dict], seen: set[tuple], entity: dict, audit: dict) -> None:
    key = (entity["start"], entity["end"], entity["label"], entity["source_label"])
    if key in seen:
        audit["duplicate_spans"] += 1
    else:
        seen.add(key)
        entities.append(entity)


def load_gretel(path: str | Path) -> tuple[list[dict], dict]:
    """Load a Gretel Parquet split; fail on unlocatable or unknown gold labels."""
    import pyarrow.parquet as parquet

    audit = {"documents": 0, "cases": 0, "source_entities": 0, "gold_spans": 0,
             "missing_entities": 0, "unbounded_fallback_entities": 0,
             "ambiguous_entities": 0, "duplicate_spans": 0,
             "nested_spans": 0, "unsupported_labels": 0, "source_labels": {}}
    labels = Counter()
    cases = []
    for row_index, row in enumerate(parquet.read_table(path).to_pylist()):
        if not all(isinstance(row.get(key), str) for key in
                   ("uid", "domain", "document_type", "text", "entities")):
            raise ValueError(f"Malformed Gretel row {row_index}: required string column missing")
        try:
            source_entities = ast.literal_eval(row["entities"])
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            raise ValueError(f"Malformed Gretel entities at row {row_index}") from None
        if not isinstance(source_entities, list):
            raise ValueError(f"Malformed Gretel entities at row {row_index}")
        text = row["text"]
        gold = []
        seen: set[tuple] = set()
        for entity_index, source in enumerate(source_entities):
            if (not isinstance(source, dict) or not isinstance(source.get("entity"), str)
                    or not source["entity"] or not isinstance(source.get("types"), list)
                    or not source["types"] or not all(isinstance(t, str) for t in source["types"])):
                raise ValueError(f"Malformed Gretel entity at row {row_index}, item {entity_index}")
            audit["source_entities"] += 1
            matches = list(re.finditer(r"(?<!\w)" + re.escape(source["entity"]) + r"(?!\w)", text))
            if not matches:
                # shortcut: unbounded exact substrings may overlabel tokens; upgrade when source offsets exist.
                matches = list(re.finditer(re.escape(source["entity"]), text))
                if matches:
                    audit["unbounded_fallback_entities"] += 1
                else:
                    audit["missing_entities"] += 1
            if len(matches) > 1:
                audit["ambiguous_entities"] += 1
            for source_label in source["types"]:
                if source_label not in GRETEL_LABELS:
                    audit["unsupported_labels"] += 1
                    continue
                labels[source_label] += 1
                for match in matches:
                    _add_unique(gold, seen, {"start": match.start(), "end": match.end(),
                                              "label": GRETEL_LABELS[source_label],
                                              "source_label": source_label}, audit)
        audit["nested_spans"] += _overlap_count(gold)
        cases.append({"text": text, "entities": gold, "uid": row["uid"],
                      "domain": row["domain"], "document_type": row["document_type"]})
    audit["documents"] = audit["cases"] = len(cases)
    audit["gold_spans"] = sum(len(case["entities"]) for case in cases)
    audit["source_labels"] = dict(sorted(labels.items()))
    if audit["missing_entities"] or audit["unsupported_labels"]:
        raise ValueError("Unresolved Gretel gold annotations: "
                         f"missing={audit['missing_entities']}, unsupported={audit['unsupported_labels']}")
    return cases, audit


def load_tab(path: str | Path) -> tuple[list[dict], dict]:
    """Load each TAB annotator layer as a separate case, retaining NO_MASK spans."""
    with open(path, encoding="utf-8") as stream:
        documents = json.load(stream)
    if not isinstance(documents, list):
        raise ValueError("TAB split must be a document list")
    audit = {"documents": len(documents), "cases": 0, "source_mentions": 0,
             "gold_spans": 0, "no_mask_spans": 0, "duplicate_spans": 0,
             "nested_spans": 0, "unsupported_labels": 0, "source_labels": {}}
    labels = Counter()
    cases = []
    for doc_index, document in enumerate(documents):
        if (not isinstance(document, dict) or not isinstance(document.get("text"), str)
                or not isinstance(document.get("doc_id"), str)
                or not isinstance(document.get("annotations"), dict)
                or not document["annotations"]):
            raise ValueError(f"Malformed TAB document {doc_index}")
        text = document["text"]
        for annotation_index, annotation in enumerate(document["annotations"].values()):
            if not isinstance(annotation, dict) or not isinstance(annotation.get("entity_mentions"), list):
                raise ValueError(f"Malformed TAB annotation at document {doc_index}")
            gold: list[dict] = []
            no_mask: list[dict] = []
            seen_gold: set[tuple] = set()
            seen_no_mask: set[tuple] = set()
            for mention_index, mention in enumerate(annotation["entity_mentions"]):
                if not isinstance(mention, dict):
                    raise ValueError(f"Malformed TAB mention at document {doc_index}, item {mention_index}")
                start, end = mention.get("start_offset"), mention.get("end_offset")
                source_label, decision = mention.get("entity_type"), mention.get("identifier_type")
                if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                        or not isinstance(mention.get("span_text"), str)
                        or text[start:end] != mention["span_text"]):
                    raise ValueError(f"Invalid TAB offset or span at document {doc_index}, item {mention_index}")
                if decision not in {"DIRECT", "QUASI", "NO_MASK"}:
                    raise ValueError(f"Unknown TAB masking decision at document {doc_index}, item {mention_index}")
                if source_label not in TAB_LABELS:
                    audit["unsupported_labels"] += 1
                    continue
                if not isinstance(mention.get("entity_id"), str):
                    raise ValueError(f"Malformed TAB entity id at document {doc_index}, item {mention_index}")
                audit["source_mentions"] += 1
                labels[source_label] += 1
                entity = {"start": start, "end": end, "label": TAB_LABELS[source_label],
                          "source_label": source_label, "identifier_type": decision,
                          "entity_id": mention["entity_id"]}
                if decision == "NO_MASK":
                    _add_unique(no_mask, seen_no_mask, entity, audit)
                else:
                    _add_unique(gold, seen_gold, entity, audit)
            audit["nested_spans"] += _overlap_count(gold)
            cases.append({"text": text, "entities": gold, "no_mask": no_mask,
                          "uid": f"{document['doc_id']}:{annotation_index}"})
    audit["cases"] = len(cases)
    audit["gold_spans"] = sum(len(case["entities"]) for case in cases)
    audit["no_mask_spans"] = sum(len(case["no_mask"]) for case in cases)
    audit["source_labels"] = dict(sorted(labels.items()))
    if audit["unsupported_labels"]:
        raise ValueError(f"Unresolved TAB gold annotations: unsupported={audit['unsupported_labels']}")
    return cases, audit
