"""Privacy scores for supplied gold cases and resolved span predictions."""

from __future__ import annotations

from collections import Counter, defaultdict

from contextveil.models import ContextVeilError, Entity
from contextveil.policy import LABEL


def _span(text: str, start: object, end: object, label: object) -> None:
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
            or not isinstance(label, str) or not LABEL.fullmatch(label)):
        raise ContextVeilError("A privacy benchmark span has an invalid offset or label.")


def _gold(text: str, entries: object) -> list[dict]:
    if type(entries) is not list:
        raise ContextVeilError("Privacy benchmark annotations must be a list.")
    result = []
    seen = set()
    for entry in entries:
        if type(entry) is not dict or set(entry) - {
            "start", "end", "label", "source_label", "identifier_type", "entity_id"
        } or not {"start", "end", "label"} <= set(entry):
            raise ContextVeilError("A privacy benchmark annotation is malformed.")
        _span(text, entry["start"], entry["end"], entry["label"])
        source = entry.get("source_label")
        decision = entry.get("identifier_type")
        entity_id = entry.get("entity_id")
        if (source is not None and (not isinstance(source, str) or not LABEL.fullmatch(source.upper()))
                or decision is not None and (not isinstance(decision, str)
                                             or decision not in {"DIRECT", "QUASI", "NO_MASK"})
                or entity_id is not None and (not isinstance(entity_id, str) or not entity_id)
                or decision in {"DIRECT", "QUASI"} and entity_id is None):
            raise ContextVeilError("A privacy benchmark annotation has invalid metadata.")
        key = (entry["start"], entry["end"], entry["label"], source, decision, entity_id)
        if key not in seen:
            seen.add(key)
            result.append(entry)
    return result


def _positions(text: str, spans) -> set[int]:
    return {index for start, end in spans for index in range(start, end) if not text[index].isspace()}


def _scores(tp: int, predicted: int, gold: int, unit: str) -> dict:
    return {
        f"gold_{unit}": gold,
        f"predicted_{unit}": predicted,
        "true_positives": tp,
        "false_positives": predicted - tp,
        "false_negatives": gold - tp,
        "precision": tp / predicted if predicted else None,
        "recall": tp / gold if gold else None,
        "f1": 2 * tp / (predicted + gold) if predicted + gold else None,
    }


def _coverage(covered: int, total: int) -> dict:
    return {"covered": covered, "total": total, "recall": covered / total if total else None}


def score_predictions(cases: list[dict], predictions: list[list[Entity]]) -> dict:
    """Score independent annotation units without returning text, IDs, or raw spans."""
    if type(cases) is not list or type(predictions) is not list or len(cases) != len(predictions):
        raise ContextVeilError("Privacy benchmark cases and predictions must be parallel lists.")

    character_counts = Counter()
    exact_counts = Counter()
    mention_counts = Counter()
    protection_counts = {kind: Counter() for kind in ("DIRECT", "QUASI")}
    no_mask_counts = Counter()
    type_counts = defaultdict(Counter)
    source_counts = defaultdict(Counter)
    case_results = []
    seen_uids = set()

    for index, (case, selected) in enumerate(zip(cases, predictions), start=1):
        if type(case) is not dict or not isinstance(case.get("text"), str):
            raise ContextVeilError("A privacy benchmark case needs string text and annotations.")
        if "uid" in case:
            uid = case["uid"]
            if not isinstance(uid, str) or not uid or uid in seen_uids:
                raise ContextVeilError("Privacy benchmark case UIDs must be unique nonempty strings.")
            seen_uids.add(uid)
        text = case["text"]
        gold = _gold(text, case.get("entities"))
        no_mask = _gold(text, case.get("no_mask", []))
        if any(entry.get("identifier_type") != "NO_MASK" for entry in no_mask):
            raise ContextVeilError("A no-mask annotation must have a NO_MASK decision.")
        if any(entry.get("identifier_type") == "NO_MASK" for entry in gold):
            raise ContextVeilError("A NO_MASK annotation cannot be sensitive gold.")
        if type(selected) is not list or any(not isinstance(entity, Entity) for entity in selected):
            raise ContextVeilError("Privacy benchmark predictions must be Entity lists.")
        for entity in selected:
            _span(text, entity.start, entity.end, entity.label)

        gold_keys = {(entry["start"], entry["end"], entry["label"]) for entry in gold}
        predicted_keys = {(entity.start, entity.end, entity.label) for entity in selected}
        predicted_offsets = {offset for start, end, _ in predicted_keys for offset in range(start, end)}
        predicted_chars = _positions(text, ((start, end) for start, end, _ in predicted_keys))
        gold_chars = _positions(text, ((start, end) for start, end, _ in gold_keys))
        char_tp = len(gold_chars & predicted_chars)
        exact_tp = len(gold_keys & predicted_keys)
        mention_covered = sum(set(range(start, end)) <= predicted_offsets for start, end, _ in gold_keys)
        character_counts.update(tp=char_tp, predicted=len(predicted_chars), gold=len(gold_chars))
        exact_counts.update(tp=exact_tp, predicted=len(predicted_keys), gold=len(gold_keys))
        mention_counts.update(covered=mention_covered, total=len(gold_keys))

        for label in {key[2] for key in gold_keys | predicted_keys}:
            label_gold = {key for key in gold_keys if key[2] == label}
            label_predicted = {key for key in predicted_keys if key[2] == label}
            chars = _positions(text, ((start, end) for start, end, _ in label_gold))
            type_counts[label].update(
                exact_tp=len(label_gold & label_predicted), gold=len(label_gold),
                predicted=len(label_predicted), chars=len(chars), covered_chars=len(chars & predicted_chars),
                covered_mentions=sum(set(range(start, end)) <= predicted_offsets
                                     for start, end, _ in label_gold),
            )

        for source in {entry["source_label"] for entry in gold if entry.get("source_label") is not None}:
            source_keys = {(entry["start"], entry["end"], entry["label"])
                           for entry in gold if entry.get("source_label") == source}
            chars = _positions(text, ((start, end) for start, end, _ in source_keys))
            source_counts[source].update(
                chars=len(chars), covered_chars=len(chars & predicted_chars), mentions=len(source_keys),
                covered_mentions=sum(set(range(start, end)) <= predicted_offsets
                                     for start, end, _ in source_keys),
            )

        for kind in protection_counts:
            entities = defaultdict(set)
            for entry in gold:
                if entry.get("identifier_type") == kind:
                    entities[entry["entity_id"]].update(range(entry["start"], entry["end"]))
            protection_counts[kind].update(
                covered=sum(offsets <= predicted_offsets for offsets in entities.values()), total=len(entities)
            )

        no_mask_keys = {(entry["start"], entry["end"], entry["label"]) for entry in no_mask}
        false_redactions = sum(bool(set(range(start, end)) & predicted_offsets)
                               for start, end, _ in no_mask_keys)
        unambiguous = {(start, end, label) for start, end, label in no_mask_keys
                       if not _positions(text, ((start, end),)) & gold_chars}
        no_mask_counts.update(
            false_redactions=false_redactions, total=len(no_mask_keys),
            annotation_conflicts=len(no_mask_keys) - len(unambiguous),
            unambiguous_total=len(unambiguous),
            unambiguous_false_redactions=sum(bool(set(range(start, end)) & predicted_offsets)
                                             for start, end, _ in unambiguous),
        )
        case_results.append({
            "index": index,
            "characters": _scores(char_tp, len(predicted_chars), len(gold_chars), "characters"),
            "exact": _scores(exact_tp, len(predicted_keys), len(gold_keys), "entities"),
            "mentions": _coverage(mention_covered, len(gold_keys)),
            "no_mask_false_redactions": false_redactions,
        })

    return {
        "schema_version": 1,
        "cases": len(cases),
        "overall": {
            "characters": _scores(character_counts["tp"], character_counts["predicted"],
                                  character_counts["gold"], "characters"),
            "exact": _scores(exact_counts["tp"], exact_counts["predicted"], exact_counts["gold"], "entities"),
            "mentions": _coverage(mention_counts["covered"], mention_counts["total"]),
            "entity_protection": {kind: _coverage(counts["covered"], counts["total"])
                                  for kind, counts in protection_counts.items()},
            "no_mask": {"false_redactions": no_mask_counts["false_redactions"],
                        "total": no_mask_counts["total"],
                        "rate": no_mask_counts["false_redactions"] / no_mask_counts["total"]
                        if no_mask_counts["total"] else None,
                        "annotation_conflicts": no_mask_counts["annotation_conflicts"],
                        "unambiguous_total": no_mask_counts["unambiguous_total"],
                        "unambiguous_false_redactions": no_mask_counts["unambiguous_false_redactions"],
                        "unambiguous_rate": no_mask_counts["unambiguous_false_redactions"]
                        / no_mask_counts["unambiguous_total"] if no_mask_counts["unambiguous_total"] else None},
        },
        "by_type": {label: {
            "exact": _scores(counts["exact_tp"], counts["predicted"], counts["gold"], "entities"),
            "sensitive_characters": _coverage(counts["covered_chars"], counts["chars"]),
            "mentions": _coverage(counts["covered_mentions"], counts["gold"]),
        } for label, counts in sorted(type_counts.items())},
        "by_source_label": {label: {
            "sensitive_characters": _coverage(counts["covered_chars"], counts["chars"]),
            "mentions": _coverage(counts["covered_mentions"], counts["mentions"]),
        } for label, counts in sorted(source_counts.items())},
        "case_results": case_results,
    }
