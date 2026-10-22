"""Gold-span evaluation of resolved detector candidates on supplied cases."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable

from .engine import resolve_entities
from .models import ContextVeilError, Detector, Entity
from .policy import LABEL

_CASE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


def _scores(true_positives: int, predicted: int, gold: int) -> dict:
    precision = true_positives / predicted if predicted else None
    recall = true_positives / gold if gold else None
    f1 = 2 * true_positives / (predicted + gold) if predicted + gold else None
    return {
        "gold_entities": gold,
        "predicted_entities": predicted,
        "true_positives": true_positives,
        "false_positives": predicted - true_positives,
        "false_negatives": gold - true_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _coverage(covered: int, total: int) -> dict:
    return {"covered": covered, "total": total, "rate": covered / total if total else None}


def _gold_entities(case: dict) -> tuple[str, list[Entity]]:
    if type(case) is not dict or set(case) - {"text", "entities", "id"}:
        raise ContextVeilError("Each evaluation case must contain only text, entities, and optional id.")
    text = case.get("text")
    entries = case.get("entities")
    if not isinstance(text, str) or type(entries) is not list:
        raise ContextVeilError("Each evaluation case needs string text and an entity list.")
    if "id" in case and (not isinstance(case["id"], str) or not _CASE_ID.fullmatch(case["id"])):
        raise ContextVeilError("An evaluation case id must be a short non-sensitive identifier.")
    gold: list[Entity] = []
    for entry in entries:
        if type(entry) is not dict or set(entry) != {"start", "end", "label"}:
            raise ContextVeilError("Each gold entity needs start, end, and label only.")
        start, end, label = entry["start"], entry["end"], entry["label"]
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                or not isinstance(label, str) or not LABEL.fullmatch(label)):
            raise ContextVeilError("A gold entity has an invalid offset or label.")
        gold.append(Entity(start, end, label, source="gold"))
    gold.sort(key=lambda entity: (entity.start, entity.end))
    if any(current.start < previous.end for previous, current in zip(gold, gold[1:])):
        raise ContextVeilError("Gold entity spans must not overlap.")
    return text, gold


def evaluate(cases: list[dict], detector: Detector) -> dict:
    """Score exact spans and gold-character coverage; never include case text or ids."""
    if type(cases) is not list:
        raise ContextVeilError("Evaluation cases must be a list.")

    aggregate_gold: Counter[str] = Counter()
    aggregate_predicted: Counter[str] = Counter()
    aggregate_true_positives: Counter[str] = Counter()
    aggregate_characters: Counter[str] = Counter()
    aggregate_covered: Counter[str] = Counter()
    case_reports: list[dict] = []

    for index, case in enumerate(cases, start=1):
        text, gold = _gold_entities(case)
        try:
            candidates = detector.detect(text)
        except Exception:
            raise ContextVeilError("The detector failed during evaluation.") from None
        if type(candidates) is not list:
            raise ContextVeilError("The detector must return an entity list.")
        selected = resolve_entities(text, candidates)

        gold_keys = {(entity.start, entity.end, entity.label) for entity in gold}
        predicted_keys = {(entity.start, entity.end, entity.label) for entity in selected}
        matches = gold_keys & predicted_keys
        gold_counts = Counter(entity.label for entity in gold)
        predicted_counts = Counter(entity.label for entity in selected)
        match_counts = Counter(label for _, _, label in matches)
        characters: Counter[str] = Counter()
        covered: Counter[str] = Counter()
        for entity in gold:
            characters[entity.label] += entity.end - entity.start
            covered[entity.label] += sum(
                max(0, min(entity.end, prediction.end) - max(entity.start, prediction.start))
                for prediction in selected
            )

        aggregate_gold.update(gold_counts)
        aggregate_predicted.update(predicted_counts)
        aggregate_true_positives.update(match_counts)
        aggregate_characters.update(characters)
        aggregate_covered.update(covered)
        case_reports.append({
            "index": index,
            "exact": _scores(len(matches), len(selected), len(gold)),
            "gold_character_coverage": _coverage(sum(covered.values()), sum(characters.values())),
        })

    types = sorted(aggregate_gold.keys() | aggregate_predicted.keys())
    return {
        "schema_version": 1,
        "cases": len(cases),
        "overall": {
            "exact": _scores(sum(aggregate_true_positives.values()), sum(aggregate_predicted.values()),
                             sum(aggregate_gold.values())),
            "gold_character_coverage": _coverage(sum(aggregate_covered.values()), sum(aggregate_characters.values())),
        },
        "by_type": {
            label: {
                "exact": _scores(aggregate_true_positives[label], aggregate_predicted[label], aggregate_gold[label]),
                "gold_character_coverage": _coverage(aggregate_covered[label], aggregate_characters[label]),
            }
            for label in types
        },
        "case_results": case_reports,
        "limits": [
            "Scores describe only the supplied gold spans and resolved detector candidates.",
            "Synthetic fixture scores are not real-world privacy validation.",
        ],
    }


def _json_key(value: object, description: str) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except Exception:
        raise ContextVeilError(description) from None


def evaluate_utility(cases: list[dict], task: Callable[[str], object]) -> dict:
    """Compare a caller-supplied task on paired text against explicit expected answers."""
    if type(cases) is not list or not callable(task):
        raise ContextVeilError("Utility evaluation needs a case list and a callable task.")

    original_correct = anonymized_correct = retained = 0
    case_results: list[dict] = []
    for index, case in enumerate(cases, start=1):
        if type(case) is not dict or set(case) != {"original", "anonymized", "expected"}:
            raise ContextVeilError("Each utility case needs original, anonymized, and expected only.")
        original, anonymized = case["original"], case["anonymized"]
        if not isinstance(original, str) or not isinstance(anonymized, str):
            raise ContextVeilError("Utility case texts must be strings.")
        expected = _json_key(case["expected"], "A utility case expected answer must be a JSON value.")
        try:
            original_answer = task(original)
            anonymized_answer = task(anonymized)
        except Exception:
            raise ContextVeilError("The utility task failed during evaluation.") from None
        original_ok = _json_key(original_answer, "A utility task answer must be a JSON value.") == expected
        anonymized_ok = _json_key(anonymized_answer, "A utility task answer must be a JSON value.") == expected
        original_correct += original_ok
        anonymized_correct += anonymized_ok
        retained += original_ok and anonymized_ok
        case_results.append({"index": index, "original_correct": original_ok, "anonymized_correct": anonymized_ok})

    count = len(cases)
    return {
        "schema_version": 1,
        "cases": count,
        "original_correct": original_correct,
        "anonymized_correct": anonymized_correct,
        "correct_original_retained": retained,
        "original_accuracy": original_correct / count if count else None,
        "anonymized_accuracy": anonymized_correct / count if count else None,
        "accuracy_delta": (anonymized_correct - original_correct) / count if count else None,
        "correct_original_retained_rate": retained / original_correct if original_correct else None,
        "case_results": case_results,
        "limits": [
            "Scores apply only to the supplied paired cases, expected answers, and caller-supplied task.",
            "Task accuracy on these cases is not a generic semantic-preservation guarantee.",
        ],
    }
