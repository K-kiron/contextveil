"""Replay cached original proposals through the real anonymization pipeline."""

from __future__ import annotations

from collections import Counter
from time import perf_counter

from contextveil.detectors import CompositeDetector, RegexDetector, SpacyDetector
from contextveil.engine import Anonymizer
from contextveil.models import ContextVeilError, Entity, ValidationError
from contextveil.neural import ContextualRulesDetector, GlinerDetector


class ReplayDetector:
    """Return cached detector results; preparation records uncached residual inputs."""

    name = "replay"

    def __init__(self, cached: dict[str, list[Entity]], contextual: bool, *, prepare: bool) -> None:
        self.cached = cached
        self.contextual = contextual
        self.prepare = prepare
        self.missing: dict[str, None] = {}

    def detect(self, text: str) -> list[Entity]:
        if text in self.cached:
            return list(self.cached[text])
        if self.prepare:
            self.missing[text] = None
            return []
        raise ValidationError("The final pipeline requested an uncached detector input.")


def select_neural_predictions(texts: list[str], candidates: list[list[Entity]],
                              threshold: float) -> list[list[Entity]]:
    """Apply production GLiNER selection and contextual rules without loading the model."""
    if (type(texts) is not list or type(candidates) is not list or len(texts) != len(candidates)
            or any(not isinstance(text, str) for text in texts)
            or any(type(entries) is not list or any(not isinstance(entity, Entity) for entity in entries)
                   for entries in candidates)
            or not isinstance(threshold, (int, float)) or not 0 < threshold < 1):
        raise ContextVeilError("Neural selection needs aligned texts, candidates, and a valid threshold.")
    selector = object.__new__(GlinerDetector)
    selector.threshold = threshold
    selector._rules = ContextualRulesDetector()
    return [selector.select(text, entries) for text, entries in zip(texts, candidates)]


def _detect_residuals(detector, texts: list[str]) -> list[list[Entity]]:
    if (type(detector) is CompositeDetector and len(detector.detectors) == 2
            and type(detector.detectors[0]) is RegexDetector
            and type(detector.detectors[1]) is SpacyDetector):
        regex, spacy = detector.detectors
        if any(len(text) > spacy._nlp.max_length for text in texts):
            raise ContextVeilError("A residual input exceeds the installed spaCy model limit.")
        return [
            regex.detect(text) + [Entity(span.start_char, span.end_char, spacy._LABELS[span.label_],
                                         source=spacy.name)
                                  for span in doc.ents if span.label_ in spacy._LABELS]
            for text, doc in zip(texts, spacy._nlp.pipe(texts, batch_size=32))
        ]
    if callable(getattr(detector, "detect_many", None)):
        result = []
        for start in range(0, len(texts), 50):
            result.extend(detector.detect_many(texts[start:start + 50]))
        return result
    return [detector.detect(text) for text in texts]


def _error_kind(error: Exception) -> str:
    if isinstance(error, ValidationError):
        message = str(error)
        if "partially overlap" in message:
            return "candidate_partial_overlap"
        if "disagree" in message:
            return "candidate_disagreement"
        if "uncached detector input" in message:
            return "uncached_detection"
        return "validation_error"
    return "detector_error" if isinstance(error, ContextVeilError) else "unexpected_error"


def run_pipeline(texts: list[str], predictions: list[list[Entity]], detector) -> dict:
    """Use real residual detection and final pipeline gates, one linkage scope per document.

    The returned outputs may contain sensitive source material and belong only in a
    caller-controlled local benchmark directory.
    """
    if (type(texts) is not list or type(predictions) is not list or len(texts) != len(predictions)
            or any(not isinstance(text, str) or not text for text in texts)
            or any(type(entries) is not list or any(not isinstance(entity, Entity) for entity in entries)
                   for entries in predictions)
            or not callable(getattr(detector, "detect", None))
            or type(getattr(detector, "contextual", None)) is not bool):
        raise ContextVeilError("Pipeline inputs need aligned nonempty texts, Entity lists, and a detector.")
    cached = {}
    for text, entries in zip(texts, predictions):
        if text in cached and cached[text] != entries:
            raise ContextVeilError("Identical pipeline inputs have conflicting cached predictions.")
        cached[text] = entries

    replay = ReplayDetector(cached, detector.contextual, prepare=True)
    preparation_failures = Counter()
    started = perf_counter()
    for text in texts:
        try:
            Anonymizer(detector=replay).anonymize_text(text)
        except Exception as error:
            preparation_failures[_error_kind(error)] += 1
    preparation_seconds = perf_counter() - started

    residual_texts = list(replay.missing)
    started = perf_counter()
    if residual_texts:
        try:
            residual_predictions = _detect_residuals(detector, residual_texts)
        except Exception:
            raise ContextVeilError("Residual detector inference failed during pipeline evaluation.") from None
        if (type(residual_predictions) is not list or len(residual_predictions) != len(residual_texts)
                or any(type(entries) is not list or any(not isinstance(entity, Entity) for entity in entries)
                       for entries in residual_predictions)):
            raise ContextVeilError("Residual detector inference returned incomplete predictions.")
        cached.update(zip(residual_texts, residual_predictions))
    residual_seconds = perf_counter() - started

    replay.prepare = False
    outputs = []
    success = []
    auto_release = []
    reports = []
    errors = Counter()
    started = perf_counter()
    for text in texts:
        try:
            result = Anonymizer(detector=replay).anonymize_text(text)
        except Exception as error:
            kind = _error_kind(error)
            outputs.append("")
            success.append(False)
            auto_release.append(False)
            reports.append({"status": "failed", "error_kind": kind})
            errors[kind] += 1
        else:
            report = result.report
            outputs.append(result.text)
            success.append(True)
            auto_release.append(report["status"] == "checks_passed")
            reports.append({
                "status": report["status"],
                "entities_replaced": report["entities_replaced"],
                "characters_replaced": report["characters_replaced"],
                "residual_candidates": report["residual_candidates"],
                "contextual_detection": report["contextual_detection"],
                "review_reasons": report["review_reasons"],
            })
    return {
        "outputs": outputs,
        "success": success,
        "auto_release": auto_release,
        "reports": reports,
        "errors": dict(errors),
        "preparation_failures": dict(preparation_failures),
        "residual_inputs": len(residual_texts),
        "seconds": {"preparation": round(preparation_seconds, 3),
                    "residual_detection": round(residual_seconds, 3),
                    "final_pipeline": round(perf_counter() - started, 3)},
    }
