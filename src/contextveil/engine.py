"""Replace selected spans without rewriting surrounding language."""

from __future__ import annotations

import math
import re
import secrets
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

from .detectors import CompositeDetector, GlossaryDetector, RegexDetector, SpacyDetector
from .formats import load_text
from .models import ContextVeilError, Detector, Document, Entity, ValidationError
from .policy import LABEL, Policy, field_key


@dataclass(frozen=True)
class UtilityCheck:
    name: str
    check: Callable[[str, str], bool] = field(repr=False)


@dataclass
class Result:
    text: str = field(repr=False)
    report: dict
    output_suffix: str
    recovery: bytes | None = field(default=None, repr=False)


def _priority(entity: Entity) -> int:
    return {"field": 4, "glossary": 3, "regex": 2, "propagated": 1}.get(entity.source, 0)


def _identity(entity: Entity, original: str) -> tuple[str, str, str]:
    return (entity.label, "canonical", entity.canonical) if entity.canonical is not None else (entity.label, "literal", original)


def resolve_entities(text: str, entities: Sequence[Entity]) -> list[Entity]:
    """Keep outer spans; reject incompatible ties and crossing boundaries."""
    if not isinstance(entities, (list, tuple)):
        raise ValidationError("A detector must return a sequence of entities.")
    for entity in entities:
        if (not isinstance(entity, Entity) or type(entity.start) is not int or type(entity.end) is not int
                or not 0 <= entity.start < entity.end <= len(text)
                or not isinstance(entity.label, str) or not LABEL.fullmatch(entity.label)
                or not isinstance(entity.source, str)
                or not isinstance(entity.score, (int, float)) or not math.isfinite(entity.score)
                or not 0 <= entity.score <= 1
                or (entity.canonical is not None and
                    (not isinstance(entity.canonical, str) or not entity.canonical.strip()))):
            raise ValidationError("A detector returned an invalid entity.")
    ordered = sorted(entities, key=lambda e: (e.start, -e.end, -_priority(e), -e.score))
    selected: list[Entity] = []
    for entity in ordered:
        if selected and entity.start < selected[-1].end:
            previous = selected[-1]
            if entity.end > previous.end:
                raise ValidationError("Detector spans partially overlap; resolve the detector or glossary conflict.")
            if ((entity.start != previous.start or entity.end != previous.end)
                    and entity.canonical is not None and entity.label == previous.label
                    and entity.canonical != previous.canonical):
                raise ValidationError("An explicit alias is nested inside a larger entity; add a whole-value glossary entry.")
            if (entity.start == previous.start and entity.end == previous.end
                    and _priority(entity) == _priority(previous)
                    and (entity.label != previous.label or entity.canonical != previous.canonical)):
                raise ValidationError("Detectors disagree on an entity; supply an explicit glossary or field policy.")
            if entity.start == previous.start and entity.end == previous.end and entity.canonical is not None:
                if entity.label != previous.label or (previous.canonical is not None and previous.canonical != entity.canonical):
                    raise ValidationError("An explicit alias conflicts with the selected entity type or identity.")
                selected[-1] = replace(previous, canonical=entity.canonical)
            continue
        selected.append(entity)
    return selected


class Anonymizer:
    """One instance is one linkage scope; a new instance starts unrelated aliases."""

    def __init__(self, detector: Detector | None = None, policy: Policy | None = None,
                 validators: Sequence[UtilityCheck] = ()) -> None:
        self.detector = detector if detector is not None else CompositeDetector([RegexDetector(), SpacyDetector()])
        self.policy = policy if policy is not None else Policy()
        self.validators = tuple(validators)
        if any(not isinstance(v, UtilityCheck) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", v.name)
               for v in self.validators):
            raise ContextVeilError("Validator names must be short, non-sensitive identifiers.")
        self._namespace = secrets.token_hex(6)
        self._tokens: dict[tuple[str, str, str], str] = {}
        self._counts: Counter = Counter()

    def anonymize_text(self, text: str, *, passphrase: str | None = None) -> Result:
        return self.anonymize(load_text(text), passphrase=passphrase)

    def anonymize(self, document: Document, *, passphrase: str | None = None) -> Result:
        return self.anonymize_many([document], passphrase=passphrase)[0]

    def anonymize_many(self, documents: Sequence[Document], *, passphrase: str | None = None) -> list[Result]:
        """Detect across the batch before replacement so exact repeated mentions stay linked."""
        originals = [segment.text for doc in documents for segment in doc.segments]
        if any(self._namespace in text for text in originals):
            raise ValidationError("Input collides with the current token namespace; start a new Anonymizer.")
        candidates: list[list[Entity]] = []
        known: dict[str, Entity] = {}
        ambiguous: set[str] = set()
        for document in documents:
            for segment in document.segments:
                entities = self.detector.detect(segment.text)
                label = self.policy.field_types.get(field_key(segment.field_name)) if segment.field_name else None
                if label and segment.text.strip():
                    left = len(segment.text) - len(segment.text.lstrip())
                    right = len(segment.text.rstrip())
                    entities.append(Entity(left, right, label, source="field"))
                selected = resolve_entities(segment.text, entities)
                candidates.append(selected)
                for entity in selected:
                    value = segment.text[entity.start:entity.end]
                    previous = known.get(value)
                    if previous and (previous.label, previous.canonical) != (entity.label, entity.canonical):
                        ambiguous.add(value)
                    else:
                        known[value] = entity
        # Exact matches propagate; coreference and abbreviated names require an explicit glossary.
        patterns = [(re.compile((r"(?<!\w)" if re.match(r"\w", value[0]) else "") + re.escape(value)
                               + (r"(?!\w)" if re.match(r"\w", value[-1]) else "")), entity)
                    for value, entity in known.items() if value not in ambiguous
                    and entity.label in {"PERSON", "ORG", "LOCATION", "ADDRESS", "EMAIL", "PHONE", "IP_ADDRESS"}]
        results: list[Result] = []
        cursor = 0
        for document in documents:
            transformed: list[str] = []
            entity_counts: Counter = Counter()
            residual_count = 0
            selected_count = 0
            replaced_characters = 0
            for segment in document.segments:
                entities = list(candidates[cursor])
                cursor += 1
                for pattern, entity in patterns:
                    entities.extend(Entity(m.start(), m.end(), entity.label, entity.score, "propagated", entity.canonical)
                                    for m in pattern.finditer(segment.text))
                entities = resolve_entities(segment.text, entities)
                parts: list[str] = []
                residual_parts: list[str] = []
                start = 0
                for entity in entities:
                    untouched = segment.text[start:entity.start]
                    parts.append(untouched)
                    residual_parts.append(untouched)
                    original = segment.text[entity.start:entity.end]
                    # Do not merge people on case, spelling or guessed gender.
                    key = _identity(entity, original)
                    if key not in self._tokens:
                        self._counts[entity.label] += 1
                        self._tokens[key] = f"[[{entity.label}_{self._counts[entity.label]}_{self._namespace}]]"
                    token = self._tokens[key]
                    parts.append(token)
                    residual_parts.append(" " * len(token))
                    entity_counts[entity.label] += 1
                    selected_count += 1
                    replaced_characters += entity.end - entity.start
                    start = entity.end
                parts.append(segment.text[start:])
                residual_parts.append(segment.text[start:])
                output = "".join(parts)
                # Check every retained interval independently of rendered token lengths.
                in_cursor = out_cursor = 0
                for entity in entities:
                    gap = entity.start - in_cursor
                    if output[out_cursor:out_cursor + gap] != segment.text[in_cursor:entity.start]:
                        raise ValidationError("Non-sensitive text changed during replacement.")
                    out_cursor += gap + len(self._tokens[_identity(entity, segment.text[entity.start:entity.end])])
                    in_cursor = entity.end
                if output[out_cursor:] != segment.text[in_cursor:]:
                    raise ValidationError("Non-sensitive text changed during replacement.")
                residual = self.detector.detect("".join(residual_parts))
                residual_count += len(resolve_entities("".join(residual_parts), residual))
                transformed.append(output)
            original_text = document.render([segment.text for segment in document.segments])
            output_text = document.render(transformed)
            structured = document.format in {"json", "jsonl", "csv", "tsv", "yaml"}
            schema_changes = {"keys_or_headers": 0, "scalar_types": 0}
            if structured:
                reparsed = load_text(output_text, document.format)
                if [segment.text for segment in reparsed.segments] != transformed:
                    raise ValidationError("Serialized output changed the transformed content or structure.")
                for before, after, changed in zip(document.segments, reparsed.segments, transformed):
                    if before.kind != after.kind:
                        raise ValidationError("Serialization changed the document structure.")
                    if before.text == changed and before.value_type != after.value_type:
                        raise ValidationError("Serialization changed a non-sensitive scalar type.")
                    schema_changes["keys_or_headers"] += before.kind == "key" and before.text != changed
                    schema_changes["scalar_types"] += before.value_type != after.value_type
            checks: dict[str, bool] = {}
            for validator in self.validators:
                try:
                    checks[validator.name] = validator.check(original_text, output_text) is True
                except Exception:
                    raise ValidationError("A utility validator failed to execute.") from None
                if not checks[validator.name]:
                    raise ValidationError("A configured utility invariant failed.")
            reasons = list(document.warnings)
            if self.policy.require_contextual and not self.detector.contextual:
                reasons.append("No contextual entity detector was used.")
            if residual_count:
                reasons.append("The residual scan found additional candidates.")
            if ambiguous:
                reasons.append("Repeated text has ambiguous entity labels or alias linkage.")
            if any(schema_changes.values()):
                reasons.append("Sensitive keys, headers or scalar types changed; review downstream schema compatibility.")
            report = {
                "schema_version": 1,
                "document_id": secrets.token_hex(16),
                "status": "review_required" if reasons else "checks_passed",
                "format": document.format,
                "segments": len(document.segments),
                "entities_replaced": selected_count,
                "entity_counts": dict(sorted(entity_counts.items())),
                "characters_replaced": replaced_characters,
                "residual_candidates": residual_count,
                "contextual_detection": bool(self.detector.contextual),
                "schema_changes": schema_changes,
                "checks": {"outside_spans_unchanged": True, "entity_consistency": not ambiguous,
                           "structured_roundtrip": True if structured else None,
                           "non_sensitive_scalar_types_preserved": True if structured else None,
                           "utility": checks},
                "review_reasons": reasons,
                "limits": ["Checks apply to detected entities and extracted text, not undetected PII.",
                           "Retained facts can still identify people through context or external data.",
                           "Downstream task accuracy requires a separate representative evaluation."],
            }
            result = Result(output_text, report, document.output_suffix)
            if passphrase is not None:
                from .vault import seal
                result.recovery = seal(original_text, output_text, passphrase, document_id=report["document_id"])
            results.append(result)
        return results


def default_detector(*, model: str = "en_core_web_sm", glossary: list[dict] | None = None,
                     rules_only: bool = False, neural_model_dir: str | None = None,
                     threshold: float = 0.65, device: str = "cpu") -> Detector:
    if neural_model_dir and rules_only:
        raise ContextVeilError("Choose either the neural detector or rules-only detection.")
    detectors: list[Detector] = []
    if neural_model_dir:
        from .neural import GlinerDetector
        detectors.append(GlinerDetector(neural_model_dir, threshold=threshold, device=device))
    else:
        detectors.append(RegexDetector())
    if not rules_only and not neural_model_dir:
        detectors.append(SpacyDetector(model))
    if glossary is not None:
        detectors.append(GlossaryDetector(glossary))
    return CompositeDetector(detectors)
