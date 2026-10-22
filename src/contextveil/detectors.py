"""Local candidate detectors; the pipeline decides what may be released."""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Mapping, Sequence

from .models import ContextVeilError, Detector, Entity

_EMAIL = re.compile(r"(?<![\w.+-])[A-Z0-9._%+-]+@(?:[A-Z0-9-]+\.)+[A-Z]{2,}(?![\w-])", re.I)
_PHONE = re.compile(
    r"(?<!\w)(?:\+1[ .-]?)?(?:\(\d{3}\)[ .-]?|\d{3}[ .-])\d{3}[ .-]\d{4}"
    r"(?:\s*(?:ext(?:ension)?\.?|x)\s*#?\s*\d+)?(?!\w)",
    re.I,
)
_SSN = re.compile(r"(?<!\d)(\d{3})-(\d{2})-(\d{4})(?!\d)")
_CARD = re.compile(r"(?<!\w)(?:\d[ -]?){12,18}\d(?!\w)")
_IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_IPV6 = re.compile(r"(?<![\w.:])(?:[A-F\d]{0,4}:){2,7}[A-F\d]{0,4}(?![\w:]|\.[\w])", re.I)
_URL = re.compile(r"(?<!\w)https?://[^\s<>\"']+", re.I)
_ID = re.compile(
    r"\b(?:account|customer|patient|employee|case|invoice|order|passport|licen[cs]e|reference)"
    r"[ _-]*(?:id|number|no\.?|#)\s*[:=#]?\s*(?P<value>[A-Z\d][A-Z\d_-]{3,})(?!\w)",
    re.I,
)
_SECRET = re.compile(
    r"\b(?:api[ _-]?key|access[ _-]?token|auth(?:entication)?[ _-]?token|secret|password|private[ _-]?key)"
    r"\s*[:=]\s*[\"']?(?P<value>[A-Z\d_./+=-]{8,})",
    re.I,
)
_TRAILING_URL_PUNCTUATION = ".,;:!?)]}"
_GLOSSARY_LABEL = re.compile(r"[A-Z][A-Z0-9_]{0,31}\Z")
_MAX_GLOSSARY_ENTRIES = 1000
_MAX_GLOSSARY_VALUE_LENGTH = 256


def _luhn(value: str) -> bool:
    digits = [int(character) for character in value if character.isdigit()]
    if len(set(digits)) == 1:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


class RegexDetector:
    """Find high-confidence structured values and explicitly labelled IDs."""

    name = "regex"
    contextual = False

    def detect(self, text: str) -> list[Entity]:
        found: list[Entity] = []

        def add(pattern: re.Pattern[str], label: str) -> None:
            found.extend(Entity(match.start(), match.end(), label, source=self.name) for match in pattern.finditer(text))

        add(_EMAIL, "EMAIL")
        add(_PHONE, "PHONE")
        found.extend(
            Entity(match.start(), match.end(), "SSN", source=self.name)
            for match in _SSN.finditer(text)
            if match.group(1) not in {"000", "666"}
            and int(match.group(1)) < 900
            and match.group(2) != "00"
            and match.group(3) != "0000"
        )
        found.extend(
            Entity(match.start(), match.end(), "CREDIT_CARD", source=self.name)
            for match in _CARD.finditer(text)
            if _luhn(match.group())
        )
        for pattern in (_IPV4, _IPV6):
            for match in pattern.finditer(text):
                try:
                    ipaddress.ip_address(match.group())
                except ValueError:
                    continue
                found.append(Entity(match.start(), match.end(), "IP_ADDRESS", source=self.name))
        for match in _URL.finditer(text):
            end = match.end()
            while end > match.start() and text[end - 1] in _TRAILING_URL_PUNCTUATION:
                end -= 1
            if end > match.start():
                found.append(Entity(match.start(), end, "URL", source=self.name))
        for pattern, label in ((_ID, "ID"), (_SECRET, "SECRET")):
            for match in pattern.finditer(text):
                end = match.end("value")
                if label == "SECRET" and text[end - 1] == ".":
                    end -= 1
                found.append(Entity(match.start("value"), end, label, source=self.name))
        return sorted(found, key=lambda entity: (entity.start, entity.end, entity.label))


class SpacyDetector:
    """Use a model the caller already installed; never fetch one implicitly."""

    name = "spacy"
    contextual = True
    _LABELS = {"PERSON": "PERSON", "ORG": "ORG", "GPE": "LOCATION", "LOC": "LOCATION", "FAC": "LOCATION"}

    def __init__(self, model: str = "en_core_web_sm") -> None:
        try:
            import spacy
        except ImportError:
            raise ContextVeilError("Install spaCy and an English model in the active environment to use SpacyDetector.") from None
        if not spacy.util.is_package(model):
            raise ContextVeilError("Install the requested spaCy model in the active environment before using SpacyDetector.")
        try:
            self._nlp = spacy.load(model)
        except (ImportError, OSError, ValueError):
            raise ContextVeilError("The installed spaCy model could not be loaded; check its compatibility with spaCy.") from None
        if not self._nlp.has_pipe("ner"):
            raise ContextVeilError("The installed spaCy model needs an enabled NER component.")

    def detect(self, text: str) -> list[Entity]:
        if len(text) > self._nlp.max_length:
            raise ContextVeilError("Input exceeds the installed spaCy model's maximum document length.")
        try:
            return [
                Entity(span.start_char, span.end_char, self._LABELS[span.label_], source=self.name)
                for span in self._nlp(text).ents
                if span.label_ in self._LABELS
            ]
        except Exception:
            raise ContextVeilError("The installed spaCy model could not process the input.") from None


class GlossaryDetector:
    """Find explicitly supplied values, including aliases sharing a canonical name."""

    name = "glossary"
    contextual = False

    def __init__(self, entries: Sequence[Mapping[str, str]]) -> None:
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ContextVeilError("Glossary entries must be a list of mappings.")
        if len(entries) > _MAX_GLOSSARY_ENTRIES:
            raise ContextVeilError("The glossary has too many entries.")
        self._patterns: list[tuple[re.Pattern[str], str, str | None]] = []
        seen: dict[str, tuple[str, str | None]] = {}
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ContextVeilError("Each glossary entry must be a mapping.")
            value = entry.get("value")
            label = entry.get("label")
            canonical = entry.get("canonical")
            if not isinstance(value, str) or not value.strip() or not isinstance(label, str) or not label.strip():
                raise ContextVeilError("Each glossary entry needs a nonempty string value and label.")
            if len(value) > _MAX_GLOSSARY_VALUE_LENGTH:
                raise ContextVeilError("A glossary value is too long.")
            if not _GLOSSARY_LABEL.fullmatch(label):
                raise ContextVeilError("A glossary label must be an uppercase identifier of at most 32 characters.")
            if canonical is not None and (
                not isinstance(canonical, str) or not canonical.strip() or len(canonical) > _MAX_GLOSSARY_VALUE_LENGTH
            ):
                raise ContextVeilError("A glossary canonical value must be a nonempty string.")
            key = value.casefold()
            identity = (label, canonical)
            if key in seen:
                if seen[key] != identity:
                    raise ContextVeilError("A glossary value has conflicting label or canonical mappings.")
                continue
            seen[key] = identity
            left = r"(?<!\w)" if re.match(r"\w", value[0]) else ""
            right = r"(?!\w)" if re.match(r"\w", value[-1]) else ""
            self._patterns.append((re.compile(left + re.escape(value) + right, re.I), label, canonical))

    def detect(self, text: str) -> list[Entity]:
        found = [
            Entity(match.start(), match.end(), label, source=self.name, canonical=canonical)
            for pattern, label, canonical in self._patterns
            for match in pattern.finditer(text)
        ]
        return sorted(found, key=lambda entity: (entity.start, entity.end, entity.label))


class CompositeDetector:
    """Collect candidates from multiple independent detectors."""

    def __init__(self, detectors: Sequence[Detector]) -> None:
        self.detectors = tuple(detectors)
        self.name = "composite"
        self.contextual = any(detector.contextual for detector in self.detectors)

    def detect(self, text: str) -> list[Entity]:
        return [entity for detector in self.detectors for entity in detector.detect(text)]
