"""Offline GLiNER PII detection with bounded, overlapping windows."""

from __future__ import annotations

import json
import math
import re
from datetime import date
from pathlib import Path

from .detectors import RegexDetector
from .model_assets import verify_gliner_bundle
from .models import ContextVeilError, Entity

# These prompts are part of the evaluated model configuration, not corpus-specific aliases.
PII_LABELS = {
    "person": "PERSON", "organization": "ORG", "address": "ADDRESS",
    "city": "LOCATION", "country": "LOCATION", "state": "LOCATION",
    "postal code": "ADDRESS", "date": "DATE", "time": "DATE",
    "email": "EMAIL", "phone number": "PHONE", "social security number": "SSN",
    "credit card number": "CREDIT_CARD", "ip address": "IP_ADDRESS", "url": "URL",
    "account number": "ID", "medical record number": "ID", "national id number": "ID",
    "tax identification number": "ID", "license plate number": "ID",
    "vehicle identification number": "ID", "driver's license number": "ID",
    "customer id": "ID", "employee id": "ID", "username": "ID",
    "device identifier": "ID", "bank routing number": "ID", "swift bic": "ID",
    "password": "SECRET", "api key": "SECRET", "cvv": "SECRET", "pin": "SECRET",
    "biometric identifier": "ID", "coordinate": "LOCATION", "unique identifier": "ID",
    "health insurance number": "ID",
}

_BIRTH_DATE = re.compile(
    r"\b(?:dob|date[ _-]*of[ _-]*birth|birth[ _-]*date|born)[\s*`\"']*"
    r"(?:is\s+|on\s+)?[:=]?[\s*`\"']*(?P<value>\d{4}-\d{2}-\d{2})(?!\d)", re.I,
)
_CONTEXT_ID = re.compile(
    r"\b(?:mrn|medical[ _-]*record(?:[ _-]*(?:number|no\.?|id))?|"
    r"(?:customer|employee|patient|account|certificate|licen[cs]e|device|beneficiary)"
    r"[ _-]*(?:id(?:entifier)?|number|no\.?)|"
    r"(?:application|case|claim|reference|docket)[ _-]*\(?[ _-]*(?:number|no\.?|id|#))"
    r"[\s*`\"']*(?:is\s+)?[:=#]?[\s*`\"']*(?P<value>[A-Z\d][A-Z\d_/-]{2,})(?!\w)", re.I,
)


class ContextualRulesDetector(RegexDetector):
    """Extend validated patterns with explicit birth-date and identifier field cues."""

    def detect(self, text: str) -> list[Entity]:
        found = super().detect(text)
        for match in _BIRTH_DATE.finditer(text):
            try:
                date.fromisoformat(match.group("value"))
            except ValueError:
                continue
            found.append(Entity(match.start("value"), match.end("value"), "DATE", source="regex"))
        for match in _CONTEXT_ID.finditer(text):
            if any(char.isdigit() for char in match.group("value")):
                found.append(Entity(match.start("value"), match.end("value"), "ID", source="regex"))
        return found


def merge_predictions(entities: list[Entity]) -> list[Entity]:
    """Union crossing automatic predictions rather than expose their uncovered edges."""
    result: list[Entity] = []
    for entity in sorted(entities, key=lambda e: (e.start, -e.end, e.source != "regex", -e.score)):
        if not result or entity.start >= result[-1].end:
            result.append(entity)
            continue
        previous = result[-1]
        if entity.end <= previous.end:
            continue
        label = previous.label if previous.label == entity.label else "SENSITIVE"
        result[-1] = Entity(previous.start, entity.end, label, max(previous.score, entity.score), "gliner")
    return result


class GlinerDetector:
    """Load a local span-model bundle; inference never downloads assets or sends text."""

    name = "gliner"
    contextual = True

    def __init__(self, model_dir: str | Path, *, threshold: float = 0.65,
                 device: str = "cpu", batch_size: int = 8) -> None:
        if (not isinstance(threshold, (int, float)) or not math.isfinite(threshold)
                or not 0 < threshold < 1 or type(batch_size) is not int or batch_size < 1
                or device not in {"cpu", "cuda"}):
            raise ContextVeilError("Invalid GLiNER threshold, batch size or device.")
        root = Path(model_dir)
        required = ("gliner_config.json", "pytorch_model.bin", "encoder/config.json",
                    "encoder/tokenizer_config.json", "encoder/spm.model")
        if not all((root / name).is_file() for name in required):
            raise ContextVeilError("GLiNER requires a complete local model and encoder tokenizer bundle.")
        self.model_reference = verify_gliner_bundle(root)
        try:
            import torch
            from gliner.config import UniEncoderSpanConfig
            from gliner.model import UniEncoderSpanGLiNER
            from transformers import DebertaV2TokenizerFast
        except ImportError:
            raise ContextVeilError("Install contextveil[neural] to use the local GLiNER detector.") from None
        try:
            supplied = json.loads((root / "gliner_config.json").read_text(encoding="utf-8"))
            encoder = json.loads((root / "encoder/config.json").read_text(encoding="utf-8"))
            if (encoder.get("model_type") != "deberta-v2" or supplied.get("span_mode") != "markerV0"
                    or supplied.get("model_name") != "microsoft/mdeberta-v3-base"
                    or supplied.get("words_splitter_type", "whitespace") != "whitespace"):
                raise ValueError("Unsupported model architecture")
            # Construct known local classes; do not load remote model/config code.
            tokenizer = DebertaV2TokenizerFast(
                vocab_file=str(root / "encoder/spm.model"), model_max_length=512,
            )
            tokenizer.add_tokens(["[FLERT]", "<<ENT>>", "<<SEP>>"], special_tokens=True)
            encoder["vocab_size"] = len(tokenizer)
            config = UniEncoderSpanConfig(
                model_name="microsoft/mdeberta-v3-base", encoder_config=encoder,
                vocab_size=len(tokenizer), class_token_index=len(tokenizer) - 2,
                max_len=384, max_width=12, hidden_size=512, span_mode="markerV0",
            )
            self._model = UniEncoderSpanGLiNER(config, tokenizer=tokenizer, backbone_from_pretrained=False)
            state = torch.load(root / "pytorch_model.bin", map_location="cpu", weights_only=True)
            self._model.model.load_state_dict(state, strict=True)
            self._model.to(device).eval()
        except Exception:
            raise ContextVeilError("The local GLiNER bundle could not be loaded; verify assets and dependencies.") from None
        self.threshold = threshold
        self.batch_size = batch_size
        self._torch = torch
        self._tokenizer = tokenizer
        self._rules = ContextualRulesDetector()
        self._prompt = [part for label in PII_LABELS for part in ("<<ENT>>", label)] + ["<<SEP>>"]

    def _windows(self, text: str) -> list[tuple[int, str]]:
        words = list(self._model.data_processor.words_splitter(text))
        windows: list[tuple[int, str]] = []
        start = 0
        while start < len(words):
            low, high = start + 1, min(start + 256, len(words))
            end = start
            while low <= high:
                middle = (low + high) // 2
                tokens = self._tokenizer(self._prompt + [word[0] for word in words[start:middle]],
                                         is_split_into_words=True, truncation=False, verbose=False)["input_ids"]
                if len(tokens) <= 512:
                    end, low = middle, middle + 1
                else:
                    high = middle - 1
            if end == start:
                raise ContextVeilError("A single input token exceeds the model window; no text was silently truncated.")
            left, right = words[start][1], words[end - 1][2]
            windows.append((left, text[left:right]))
            if end == len(words):
                break
            start = max(start + 1, end - 48)
        return windows

    def candidates_many(self, texts: list[str], *, threshold: float = 0.1) -> list[list[Entity]]:
        """Return model proposals for calibration; production callers use detect/detect_many."""
        if (not isinstance(texts, list) or any(not isinstance(text, str) for text in texts)
                or not isinstance(threshold, (int, float)) or not math.isfinite(threshold)
                or not 0 < threshold < 1):
            raise ContextVeilError("Invalid GLiNER text batch or candidate threshold.")
        chunks = [(index, offset, chunk) for index, text in enumerate(texts) for offset, chunk in self._windows(text)]
        result: list[list[Entity]] = [[] for _ in texts]
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start:start + self.batch_size]
            try:
                with self._torch.inference_mode():
                    outputs = self._model.inference([item[2] for item in batch], list(PII_LABELS),
                                                    threshold=threshold, flat_ner=False,
                                                    multi_label=True, batch_size=self.batch_size)
            except Exception:
                raise ContextVeilError("The local GLiNER model could not process this batch.") from None
            if len(outputs) != len(batch):
                raise ContextVeilError("GLiNER returned an incomplete batch.")
            for (index, offset, chunk), entries in zip(batch, outputs):
                for entry in entries:
                    left, right, label, score = entry["start"], entry["end"], entry["label"], entry["score"]
                    if (type(left) is not int or type(right) is not int or not 0 <= left < right <= len(chunk)
                            or label not in PII_LABELS or not math.isfinite(score) or not 0 <= score <= 1):
                        raise ContextVeilError("GLiNER returned an invalid entity.")
                    result[index].append(Entity(offset + left, offset + right, PII_LABELS[label], score, "gliner"))
        return result

    def select(self, text: str, candidates: list[Entity]) -> list[Entity]:
        return merge_predictions(self._rules.detect(text) + [e for e in candidates if e.score >= self.threshold])

    def detect_many(self, texts: list[str]) -> list[list[Entity]]:
        return [self.select(text, candidates) for text, candidates in
                zip(texts, self.candidates_many(texts, threshold=self.threshold))]

    def detect(self, text: str) -> list[Entity]:
        return self.detect_many([text])[0]
