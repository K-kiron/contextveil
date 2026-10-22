"""ContextVeil: local, policy-driven text de-identification."""

from .detectors import CompositeDetector, GlossaryDetector, RegexDetector, SpacyDetector
from .engine import Anonymizer, Result, UtilityCheck, default_detector
from .formats import load_document, load_text
from .models import ContextVeilError, Detector, Document, Entity, Segment, ValidationError
from .neural import GlinerDetector
from .policy import Policy

__all__ = ["Anonymizer", "CompositeDetector", "ContextVeilError", "Detector", "Document", "Entity",
           "GlinerDetector", "GlossaryDetector", "Policy", "RegexDetector", "Result", "Segment", "SpacyDetector",
           "UtilityCheck", "ValidationError", "default_detector", "load_document", "load_text"]
