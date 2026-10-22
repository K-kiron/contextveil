"""Small, explicit contracts shared by detectors, adapters and the pipeline."""

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence


@dataclass(frozen=True)
class Entity:
    start: int
    end: int
    label: str
    score: float = 1.0
    source: str = "custom"
    canonical: str | None = None


class Detector(Protocol):
    name: str
    contextual: bool

    def detect(self, text: str) -> list[Entity]: ...


@dataclass(frozen=True)
class Segment:
    id: str
    text: str
    field_name: str | None = None
    kind: str = "text"
    value_type: str = "str"


@dataclass
class Document:
    format: str
    segments: list[Segment]
    render: Callable[[Sequence[str]], str]
    output_suffix: str
    warnings: list[str] = field(default_factory=list)


class ContextVeilError(ValueError):
    """Safe diagnostic without raw document content."""


class ValidationError(ContextVeilError):
    """The configured release contract could not be verified."""
