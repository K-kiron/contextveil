"""Bounded, text-only input adapters for the redaction pipeline."""

from __future__ import annotations

import csv
import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Sequence
from xml.etree import ElementTree as ET

from .models import ContextVeilError, Document, Segment

MAX_SOURCE_BYTES = 20 * 1024 * 1024
MAX_EXPANDED_BYTES = 40 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 500
MAX_PDF_PAGES = 100


def _checked(values: Sequence[str], count: int) -> None:
    if isinstance(values, (str, bytes)) or len(values) != count or any(not isinstance(value, str) for value in values):
        raise ContextVeilError("The transformed segment count or type is invalid.")
    if any("\x00" in value or any(0xD800 <= ord(char) <= 0xDFFF for char in value) for value in values):
        raise ContextVeilError("Text contains a NUL or unpaired surrogate.")


def _plain(text: str, format: str) -> Document:
    segments = [Segment("s0", text)]

    def render(values: Sequence[str]) -> str:
        _checked(values, 1)
        return values[0]

    return Document(format, segments, render, "." + format)


@dataclass(frozen=True)
class _Slot:
    index: int
    original: str
    value: Any
    key: bool = False


def _json_dumps(value: Any, *, compact: bool = False, level: int = 0) -> str:
    """Serialize Decimal numbers as JSON numbers without a float conversion."""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        if not value:
            return "{}"
        separator = "," if compact else ",\n"
        entries = [(json.dumps(key, ensure_ascii=False) + (":" if compact else ": ")
                    + _json_dumps(item, compact=compact, level=level + 1))
                   for key, item in value.items()]
        if compact:
            return "{" + separator.join(entries) + "}"
        return "{\n" + separator.join("  " * (level + 1) + entry for entry in entries) + "\n" + "  " * level + "}"
    if isinstance(value, list):
        if not value:
            return "[]"
        separator = "," if compact else ",\n"
        entries = [_json_dumps(item, compact=compact, level=level + 1) for item in value]
        if compact:
            return "[" + separator.join(entries) + "]"
        return "[\n" + separator.join("  " * (level + 1) + entry for entry in entries) + "\n" + "  " * level + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _structured(data: Any, format: str, *, compact: bool = False) -> Document:
    segments: list[Segment] = []

    def add(value: Any, hint: str | None = None, key: bool = False) -> _Slot:
        original = value if isinstance(value, str) else _json_dumps(value)
        if "\x00" in original or any(0xD800 <= ord(char) <= 0xDFFF for char in original):
            raise ContextVeilError("Structured text contains a NUL or unpaired surrogate.")
        slot = _Slot(len(segments), original, value, key)
        value_type = "float" if isinstance(value, Decimal) else type(value).__name__
        segments.append(Segment(f"s{slot.index}", original, hint,
                                "key" if key else "value", value_type))
        return slot

    def prepare(value: Any, hint: str | None = None) -> Any:
        if isinstance(value, dict):
            if any(not isinstance(k, str) for k in value):
                raise ContextVeilError("Structured mapping keys must be strings.")
            return [(add(k, key=True), prepare(v, k)) for k, v in value.items()]
        if isinstance(value, list):
            return ("list", [prepare(item, hint) for item in value])
        if value is None:
            return None
        if isinstance(value, Decimal) and not value.is_finite() or isinstance(value, float) and not math.isfinite(value):
            raise ContextVeilError("Non-finite numbers are unsupported.")
        if isinstance(value, (str, int, float, bool, Decimal)):
            return add(value, hint)
        raise ContextVeilError("The structured document contains an unsupported value.")

    prepared = prepare(data)

    def restore(node: Any, values: Sequence[str]) -> Any:
        if isinstance(node, _Slot):
            changed = values[node.index]
            if node.key:
                return changed
            return node.value if changed == node.original else changed
        if isinstance(node, list):
            result: dict[str, Any] = {}
            for key, child in node:
                new_key = restore(key, values)
                if new_key in result:
                    raise ContextVeilError("Transformed mapping keys collide.")
                result[new_key] = restore(child, values)
            return result
        if isinstance(node, tuple):
            return [restore(child, values) for child in node[1]]
        return node

    def render(values: Sequence[str]) -> str:
        _checked(values, len(segments))
        restored = restore(prepared, values)
        if format == "yaml":
            import yaml

            class DecimalDumper(yaml.SafeDumper):
                pass

            DecimalDumper.add_representer(Decimal, lambda dumper, number:
                                          dumper.represent_scalar("tag:yaml.org,2002:float", str(number)))
            return yaml.dump(restored, Dumper=DecimalDumper, allow_unicode=True, sort_keys=False)
        return _json_dumps(restored, compact=compact) + ("" if compact else "\n")

    return Document(format, segments, render, ".yaml" if format == "yaml" else ".json")


def _parse_json(text: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContextVeilError("Duplicate JSON mapping key.")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ContextVeilError("Non-finite JSON numbers are unsupported.")

    def exact_float(value: str) -> Decimal:
        try:
            return Decimal(value)
        except InvalidOperation:
            raise ContextVeilError("Unsupported JSON number.") from None

    return json.loads(text, object_pairs_hook=unique, parse_constant=reject_constant,
                      parse_float=exact_float)


def _jsonl(text: str) -> Document:
    lines = text.splitlines()
    documents: list[Document | None] = []
    for line in lines:
        if not line.strip():
            documents.append(None)
            continue
        try:
            documents.append(_structured(_parse_json(line), "json", compact=True))
        except (ValueError, TypeError, RecursionError):
            raise ContextVeilError("Invalid JSONL record.") from None
    segments: list[Segment] = []
    for document in documents:
        if document:
            start = len(segments)
            segments.extend(Segment(f"s{start + i}", s.text, s.field_name, s.kind, s.value_type)
                            for i, s in enumerate(document.segments))

    def render(values: Sequence[str]) -> str:
        _checked(values, len(segments))
        output: list[str] = []
        index = 0
        for document in documents:
            if document is None:
                output.append("")
            else:
                count = len(document.segments)
                output.append(document.render(values[index:index + count]))
                index += count
        return "\n".join(output) + ("\n" if text.endswith(("\n", "\r")) else "")

    return Document("jsonl", segments, render, ".jsonl")


def _delimited(text: str, format: str) -> Document:
    delimiter = "\t" if format == "tsv" else ","
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True))
    except csv.Error:
        raise ContextVeilError("Invalid delimited text.") from None
    if rows and any(len(row) != len(rows[0]) for row in rows):
        raise ContextVeilError("Delimited rows have inconsistent column counts.")
    segments: list[Segment] = []
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            segments.append(Segment(f"s{len(segments)}", value,
                                    rows[0][j] if i else None, "value" if i else "key", "str"))

    def render(values: Sequence[str]) -> str:
        _checked(values, len(segments))
        output = io.StringIO(newline="")
        writer = csv.writer(output, delimiter=delimiter, lineterminator="\n")
        index = 0
        for row in rows:
            writer.writerow(values[index:index + len(row)])
            index += len(row)
        return output.getvalue()

    return Document(format, segments, render, "." + format)


def _yaml(text: str) -> Document:
    try:
        import yaml
    except ImportError:
        raise ContextVeilError("YAML support requires PyYAML.") from None

    class StrictLoader(yaml.SafeLoader):
        def compose_node(self, parent: Any, index: Any) -> Any:
            if self.check_event(yaml.AliasEvent):
                raise ContextVeilError("YAML aliases are unsupported.")
            return super().compose_node(parent, index)

        def construct_mapping(self, node: Any, deep: bool = False) -> Any:
            mapping: dict[str, Any] = {}
            for key_node, value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                if not isinstance(key, str) or key == "<<":
                    raise ContextVeilError("YAML mapping keys must be plain strings.")
                if key in mapping:
                    raise ContextVeilError("Duplicate YAML mapping key.")
                mapping[key] = self.construct_object(value_node, deep=deep)
            return mapping

    def exact_yaml_float(loader: StrictLoader, node: Any) -> Decimal:
        source = loader.construct_scalar(node).replace("_", "")
        if ":" in source:
            raise ContextVeilError("YAML sexagesimal numbers are unsupported.")
        if source.lower().lstrip("+-") in {".nan", ".inf"}:
            raise ContextVeilError("Non-finite numbers are unsupported.")
        try:
            number = Decimal(source)
        except InvalidOperation:
            raise ContextVeilError("Invalid or unsupported YAML number.") from None
        if not number.is_finite():
            raise ContextVeilError("Non-finite numbers are unsupported.")
        return number

    StrictLoader.add_constructor("tag:yaml.org,2002:float", exact_yaml_float)
    StrictLoader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(r"[-+]?[0-9][0-9_]*(?:\.[0-9_]*)?[eE][-+]?[0-9]+\Z"),
        list("-+0123456789"),
    )

    try:
        data = yaml.load(text, Loader=StrictLoader)
        document = _structured(data, "yaml")
        covered = 0
        for token in yaml.scan(text):
            if "#" in text[covered:token.start_mark.index]:
                document.warnings.append("YAML comments are omitted from the output.")
                break
            covered = max(covered, token.end_mark.index)
        return document
    except ContextVeilError:
        raise
    except (yaml.YAMLError, ValueError, RecursionError):
        raise ContextVeilError("Invalid or unsupported YAML.") from None


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_parts: list[str] = []
        self.extras: list[str] = []
        self.current: list[str] = []

    def _flush(self) -> None:
        content = "".join(self.current).strip()
        if content:
            self.text_parts.append(content)
        self.current.clear()

    @property
    def parts(self) -> list[str]:
        self._flush()
        return self.text_parts + self.extras

    def handle_data(self, data: str) -> None:
        self.current.append(data)

    def handle_comment(self, data: str) -> None:
        if data.strip():
            self.extras.append(data.strip())

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"p", "div", "li", "br", "hr", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._flush()
        self.extras.extend(value for _, value in attrs if value and value.strip())

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._flush()


def _extracted(parts: list[str], format: str) -> Document:
    segments = [Segment(f"s{i}", part) for i, part in enumerate(parts)]

    def render(values: Sequence[str]) -> str:
        _checked(values, len(segments))
        return "\n".join(values) + ("\n" if values else "")

    return Document(format, segments, render, ".txt",
                    ["Text extraction is unverified; layout, embedded content, metadata and other source surfaces may be omitted."])


def _xml(text: str) -> Document:
    if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", text, re.IGNORECASE):
        raise ContextVeilError("XML DTDs and entities are unsupported.")
    try:
        root = ET.fromstring(text, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    except ET.ParseError:
        raise ContextVeilError("Invalid XML.") from None
    extras: list[str] = []
    for element in root.iter():
        extras.extend(value for value in element.attrib.values() if value.strip())
        if element.tag is ET.Comment and element.text and element.text.strip():
            extras.append(element.text.strip())

    def inner(element: ET.Element) -> str:
        content = [element.text or ""]
        for child in element:
            if child.tag is not ET.Comment:
                content.append(inner(child))
            content.append(child.tail or "")
        return "".join(content)

    parts: list[str] = []

    def walk(element: ET.Element) -> None:
        children = [child for child in element if child.tag is not ET.Comment]
        tag = element.tag.rsplit("}", 1)[-1].lower()
        mixed = bool((element.text or "").strip()) or any((child.tail or "").strip() for child in element)
        if mixed or (children and tag in {"p", "paragraph", "text", "title", "heading", "div", "li", "cell"}):
            content = inner(element).strip()
            if content:
                parts.append(content)
        elif children:
            for child in children:
                walk(child)
        elif element.text and element.text.strip():
            parts.append(element.text.strip())

    walk(root)
    return _extracted(parts + extras, "xml")


def _docx(raw: bytes) -> Document:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            if len(members) > MAX_ARCHIVE_MEMBERS or sum(m.file_size for m in members) > MAX_EXPANDED_BYTES:
                raise ContextVeilError("DOCX archive exceeds the extraction limit.")
            names = {member.filename for member in members}
            if "word/document.xml" not in names or "[Content_Types].xml" not in names:
                raise ContextVeilError("Invalid DOCX archive.")
            parts: list[str] = []
            for member in members:
                if not (member.filename.startswith("word/") and member.filename.endswith(".xml")):
                    continue
                if member.file_size > MAX_EXPANDED_BYTES or member.compress_size == 0 and member.file_size:
                    raise ContextVeilError("DOCX archive exceeds the extraction limit.")
                xml = archive.read(member).decode("utf-8-sig")
                if "\x00" in xml or re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", xml, re.IGNORECASE):
                    raise ContextVeilError("DOCX XML entities are unsupported.")
                root = ET.fromstring(xml)
                namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
                for paragraph in root.iter(namespace + "p"):
                    paragraph_text = "".join(element.text or "" for element in paragraph.iter()
                                             if element.tag in (namespace + "t", namespace + "instrText"))
                    if paragraph_text.strip():
                        parts.append(paragraph_text)
            if not parts:
                raise ContextVeilError("DOCX contains no extractable text.")
            return _extracted(parts, "docx")
    except (zipfile.BadZipFile, ET.ParseError, RuntimeError, EOFError, UnicodeError):
        raise ContextVeilError("Invalid DOCX archive.") from None


def _pdf(raw: bytes) -> Document:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ContextVeilError("PDF support requires pypdf.") from None
    try:
        reader = PdfReader(io.BytesIO(raw), strict=True)
        if reader.is_encrypted:
            raise ContextVeilError("Encrypted PDFs are unsupported.")
        if not reader.pages or len(reader.pages) > MAX_PDF_PAGES:
            raise ContextVeilError("PDF page count is unsupported.")
        parts: list[str] = []
        total = 0
        for page in reader.pages:
            extracted = page.extract_text() or ""
            if not extracted.strip():
                raise ContextVeilError("PDF has a page without extractable text; OCR is unsupported.")
            total += len(extracted.encode("utf-8"))
            if total > MAX_EXPANDED_BYTES:
                raise ContextVeilError("Extracted PDF text exceeds the limit.")
            parts.append(extracted)
        return _extracted(parts, "pdf")
    except ContextVeilError:
        raise
    except Exception:
        raise ContextVeilError("Invalid or unsupported PDF.") from None


def load_text(text: str, format: str = "txt") -> Document:
    """Load supported UTF-8 text formats without retaining the source container."""
    try:
        invalid = not isinstance(text, str) or "\x00" in text or len(text.encode("utf-8")) > MAX_SOURCE_BYTES
    except UnicodeEncodeError:
        invalid = True
    if invalid:
        raise ContextVeilError("Text input is invalid or exceeds the size limit.")
    format = format.lower().lstrip(".")
    if format in {"txt", "md", "log"}:
        return _plain(text, format)
    if format == "json":
        try:
            return _structured(_parse_json(text), "json")
        except ContextVeilError:
            raise
        except (ValueError, TypeError, RecursionError):
            raise ContextVeilError("Invalid JSON.") from None
    if format == "jsonl":
        return _jsonl(text)
    if format in {"csv", "tsv"}:
        return _delimited(text, format)
    if format in {"yaml", "yml"}:
        return _yaml(text)
    if format == "html" or format == "htm":
        parser = _HTMLText()
        parser.feed(text)
        return _extracted(parser.parts, "html")
    if format == "xml":
        return _xml(text)
    raise ContextVeilError("Unsupported text format.")


def load_document(path: Path | str) -> Document:
    """Read a local file into independently transformable text segments."""
    source = Path(path)
    format = source.suffix.lower().lstrip(".")
    if format not in {"txt", "md", "log", "json", "jsonl", "csv", "tsv",
                      "yaml", "yml", "html", "htm", "xml", "docx", "pdf"}:
        raise ContextVeilError("Unsupported file format.")
    try:
        if source.stat().st_size > MAX_SOURCE_BYTES:
            raise ContextVeilError("Source file exceeds the size limit.")
        raw = source.read_bytes()
    except OSError:
        raise ContextVeilError("Unable to read source file.") from None
    if len(raw) > MAX_SOURCE_BYTES:
        raise ContextVeilError("Source file exceeds the size limit.")
    if format == "docx":
        return _docx(raw)
    if format == "pdf":
        return _pdf(raw)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ContextVeilError("Text input must be UTF-8.") from None
    return load_text(text, format)
