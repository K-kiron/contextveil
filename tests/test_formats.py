import io
import json
import zipfile
from decimal import Decimal

import pytest

from contextveil.formats import load_document, load_text
from contextveil.models import ContextVeilError


@pytest.mark.parametrize("format", ["txt", "md", "log"])
def test_plain_text_round_trip(format):
    document = load_text("Alice\nLondon", format)
    assert document.output_suffix == "." + format
    assert document.render(["[PERSON]\nLondon"]) == "[PERSON]\nLondon"
    with pytest.raises(ContextVeilError):
        document.render([])


def test_json_scans_keys_values_and_preserves_unchanged_scalar_types():
    document = load_text('{"Alice":{"age":42,"active":true,"note":null}}', "json")
    assert [segment.id for segment in document.segments] == [f"s{i}" for i in range(6)]
    assert [segment.field_name for segment in document.segments] == [None, None, "age", None, "active", None]
    assert [segment.kind for segment in document.segments] == ["key", "key", "value", "key", "value", "key"]
    assert [segment.value_type for segment in document.segments] == ["str", "str", "int", "str", "bool", "str"]
    assert json.loads(document.render([s.text for s in document.segments])) == {
        "Alice": {"age": 42, "active": True, "note": None}}
    assert json.loads(document.render(["[NAME]", "age", "[AGE]", "active", "true", "note"])) == {
        "[NAME]": {"age": "[AGE]", "active": True, "note": None}}
    assert "Alice" not in " ".join(segment.id for segment in document.segments)


def test_json_key_collision_and_duplicate_source_keys_fail():
    document = load_text('{"one":1,"two":2}', "json")
    with pytest.raises(ContextVeilError, match="collide"):
        document.render(["same", "1", "same", "2"])
    with pytest.raises(ContextVeilError):
        load_text('{"one":1,"one":2}', "json")
    assert load_text('{"value":1e999}', "json").segments[1].value_type == "float"


def test_jsonl_preserves_records_and_segment_identity():
    document = load_text('{"name":"Alice"}\n\n{"name":"Bob"}\n', "jsonl")
    assert len({segment.id for segment in document.segments}) == len(document.segments)
    assert [segment.kind for segment in document.segments] == ["key", "value", "key", "value"]
    assert [segment.value_type for segment in document.segments] == ["str"] * 4
    output = document.render(["name", "[PERSON]", "name", "[PERSON]"])
    assert output == '{"name":"[PERSON]"}\n\n{"name":"[PERSON]"}\n'
    with pytest.raises(ContextVeilError):
        load_text('{"x":1}\ninvalid', "jsonl")


@pytest.mark.parametrize("format,content", [
    ("json", '{"precise":0.12345678901234567890,"tiny":1e-400}'),
    ("jsonl", '{"precise":0.12345678901234567890}\n{"tiny":1e-400}\n'),
    ("yaml", "precise: 0.12345678901234567890\ntiny: 1e-400\n"),
])
def test_unchanged_fractional_values_keep_exact_decimal_value(format, content):
    if format == "yaml":
        pytest.importorskip("yaml")
    document = load_text(content, format)
    assert [(segment.text, segment.value_type) for segment in document.segments if segment.kind == "value"] == [
        ("0.12345678901234567890", "float"), ("1E-400", "float")]
    output = document.render([segment.text for segment in document.segments])
    reparsed = load_text(output, format)
    assert [segment.text for segment in reparsed.segments] == [segment.text for segment in document.segments]
    if format == "json":
        values = json.loads(output, parse_float=Decimal)
        assert values == {"precise": Decimal("0.12345678901234567890"), "tiny": Decimal("1e-400")}


@pytest.mark.parametrize("format,content", [
    ("csv", 'name,age\n"Alice, A",42\n'),
    ("tsv", 'name\tage\nAlice\t42\n'),
])
def test_delimited_header_hints_and_shape(format, content):
    document = load_text(content, format)
    assert document.segments[2].field_name == "name"
    assert document.segments[3].field_name == "age"
    assert [segment.kind for segment in document.segments] == ["key", "key", "value", "value"]
    assert all(segment.value_type == "str" for segment in document.segments)
    assert "[PERSON]" in document.render(["name", "age", "[PERSON]", "42"])
    with pytest.raises(ContextVeilError):
        load_text("a" + ("\t" if format == "tsv" else ",") + "b\n1\n", format)


def test_yaml_rejects_aliases_duplicates_and_tags():
    pytest.importorskip("yaml")
    document = load_text("name: Alice\nage: 42\n", "yaml")
    assert document.output_suffix == ".yaml"
    assert [(segment.kind, segment.value_type) for segment in document.segments] == [
        ("key", "str"), ("value", "str"), ("key", "str"), ("value", "int")]
    assert "[PERSON]" in document.render(["name", "[PERSON]", "age", "42"])
    for content in ("a: &x Alice\nb: *x\n", "a: 1\na: 2\n", "a: !custom foo\n"):
        with pytest.raises(ContextVeilError):
            load_text(content, "yaml")
    with pytest.raises(ContextVeilError, match="Non-finite"):
        load_text("value: .nan\n", "yaml")


def test_yaml_comment_warning_only_for_actual_comments():
    pytest.importorskip("yaml")
    assert not load_text('name: "Alice # Smith"\n', "yaml").warnings
    assert not load_text("note: |\n  # literal content\n", "yaml").warnings
    assert load_text("name: Alice # private note\n", "yaml").warnings


@pytest.mark.parametrize("format,content", [
    ("json", '{"name":"\\u0000"}'),
    ("json", '{"\\ud800":"Alice"}'),
    ("jsonl", '{"name":"\\ud800"}\n'),
    ("yaml", 'name: "\\0"\n'),
])
def test_structured_strings_reject_embedded_nul_or_surrogate(format, content):
    if format == "yaml":
        pytest.importorskip("yaml")
    with pytest.raises(ContextVeilError):
        load_text(content, format)


@pytest.mark.parametrize("format,content", [
    ("html", '<p title="Alice">Hello <b>Bob</b></p><!-- Carol -->'),
    ("xml", '<root label="Alice">Hello <child>Bob</child><!-- Carol --></root>'),
])
def test_markup_extracts_text_and_attributes_to_fresh_txt(format, content):
    document = load_text(content, format)
    assert document.output_suffix == ".txt"
    assert document.warnings
    assert {"Alice", "Hello Bob", "Carol"} <= set(s.text for s in document.segments)
    output = document.render(["[REDACTED]"] * len(document.segments))
    assert "Alice" not in output and "<" not in output


def test_inline_markup_stays_together_and_nested_xml_keeps_reading_order():
    html = load_text("<p>John <b>Smith</b></p>", "html")
    xml = load_text("<p>A<b>B<c>D</c>E</b>F</p>", "xml")
    assert [segment.text for segment in html.segments] == ["John Smith"]
    assert [segment.text for segment in xml.segments] == ["ABDEF"]


def test_xml_rejects_dtd():
    with pytest.raises(ContextVeilError):
        load_text('<!DOCTYPE root [<!ENTITY secret SYSTEM "file:///x">]><root>&secret;</root>', "xml")


def test_docx_extracts_paragraph_across_runs_without_original_archive(tmp_path):
    path = tmp_path / "sample.docx"
    xml = ('<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
           '<w:body><w:p><w:r><w:t>Ali</w:t></w:r><w:r><w:t>ce</w:t></w:r></w:p>'
           '</w:body></w:document>')
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", xml)
        archive.writestr("docProps/core.xml", "private metadata")
    document = load_document(path)
    assert [s.text for s in document.segments] == ["Alice"]
    assert document.output_suffix == ".txt"
    assert document.render(["[PERSON]"]) == "[PERSON]\n"
    assert "private metadata" not in document.render(["[PERSON]"])


def test_docx_cannot_bypass_entity_guard_using_utf16(tmp_path):
    path = tmp_path / "encoded.docx"
    xml = '<?xml version="1.0" encoding="UTF-16"?><!DOCTYPE root [<!ENTITY x "secret">]><root>&x;</root>'
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", xml.encode("utf-16"))
    with pytest.raises(ContextVeilError):
        load_document(path)


def test_pdf_rejects_page_without_text(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    stream = io.BytesIO()
    writer.write(stream)
    path = tmp_path / "blank.pdf"
    path.write_bytes(stream.getvalue())
    with pytest.raises(ContextVeilError, match="OCR"):
        load_document(path)


def test_pdf_extracts_text_to_fresh_txt(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=200, height=100)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    content = DecodedStreamObject()
    content.set_data(b"BT /F1 12 Tf 10 50 Td (Alice) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(content)
    path = tmp_path / "text.pdf"
    with path.open("wb") as output:
        writer.write(output)
    document = load_document(path)
    assert document.output_suffix == ".txt" and document.warnings
    assert "Alice" in document.segments[0].text
    assert "Alice" not in document.render(["[PERSON]"])


def test_file_input_requires_utf8_and_size_bound(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff")
    with pytest.raises(ContextVeilError, match="UTF-8"):
        load_document(path)
    with pytest.raises(ContextVeilError, match="size limit"):
        load_text("x" * (20 * 1024 * 1024 + 1))
