"""The command line publishes only complete, reviewable output bundles."""

import json
import re
from pathlib import Path

from contextveil import cli


def _args(*sources: Path, output: Path, options: tuple[str, ...] = ()) -> list[str]:
    return ["anonymize", *(str(source) for source in sources), "--output-dir", str(output), *options]


def test_batch_review_override_writes_neutral_files_and_links_aliases(tmp_path, capsys):
    first = tmp_path / "Alice Nguyen private.txt"
    second = tmp_path / "Dr. Nguyen private.md"
    first.write_text("Alice Nguyen approved the plan.", encoding="utf-8")
    second.write_text("Dr. Nguyen approved the plan.", encoding="utf-8")
    glossary = tmp_path / "glossary.json"
    glossary.write_text(json.dumps([
        {"value": "Alice Nguyen", "label": "PERSON", "canonical": "person-1"},
        {"value": "Dr. Nguyen", "label": "PERSON", "canonical": "person-1"},
    ]), encoding="utf-8")
    output = tmp_path / "released"

    code = cli.main(_args(first, second, output=output,
                          options=("--rules-only", "--allow-review", "--glossary", str(glossary))))
    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out)["status"] == "review_required"
    assert "Alice Nguyen" not in captured.out + captured.err
    assert "Dr. Nguyen" not in captured.out + captured.err
    assert {path.name for path in output.iterdir()} == {
        "document-001.txt", "document-001.report.json", "document-002.md", "document-002.report.json"}
    texts = [(output / "document-001.txt").read_text(encoding="utf-8"),
             (output / "document-002.md").read_text(encoding="utf-8")]
    tokens = [re.search(r"\[\[PERSON_\d+_[0-9a-f]+\]\]", text).group() for text in texts]
    assert tokens[0] == tokens[1]
    assert "Alice Nguyen" not in texts[0] and "Dr. Nguyen" not in texts[1]
    for report in output.glob("*.report.json"):
        content = report.read_text(encoding="utf-8")
        assert "Alice Nguyen" not in content and "Dr. Nguyen" not in content
        assert json.loads(content)["status"] == "review_required"
    assert not any(path.name.startswith(".contextveil-") for path in tmp_path.iterdir())


def test_strict_review_gate_emits_no_output(tmp_path, capsys):
    source = tmp_path / "source.txt"
    source.write_text("Contact alice@example.test.", encoding="utf-8")
    output = tmp_path / "unreleased"
    assert cli.main(["anonymize", str(source), "--output-dir", str(output), "--rules-only"]) == 2
    captured = capsys.readouterr()
    assert not output.exists()
    assert "alice@example.test" not in captured.out + captured.err
    assert json.loads(captured.err)["status"] == "review_required"


def test_missing_model_does_not_fall_back_to_rules(tmp_path, capsys):
    source = tmp_path / "source.txt"
    source.write_text("Contact alice@example.test.", encoding="utf-8")
    output = tmp_path / "unreleased"
    code = cli.main(["anonymize", str(source), "--output-dir", str(output),
                     "--model", "certainly_missing_contextveil_test_model"])
    assert code == 2 and not output.exists()
    assert "model" in capsys.readouterr().err.lower()


def test_invalid_input_prevents_any_batch_artifact(tmp_path, capsys):
    good = tmp_path / "good.txt"
    bad = tmp_path / "bad.json"
    good.write_text("Contact alice@example.test.", encoding="utf-8")
    bad.write_text('{"broken":', encoding="utf-8")
    output = tmp_path / "released"
    code = cli.main(["anonymize", str(good), str(bad), "--output-dir", str(output),
                     "--rules-only", "--allow-review"])
    assert code == 2 and not output.exists()
    assert "alice@example.test" not in capsys.readouterr().err
    assert not any(path.name.startswith(".contextveil-") for path in tmp_path.iterdir())


def test_existing_output_directory_is_preserved(tmp_path, capsys):
    source = tmp_path / "source.txt"
    source.write_text("Contact alice@example.test.", encoding="utf-8")
    output = tmp_path / "released"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve this", encoding="utf-8")
    code = cli.main(["anonymize", str(source), "--output-dir", str(output),
                     "--rules-only", "--allow-review"])
    assert code == 2
    assert marker.read_text(encoding="utf-8") == "preserve this"
    assert list(output.iterdir()) == [marker]
    assert "already exists" in capsys.readouterr().err


def test_write_failure_cleans_owned_staging_directory(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.txt"
    source.write_text("Contact alice@example.test.", encoding="utf-8")
    output = tmp_path / "released"
    original_open = Path.open

    def fail_report_write(path, *args, **kwargs):
        if path.parent.name.startswith(".contextveil-") and path.name.endswith(".report.json"):
            raise OSError("synthetic write failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_report_write)
    code = cli.main(["anonymize", str(source), "--output-dir", str(output),
                     "--rules-only", "--allow-review"])
    assert code == 2 and not output.exists()
    assert not any(path.name.startswith(".contextveil-") for path in tmp_path.iterdir())
    assert "alice@example.test" not in capsys.readouterr().err


def test_recoverable_env_passphrase_restores_and_wrong_password_writes_nothing(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.txt"
    original = "Contact alice@example.test."
    source.write_text(original, encoding="utf-8")
    output = tmp_path / "released"
    monkeypatch.setenv("CONTEXTVEIL_TEST_PASSPHRASE", "correct horse battery staple")
    code = cli.main(["anonymize", str(source), "--output-dir", str(output),
                     "--rules-only", "--allow-review", "--recoverable",
                     "--passphrase-env", "CONTEXTVEIL_TEST_PASSPHRASE"])
    assert code == 0
    assert (output / "document-001.vault").is_file()
    redacted = output / "document-001.txt"
    assert "alice@example.test" not in redacted.read_text(encoding="utf-8")
    capsys.readouterr()

    recovered = tmp_path / "restored"
    restore_args = ["restore", str(redacted), "--vault", str(output / "document-001.vault"),
                    "--output-dir", str(recovered), "--passphrase-env", "CONTEXTVEIL_TEST_PASSPHRASE"]
    assert cli.main(restore_args) == 0
    assert (recovered / "restored.txt").read_text(encoding="utf-8") == original
    capsys.readouterr()

    monkeypatch.setenv("CONTEXTVEIL_TEST_PASSPHRASE", "another lengthy private passphrase")
    failed = tmp_path / "wrong-password"
    restore_args[restore_args.index(str(recovered))] = str(failed)
    assert cli.main(restore_args) == 2
    assert not failed.exists()
    assert "alice@example.test" not in capsys.readouterr().err


def test_evaluate_threshold_failure_and_invalid_threshold(tmp_path, capsys):
    dataset = tmp_path / "gold.jsonl"
    dataset.write_text(json.dumps({"id": "case-1", "text": "Alice replied.",
                                   "entities": [{"start": 0, "end": 5, "label": "PERSON"}]}) + "\n",
                       encoding="utf-8")
    args = ["evaluate", str(dataset), "--rules-only", "--min-recall", "0.8"]
    assert cli.main(args) == 3
    report = json.loads(capsys.readouterr().out)
    assert report["overall"]["exact"]["recall"] == 0
    assert "Alice" not in json.dumps(report)

    assert cli.main(["evaluate", str(dataset), "--rules-only", "--min-recall", "-0.1"]) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert "threshold" in captured.err.lower()
