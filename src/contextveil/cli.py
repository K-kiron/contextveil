"""Local commands with neutral filenames and all-or-nothing output bundles."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from .engine import Anonymizer, default_detector
from .formats import MAX_SOURCE_BYTES, load_document
from .models import ContextVeilError, ValidationError
from .policy import Policy


def _json_file(path: str):
    try:
        source = Path(path)
        if source.stat().st_size > MAX_SOURCE_BYTES:
            raise ContextVeilError("Configuration or evaluation input exceeds the size limit.")
        return json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, UnicodeError):
        raise ContextVeilError("Unable to read a valid JSON configuration file.") from None


def _passphrase(variable: str | None) -> str:
    if variable:
        value = os.environ.get(variable)
        if value is None:
            raise ContextVeilError("The requested passphrase environment variable is not set.")
        return value
    return getpass.getpass("Recovery passphrase (at least 16 characters): ")


def _bundle(destination: Path, files: dict[str, bytes]) -> None:
    """Publish a newly named bundle only after every member has been written."""
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise ContextVeilError("Output directory already exists; choose a new directory.")
    if not destination.parent.is_dir():
        raise ContextVeilError("Output parent directory does not exist.")
    stage = Path(tempfile.mkdtemp(prefix=".contextveil-", dir=destination.parent))
    try:
        for name, data in files.items():
            if Path(name).name != name:
                raise ContextVeilError("Invalid output member name.")
            with (stage / name).open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        if destination.exists() or destination.is_symlink():
            raise ContextVeilError("Output directory already exists; choose a new directory.")
        stage.rename(destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _add_detector_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="en_core_web_sm", help="Installed spaCy model package")
    parser.add_argument("--glossary", help="Local JSON list of explicit entities and canonical aliases")
    parser.add_argument("--rules-only", action="store_true", help="Use pattern detection; results require review by default")
    parser.add_argument("--neural-model-dir", help="Verified local GLiNER PII model bundle")
    parser.add_argument("--threshold", type=float, default=0.65, help="Neural confidence threshold (default: 0.65)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Local neural inference device")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="contextveil", description="Local text de-identification with explicit fidelity checks")
    commands = parser.add_subparsers(dest="command", required=True)
    redact = commands.add_parser("anonymize", help="De-identify a batch using one entity mapping")
    redact.add_argument("inputs", nargs="+")
    redact.add_argument("--output-dir", required=True, help="New directory; existing directories are never overwritten")
    redact.add_argument("--policy", help="Local JSON policy overrides")
    redact.add_argument("--allow-review", action="store_true", help="Write outputs explicitly marked review_required")
    redact.add_argument("--recoverable", action="store_true", help="Write encrypted recovery vaults")
    redact.add_argument("--passphrase-env", help="Read recovery passphrase from this environment variable")
    _add_detector_options(redact)
    recover = commands.add_parser("restore", help="Recover normalized text using a vault and passphrase")
    recover.add_argument("input")
    recover.add_argument("--vault", required=True)
    recover.add_argument("--report", help="Matching report; defaults to INPUT stem + .report.json")
    recover.add_argument("--output-dir", required=True)
    recover.add_argument("--passphrase-env")
    evaluate = commands.add_parser("evaluate", help="Measure detection against independently labeled JSONL")
    evaluate.add_argument("dataset")
    evaluate.add_argument("--min-recall", type=float)
    evaluate.add_argument("--min-precision", type=float)
    _add_detector_options(evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "restore":
            from .vault import MAX_VAULT_BYTES, restore
            source, vault = Path(args.input), Path(args.vault)
            if source.stat().st_size > MAX_VAULT_BYTES or vault.stat().st_size > MAX_VAULT_BYTES:
                raise ContextVeilError("Recovery input exceeds the size limit.")
            output = source.read_bytes().decode("utf-8")
            report = _json_file(args.report or str(source.with_suffix(".report.json")))
            if not isinstance(report, dict) or not isinstance(report.get("document_id"), str):
                raise ContextVeilError("A matching report with document_id is required for recovery.")
            original = restore(output, vault.read_bytes(), _passphrase(args.passphrase_env), document_id=report["document_id"])
            _bundle(Path(args.output_dir), {"restored" + source.suffix: original.encode("utf-8")})
            print(json.dumps({"status": "restored", "scope": "normalized text"}))
            return 0
        glossary = _json_file(args.glossary) if args.glossary else None
        detector = default_detector(model=args.model, glossary=glossary, rules_only=args.rules_only,
                                    neural_model_dir=args.neural_model_dir, threshold=args.threshold, device=args.device)
        if args.command == "evaluate":
            from .evaluation import evaluate
            for threshold in (args.min_recall, args.min_precision):
                if threshold is not None and not 0 <= threshold <= 1:
                    raise ContextVeilError("Evaluation thresholds must be between zero and one.")
            dataset = Path(args.dataset)
            if dataset.stat().st_size > MAX_SOURCE_BYTES:
                raise ContextVeilError("Evaluation dataset exceeds the size limit.")
            try:
                cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            except (ValueError, UnicodeError):
                raise ContextVeilError("Invalid evaluation JSONL.") from None
            report = evaluate(cases, detector)
            print(json.dumps(report, indent=2))
            # Metric names are intentionally exact-match entity precision and recall.
            metrics = report["overall"]["exact"]
            failed = any(target is not None and (metrics[key] is None or metrics[key] < target)
                         for key, target in (("recall", args.min_recall), ("precision", args.min_precision)))
            return 3 if failed else 0
        if args.passphrase_env and not args.recoverable:
            raise ContextVeilError("A passphrase is only used with --recoverable.")
        policy = Policy.from_dict(_json_file(args.policy)) if args.policy else Policy()
        documents = [load_document(path) for path in args.inputs]
        passphrase = _passphrase(args.passphrase_env) if args.recoverable else None
        results = Anonymizer(detector, policy).anonymize_many(documents, passphrase=passphrase)
        review = sum(result.report["status"] == "review_required" for result in results)
        if review and not args.allow_review:
            reasons = sorted({reason for result in results for reason in result.report["review_reasons"]})
            print(json.dumps({"status": "review_required", "documents": review, "reasons": reasons,
                              "action": "Review these limits; --allow-review writes explicitly marked review outputs."}), file=sys.stderr)
            return 2
        files: dict[str, bytes] = {}
        for index, result in enumerate(results, 1):
            stem = f"document-{index:03d}"
            files[stem + result.output_suffix] = result.text.encode("utf-8")
            files[stem + ".report.json"] = (json.dumps(result.report, indent=2) + "\n").encode("utf-8")
            if result.recovery is not None:
                files[stem + ".vault"] = result.recovery
        _bundle(Path(args.output_dir), files)
        print(json.dumps({"status": "review_required" if review else "checks_passed", "documents": len(results),
                          "entities_replaced": sum(result.report["entities_replaced"] for result in results)}))
        return 0
    except (ContextVeilError, ValidationError) as error:
        print(f"contextveil: {error}", file=sys.stderr)
        return 2
    except (OSError, UnicodeError):
        print("contextveil: Local input or output could not be accessed safely.", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("contextveil: Interrupted.", file=sys.stderr)
        return 130
