"""Score regex + spaCy proposals on an explicitly supplied public data cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path,
                        help="Local evaluation root containing data/manifest.json")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--source", type=Path,
                        help="Optional source snapshot root containing the contextveil package")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def _documents(cases: list[dict], tab: bool) -> list[tuple[str, str]]:
    documents = {}
    for case in cases:
        uid = case["uid"]
        document_id = uid.rsplit(":", 1)[0] if tab else uid
        if document_id in documents and documents[document_id] != case["text"]:
            raise ValueError("A baseline document has inconsistent annotation text.")
        documents[document_id] = case["text"]
    return list(documents.items())


def _predict(documents: list[tuple[str, str]], spacy_detector) -> tuple[dict, dict]:
    from contextveil.detectors import RegexDetector
    from contextveil.models import Entity

    regex = RegexDetector()
    predictions = {}
    counts = Counter()
    started = perf_counter()
    for index, ((doc_id, text), doc) in enumerate(
        zip(documents, spacy_detector._nlp.pipe((text for _, text in documents), batch_size=32)), start=1
    ):
        regex_candidates = regex.detect(text)
        spacy_candidates = [
            Entity(span.start_char, span.end_char, spacy_detector._LABELS[span.label_], source="spacy")
            for span in doc.ents if span.label_ in spacy_detector._LABELS
        ]
        predictions[doc_id] = regex_candidates + spacy_candidates
        counts.update(regex=len(regex_candidates), spacy=len(spacy_candidates))
        if index % (10 if len(documents) < 500 else 250) == 0 or index == len(documents):
            print(f"Processed {index}/{len(documents)} unique documents in {perf_counter() - started:.1f}s", flush=True)
    return predictions, {"seconds": round(perf_counter() - started, 3), "candidate_counts": dict(counts)}


def _score(name: str, cases: list[dict], audit: dict, tab: bool,
           spacy_detector, source_hashes: dict, output: Path) -> dict:
    from benchmarks.privacy_metrics import score_predictions
    from contextveil.engine import resolve_entities
    from contextveil.models import ValidationError

    documents = _documents(cases, tab)
    if tab and len(documents) != audit["documents"]:
        raise ValueError("The TAB document count differs from the loader audit.")
    if any(len(text) > spacy_detector._nlp.max_length for _, text in documents):
        raise ValueError("A baseline document exceeds the installed spaCy model limit.")
    print(f"Starting {name}: {len(cases)} cases, {len(documents)} unique documents", flush=True)
    raw_by_document, prediction_run = _predict(documents, spacy_detector)
    raw_by_uid = {case["uid"]: raw_by_document[case["uid"].rsplit(":", 1)[0] if tab else case["uid"]]
                  for case in cases}
    if len(raw_by_uid) != len(cases):
        raise ValueError("Baseline annotation UIDs are not unique.")
    raw_predictions = [raw_by_uid[case["uid"]] for case in cases]
    raw_report = score_predictions(cases, raw_predictions)

    resolution = Counter()
    for doc_id, text in documents:
        try:
            selected = resolve_entities(text, raw_by_document[doc_id])
        except ValidationError as error:
            message = str(error)
            if "partially overlap" in message:
                resolution["partial_overlap_documents"] += 1
            elif "disagree" in message:
                resolution["equal_span_disagreement_documents"] += 1
            else:
                resolution["other_conflict_documents"] += 1
        else:
            resolution["accepted_documents"] += 1
            resolution["accepted_spans"] += len(selected)
            resolution["discarded_proposals"] += len(raw_by_document[doc_id]) - len(selected)

    prediction_artifact = {
        "schema_version": 1,
        "split": name,
        "kind": "raw_regex_then_spacy_proposals",
        "predictions_by_uid": {
            uid: [[entity.start, entity.end, entity.label, entity.source] for entity in entities]
            for uid, entities in raw_by_uid.items()
        },
    }
    _write_json(output / f"{name}_predictions.json", prediction_artifact)
    summary = {
        "schema_version": 1,
        "split": name,
        "status": "complete_proposal_only",
        "cases": len(cases),
        "unique_documents": len(documents),
        "source_sha256": source_hashes,
        "model": "en_core_web_sm",
        "model_version": spacy_detector._nlp.meta.get("version"),
        "spacy_batch_size": 32,
        "prediction_run": prediction_run,
        "resolver": dict(resolution),
        "raw_proposal_scores": raw_report,
    }
    _write_json(output / f"{name}_summary.json", summary)
    overall = raw_report["overall"]
    print(json.dumps({
        "split": name, "status": summary["status"], "cases": len(cases),
        "unique_documents": len(documents), "prediction_run": prediction_run,
        "resolver": dict(resolution), "raw_characters": overall["characters"],
        "raw_exact": overall["exact"], "raw_mentions": overall["mentions"],
        "raw_entity_protection": overall["entity_protection"], "raw_no_mask": overall["no_mask"],
    }, sort_keys=True), flush=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    work_dir = args.work_dir.resolve()
    data = work_dir / "data"
    output = work_dir / f"baseline-{args.split}"
    sys.path.insert(0, str(ROOT))
    if args.source is not None:
        source = args.source.resolve()
        if not (source / "contextveil" / "detectors.py").is_file():
            raise ValueError("The supplied source snapshot does not contain the contextveil package.")
        sys.path.insert(0, str(source))

    import contextveil
    from benchmarks.public_data import load_gretel, load_tab
    from contextveil.detectors import SpacyDetector

    package = Path(contextveil.__file__).resolve().parent
    if args.source is not None and package != source / "contextveil":
        raise ValueError("The supplied source snapshot was not loaded.")
    manifest = json.loads(Path(__file__).with_name("assets.json").read_text(encoding="utf-8"))["data"]
    gretel_split = "validation" if args.split == "dev" else "test"
    paths = {f"gretel_{gretel_split}": data / "gretel" / f"{gretel_split}.parquet",
             f"tab_{args.split}": data / "tab" / f"{args.split}.json"}
    hashes = {name: _sha256(path) for name, path in paths.items()}
    for name, path in paths.items():
        key = f"data/gretel/{gretel_split}.parquet" if name.startswith("gretel_") else f"data/tab/{args.split}.json"
        if hashes[name] != manifest[key]["sha256"] or path.stat().st_size != manifest[key]["bytes"]:
            raise ValueError("A cached benchmark file failed its pinned integrity check.")
    spacy_detector = SpacyDetector()
    source_hashes = {"detectors": _sha256(package / "detectors.py"),
                     "resolver": _sha256(package / "engine.py")}
    gretel_name = f"gretel_{gretel_split}"
    gretel, gretel_audit = load_gretel(paths[gretel_name])
    _score(gretel_name, gretel, gretel_audit, False, spacy_detector,
           {"data": hashes[gretel_name], **source_hashes}, output)
    tab_name = f"tab_{args.split}"
    tab_cases, tab_audit = load_tab(paths[tab_name])
    _score(tab_name, tab_cases, tab_audit, True, spacy_detector,
           {"data": hashes[tab_name], **source_hashes}, output)


if __name__ == "__main__":
    main()
