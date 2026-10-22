"""Evaluate cached public corpora with detector scores and actual pipeline outcomes.

Outputs, which may retain source text, are written only under --work-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from benchmarks.privacy_metrics import score_predictions
from benchmarks.public_data import load_gretel, load_tab
from benchmarks.run_pipeline import run_pipeline, select_neural_predictions
from benchmarks.utility import evaluate_classifier
from contextveil.detectors import CompositeDetector, RegexDetector, SpacyDetector
from contextveil.models import Entity
from contextveil.neural import PII_LABELS, GlinerDetector

REPOSITORY = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _documents(cases: list[dict], tab: bool) -> list[tuple[str, str]]:
    documents = {}
    for case in cases:
        uid = case["uid"].rsplit(":", 1)[0] if tab else case["uid"]
        if uid in documents and documents[uid] != case["text"]:
            raise ValueError("Annotation layers disagree on document text.")
        documents[uid] = case["text"]
    return list(documents.items())


def _load_cases(root: Path, dataset: str, split: str) -> tuple[list[dict], dict, str]:
    registry = json.loads(Path(__file__).with_name("assets.json").read_text(encoding="utf-8"))["data"]
    if dataset == "ag_news":
        if split != "test":
            raise ValueError("AG News is evaluated only on the test split.")
        import pyarrow.parquet as parquet

        path = root / "data/ag_news/test.parquet"
        pinned = registry["data/ag_news/test.parquet"]
        source_hash = _sha256(path)
        if path.stat().st_size != pinned["bytes"] or source_hash != pinned["sha256"]:
            raise ValueError("AG News test data failed its pinned integrity check.")
        table = parquet.read_table(path, columns=["text", "label"])
        if table.num_rows != 7600:
            raise ValueError("AG News test row count differs from the pinned manifest.")
        cases = [{"uid": f"ag_news:{index}", "text": row["text"], "label": row["label"]}
                 for index, row in enumerate(table.to_pylist())]
        if any(not isinstance(case["text"], str) or not case["text"]
               or type(case["label"]) is not int for case in cases):
            raise ValueError("AG News test rows have invalid text or labels.")
        return cases, {"documents": len(cases), "cases": len(cases)}, source_hash

    corpus_split = "validation" if dataset == "gretel" and split == "dev" else split
    suffix = ".parquet" if dataset == "gretel" else ".json"
    path = root / "data" / dataset / (corpus_split + suffix)
    pinned = registry[f"data/{dataset}/{corpus_split}{suffix}"]
    source_hash = _sha256(path)
    if path.stat().st_size != pinned["bytes"] or source_hash != pinned["sha256"]:
        raise ValueError("A public corpus failed its pinned integrity check.")
    cases, audit = (load_gretel if dataset == "gretel" else load_tab)(path)
    return cases, audit, source_hash


def _load_classifier(root: Path, dataset: str, split: str, source_hash: str):
    import joblib

    name = "ag_news_classifier" if dataset == "ag_news" else "gretel_domain_classifier"
    model_path = root / "models" / f"{name}.joblib"
    metadata = json.loads((root / "models" / f"{name}.metadata.json").read_text(encoding="utf-8"))
    if _sha256(model_path) != metadata["model_sha256"]:
        raise ValueError("The frozen classifier failed its metadata hash check.")
    if dataset == "ag_news":
        if (metadata["test_parquet_sha256"] != source_hash
                or metadata["source_label_names"] != ["World", "Sports", "Business", "Sci/Tech"]):
            raise ValueError("AG News classifier metadata does not match its test data or label mapping.")
    elif split == "dev" and metadata["validation_parquet_sha256"] != source_hash:
        raise ValueError("Gretel classifier metadata does not match its validation data.")
    model = joblib.load(model_path)
    if model.classes_.tolist() != metadata["classes"]:
        raise ValueError("The frozen classifier classes differ from its metadata.")
    return model, metadata


def _baseline_predictions(root: Path, split: str, dataset: str, documents: list[tuple[str, str]],
                          source_hash: str, detector) -> list[list[Entity]]:
    dest = root / f"baseline-{split}"
    corpus_split = "validation" if dataset == "gretel" and split == "dev" else split
    name = f"{dataset}_{corpus_split}"
    path = dest / f"{name}_predictions.json"
    detector_hash = _sha256(REPOSITORY / "src/contextveil/detectors.py")
    model_version = detector.detectors[1]._nlp.meta.get("version")
    if path.exists():
        cache = json.loads(path.read_text(encoding="utf-8"))
        if cache.get("kind") != "raw_regex_then_spacy_proposals" or cache.get("split") != name:
            raise ValueError("The baseline proposal cache has an unexpected format.")
        if dataset != "ag_news":
            summary = json.loads((dest / f"{name}_summary.json").read_text(encoding="utf-8"))
            hashes = summary["source_sha256"]
            if (hashes["data"] != source_hash or hashes["detectors"] != detector_hash
                    or summary.get("model_version") != model_version):
                raise ValueError("Baseline cache provenance differs from current detector, model, or data.")
        elif (cache.get("source_sha256") != source_hash
              or cache.get("detectors_sha256") != detector_hash
              or cache.get("model_version") != model_version):
            raise ValueError("AG News baseline cache does not match the requested data.")
        serialized = cache["predictions_by_uid"]
        if dataset == "tab":
            # The cache stores one prediction list per annotator; each document must agree.
            chosen = {}
            for uid, entries in serialized.items():
                doc_id = uid.rsplit(":", 1)[0]
                if doc_id in chosen and chosen[doc_id] != entries:
                    raise ValueError("TAB baseline annotations have inconsistent predictions.")
                chosen[doc_id] = entries
            serialized = chosen
        if set(serialized) != {uid for uid, _ in documents}:
            raise ValueError("Baseline cache does not cover exactly the requested documents.")
        return [[Entity(start, end, label, source=source) for start, end, label, source in serialized[uid]]
                for uid, _ in documents]

    if dataset != "ag_news" or split != "test":
        raise ValueError("Required baseline predictions are missing; run predict_baseline first.")
    snapshot = root / "baseline-src/contextveil/detectors.py"
    if snapshot.is_file() and _sha256(snapshot) != detector_hash:
        raise ValueError("Current baseline detector differs from the original snapshot.")
    regex, spacy = detector.detectors
    texts = [text for _, text in documents]
    predictions = [regex.detect(text) + [
        Entity(span.start_char, span.end_char, spacy._LABELS[span.label_], source=spacy.name)
        for span in doc.ents if span.label_ in spacy._LABELS]
        for text, doc in zip(texts, spacy._nlp.pipe(texts, batch_size=32))]
    if len(predictions) != len(documents):
        raise ValueError("The baseline detector returned incomplete AG News predictions.")
    _write_json(path, {"schema_version": 1, "split": name, "kind": "raw_regex_then_spacy_proposals",
                       "source_sha256": source_hash, "detectors_sha256": detector_hash,
                       "model_version": model_version,
                       "predictions_by_uid": {
                           uid: [[e.start, e.end, e.label, e.source] for e in entries]
                           for (uid, _), entries in zip(documents, predictions)}})
    return predictions


def _neural_predictions(root: Path, split: str, dataset: str, documents: list[tuple[str, str]],
                        source_hash: str, detector: GlinerDetector, threshold: float,
                        signature: dict) -> list[list[Entity]]:
    path = root / f"neural-{split}" / f"{dataset}_candidates.json"
    if path.exists():
        cache = json.loads(path.read_text(encoding="utf-8"))
        if (cache.get("source_sha256") != source_hash or cache.get("candidate_threshold", 1) > threshold
                or "candidate_inference" in cache and cache["candidate_inference"] != signature
                or split == "test" and cache.get("candidate_inference") != signature):
            raise ValueError("Neural candidate cache does not match the requested data or threshold.")
        candidates = {uid: [Entity(**entry) for entry in entries]
                      for uid, entries in cache["predictions"].items()}
        if set(candidates) != {uid for uid, _ in documents}:
            raise ValueError("Neural candidate cache does not cover exactly the requested documents.")
    else:
        candidates = {}
        started = perf_counter()
        for start in range(0, len(documents), 50):
            batch = documents[start:start + 50]
            outputs = detector.candidates_many([text for _, text in batch], threshold=0.1)
            if len(outputs) != len(batch):
                raise ValueError("Neural inference returned incomplete candidate batches.")
            candidates.update({uid: entries for (uid, _), entries in zip(batch, outputs)})
            print(json.dumps({"dataset": dataset, "candidate_documents": len(candidates),
                              "total": len(documents)}), flush=True)
        _write_json(path, {"source_sha256": source_hash, "candidate_threshold": 0.1,
                           "candidate_inference": signature,
                           "seconds": round(perf_counter() - started, 3),
                           "predictions": {uid: [asdict(entity) for entity in entries]
                                           for uid, entries in candidates.items()}})
    return select_neural_predictions([text for _, text in documents],
                                     [candidates[uid] for uid, _ in documents], threshold)


def _utility(model, metadata: dict, dataset: str, cases: list[dict],
             documents: list[tuple[str, str]], run: dict) -> dict:
    if dataset == "ag_news":
        label_names = metadata["source_label_names"]
        if any(case["label"] not in range(len(label_names)) for case in cases):
            raise ValueError("AG News labels fall outside the frozen classifier mapping.")
        labels = [label_names[case["label"]] for case in cases]
    else:
        labels = [case["domain"] for case in cases]
    if len(labels) != len(documents):
        raise ValueError("Utility labels do not align with unique documents.")
    originals = [text for _, text in documents]
    return {
        "successful_outputs_including_review": evaluate_classifier(
            model, originals, run["outputs"], labels, available=run["success"]),
        "automatic_release_only": evaluate_classifier(
            model, originals, run["outputs"], labels, available=run["auto_release"]),
    }


def _evaluate(root: Path, split: str, dataset: str, threshold: float, backend: str,
              cases: list[dict], audit: dict, source_hash: str, documents: list[tuple[str, str]],
              detector, classifier, signature: dict) -> None:
    started = perf_counter()
    if backend == "baseline":
        selected = _baseline_predictions(root, split, dataset, documents, source_hash, detector)
    else:
        selected = _neural_predictions(root, split, dataset, documents, source_hash, detector,
                                       threshold, signature)
    lookup = {uid: prediction for (uid, _), prediction in zip(documents, selected)}
    detector_scores = None if dataset == "ag_news" else score_predictions(
        cases, [lookup[case["uid"].rsplit(":", 1)[0] if dataset == "tab" else case["uid"]]
                for case in cases])
    run = run_pipeline([text for _, text in documents], selected, detector)
    counts = Counter(report["status"] for report in run["reports"])
    if (len(run["outputs"]) != len(documents) or sum(counts.values()) != len(documents)
            or sum(run["success"]) + counts["failed"] != len(documents)):
        raise ValueError("Pipeline statuses do not cover every requested document.")
    destination = root / f"pipeline-{split}"
    utility = _utility(*classifier, dataset, cases, documents, run) if classifier else None
    summary = {
        "schema_version": 1, "split": split, "dataset": dataset, "backend": backend,
        "threshold": threshold if backend == "neural" else None,
        "source_sha256": source_hash, "audit": audit, "annotation_cases": len(cases),
        "unique_documents": len(documents),
        "detector_scope": "Cached original-text proposals; not output redaction coverage.",
        "detector_scores": detector_scores,
        "pipeline": {
            "status_counts": dict(counts), "failures": run["errors"],
            "preparation_failures": run["preparation_failures"],
            "successful_outputs": sum(run["success"]),
            "automatic_releases": sum(run["auto_release"]),
            "residual_inputs": run["residual_inputs"], "seconds": run["seconds"],
            "per_document": [{"index": index, **report, "success": success,
                              "auto_release": release} for index, (report, success, release) in enumerate(
                                  zip(run["reports"], run["success"], run["auto_release"]), start=1)],
        },
        "utility": utility,
        "total_seconds": round(perf_counter() - started, 3),
    }
    _write_json(destination / f"{dataset}_{backend}_outputs.json", {
        "schema_version": 1, "dataset": dataset, "backend": backend,
        "documents": [{"uid": uid, "output": output, "success": success,
                       "auto_release": release} for (uid, _), output, success, release in
                      zip(documents, run["outputs"], run["success"], run["auto_release"])]})
    _write_json(destination / f"{dataset}_{backend}_report.json", summary)
    print(json.dumps({"dataset": dataset, "backend": backend, "documents": len(documents),
                      "statuses": dict(counts), "automatic_releases": sum(run["auto_release"]),
                      "seconds": summary["total_seconds"]}), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--datasets", default="gretel,tab",
                        help="Comma-separated selection from gretel,tab,ag_news")
    args = parser.parse_args(argv)
    root = args.work_dir.resolve()
    if root == REPOSITORY or REPOSITORY in root.parents:
        raise ValueError("Work directory must be outside the repository.")
    names = args.datasets.split(",")
    if (not names or len(set(names)) != len(names)
            or any(name not in {"gretel", "tab", "ag_news"} for name in names)
            or "ag_news" in names and args.split != "test"
            or not 0 < args.threshold < 1):
        raise ValueError("Invalid dataset selection, split, or threshold.")
    import torch

    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline = CompositeDetector([RegexDetector(), SpacyDetector()])
    neural = GlinerDetector(root / "models/gliner-pii", device=device, threshold=args.threshold)
    signature = {"model": neural.model_reference, "labels": PII_LABELS, "candidate_threshold": 0.1,
                 "window_words": 256, "overlap_words": 48, "encoder_tokens": 512,
                 "flat_ner": False, "multi_label": True}
    if args.split == "test":
        frozen = json.loads((root / "frozen-config.json").read_text(encoding="utf-8"))
        if frozen["threshold"] != args.threshold or frozen["candidate_inference"] != signature:
            raise ValueError("Test inference differs from the frozen development configuration.")
        for relative, digest in frozen["source_hashes"].items():
            path = (REPOSITORY / relative).resolve()
            if (not any(path.is_relative_to(REPOSITORY / folder)
                        for folder in ("src", "benchmarks"))
                    or path.suffix != ".py"
                    or _sha256(path) != digest):
                raise ValueError("Evaluation source changed after configuration freeze.")
        versions = frozen["environment_versions"]
        required_versions = {"spacy", "en-core-web-sm", "torch", "gliner", "transformers",
                             "scikit-learn", "pyarrow", "numpy"}
        if not required_versions <= set(versions):
            raise ValueError("The frozen environment is incomplete.")
        for package, version in versions.items():
            if importlib.metadata.version(package) != version:
                raise ValueError("A dependency version changed after configuration freeze.")
        classifier_hashes = frozen["classifier_hashes"]
        required = ({"gretel_domain_classifier.joblib"} if "gretel" in names else set()) | (
            {"ag_news_classifier.joblib"} if "ag_news" in names else set())
        if not required <= set(classifier_hashes):
            raise ValueError("The test classifier is missing from the frozen configuration.")
        for filename, digest in classifier_hashes.items():
            if Path(filename).name != filename or _sha256(root / "models" / filename) != digest:
                raise ValueError("A classifier changed after configuration freeze.")
    for dataset in names:
        cases, audit, source_hash = _load_cases(root, dataset, args.split)
        documents = _documents(cases, dataset == "tab")
        if dataset == "tab" and len(documents) != audit["documents"]:
            raise ValueError("TAB annotation layers do not match the document count.")
        classifier = _load_classifier(root, dataset, args.split, source_hash) if dataset != "tab" else None
        for backend, detector in (("baseline", baseline), ("neural", neural)):
            _evaluate(root, args.split, dataset, args.threshold, backend,
                      cases, audit, source_hash, documents, detector, classifier, signature)


if __name__ == "__main__":
    main()
