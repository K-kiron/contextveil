"""Cache complete public-corpus proposals and calibrate thresholds on development data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from dataclasses import asdict
from pathlib import Path

from benchmarks.download import _verify
from benchmarks.privacy_metrics import score_predictions
from benchmarks.public_data import load_gretel, load_tab
from contextveil.models import Entity
from contextveil.neural import PII_LABELS, GlinerDetector


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--dataset", choices=("gretel", "tab", "both"), default="both")
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    root = args.work_dir
    detector = GlinerDetector(root / "models/gliner-pii", device="cuda", threshold=args.threshold)
    signature = {"model": detector.model_reference, "labels": PII_LABELS, "candidate_threshold": 0.1,
                 "window_words": 256, "overlap_words": 48, "encoder_tokens": 512,
                 "flat_ner": False, "multi_label": True}
    registry = json.loads(Path(__file__).with_name("assets.json").read_text(encoding="utf-8"))
    if args.split == "test":
        frozen = json.loads((root / "frozen-config.json").read_text(encoding="utf-8"))
        if frozen["threshold"] != args.threshold or frozen["candidate_inference"] != signature:
            raise ValueError("Test inference differs from the frozen development configuration.")
        for relative, digest in frozen["source_hashes"].items():
            path = (Path(__file__).parents[1] / relative).resolve()
            repository = Path(__file__).resolve().parents[1]
            if (Path(relative).parts[0] not in {"src", "benchmarks"} or repository not in path.parents
                    or path.suffix != ".py" or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
                raise ValueError("Detector source changed after configuration freeze.")
        for name, version in frozen["environment_versions"].items():
            if importlib.metadata.version(name) != version:
                raise ValueError("Evaluation dependency changed after configuration freeze.")
    for name in ("gretel", "tab"):
        if args.dataset not in {name, "both"}:
            continue
        split = ("validation" if name == "gretel" else "dev") if args.split == "dev" else "test"
        source = root / "data" / name / (split + (".parquet" if name == "gretel" else ".json"))
        spec = registry["data"][source.relative_to(root).as_posix()]
        if not _verify(source, spec["sha256"], spec["bytes"]):
            raise ValueError("The dataset differs from the pinned public asset.")
        cases, audit = (load_gretel if name == "gretel" else load_tab)(source)
        by_id = {}
        for case in cases:
            uid = case["uid"].rsplit(":", 1)[0] if name == "tab" else case["uid"]
            if uid in by_id and by_id[uid] != case["text"]:
                raise ValueError("Annotation layers disagree on document text.")
            by_id[uid] = case["text"]
        documents = list(by_id.items())
        dest = root / f"neural-{args.split}"
        cache_path = dest / f"{name}_candidates.json"
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        started = time.monotonic()
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else None
        if cache is not None and cache.get("candidate_inference") == signature:
            if cache["source_sha256"] != source_hash:
                raise ValueError("Cached predictions are for different corpus bytes.")
            candidates = {uid: [Entity(**entry) for entry in entries] for uid, entries in cache["predictions"].items()}
            if set(candidates) != set(by_id):
                raise ValueError("Cached predictions do not cover the complete corpus.")
        else:
            candidates = {}
            for start in range(0, len(documents), 50):
                batch = documents[start:start + 50]
                outputs = detector.candidates_many([text for _, text in batch], threshold=0.1)
                candidates.update({uid: entries for (uid, _), entries in zip(batch, outputs)})
                print(json.dumps({"dataset": name, "split": split, "documents": len(candidates),
                                  "total": len(documents), "seconds": round(time.monotonic() - started, 1)}), flush=True)
            cache = {"source_sha256": source_hash, "candidate_threshold": 0.1, "candidate_inference": signature,
                     "seconds": time.monotonic() - started,
                     "predictions": {uid: [asdict(e) for e in entries] for uid, entries in candidates.items()}}
            write_json(cache_path, cache)
        reports = []
        for threshold in ((0.2, 0.35, 0.5, 0.65, 0.8) if args.split == "dev" else (args.threshold,)):
            detector.threshold = threshold
            selected = {uid: detector.select(text, candidates[uid]) for uid, text in documents}
            predictions = [selected[case["uid"].rsplit(":", 1)[0] if name == "tab" else case["uid"]] for case in cases]
            report = score_predictions(cases, predictions)
            reports.append({"threshold": threshold, "scores": report})
            print(json.dumps({"dataset": name, "threshold": threshold, "overall": report["overall"]}), flush=True)
        write_json(dest / f"{name}_scores.json", {"audit": audit, "reports": reports,
                                                 "source_sha256": source_hash, "inference_seconds": cache["seconds"]})


if __name__ == "__main__":
    main()
