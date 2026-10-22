"""Train the frozen local domain classifier and check raw validation accuracy.

Run from the repository root: python -m benchmarks.train_utility --work-dir PATH
Outputs stay in the supplied external work directory. Never load an untrusted joblib file.
"""

from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "8"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from benchmarks.public_data import load_gretel  # noqa: E402
from benchmarks.utility import fit_classifier  # noqa: E402

REPOSITORY = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).with_name("assets.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True, help="Public evaluation directory outside repository")
    work_dir = parser.parse_args().work_dir.resolve()
    if work_dir == REPOSITORY or REPOSITORY in work_dir.parents:
        raise ValueError("Work directory must be outside the repository")
    assets = json.loads(ASSETS.read_text(encoding="utf-8"))["data"]
    train_path = work_dir / "data" / "gretel" / "train.parquet"
    validation_path = work_dir / "data" / "gretel" / "validation.parquet"
    revisions = set()
    for split, path in (("train", train_path), ("validation", validation_path)):
        source = assets[f"data/gretel/{split}.parquet"]
        if path.stat().st_size != source["bytes"] or _sha256(path) != source["sha256"]:
            raise ValueError(f"Gretel {split} does not match pinned assets")
        revisions.add(source["url"].split("/resolve/")[1].split("/")[0])
    if len(revisions) != 1:
        raise ValueError("Gretel train and validation assets have different revisions")
    revision = revisions.pop()

    train, _ = load_gretel(train_path)
    validation, _ = load_gretel(validation_path)
    if len(train) != 50_000 or len(validation) != 5_000:
        raise ValueError("Unexpected Gretel train or validation row count")
    train_ids = [case["uid"] for case in train]
    validation_ids = [case["uid"] for case in validation]
    if (len(set(train_ids)) != len(train_ids) or len(set(validation_ids)) != len(validation_ids)
            or set(train_ids) & set(validation_ids)):
        raise ValueError("Gretel train and validation IDs must be unique and disjoint")

    train_texts = [case["text"] for case in train]
    train_labels = [case["domain"] for case in train]
    validation_texts = [case["text"] for case in validation]
    validation_labels = [case["domain"] for case in validation]
    with threadpool_limits(limits=8):
        model = fit_classifier(train_texts, train_labels)
        predictions = model.predict(validation_texts)
    if len(predictions) != len(validation_labels):
        raise ValueError("Classifier returned the wrong number of validation predictions")

    document_type_counts = Counter(case["document_type"] for case in train)
    model_dir = work_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "gretel_domain_classifier.joblib"
    temporary_model = model_dir / "gretel_domain_classifier.joblib.tmp"
    joblib.dump(model, temporary_model, compress=3)
    metadata = {
        "schema_version": 1,
        "task": "domain_classification",
        "dataset": "gretelai/gretel-pii-masking-en-v1",
        "revision": revision,
        "train_split": "train",
        "train_rows": len(train),
        "train_parquet_sha256": assets["data/gretel/train.parquet"]["sha256"],
        "validation_split": "validation",
        "validation_rows": len(validation),
        "validation_parquet_sha256": assets["data/gretel/validation.parquet"]["sha256"],
        "classes": model.classes_.tolist(),
        "train_class_support": dict(sorted(Counter(train_labels).items())),
        "tfidf": {"ngram_range": [1, 2], "min_df": 2, "max_features": 100_000,
                  "sublinear_tf": True},
        "logistic_regression": {"C": 4, "max_iter": 1000, "random_state": 20261022},
        "sklearn_version": sklearn.__version__,
        "model_sha256": _sha256(temporary_model),
        "raw_validation_accuracy": float(accuracy_score(validation_labels, predictions)),
        "raw_validation_macro_f1": float(f1_score(validation_labels, predictions, average="macro")),
        "train_document_type_classes": len(document_type_counts),
        "train_document_type_min_support": min(document_type_counts.values()),
        "train_document_type_max_support": max(document_type_counts.values()),
    }
    temporary_metadata = model_dir / "gretel_domain_classifier.metadata.json.tmp"
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary_model.replace(model_path)
    temporary_metadata.replace(model_dir / "gretel_domain_classifier.metadata.json")
    print(json.dumps({"train_rows": metadata["train_rows"], "classes": len(metadata["classes"]),
                      "validation_rows": metadata["validation_rows"],
                      "raw_validation_accuracy": metadata["raw_validation_accuracy"],
                      "raw_validation_macro_f1": metadata["raw_validation_macro_f1"]}))


if __name__ == "__main__":
    main()
