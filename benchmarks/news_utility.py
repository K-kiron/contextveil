"""Train a frozen AG News topic classifier on the full public training split.

Run from the repository root: python -m benchmarks.news_utility --work-dir PATH
The pinned train/test Parquet files must already be in PATH/data/ag_news. Test rows are never read.
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
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as parquet  # noqa: E402
import sklearn  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from benchmarks.utility import fit_classifier  # noqa: E402

REPOSITORY = Path(__file__).resolve().parents[1]
REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"
LABELS = ("World", "Sports", "Business", "Sci/Tech")
ASSETS = {
    "train": {"url": f"https://huggingface.co/datasets/fancyzhx/ag_news/resolve/{REVISION}/"
                     "data/train-00000-of-00001.parquet",
              "bytes": 18_585_438,
              "sha256": "fc508d6d9868594e3da960a8cfeb63ab5a4746598b93428c224397080c1f52ee",
              "rows": 120_000},
    "test": {"url": f"https://huggingface.co/datasets/fancyzhx/ag_news/resolve/{REVISION}/"
                    "data/test-00000-of-00001.parquet",
             "bytes": 1_234_829,
             "sha256": "71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb",
             "rows": 7_600},
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify(path: Path, split: str) -> None:
    expected = ASSETS[split]
    if path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"]:
        raise ValueError(f"AG News {split} does not match the pinned asset")
    if parquet.read_metadata(path).num_rows != expected["rows"]:
        raise ValueError(f"Unexpected AG News {split} row count")
    schema = parquet.read_schema(path)
    if schema.names != ["text", "label"] or schema.field("text").type != pa.string() or \
            schema.field("label").type != pa.int64():
        raise ValueError(f"Unexpected AG News {split} schema")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True, help="Public evaluation directory outside repository")
    work_dir = parser.parse_args().work_dir.resolve()
    if work_dir == REPOSITORY or REPOSITORY in work_dir.parents:
        raise ValueError("Work directory must be outside the repository")
    data_dir = work_dir / "data" / "ag_news"
    train_path = data_dir / "train.parquet"
    _verify(train_path, "train")
    _verify(data_dir / "test.parquet", "test")  # Metadata and hash only: preserve held-out test.

    train = parquet.read_table(train_path, columns=["text", "label"]).to_pydict()
    texts, source_labels = train["text"], train["label"]
    if any(not isinstance(text, str) or not text for text in texts) or \
            any(type(label) is not int or label not in range(4) for label in source_labels):
        raise ValueError("AG News train data contains invalid text or class labels")
    labels = [LABELS[label] for label in source_labels]
    if len(set(labels)) != 4:
        raise ValueError("AG News train data does not contain all four classes")
    with threadpool_limits(limits=8):
        model = fit_classifier(texts, labels)

    model_dir = work_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    temporary_model = model_dir / "ag_news_classifier.joblib.tmp"
    model_path = model_dir / "ag_news_classifier.joblib"
    joblib.dump(model, temporary_model, compress=3)
    metadata = {
        "schema_version": 1,
        "task": "ag_news_topic_classification",
        "dataset": "fancyzhx/ag_news",
        "revision": REVISION,
        "source_author": "Xiang Zhang",
        "license_metadata": "unknown; source corpus described for research and non-commercial activity",
        "train_rows": len(texts),
        "test_rows_verified_not_evaluated": ASSETS["test"]["rows"],
        "train_parquet_sha256": ASSETS["train"]["sha256"],
        "test_parquet_sha256": ASSETS["test"]["sha256"],
        "source_label_names": list(LABELS),
        "classes": model.classes_.tolist(),
        "train_class_support": dict(sorted(Counter(labels).items())),
        "tfidf": {"ngram_range": [1, 2], "min_df": 2, "max_features": 100_000,
                  "sublinear_tf": True},
        "logistic_regression": {"C": 4, "max_iter": 1000, "random_state": 20261022},
        "sklearn_version": sklearn.__version__,
        "model_sha256": _sha256(temporary_model),
        "test_evaluated": False,
    }
    temporary_metadata = model_dir / "ag_news_classifier.metadata.json.tmp"
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary_model.replace(model_path)
    temporary_metadata.replace(model_dir / "ag_news_classifier.metadata.json")
    (data_dir / "manifest.json").write_text(json.dumps(ASSETS, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train_rows": len(texts), "classes": len(model.classes_),
                      "test_rows_verified_not_evaluated": ASSETS["test"]["rows"]}))


if __name__ == "__main__":
    main()
