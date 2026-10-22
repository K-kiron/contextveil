"""Count exact text and ID overlap across pinned public benchmark splits.

Only aggregate counts leave this process. Text is hashed as raw UTF-8 with no
Unicode, whitespace, or case normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import pyarrow.parquet as parquet


def _digest(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("A split contains non-string text")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parquet(path: Path, id_column: str | None = None) -> tuple[Counter, Counter | None]:
    columns = ["text"] + ([id_column] if id_column else [])
    table = parquet.read_table(path, columns=columns)
    texts = Counter()
    ids = Counter() if id_column else None
    for row in table.to_pylist():
        texts[_digest(row["text"])] += 1
        if ids is not None:
            uid = row[id_column]
            if not isinstance(uid, str):
                raise ValueError("A split contains a non-string ID")
            ids[uid] += 1
    return texts, ids


def _tab(path: Path) -> tuple[Counter, Counter]:
    # TAB is a JSON list, so standard-library loading parses the container;
    # the audit accesses only doc_id and text, never annotation or task fields.
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("TAB split is not a document list")
    texts = Counter()
    ids = Counter()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("doc_id"), str):
            raise ValueError("TAB document has an invalid ID")
        texts[_digest(row.get("text"))] += 1
        ids[row["doc_id"]] += 1
    return texts, ids


def _within(counts: Counter) -> dict:
    rows = sum(counts.values())
    unique = len(counts)
    assert 0 <= unique <= rows
    return {"rows": rows, "unique": unique, "duplicate_rows": rows - unique,
            "duplicate_groups": sum(value > 1 for value in counts.values()),
            "rows_in_duplicate_groups": sum(value for value in counts.values() if value > 1)}


def _between(left: Counter, right: Counter) -> dict:
    common = left.keys() & right.keys()
    return {"shared_unique_values": len(common),
            "left_rows_with_shared_value": sum(left[value] for value in common),
            "right_rows_with_shared_value": sum(right[value] for value in common)}


def _dataset(splits: dict[str, tuple[Counter, Counter | None]]) -> dict:
    return {
        "splits": {name: {"text": _within(texts),
                          **({"id": _within(ids)} if ids is not None else {})}
                   for name, (texts, ids) in splits.items()},
        "cross_split": {
            f"{left_name}:{right_name}": {
                "text": _between(left_texts, right_texts),
                **({"id": _between(left_ids, right_ids)}
                   if left_ids is not None and right_ids is not None else {}),
            }
            for (left_name, (left_texts, left_ids)), (right_name, (right_texts, right_ids))
            in combinations(splits.items(), 2)
        },
    }


def audit(data_root: Path) -> dict:
    gretel = {split: _parquet(data_root / "gretel" / f"{split}.parquet", "uid")
              for split in ("train", "validation", "test")}
    tab = {split: _tab(data_root / "tab" / f"{split}.json")
           for split in ("train", "dev", "test")}
    news = {split: _parquet(data_root / "ag_news" / f"{split}.parquet")
            for split in ("train", "test")}
    return {"schema_version": 1, "text_hash": "SHA-256 of exact UTF-8 text; no normalization",
            "gretel": _dataset(gretel), "tab": _dataset(tab), "ag_news": _dataset(news)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output == repository or repository in output.parents:
        parser.error("Aggregate reports must be written outside the repository")
    report = audit(args.data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Aggregate split audit saved: {output}")


if __name__ == "__main__":
    main()
