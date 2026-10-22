"""Lock development-selected settings, code, assets and classifiers before test evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.work_dir.resolve()
    repository = Path(__file__).resolve().parents[1]
    if root == repository or repository in root.parents:
        raise ValueError("Work directory must be outside the repository.")
    path = root / "frozen-config.json"
    if path.exists():
        raise ValueError("A frozen protocol already exists; preserve it and use a new work directory.")
    scores = [json.loads((root / "neural-dev" / f"{name}_scores.json").read_text(encoding="utf-8"))
              for name in ("gretel", "tab")]
    grid = [report["threshold"] for report in scores[0]["reports"]]
    if any([r["threshold"] for r in item["reports"]] != grid for item in scores):
        raise ValueError("Development threshold grids differ.")
    average = [sum(item["reports"][i]["scores"]["overall"]["characters"]["f1"] for item in scores) / len(scores)
               for i in range(len(grid))]
    threshold = grid[max(range(len(grid)), key=lambda i: average[i])]
    candidate = json.loads((root / "neural-dev/gretel_candidates.json").read_text(encoding="utf-8"))
    packages = ("spacy", "en-core-web-sm", "torch", "gliner", "transformers", "scikit-learn", "pyarrow", "numpy")
    sources = list((repository / "src/contextveil").glob("*.py")) + list((repository / "benchmarks").glob("*.py"))
    classifiers = ("gretel_domain_classifier.joblib", "ag_news_classifier.joblib")
    result = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "Maximum equally weighted mean Gretel/TAB development nonwhitespace character F1; first grid tie wins.",
        "threshold_grid": grid, "development_mean_f1": average, "threshold": threshold,
        "candidate_inference": candidate["candidate_inference"],
        "source_hashes": {file.relative_to(repository).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
                          for file in sorted(sources)},
        "classifier_hashes": {name: hashlib.sha256((root / "models" / name).read_bytes()).hexdigest() for name in classifiers},
        "environment_versions": {name: importlib.metadata.version(name) for name in packages},
        "utility_acceptance": {"maximum_accuracy_loss": 0.02,
                               "criterion": "Paired bootstrap 95% lower bound >= -0.02 for overall AND changed outputs.",
                               "bootstrap_draws": 2000, "seed": 20261022},
        "test_sets": {"gretel": 5000, "tab_unique_documents": 127, "ag_news": 7600},
    }
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"threshold": threshold, "source_files": len(sources), "status": "frozen"}))


if __name__ == "__main__":
    main()
