# Contributing

Use Python 3.11 or newer. Install the development extras and the explicitly chosen
English model as described in the README. Run `python -m pytest -q`,
`python -m ruff check src tests examples benchmarks`, and `python -m build` before submitting
a change.

Use synthetic data in fixtures. Bug reports should describe the expected
behavior, a minimal synthetic reproduction, and the detector/model and policy
versions. Avoid reporting source text, filenames or vault secrets in diagnostics.

New detectors must return validated Unicode character offsets without rewriting
the input. Add both positive detection cases and negative utility cases. New file
adapters need malformed-input tests, output parsing checks and a documented
extraction boundary. Never silently claim complete extraction or universal PII
coverage. Include paired task evaluation when changing a domain utility policy.

Keep public datasets, checkpoints, classifier artifacts and transformed corpus
outputs outside the repository. Use the pinned downloader and separate development
and test splits described in `docs/public-evaluation.md`. Never tune on the published
test results and then describe those same results as an untouched held-out test.
