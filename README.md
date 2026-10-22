# ContextVeil

Local text de-identification that preserves the words around sensitive entities,
keeps repeated entities linked, and makes verification limits explicit.

ContextVeil is an English-first Python SDK and CLI under the MIT license. It uses
local GLiNER or spaCy named-entity recognition, validated patterns, structured field
hints, and optional entity glossaries. Document contents never need to leave your machine.
This repository is an initial framework, not a certified anonymization product.

[中文说明](README.zh-CN.md) · [Assurance and limitations](docs/assurance.md) · [Security](SECURITY.md)

Version 0.2 includes complete held-out Gretel and TAB detection evaluations and
paired classification on Gretel and AG News. See the [public evaluation report](docs/public-evaluation.md)
for measured improvements, accuracy losses, failures, and reproducible commands.

## Why consistent replacement matters

```text
Before: Alice Morgan paid Robert Chen $125.50. Robert Chen did not refund it.
After:  [[PERSON_1_<run>]] paid [[PERSON_2_<run>]] $125.50. [[PERSON_2_<run>]] did not refund it.
```

The payment direction, amount, negation and repeated recipient remain intact.
ContextVeil does not ask a language model to rewrite the document. Entity tokens
are typed and unique within a linkage scope; their random run namespace avoids
deterministic linkage across fresh runs. An explicit glossary can link aliases
such as a full name and a nickname without guessing their identity.

The guarantee is deliberately narrow: **outside the selected entity spans, the
extracted segment text is unchanged**. This does not prove that entity selection
was correct, that every PII occurrence was found, or that every downstream task
retains its accuracy. Those are separate evaluations.

## Install and run

Python 3.11 or newer:

```sh
git clone https://github.com/K-kiron/contextveil.git
cd contextveil
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -e ".[nlp,formats,vault,benchmark,dev]"
python -m spacy download en_core_web_sm

contextveil anonymize examples/story.txt --glossary examples/glossary.json --output-dir output
```

Setup downloads dependencies and the explicitly requested model. Inference uses
the installed model locally; there is no runtime model download, telemetry, cloud
API, or silent fallback when the model is missing. A stronger installed spaCy
model can be selected with `--model` and evaluated on the same labeled data.

For the evaluated PII-specific neural backend, run these commands from the source checkout:

```sh
python -m pip install -e ".[neural,formats,vault]"
python -m benchmarks.download --dest ../contextveil-assets model
contextveil anonymize examples/story.txt --neural-model-dir ../contextveil-assets/models/gliner-pii --output-dir neural-output
```

The explicit setup download is about 1.16 GB. The runtime verifies pinned weight
and tokenizer hashes and uses local classes with weights-only loading. It supports
CPU inference; add `--device cuda` with a compatible PyTorch CUDA installation.
Long text is scanned in overlapping windows bounded by both word and encoder-token
limits. The evaluated threshold is `0.65`. Unlike the lightweight default backend,
this profile also selects dates and contextual identifiers. It may remove information
needed for chronology or other tasks. It does not resolve target-specific legal
masking policies, and many benchmark outputs require review.

The SDK exposes `GlinerDetector(model_dir, threshold=0.65, device="cpu")`; pass it
to `Anonymizer`. Neural dependencies and model assets remain optional.

The output directory must not exist. Outputs use neutral names such as
`document-001.txt` and `document-001.report.json`, so source filenames are not
copied. Inputs are not overwritten. A batch is published only after all its
files pass the configured checks and all output members have been written.

For a dependency-free pattern-only run:

```sh
python -m pip install -e .
contextveil anonymize examples/story.txt --rules-only --glossary examples/glossary.json --allow-review --output-dir review-output
```

`--rules-only` is explicitly limited. Without a contextual detector the default
policy marks results `review_required`. The CLI refuses to write those results
unless `--allow-review` is supplied; that option does not turn them into verified
anonymization. SDK callers receive the status and must enforce their own release
policy.

## File support

| Input | Output | Contract |
| --- | --- | --- |
| TXT, Markdown, log | Same text format | Unselected text remains exactly the same |
| JSON, JSONL | Same format | Values, order, and non-sensitive scalar types preserved; exact decimal numbers |
| CSV, TSV | Same format | Rows, columns, headers and cell content preserved except selected spans |
| YAML | YAML | Safe subset, exact numeric values; aliases/custom tags rejected; omitted comments trigger review |
| HTML, XML | Fresh TXT | Inline text assembled for detection; attributes/comments extracted separately |
| DOCX | Fresh TXT | Paragraph text, including text in supported document XML parts |
| Text-based PDF | Fresh TXT | Text layer extracted; encrypted or textless-page inputs rejected |

Structured formats are reserialized, so original whitespace, quoting, and style
are not retained. Sensitive keys/headers or numeric IDs converted to string
tokens are reported as schema changes and require review. HTML/XML/DOCX/PDF
conversion always requires review: reading order, omitted surfaces and layout
cannot be certified. No source binary container, hidden layer or attachment is
copied into the output. No OCR, images, audio or video are supported.

Text input is UTF-8. Inputs are bounded to 20 MiB; DOCX expansion and extracted
PDF text have additional bounds. See [the assurance document](docs/assurance.md)
for the exact scope and resource limits.

## Python SDK

```python
from contextveil import Anonymizer, UtilityCheck, default_detector, load_document

detector = default_detector(glossary=[
    {"value": "Alice Morgan", "label": "PERSON", "canonical": "person-a"},
    {"value": "A. Morgan", "label": "PERSON", "canonical": "person-a"},
])

# A domain-specific invariant supplements the built-in edit checks.
amount_check = UtilityCheck(
    "invoice-amount",
    lambda original, output: ("$125.50" in original) == ("$125.50" in output),
)
engine = Anonymizer(detector=detector, validators=[amount_check])
result = engine.anonymize(load_document("examples/story.txt"))
assert result.report["checks"]["outside_spans_unchanged"]
if result.report["status"] != "checks_passed":
    raise RuntimeError("Review the declared limitations before releasing this output")
print(result.text)  # An explicit application decision to display sanitized text.
```

Use `anonymize_many(documents)` for a linked batch: detection happens across the
batch before replacement, and exact detected named-entity/contact mentions can
propagate to other documents. IDs, dates and secret values are not propagated
into arbitrary matching numbers in prose. Reusing one `Anonymizer` retains its
token mapping in memory; use a new instance for an unrelated linkage scope.
It is a stateful, single-threaded object. Do not log its internals.

Custom detectors implement `name`, `contextual` and `detect(text) -> list[Entity]`.
Offsets use Python Unicode character indices in the original text. Invalid
offsets, conflicting canonical identities and partially crossing spans fail
closed. Detectors and utility callbacks are trusted local code, not a sandbox.

## Policies and aliases

```json
{
  "field_types": {
    "memberId": "ID",
    "deliveryAddress": "ADDRESS",
    "diagnosis": "SENSITIVE"
  },
  "require_contextual": true
}
```

Pass the file using `--policy`. It extends the default sensitive-field names;
snake_case, spaced, hyphenated and camelCase fields normalize to the same key.
For full replacement of the defaults, SDK callers can construct `Policy` directly.
Glossary files use the list format shown in [the example](examples/glossary.json).
Policy and glossary inputs may themselves be sensitive; store them accordingly.

Dates, ages, ordinary amounts and measurements are retained unless explicitly
selected by a field policy, glossary or custom detector. Recognized DOB fields
are replaced. This preserves useful facts but can leave indirect identifiers:
choose and evaluate a domain policy before release. Automatic date shifting,
gender inference and guessed coreference are not part of this version.

## Verify detection and downstream accuracy separately

```sh
contextveil evaluate examples/evaluation.jsonl
contextveil evaluate your-labeled-cases.jsonl --min-recall 0.99 --min-precision 0.95
python examples/utility_evaluation.py
```

`evaluate` accepts independent gold character spans. It reports exact entity
precision/recall/F1 and sensitive-character coverage, overall and by entity type.
Threshold failure exits with code 3. The included 13-case synthetic fixture is
for regression and integration; it is too small to validate real-world accuracy.
The baseline model has both false positives and false negatives on it. No sample
score is a universal privacy or semantic-accuracy claim.

The [public benchmark suite](docs/public-evaluation.md) uses all 5,000 Gretel test
documents, all 127 TAB test documents (555 annotator layers), and all 7,600 AG News
test documents for downstream classification. It freezes detection thresholds,
source hashes, dependency versions, and classifiers before test evaluation.
Failed outputs and review-required outputs are reported explicitly; automatic
release utility treats blocked documents as incorrect abstentions.

For downstream tasks, `contextveil.evaluation.evaluate_utility(cases, task)` runs
the **same task** on original and sanitized text and compares both answers with
the supplied ground truth. It returns accuracy before/after, accuracy difference,
and retention among originally correct cases. Plug in the task you actually use
(question answering, extraction, retrieval, classification); audit any callback
that would send original text to an external service. The shipped utility example
is a local extraction check, not an LLM benchmark.

Reports contain counts and check outcomes, never source excerpts, paths, field
names, mappings, gold text or task answers. Counts and entity types can themselves
be sensitive metadata. The residual detector runs again on retained text, but
using the same detector cannot measure its own false negatives.

## Optional encrypted recovery

By default, no recovery file or plaintext mapping is persisted. In-memory source
data is not securely erased, and semantic context may still identify subjects.
This default is not a mathematical claim of irreversible anonymization.

```sh
contextveil anonymize examples/story.txt --glossary examples/glossary.json --recoverable --output-dir sealed-output
contextveil restore sealed-output/document-001.txt --vault sealed-output/document-001.vault --output-dir restored-output
```

Both commands prompt for a passphrase of at least 16 characters. For automation,
`--passphrase-env VARIABLE_NAME` reads a passphrase from an existing environment
variable; no secret belongs in command-line arguments. Use a strong passphrase
and keep the vault separate from released documents.

The vault uses AES-256-GCM, a random salt and nonce, and scrypt key derivation.
It encrypts the full **normalized original text**, not merely a token dictionary,
so different alias spellings, numeric types and escaping restore correctly.
Keep the matching report with the output: its random `document_id` is also
authenticated, so identical output text cannot accidentally use another
document's vault. The CLI finds the matching `.report.json` automatically;
use `--report` if it was moved. SDK recovery requires the matching `document_id`.
Authentication or changed-output checks reject a mismatched recovery. Recovery
does not recreate the original DOCX/PDF container, layout or metadata.

## Develop

```sh
python -m pytest -q
python -m ruff check src tests examples benchmarks
python -m build
```

Tests cover span conflicts, exact decimals, structured schemas, alias linkage,
meaningful retained facts, actual installed English NER, malformed documents,
privacy-safe diagnostics, authenticated recovery and transactional CLI output.
CI installs the development, benchmark, format, vault and spaCy extras plus the
English model. Neural adapter unit tests use a stub model; real GLiNER checkpoint
and GPU evaluation are documented in the public evaluation report and are not
part of CI. A subset of integration tests skips missing optional packages, so a
minimal core installation is not a full test environment.

The framework reuses [spaCy](https://spacy.io/usage/models),
[pypdf](https://pypdf.readthedocs.io/en/stable/user/extract-text.html),
[PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation), and
[cryptography](https://cryptography.io/en/latest/hazmat/primitives/aead/).
The separation between candidate detection, transformation and explicit
evaluation also follows the limits documented by
[Presidio](https://github.com/data-privacy-stack/presidio/blob/main/docs/faq.md).
