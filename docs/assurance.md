# Assurance contract

## What is checked

The pipeline first selects spans, resolves contained matches, rejects crossing
matches, assigns consistent typed tokens, and concatenates untouched intervals
with replacements. It then checks every retained interval and scans the residual
text. Structured outputs are reparsed and compared with the transformed segments;
unselected scalar types must remain the same. Explicit utility callbacks can
reject output when domain invariants fail.

`checks_passed` describes successful execution of these checks for the selected
policy and extracted representation. It never means `PII-free`, anonymous under
all auxiliary information, legally compliant, or semantically equivalent for
all possible uses. A caller must decide whether the declared scope is sufficient.

`review_required` is produced for missing contextual detection, additional
residual candidates, ambiguous linkage, extraction limitations, schema changes,
or omitted YAML comments. The CLI writes nothing by default for these cases;
`--allow-review` permits a marked review artifact. Invalid parsing, overlap
conflicts, failed utility checks and recovery authentication failures cannot be
overridden with that flag.

## Representation fidelity

| Property | Guarantee and boundary |
| --- | --- |
| Unselected prose | Exact Unicode character sequence within extracted segments |
| Repeated entities | Same label and literal value or explicit canonical group receives one token within a scope |
| Different entities | Distinct identity keys receive distinct tokens; identical names belonging to different people require a custom detector/canonical policy |
| Negation, numbers, units, time order | Retained exactly when outside selected spans; detector false positives can still select them |
| Numeric precision | JSON/JSONL/YAML decimal values are parsed losslessly and serialized as numbers |
| Structure | Container shape/order and unselected scalar types retained; selected keys/headers/types can change and require review |
| Layout | Original structured formatting and document layout are not preserved |
| Semantics | Task-dependent; verified only by a supplied utility invariant or independent paired task evaluation |

Replacing an identity necessarily removes some information. A location may be
both identifying and relevant to a task; a date may be essential for chronology
and identifying in combination with other facts. The framework cannot infer the
right tradeoff for every domain. Customize field labels, detectors and gold data.

No approximate name merging, gender inference or automatic coreference is
performed. Exact propagation can over-redact homonyms; provide a contextual
detector and review ambiguous cases. Dates are not shifted automatically. Dates
and other quasi-identifiers outside explicit policy selection remain present.

The optional GLiNER profile explicitly selects dates and additional identifiers.
It unions crossing automatic model predictions conservatively before the general
pipeline; explicit glossary conflicts and propagation conflicts still fail closed.
Its published [paired task results](public-evaluation.md) measure classification,
not universal semantic equivalence. A high review rate or failed accuracy gate is
an evaluation result, not grounds to weaken release checks.

## File boundaries

UTF-8 inputs only, at most 20 MiB per file. DOCX supports at most 500 archive
members and 40 MiB declared expanded data; PDF supports at most 100 pages and
40 MiB extracted text. Encrypted PDFs and any page without extractable text are
rejected. The spaCy adapter rejects text beyond its model's input limit instead
of truncating it. Limits do not constitute a resource-isolation sandbox: parsers
and models process data in the calling process, and compressed inputs can be
expensive before extraction limits are reached.

JSON duplicate keys and nonfinite numbers are rejected. JSONL preserves record
ordering and blank records. CSV/TSV require consistent row widths and have no
spreadsheet formula execution. Outputs retain non-PII formulas as text; importing
untrusted CSV into a spreadsheet remains an application-level decision.

YAML uses a safe subset: aliases, merge keys, non-string mapping keys, unsupported
objects and nonfinite numbers are rejected. Unquoted date objects are rejected;
quote date values to make them explicit strings. Comments are omitted and trigger
review. Presentation syntax is normalized.

HTML and XML are never executed; XML DTDs/entities are rejected. Inline text is
assembled before detection. Attributes and comments may become separate text
segments. DOCX XML parts must be UTF-8. Text is read from supported Word XML paragraphs; tracked changes,
embedded objects, drawings and other surfaces are not claimed complete. PDF
extraction uses its text layer and can have incorrect reading order. All four
formats produce fresh plain text and always require review. Their fidelity
contract starts at the extracted segments, not at the entire source document.

## Evaluation

The span evaluator uses independently supplied, nonoverlapping gold spans. Exact
precision/recall measures type and boundary correctness. Sensitive-character
coverage measures how much labeled sensitive text lies inside selected spans;
it does not reward correctness of labels or penalize excess removal, so inspect
it together with precision. Zero-denominator metrics are null. Evaluation uses
resolved detector spans, not the separate structured field/propagation stage.

The paired utility evaluator runs the same task before and after transformation
against the same expected answer. Accuracy difference and retention of originally
correct answers are meaningful only for that task, sample and answer comparator.
The included example deliberately demonstrates that a negation change can lose
task accuracy despite high string similarity. No generic similarity score is
used as a substitute for privacy or utility evaluation.

## Recovery and local privacy

Default processing persists no recovery data. The engine holds originals and
token keys in memory while it operates; Python does not guarantee secure memory
erasure. Crash dumps, swap, backups, caller logging and debugger inspection are
outside this library's control. Source filenames are not copied into CLI output
names or reports. Reports omit raw text and mapping values, but their counts and
types still reveal metadata.

Recovery vaults encrypt the normalized original and a SHA-256 digest of the exact
sanitized output. AES-256-GCM uses a fresh 96-bit nonce, a fresh 128-bit salt,
and scrypt (N=32768, r=8, p=1, 32-byte key); the format version and salt are
authenticated together with a random per-document ID from the matching report.
Keep that report paired with its output; SDK recovery requires its `document_id`.
Wrong passphrases, swapped document vaults, changed output and ciphertext tampering fail.
The fixed derivation parameters are not accepted from an untrusted header.
Length limits and minimum passphrase length are not a substitute for a strong
passphrase or deployment-specific cryptographic review.

Only ciphertext is staged for vaults. Restoration intentionally writes sensitive
normalized plaintext into a new directory. Protect that destination. The CLI
stages sanitized outputs in a temporary sibling directory, flushes files, and
renames the bundle after success; ordinary exceptions clean up the owned staging directory.
An abrupt process termination can leave a staging directory behind.
No existing output directory is reused. This is an application-level transaction,
not a guarantee against power loss, hostile concurrent filesystem mutation or
OS-level observation.
