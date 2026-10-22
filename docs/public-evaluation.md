# Complete public evaluation — ContextVeil 0.2

The neural profile improves PII recall and reduces Gretel classification loss. AG News still loses substantial accuracy, and its changed-output subset and macro-F1 do not improve over the baseline. These results do **not** establish lossless semantic preservation or unattended release readiness. Failed documents and review requirements are included; the configured release gate was not weakened.

This evaluation was run locally on Windows, Python 3.11, and an NVIDIA RTX 4090. All test documents were evaluated after freezing the detector, threshold, classifiers, code hashes and key dependency versions. No detector or threshold was changed after observing test scores. The complete aggregate evidence, per-type scores, audits, class supports and frozen protocol are in [public-evaluation.json](public-evaluation.json). Raw text, predictions with source offsets, checkpoints and transformed corpora remain outside the repository.

## Data and split boundaries

| Corpus | Training | Development | Complete test | Purpose |
| --- | ---: | ---: | ---: | --- |
| [Gretel English PII](https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1) | 50,000 | 5,000 | 5,000 | 42 PII categories; 47-class domain classification |
| [TAB](https://github.com/NorskRegnesentral/text-anonymization-benchmark) | 1,014 | 127 | 127 | Real English legal PII; 555 test annotator layers |
| [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) | 120,000 | None used | 7,600 | Four-class news classification |

Gretel is synthetic and Apache-2.0 licensed. TAB supplies real legal text and an MIT-licensed repository. AG News is public, but its dataset card marks the license as unknown and the source describes research use; the corpus is not redistributed here. Framework licensing does not grant rights to external datasets or models. Exact revisions, byte lengths and SHA-256 hashes are pinned in [assets.json](../benchmarks/assets.json).

Gretel provides entity strings rather than offsets. The adapter expands all exact word-bounded occurrences; when no bounded occurrence exists it uses an audited exact-substring fallback. No missing entities or unknown labels were silently discarded. Test audit: 21,378 gold spans; 39 annotations required fallback; 0 annotations had multiple matching occurrences. These reconstructed offsets and potentially incomplete synthetic annotations limit interpretation of boundary scores.

TAB layers are scored separately, without pooling contradictory annotations into one gold mask. DIRECT and QUASI are positive masking targets; NO_MASK is retained as an explicit negative. All categories are included, including DEM and MISC even where the detector is weak. There are 9 test NO_MASK spans overlapping positive gold; both inclusive and conflict-excluded rates are reported. TAB is target-relative anonymization: recognizing a public institution correctly can still be an unnecessary masking decision.

Exact UTF-8 text hashes found zero duplicates within or across each corpus's splits. Gretel UIDs and TAB document IDs were also disjoint. This does not test near duplicates, paraphrases or model pretraining contamination.

## Configuration and development selection

The baseline is the original validated regex detector plus spaCy `en_core_web_sm` 3.8.0. The new profile uses the pinned [GLiNER PII model](https://huggingface.co/urchade/gliner_multi_pii-v1), validated regex patterns, and development-derived context rules for birth dates and structured identifiers. The model card reports training on `urchade/synthetic-pii-ner-mistral-v1`; this evaluation does not fine-tune it on Gretel or TAB. Its complete training-data overlap cannot be independently ruled out.

The grid was 0.20, 0.35, 0.50, 0.65 and 0.80. Threshold **0.65** maximized the equally weighted mean Gretel/TAB development nonwhitespace-character F1 after context-rule fixes. The rules handle quoted and Markdown field labels and parenthesized document numbers; they do not memorize entity values. The 36-label prompt is evaluated as a whole, despite the checkpoint's 25-type training setting. Long documents use at most 256 model words and 512 encoded tokens per window, including the label prompt, with 48-word overlap. Oversized individual tokens fail explicitly. Overlapping automatic candidates are conservatively unioned; explicit or propagation conflicts still fail closed.

## PII detection on all test annotations

These are **source-text detector proposal scores**, not proof that released outputs contain no PII. Character precision/recall count nonwhitespace Unicode characters, irrespective of the predicted type; exact entity scores require matching start, end and canonical type. Full-mention recall requires every character in a gold span to be covered. The JSON includes all 42 Gretel source categories, not just those supported well.

| Test corpus | Detector | Character precision | Character recall | Character F1 | FP characters | FN characters | Full-mention recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRETEL | baseline | 73.24% | 47.46% | 57.59% | 47,417 | 143,671 | 37.21% |
| GRETEL | neural | 85.54% | 87.52% | 86.52% | 40,451 | 34,138 | 85.99% |
| TAB | baseline | 30.52% | 33.38% | 31.89% | 184,050 | 161,410 | 20.70% |
| TAB | neural | 45.06% | 75.18% | 56.35% | 222,098 | 60,129 | 71.67% |

| Test corpus | Detector | Exact precision | Exact recall | Exact F1 |
| --- | --- | ---: | ---: | ---: |
| GRETEL | baseline | 32.84% | 23.00% | 27.06% |
| GRETEL | neural | 71.89% | 69.92% | 70.89% |
| TAB | baseline | 12.41% | 12.26% | 12.34% |
| TAB | neural | 43.84% | 66.37% | 52.80% |

Gretel improves both the false-positive and false-negative character counts. TAB recall and precision improve, but **absolute false-positive characters increase**: the broader profile masks more text overall. A generic detector does not implement TAB's full target-relative policy.

| TAB entity protection / negative decisions | Baseline | Neural |
| --- | ---: | ---: |
| All mentions of a DIRECT entity covered | 7.06% | 90.70% |
| All mentions of a QUASI entity covered | 15.65% | 68.17% |
| NO_MASK overlap (inclusive) | 68.31% | 61.76% |
| NO_MASK overlap (excluding annotation conflicts) | 68.33% | 61.77% |

## Actual downstream accuracy

Two frozen TF-IDF (word unigrams/bigrams, min_df=2, max_features=100,000, sublinear TF) + logistic-regression (C=4, max_iter=1,000, seed=20261022) classifiers were trained on the complete original training splits. No classifier was trained or adapted on anonymized test text. The same model processes each original document and its actual `Anonymizer` output. Each unrelated document receives a separate linkage scope. A preparatory pass only batches residual inputs; a strict final replay uses real residual-detector results and produces the scored output. It does not replace residual checks with an always-pass stub.

The following table includes transformed outputs requiring review so that transformation quality can be assessed separately. Failed outputs remain in the denominator as incorrect abstentions. They are not empty-text guesses. This is **not automatic-release accuracy**. All 5,000 and 7,600 test rows are retained.

| Task | Detector | Original accuracy | Transformed accuracy | Delta (pp) | Paired 95% CI (pp) | Original macro-F1 | Transformed macro-F1 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| gretel | baseline | 76.82% | 72.68% | -4.14 | [-4.98, -3.32] | 76.79% | 72.82% |
| gretel | neural | 76.82% | 74.76% | -2.06 | [-2.90, -1.24] | 76.79% | 74.78% |
| ag_news | baseline | 92.32% | 87.57% | -4.75 | [-5.30, -4.21] | 92.30% | 87.84% |
| ag_news | neural | 92.32% | 87.75% | -4.57 | [-5.13, -3.97] | 92.30% | 87.76% |

| Actually changed documents | Detector | Count | Original accuracy | Transformed accuracy | Paired delta CI (pp) |
| --- | --- | ---: | ---: | ---: | --- |
| gretel | baseline | 4,599 | 76.99% | 72.65% | [-5.18, -3.50] |
| gretel | neural | 4,941 | 76.81% | 74.72% | [-2.89, -1.30] |
| ag_news | baseline | 7,259 | 92.49% | 88.18% | [-4.89, -3.76] |
| ag_news | neural | 7,530 | 92.34% | 87.77% | [-5.13, -4.01] |

Confidence intervals use 2,000 paired document bootstrap samples with seed 20261022. They quantify this test sample and this frozen model, not variability across model training seeds or other tasks. The pre-test acceptance rule was a lower 95% confidence bound of at least -2 percentage points for **both** the full corpus and changed-output subset.

| Task | Detector | Accuracy-margin gate |
| --- | --- | --- |
| gretel | baseline | FAIL |
| gretel | neural | FAIL |
| ag_news | baseline | FAIL |
| ag_news | neural | FAIL |

## Release coverage and failures

| Corpus | Detector | Checks passed | Review required | Failed | Automatic-release rate |
| --- | --- | ---: | ---: | ---: | ---: |
| gretel | baseline | 4,561 | 428 | 11 | 91.22% |
| gretel | neural | 1,080 | 3,920 | 0 | 21.60% |
| tab | baseline | 68 | 58 | 1 | 53.54% |
| tab | neural | 0 | 110 | 17 | 0.00% |
| ag_news | baseline | 7,092 | 460 | 48 | 93.32% |
| ag_news | neural | 1,472 | 6,125 | 3 | 19.37% |

Review includes new candidates from the genuine residual scan. Failures include unresolved entity-span conflicts. The neural profile's high review rate is a material operational limitation, not a hidden successful result. When both review-required and failed documents are counted as incorrect abstentions, end-to-end automatic-release task accuracy is:

| Task | Baseline | Neural |
| --- | ---: | ---: |
| gretel | 66.48% | 16.30% |
| ag_news | 82.11% | 16.28% |

## Reproduce

Run from the source checkout or extracted source distribution. Keep the workspace outside the repository. The commands below require a CUDA-capable setup for the benchmark predictor; the SDK also supports CPU inference. Install a compatible PyTorch CUDA build explicitly; the recorded run used torch 2.8.0+cu128. Package/model versions are in the JSON protocol. The model is downloaded only by the explicit setup command.

```sh
python -m pip install -e ".[nlp,neural,formats,vault,benchmark,dev]"
python -m spacy download en_core_web_sm
python -m benchmarks.download --dest ../contextveil-eval all
python -m benchmarks.audit_splits --data-root ../contextveil-eval/data --output ../contextveil-eval/split-audit.json
python -m benchmarks.train_utility --work-dir ../contextveil-eval
python -m benchmarks.news_utility --work-dir ../contextveil-eval
python -m benchmarks.predict_baseline --work-dir ../contextveil-eval --split dev
python -m benchmarks.predict_neural --work-dir ../contextveil-eval --split dev
python -m benchmarks.evaluate_public --work-dir ../contextveil-eval --split dev --datasets gretel,tab
python -m benchmarks.freeze --work-dir ../contextveil-eval
python -m benchmarks.predict_baseline --work-dir ../contextveil-eval --split test
python -m benchmarks.predict_neural --work-dir ../contextveil-eval --split test --threshold 0.65
python -m benchmarks.evaluate_public --work-dir ../contextveil-eval --split test --threshold 0.65 --datasets gretel,tab,ag_news
```

Use a fresh external workspace for a new protocol. The freeze command refuses to overwrite an existing protocol. The final test evaluator rejects changed detection/scoring code, recorded dependency versions, thresholds or frozen classifiers. Future tuning on these reported results must not be presented as evaluation on a still-untouched test set.

## Limits

The utility evidence covers two classification tasks, not question answering, retrieval, chronology, legal reasoning or arbitrary LLM workflows. AG News has no PII gold in this evaluation and is used only for task utility; its names need not be private in their original public-news context. The same-detector residual scan cannot prove absence of undetected PII and can overreact when masking removes context. The fast tokenizer lacks SentencePiece byte fallback; uncommon Unicode requires further testing. Format fidelity tests remain separate from these text-corpus benchmarks. Neither source-character coverage nor a classification score proves resistance to contextual reidentification.
