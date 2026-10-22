"""Behavior checks for the local paired classification benchmark."""

import pytest

from benchmarks.utility import evaluate_classifier, fit_classifier


class WordClassifier:
    classes_ = ["negative", "positive"]

    def predict(self, texts):
        return ["positive" if "good" in text else "negative" for text in texts]


def test_fit_classifier_uses_training_text_and_requested_settings():
    texts = ["good service", "good price", "excellent good", "good support",
             "bad service", "bad price", "terrible bad", "bad support"]
    labels = ["positive"] * 4 + ["negative"] * 4

    model = fit_classifier(texts, labels)

    vectorizer = model.named_steps["tfidfvectorizer"]
    classifier = model.named_steps["logisticregression"]
    assert vectorizer.ngram_range == (1, 2)
    assert vectorizer.min_df == 2
    assert vectorizer.max_features == 100_000
    assert vectorizer.sublinear_tf is True
    assert classifier.C == 4
    assert classifier.max_iter == 1000
    assert classifier.random_state == 20261022
    assert model.predict(texts).tolist() == labels


def test_changed_subgroup_exposes_a_loss_hidden_by_unchanged_inputs():
    originals = ["good"] * 20
    outputs = ["good"] * 19 + ["bad"]
    labels = ["positive"] * 20

    report = evaluate_classifier(WordClassifier(), originals, outputs, labels)

    assert report["overall"]["count"] == 20
    assert report["overall"]["accuracy_delta"] == pytest.approx(-0.05)
    assert report["changed"]["count"] == 1
    assert report["changed"]["accuracy_delta"] == -1
    assert report["changed"]["original_correct_retention"] == 0
    assert report["changed"]["accuracy_delta_ci_95"] == [-1.0, -1.0]
    assert report["unchanged_count"] == 19
    assert report["by_class"][1]["changed"]["support"] == 1
    assert report["overall"]["accuracy_delta_ci_95"] == evaluate_classifier(
        WordClassifier(), originals, outputs, labels)["overall"]["accuracy_delta_ci_95"]


def test_paired_gains_and_losses_do_not_cancel_in_transition_counts():
    report = evaluate_classifier(
        WordClassifier(), ["good", "bad"], ["bad", "good"], ["positive", "positive"])

    assert report["overall"]["accuracy_delta"] == 0
    assert report["overall"]["original_correct_to_wrong"] == 1
    assert report["overall"]["original_wrong_to_correct"] == 1
    assert report["overall"]["original_correct_retention"] == 0
    assert report["changed"]["count"] == 2


def test_empty_changed_group_has_no_invented_accuracy():
    report = evaluate_classifier(WordClassifier(), ["good"], ["good"], ["positive"])

    assert report["changed"]["count"] == 0
    assert report["changed"]["accuracy_delta"] is None
    assert report["changed"]["accuracy_delta_ci_95"] is None
    assert report["unchanged_count"] == 1


def test_unavailable_empty_output_abstains_even_if_classifier_would_guess_right():
    class RejectEmpty(WordClassifier):
        def predict(self, texts):
            assert "" not in texts
            return super().predict(texts)

    without_mask = evaluate_classifier(WordClassifier(), ["bad"], [""], ["negative"])
    with_mask = evaluate_classifier(RejectEmpty(), ["bad"], [""], ["negative"], available=[False])

    assert without_mask["overall"]["anonymized_accuracy"] == 1
    assert with_mask["overall"]["anonymized_accuracy"] == 0
    assert with_mask["overall"]["original_correct_to_wrong"] == 1
    assert with_mask["changed"]["count"] == 0
    assert with_mask["unchanged_count"] == 0
    assert with_mask["failed_output_count"] == with_mask["abstention_count"] == 1


def test_available_output_is_scored_while_missing_output_stays_in_denominator():
    report = evaluate_classifier(
        WordClassifier(), ["good", "bad"], ["bad", ""],
        ["positive", "negative"], available=[True, False])

    assert report["overall"]["count"] == 2
    assert report["overall"]["accuracy_delta"] == -1
    assert report["changed"]["count"] == 1
    assert report["changed"]["accuracy_delta"] == -1
    assert report["failed_output_count"] == 1


@pytest.mark.parametrize("available", [[True], [1, True], (True, True)])
def test_availability_requires_one_real_boolean_per_example(available):
    with pytest.raises(ValueError, match="Availability"):
        evaluate_classifier(
            WordClassifier(), ["good", "bad"], ["good", "bad"],
            ["positive", "negative"], available=available)
