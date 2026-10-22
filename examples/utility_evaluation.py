"""A local task accuracy example, not a real-world semantic benchmark."""

import json

from contextveil import Anonymizer, GlossaryDetector, Policy
from contextveil.evaluation import evaluate_utility

source = "Alice Morgan did not refund $125.50."
engine = Anonymizer(
    detector=GlossaryDetector([{"value": "Alice Morgan", "label": "PERSON"}]),
    policy=Policy(require_contextual=False),
)
sanitized = engine.anonymize_text(source).text


def refund_task(text):
    """Illustrative deterministic task; replace with your actual application task."""
    return {"refunded": "did not refund" not in text, "amount_125_50": "$125.50" in text}


correct = {"refunded": False, "amount_125_50": True}
preserved = evaluate_utility([
    {"original": source, "anonymized": sanitized, "expected": correct},
], refund_task)
damaged = evaluate_utility([
    {"original": source, "anonymized": sanitized.replace("did not refund", "refunded"), "expected": correct},
], refund_task)
print(json.dumps({"preserved": preserved, "negation_damaged": damaged}, indent=2))
