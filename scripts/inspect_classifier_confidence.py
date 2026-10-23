
from pathlib import Path

import joblib
import numpy as np


INTENT_MODEL_PATH = (
    Path("artifacts")
    / "intent_baseline"
    / "intent_model.joblib"
)

CATEGORY_MODEL_PATH = (
    Path("artifacts")
    / "category_baseline"
    / "category_model.joblib"
)


TEST_MESSAGES = [
    (
        "clear_refund",
        "I want a refund for my order.",
    ),
    (
        "clear_order_tracking",
        "Where is my order? It has not arrived yet.",
    ),
    (
        "clear_shipping",
        "I need to change my shipping address.",
    ),
    (
        "clear_account",
        "Please help me delete my account.",
    ),
    (
        "clear_payment",
        "My payment failed but the money was deducted.",
    ),
    (
        "ambiguous_refund",
        "I have a problem with my order and want my money back.",
    ),
    (
        "multiple_intents",
        "Track my package and change the delivery address.",
    ),
    (
        "vague_request",
        "Can you help me with something?",
    ),
    (
        "greeting",
        "Hi, hello!",
    ),
    (
        "unrelated_weather",
        "What will the weather be like tomorrow?",
    ),
    (
        "unrelated_trivia",
        "What is the capital of France?",
    ),
    (
        "unrelated_programming",
        "Help me debug a Python function.",
    ),
]


def top_predictions(model, message, limit=3):
    probabilities = model.predict_proba([message])[0]
    classes = model.named_steps["classifier"].classes_

    indices = np.argsort(probabilities)[::-1][:limit]

    return [
        (str(classes[index]), float(probabilities[index]))
        for index in indices
    ]


def print_predictions(label, model, message):
    predictions = top_predictions(model, message)

    top_label, top_probability = predictions[0]
    second_probability = (
        predictions[1][1] if len(predictions) > 1 else 0.0
    )
    margin = top_probability - second_probability

    print(f"  {label}:")
    print(f"    Top prediction: {top_label}")
    print(f"    Top probability: {top_probability:.4f}")
    print(f"    Top-two margin: {margin:.4f}")
    print("    Top three:")
    for name, probability in predictions:
        print(f"      {name}: {probability:.4f}")


def main():
    if not INTENT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Intent model not found: {INTENT_MODEL_PATH}"
        )

    if not CATEGORY_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Category model not found: {CATEGORY_MODEL_PATH}"
        )

    intent_bundle = joblib.load(INTENT_MODEL_PATH)
    category_bundle = joblib.load(CATEGORY_MODEL_PATH)

    intent_model = intent_bundle["model"]
    category_model = category_bundle["model"]

    print("=== Classifier confidence inspection ===")
    print("These are diagnostic examples, not a formal evaluation.")
    print("Probabilities are not guaranteed to be calibrated.")

    for case_id, message in TEST_MESSAGES:
        print(f"\n{'=' * 72}")
        print(f"Case: {case_id}")
        print(f"Message: {message}")

        print_predictions("Intent", intent_model, message)
        print_predictions("Category", category_model, message)


if __name__ == "__main__":
    main()
