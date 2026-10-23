
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTENT_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "grouped_baselines"
    / "intent"
    / "model.joblib"
)

CATEGORY_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "grouped_baselines"
    / "category"
    / "model.joblib"
)


@lru_cache(maxsize=1)
def load_classifiers() -> tuple[Any, Any]:
    """Load both trained classifiers once per process."""
    for path in (INTENT_MODEL_PATH, CATEGORY_MODEL_PATH):
        if not path.is_file():
            raise FileNotFoundError(
                f"Trained classifier not found: {path}. "
                "Run scripts.train_grouped_baselines first."
            )

    intent_bundle = joblib.load(INTENT_MODEL_PATH)
    category_bundle = joblib.load(CATEGORY_MODEL_PATH)

    return intent_bundle["model"], category_bundle["model"]


def _top_predictions(model, message: str, limit: int = 3) -> list[dict]:
    probabilities = model.predict_proba([message])[0]
    classes = model.named_steps["classifier"].classes_

    order = np.argsort(probabilities)[::-1][:limit]

    return [
        {
            "label": str(classes[index]),
            "probability": round(float(probabilities[index]), 6),
        }
        for index in order
    ]


def classify_support_message(message: str) -> dict:
    """
    Return model suggestions only.

    This function does not determine whether a request is in scope,
    authorize an action, access customer data, or approve a refund.
    """
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")

    intent_model, category_model = load_classifiers()
    cleaned_message = message.strip()

    intent_predictions = _top_predictions(intent_model, cleaned_message)
    category_predictions = _top_predictions(category_model, cleaned_message)

    intent_top = intent_predictions[0]
    category_top = category_predictions[0]

    intent_margin = (
        intent_top["probability"] - intent_predictions[1]["probability"]
    )
    category_margin = (
        category_top["probability"] - category_predictions[1]["probability"]
    )

    return {
        "message": cleaned_message,
        "intent_suggestion": intent_top["label"],
        "intent_probability": intent_top["probability"],
        "intent_margin": round(intent_margin, 6),
        "intent_alternatives": intent_predictions,
        "category_suggestion": category_top["label"],
        "category_probability": category_top["probability"],
        "category_margin": round(category_margin, 6),
        "category_alternatives": category_predictions,
        "routing_status": "suggestion_only",
        "review_required": True,
    }
