
import json
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.db import engine


RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

ARTIFACT_DIR = Path("artifacts") / "intent_baseline"
MODEL_PATH = ARTIFACT_DIR / "intent_model.joblib"


def load_dataset() -> pd.DataFrame:
    query = text("""
        SELECT instruction, intent
        FROM bitext_messages
        WHERE instruction IS NOT NULL
          AND intent IS NOT NULL
        ORDER BY id
    """)

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    df = pd.DataFrame([dict(row) for row in rows])

    df["instruction"] = df["instruction"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()

    df = df[
        (df["instruction"] != "")
        & (df["intent"] != "")
    ].reset_index(drop=True)

    return df


def print_metrics(name: str, y_true, y_pred) -> dict:
    print(f"\n=== {name} ===")
    print(f"Accuracy:   {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro-F1:   {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Weighted-F1:{f1_score(y_true, y_pred, average='weighted'):.4f}")

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    print("\nPer-intent classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": report,
    }


def main():
    df = load_dataset()

    print("=== Dataset ===")
    print(f"Usable messages: {len(df):,}")
    print(f"Distinct intents: {df['intent'].nunique()}")
    print(f"Random seed: {RANDOM_STATE}")

    X = df["instruction"]
    y = df["intent"]

    # First, hold out 15% for testing.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Split the remaining 85% into training and validation.
    # 0.15 / 0.85 gives a validation share of 15% of the full dataset.
    validation_fraction_of_remaining = VALIDATION_SIZE / (1 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_fraction_of_remaining,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    print("\n=== Split sizes ===")
    print(f"Training:   {len(X_train):,} ({len(X_train) / len(df):.1%})")
    print(f"Validation: {len(X_val):,} ({len(X_val) / len(df):.1%})")
    print(f"Test:       {len(X_test):,} ({len(X_test) / len(df):.1%})")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=50_000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2_000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    print("\nTraining TF-IDF + Logistic Regression...")
    model.fit(X_train, y_train)

    # Use validation for development checks.
    val_predictions = model.predict(X_val)
    validation_metrics = print_metrics(
        "Validation results",
        y_val,
        val_predictions,
    )

    # This is the initial baseline with fixed settings; evaluate the test
    # split once to establish a held-out benchmark.
    test_predictions = model.predict(X_test)
    test_metrics = print_metrics(
        "Held-out test results",
        y_test,
        test_predictions,
    )

    print("\n=== Confusion matrix (test set) ===")
    labels = sorted(y.unique())
    matrix = confusion_matrix(y_test, test_predictions, labels=labels)
    print("Label order:", labels)
    print(matrix)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "target": "intent",
            "input_column": "instruction",
            "random_state": RANDOM_STATE,
            "labels": labels,
        },
        MODEL_PATH,
    )

    results = {
        "dataset": "bitext_messages",
        "rows_used": int(len(df)),
        "number_of_intents": int(y.nunique()),
        "split_sizes": {
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "random_state": RANDOM_STATE,
        "model": "TF-IDF + Logistic Regression",
        "validation": validation_metrics,
        "test": test_metrics,
    }

    metrics_path = ARTIFACT_DIR / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\n=== Saved artifacts ===")
    print(f"Model:   {MODEL_PATH}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()