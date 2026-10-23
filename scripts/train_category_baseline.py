
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
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.db import engine


RANDOM_STATE = 42
ARTIFACT_DIR = Path("artifacts") / "category_baseline"


def main():
    query = text("""
        SELECT instruction, category
        FROM bitext_messages
        WHERE instruction IS NOT NULL
          AND category IS NOT NULL
        ORDER BY id
    """)

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    df = pd.DataFrame([dict(row) for row in rows])
    df["instruction"] = df["instruction"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[
        (df["instruction"] != "")
        & (df["category"] != "")
    ].reset_index(drop=True)

    X = df["instruction"]
    y = df["category"]

    print("=== Category dataset ===")
    print(f"Messages: {len(df):,}")
    print(f"Distinct categories: {y.nunique()}")
    print("\nCategory counts:")
    print(y.value_counts().to_string())

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    validation_fraction = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_fraction,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    print("\n=== Split sizes ===")
    print(f"Training:   {len(X_train):,}")
    print(f"Validation: {len(X_val):,}")
    print(f"Test:       {len(X_test):,}")

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

    print("\nTraining category classifier...")
    model.fit(X_train, y_train)

    def evaluate(name, X_eval, y_eval):
        predictions = model.predict(X_eval)

        print(f"\n=== {name} ===")
        print(f"Accuracy: {accuracy_score(y_eval, predictions):.4f}")
        print(
            "Macro-F1: "
            f"{f1_score(y_eval, predictions, average='macro'):.4f}"
        )
        print(
            "Weighted-F1: "
            f"{f1_score(y_eval, predictions, average='weighted'):.4f}"
        )
        print("\nPer-category report:")
        print(classification_report(y_eval, predictions, zero_division=0))

        return {
            "accuracy": float(accuracy_score(y_eval, predictions)),
            "macro_f1": float(
                f1_score(y_eval, predictions, average="macro")
            ),
            "weighted_f1": float(
                f1_score(y_eval, predictions, average="weighted")
            ),
            "classification_report": classification_report(
                y_eval,
                predictions,
                output_dict=True,
                zero_division=0,
            ),
        }

    validation_metrics = evaluate("Validation results", X_val, y_val)
    test_metrics = evaluate("Held-out test results", X_test, y_test)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACT_DIR / "category_model.joblib"
    metrics_path = ARTIFACT_DIR / "metrics.json"

    joblib.dump(
        {
            "model": model,
            "target": "category",
            "input_column": "instruction",
            "random_state": RANDOM_STATE,
            "labels": sorted(y.unique()),
        },
        model_path,
    )

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset": "bitext_messages",
                "rows_used": len(df),
                "number_of_categories": int(y.nunique()),
                "split_sizes": {
                    "train": len(X_train),
                    "validation": len(X_val),
                    "test": len(X_test),
                },
                "random_state": RANDOM_STATE,
                "model": "TF-IDF + Logistic Regression",
                "validation": validation_metrics,
                "test": test_metrics,
            },
            file,
            indent=2,
        )

    print("\n=== Saved artifacts ===")
    print(f"Model:   {model_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
