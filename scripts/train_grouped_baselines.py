
import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42

CSV_PATH = (
    Path("data/raw/bitext")
    / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)

ARTIFACT_ROOT = Path("artifacts") / "grouped_baselines"


def normalize_for_grouping(value):
    value = str(value).lower().strip()
    value = re.sub(r"[^\w\s]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def make_model():
    return Pipeline([
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


def group_split(X, groups, test_size, seed):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )
    train_indices, test_indices = next(
        splitter.split(X, groups=groups)
    )
    return train_indices, test_indices


def evaluate(model, X, y, split_name):
    predictions = model.predict(X)

    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(
            f1_score(y, predictions, average="macro")
        ),
        "weighted_f1": float(
            f1_score(y, predictions, average="weighted")
        ),
        "classification_report": classification_report(
            y,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    print(f"\n=== {split_name} ===")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro-F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted-F1: {metrics['weighted_f1']:.4f}")
    print(
        classification_report(
            y,
            predictions,
            zero_division=0,
        )
    )

    return metrics


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required = {"instruction", "intent", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.dropna(subset=["instruction", "intent", "category"]).copy()
    df["instruction"] = df["instruction"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()

    df = df[
        (df["instruction"] != "")
        & (df["intent"] != "")
        & (df["category"] != "")
    ].reset_index(drop=True)

    # Identical normalized instructions must not cross split boundaries.
    df["group_key"] = df["instruction"].map(normalize_for_grouping)

    conflicts = df.groupby("group_key").agg(
        intents=("intent", "nunique"),
        categories=("category", "nunique"),
    )
    conflicting = conflicts[
        (conflicts["intents"] > 1)
        | (conflicts["categories"] > 1)
    ]

    if not conflicting.empty:
        raise ValueError(
            "Conflicting labels found for normalized instructions. "
            "Resolve these before training."
        )

    X = df["instruction"]
    groups = df["group_key"]

    # Hold out approximately 15% of rows by normalized instruction group.
    train_val_idx, test_idx = group_split(
        X,
        groups,
        test_size=0.15,
        seed=RANDOM_STATE,
    )

    # From the remaining data, hold out enough groups for validation.
    # 0.15 / 0.85 targets validation at approximately 15% overall.
    relative_validation_size = 0.15 / 0.85

    train_relative_idx, val_relative_idx = group_split(
        X.iloc[train_val_idx],
        groups.iloc[train_val_idx],
        test_size=relative_validation_size,
        seed=RANDOM_STATE,
    )

    train_idx = train_val_idx[train_relative_idx]
    val_idx = train_val_idx[val_relative_idx]

    train_groups = set(groups.iloc[train_idx])
    val_groups = set(groups.iloc[val_idx])
    test_groups = set(groups.iloc[test_idx])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    print("=== Grouped dataset ===")
    print(f"Usable rows: {len(df):,}")
    print(f"Unique normalized instructions: {groups.nunique():,}")
    print(f"Intent classes: {df['intent'].nunique()}")
    print(f"Category classes: {df['category'].nunique()}")
    print("\n=== Actual split sizes ===")
    print(f"Training:   {len(train_idx):,}")
    print(f"Validation: {len(val_idx):,}")
    print(f"Test:       {len(test_idx):,}")
    print("\nGroup overlap check: passed")

    split_indices = {
        "train": train_idx,
        "validation": val_idx,
        "test": test_idx,
    }

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    for target in ("intent", "category"):
        print(f"\n{'#' * 72}")
        print(f"TARGET: {target.upper()}")
        print(f"{'#' * 72}")

        y = df[target]

        for split_name, indices in split_indices.items():
            missing_classes = set(y.unique()) - set(y.iloc[indices].unique())
            if missing_classes:
                print(
                    f"WARNING: {split_name} is missing classes: "
                    f"{sorted(missing_classes)}"
                )

        model = make_model()

        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
        )

        validation_metrics = evaluate(
            model,
            X.iloc[val_idx],
            y.iloc[val_idx],
            "Validation",
        )

        # Fixed model settings: evaluate the held-out test set once.
        test_metrics = evaluate(
            model,
            X.iloc[test_idx],
            y.iloc[test_idx],
            "Held-out test",
        )

        target_dir = ARTIFACT_ROOT / target
        target_dir.mkdir(parents=True, exist_ok=True)

        model_path = target_dir / "model.joblib"
        metrics_path = target_dir / "metrics.json"

        joblib.dump(
            {
                "model": model,
                "target": target,
                "input_column": "instruction",
                "random_state": RANDOM_STATE,
                "grouping": "normalized instruction",
                "labels": sorted(y.unique()),
            },
            model_path,
        )

        metrics = {
            "dataset": str(CSV_PATH),
            "rows_used": int(len(df)),
            "unique_normalized_instructions": int(groups.nunique()),
            "target": target,
            "model": "TF-IDF + Logistic Regression",
            "split_sizes": {
                name: int(len(indices))
                for name, indices in split_indices.items()
            },
            "random_state": RANDOM_STATE,
            "group_overlap": False,
            "validation": validation_metrics,
            "test": test_metrics,
        }

        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)

        print(f"\nSaved model:   {model_path}")
        print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
