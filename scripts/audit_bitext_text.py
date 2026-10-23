
from pathlib import Path

import pandas as pd


CSV_PATH = (
    Path("data/raw/bitext")
    / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)


def normalize_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Source dataset not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required = {"instruction", "category", "intent", "response"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    text = normalize_text(df["instruction"])
    normalized = text.str.replace(r"[^\w\s]", "", regex=True)
    normalized = normalized.str.replace(r"\s+", " ", regex=True).str.strip()

    print("=== Source dataset audit ===")
    print(f"Rows: {len(df):,}")
    print(f"Distinct intents: {df['intent'].nunique()}")
    print(f"Distinct categories: {df['category'].nunique()}")

    print("\n=== Duplicate instructions ===")
    print(f"Exact duplicate rows: {int(df.duplicated().sum()):,}")
    print(f"Repeated instruction strings: {int(text.duplicated().sum()):,}")
    print(
        "Repeated instructions after punctuation normalization: "
        f"{int(normalized.duplicated().sum()):,}"
    )
    print(
        "Unique normalized instructions: "
        f"{normalized.nunique():,}"
    )

    # Check whether identical normalized instructions have conflicting labels.
    label_check = pd.DataFrame({
        "instruction": normalized,
        "intent": df["intent"].astype(str),
        "category": df["category"].astype(str),
    })

    label_counts = label_check.groupby("instruction").agg(
        intent_count=("intent", "nunique"),
        category_count=("category", "nunique"),
        rows=("intent", "size"),
    )

    conflicting = label_counts[
        (label_counts["intent_count"] > 1)
        | (label_counts["category_count"] > 1)
    ]

    print("\n=== Label consistency ===")
    print(
        "Normalized instructions with conflicting intent/category labels: "
        f"{len(conflicting):,}"
    )

    if not conflicting.empty:
        print("\nExamples of conflicting instructions:")
        examples = conflicting.head(10).index
        print(
            label_check[label_check["instruction"].isin(examples)]
            .to_string(index=False)
        )

    print("\n=== Template indicators ===")
    placeholder_pattern = r"\{\{.*?\}\}"
    placeholder_rows = text.str.contains(
        placeholder_pattern, regex=True, na=False
    )
    print(
        "Instructions containing {{...}} placeholders: "
        f"{int(placeholder_rows.sum()):,}"
    )

    print("\n=== Instruction length ===")
    lengths = text.str.split().str.len()
    print(lengths.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).to_string())

    print("\n=== Most frequent normalized instructions ===")
    print(
        normalized.value_counts()
        .head(15)
        .to_string()
    )

    print("\n=== Examples by category ===")
    for category in sorted(df["category"].dropna().unique()):
        examples = df.loc[
            df["category"] == category, "instruction"
        ].head(2)
        print(f"\n{category}:")
        for example in examples:
            print(f"  - {example}")


if __name__ == "__main__":
    main()
