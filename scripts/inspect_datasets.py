
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Columns that are likely to identify records or link tables.
ID_COLUMNS = {
    "customer_id",
    "order_id",
    "order_item_id",
    "payment_sequential",
    "product_id",
    "seller_id",
    "review_id",
    "geolocation_zip_code_prefix",
}


def inspect_csv(path: Path) -> None:
    print("\n" + "=" * 90)
    print(f"FILE: {path.relative_to(PROJECT_ROOT)}")
    print(f"SIZE: {path.stat().st_size:,} bytes")

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"Could not read file: {type(exc).__name__}: {exc}")
        return

    print(f"ROWS: {len(df):,}")
    print(f"COLUMNS: {len(df.columns)}")
    print("\nCOLUMN NAMES:")
    print(", ".join(str(column) for column in df.columns))

    print("\nDATA TYPES:")
    print(df.dtypes.to_string())

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    print("\nMISSING VALUES:")
    if missing.empty:
        print("No missing values detected.")
    else:
        for column, count in missing.items():
            print(f"  {column}: {count:,} ({count / len(df):.1%})")

    likely_ids = [
        column
        for column in df.columns
        if column in ID_COLUMNS or column.endswith("_id")
    ]

    print("\nDUPLICATE IDENTIFIER CHECKS:")
    checked = False

    for column in likely_ids:
        # Composite identifiers need separate handling.
        if column in {"order_item_id", "payment_sequential"}:
            continue

        checked = True
        non_null = df[column].dropna()
        duplicate_count = int(non_null.duplicated().sum())
        unique_count = int(non_null.nunique())

        print(
            f"  {column}: unique={unique_count:,}, "
            f"duplicate non-null values={duplicate_count:,}"
        )

    if not checked:
        print("No likely identifier columns found.")

    print("\nSAMPLE ROWS (up to 3):")
    print(df.head(3).to_string(index=False, max_cols=12))

    # Show likely intent/category fields for support-message datasets.
    text_columns = [
        column
        for column in df.columns
        if any(term in column.lower() for term in [
            "intent", "category", "instruction", "question", "response"
        ])
    ]

    if text_columns:
        print("\nSUPPORT/CLASSIFICATION FIELD SUMMARY:")
        for column in text_columns:
            print(f"  {column}:")
            if df[column].dtype == "object":
                print(
                    df[column]
                    .value_counts(dropna=False)
                    .head(12)
                    .to_string()
                )


def main() -> None:
    csv_files = sorted(RAW_DIR.rglob("*.csv"))

    if not csv_files:
        print(f"No CSV files found under: {RAW_DIR}")
        return

    print(f"Found {len(csv_files)} CSV files.")
    for path in csv_files:
        inspect_csv(path)

    print("\n" + "=" * 90)
    print("Inspection complete. No database records were changed.")


if __name__ == "__main__":
    main()
