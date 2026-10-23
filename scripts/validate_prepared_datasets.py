
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

FILES_AND_KEYS = {
    "olist_orders.csv": ["order_id"],
    "olist_customers.csv": ["customer_id"],
    "olist_products.csv": ["product_id"],
    "olist_sellers.csv": ["seller_id"],
    "olist_order_items.csv": ["order_id", "order_item_id"],
    "olist_order_payments.csv": ["order_id", "payment_sequential"],
    "product_category_translations.csv": ["product_category_name"],
    "bitext_messages.csv": [],
    "olist_order_reviews.csv": [],
}

failed = False

for filename, key_columns in FILES_AND_KEYS.items():
    path = DATA_DIR / filename

    if not path.exists():
        print(f"FAIL: Missing file: {filename}")
        failed = True
        continue

    df = pd.read_csv(path, low_memory=False)

    print(f"\n--- {filename} ---")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    missing_keys = [col for col in key_columns if col not in df.columns]
    if missing_keys:
        print(f"FAIL: Missing key columns: {missing_keys}")
        failed = True
        continue

    if key_columns:
        null_key_rows = df[key_columns].isna().any(axis=1).sum()
        duplicate_key_rows = df.duplicated(subset=key_columns).sum()

        print(f"Rows with missing key values: {null_key_rows:,}")
        print(f"Duplicate key rows: {duplicate_key_rows:,}")

        if null_key_rows or duplicate_key_rows:
            failed = True

    missing_cells = int(df.isna().sum().sum())
    print(f"Missing cells across all columns: {missing_cells:,}")


# Extra checks for reviews: do not assume review_id is unique.
reviews_path = DATA_DIR / "olist_order_reviews.csv"
orders_path = DATA_DIR / "olist_orders.csv"

if reviews_path.exists() and orders_path.exists():
    reviews = pd.read_csv(reviews_path, low_memory=False)
    orders = pd.read_csv(orders_path, usecols=["order_id"])

    required_columns = {"review_id", "order_id"}
    if not required_columns.issubset(reviews.columns):
        print("\nFAIL: Reviews file is missing required columns.")
        failed = True
    else:
        missing_order_ids = int(reviews["order_id"].isna().sum())
        unmatched_orders = int(
            (~reviews["order_id"].isin(orders["order_id"])).sum()
        )

        print("\n--- Review relationship checks ---")
        print(f"Review rows: {len(reviews):,}")
        print(f"Rows with missing order_id: {missing_order_ids:,}")
        print(f"Rows referencing orders outside the sample: {unmatched_orders:,}")

        if missing_order_ids or unmatched_orders:
            failed = True


if failed:
    print("\nVALIDATION FAILED. Review the issues above.")
    raise SystemExit(1)

print("\nVALIDATION PASSED.")
print("No missing files, missing key values, or duplicate declared keys.")
