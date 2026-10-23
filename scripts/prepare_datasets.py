
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_OLIST = ROOT / "data" / "raw" / "olist"
RAW_BITEXT = ROOT / "data" / "raw" / "bitext"
OUTPUT = ROOT / "data" / "processed"

ORDER_SAMPLE_SIZE = 10_000
BITEXT_SAMPLE_SIZE = 5_000
RANDOM_SEED = 42


def read_olist(filename: str) -> pd.DataFrame:
    path = RAW_OLIST / filename
    if not path.exists():
        raise FileNotFoundError(f"Required Olist file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def save_csv(df: pd.DataFrame, filename: str) -> None:
    path = OUTPUT / filename
    df.to_csv(path, index=False)
    print(f"{filename}: {len(df):,} rows")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Loading Olist source files...")

    orders = read_olist("olist_orders_dataset.csv")
    customers = read_olist("olist_customers_dataset.csv")
    items = read_olist("olist_order_items_dataset.csv")
    payments = read_olist("olist_order_payments_dataset.csv")
    reviews = read_olist("olist_order_reviews_dataset.csv")
    products = read_olist("olist_products_dataset.csv")
    sellers = read_olist("olist_sellers_dataset.csv")
    translations = read_olist("product_category_name_translation.csv")

    if orders["order_id"].isna().any():
        raise ValueError("Olist orders contain missing order IDs.")

    if orders["order_id"].duplicated().any():
        raise ValueError("Olist order IDs are not unique.")

    sample_size = min(ORDER_SAMPLE_SIZE, len(orders))

    selected_orders = orders.sample(
        n=sample_size,
        random_state=RANDOM_SEED,
    ).copy()

    order_ids = set(selected_orders["order_id"])

    selected_items = items[items["order_id"].isin(order_ids)].copy()
    selected_payments = payments[
        payments["order_id"].isin(order_ids)
    ].copy()
    selected_reviews = reviews[
        reviews["order_id"].isin(order_ids)
    ].copy()

    customer_ids = set(selected_orders["customer_id"])
    product_ids = set(selected_items["product_id"])
    seller_ids = set(selected_items["seller_id"])

    selected_customers = customers[
        customers["customer_id"].isin(customer_ids)
    ].copy()

    selected_products = products[
        products["product_id"].isin(product_ids)
    ].copy()

    selected_sellers = sellers[
        sellers["seller_id"].isin(seller_ids)
    ].copy()

    # Keep the complete translation reference table.
    selected_translations = translations.copy()

    # Validate order references before saving.
    checks = {
        "orders -> customers": (
            set(selected_orders["customer_id"])
            <= set(selected_customers["customer_id"])
        ),
        "items -> orders": (
            set(selected_items["order_id"]) <= order_ids
        ),
        "payments -> orders": (
            set(selected_payments["order_id"]) <= order_ids
        ),
        "reviews -> orders": (
            set(selected_reviews["order_id"]) <= order_ids
        ),
        "items -> products": (
            set(selected_items["product_id"])
            <= set(selected_products["product_id"])
        ),
        "items -> sellers": (
            set(selected_items["seller_id"])
            <= set(selected_sellers["seller_id"])
        ),
    }

    print("\nRelationship validation:")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")

    if not all(checks.values()):
        raise ValueError(
            "One or more relationships failed validation. "
            "No output files have been written."
        )

    print("\nSaving Olist subset:")
    save_csv(selected_orders, "olist_orders.csv")
    save_csv(selected_customers, "olist_customers.csv")
    save_csv(selected_items, "olist_order_items.csv")
    save_csv(selected_payments, "olist_order_payments.csv")
    save_csv(selected_reviews, "olist_order_reviews.csv")
    save_csv(selected_products, "olist_products.csv")
    save_csv(selected_sellers, "olist_sellers.csv")
    save_csv(
        selected_translations,
        "product_category_translations.csv",
    )

    bitext_files = sorted(RAW_BITEXT.glob("*.csv"))
    if not bitext_files:
        raise FileNotFoundError(
            f"No Bitext CSV found in {RAW_BITEXT}"
        )

    # Use the first CSV in the directory, which should be the
    # downloaded Bitext sample/training dataset.
    bitext_path = bitext_files[0]
    print(f"\nLoading Bitext: {bitext_path.name}")

    bitext = pd.read_csv(bitext_path, low_memory=False)

    required_columns = {"instruction", "category", "intent"}
    missing_columns = required_columns - set(bitext.columns)

    if missing_columns:
        raise ValueError(
            f"Bitext is missing required columns: {missing_columns}"
        )

    bitext_size = min(BITEXT_SAMPLE_SIZE, len(bitext))
    bitext_sample = bitext.sample(
        n=bitext_size,
        random_state=RANDOM_SEED,
    ).copy()

    print("\nSaving Bitext sample:")
    save_csv(bitext_sample, "bitext_messages.csv")

    manifest = {
        "selection_seed": RANDOM_SEED,
        "requested_olist_orders": ORDER_SAMPLE_SIZE,
        "actual_olist_orders": len(selected_orders),
        "requested_bitext_messages": BITEXT_SAMPLE_SIZE,
        "actual_bitext_messages": len(bitext_sample),
        "source_files": [
            str(path.relative_to(ROOT))
            for path in [
                RAW_OLIST / "olist_orders_dataset.csv",
                RAW_OLIST / "olist_customers_dataset.csv",
                RAW_OLIST / "olist_order_items_dataset.csv",
                RAW_OLIST / "olist_order_payments_dataset.csv",
                RAW_OLIST / "olist_order_reviews_dataset.csv",
                RAW_OLIST / "olist_products_dataset.csv",
                RAW_OLIST / "olist_sellers_dataset.csv",
                bitext_path,
            ]
        ],
        "note": (
            "Historical Olist records and Bitext examples are separate "
            "sources. Bitext messages are not linked to Olist orders. "
            "No refund approvals or inspection outcomes were inferred."
        ),
    }

    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"\nManifest saved: {manifest_path}")
    print("\nDataset preparation completed.")
    print("Original files and PostgreSQL were not modified.")


if __name__ == "__main__":
    main()
