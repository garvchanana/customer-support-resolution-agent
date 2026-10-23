
import csv
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from sqlalchemy import BigInteger, DateTime, Integer, Numeric, inspect, text
from sqlalchemy.dialects.postgresql import insert

from app.db import Base, engine
from app import models  # Register existing application models.
from app import olist_models  # Register historical-data models.


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

# Parent tables must be imported before tables referencing them.
IMPORTS = [
    ("product_category_translations.csv", "olist_product_category_translations"),
    ("olist_customers.csv", "olist_customers"),
    ("olist_products.csv", "olist_products"),
    ("olist_sellers.csv", "olist_sellers"),
    ("olist_orders.csv", "olist_orders"),
    ("olist_order_items.csv", "olist_order_items"),
    ("olist_order_payments.csv", "olist_order_payments"),
    ("olist_order_reviews.csv", "olist_order_reviews"),
    ("bitext_messages.csv", "bitext_messages"),
]

GENERATED_ID_TABLES = {
    "olist_order_reviews",
    "bitext_messages",
}



def convert_value(value, column):
    """Convert CSV text to a value appropriate for its SQLAlchemy column."""
    if value is None or value.strip() == "":
        return None

    column_type = column.type

    if isinstance(column_type, DateTime):
        return datetime.fromisoformat(value.strip())

    if isinstance(column_type, (Integer, BigInteger)):
        number = Decimal(value.strip())

        if number != number.to_integral_value():
            raise ValueError(
                f"Expected a whole number for column {column.name}, "
                f"got {value!r}"
            )

        return int(number)

    if isinstance(column_type, Numeric):
        return Decimal(value.strip())

    return value




def load_csv_records(path, table):
    """Read a CSV and convert its values using the target table schema."""
    records = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        expected_columns = {
            column.name for column in table.columns
            if column.name != "id" or table.name not in GENERATED_ID_TABLES
        }

        actual_columns = set(reader.fieldnames or [])
        missing_columns = expected_columns - actual_columns

        if missing_columns:
            raise ValueError(
                f"{path.name} is missing columns: {sorted(missing_columns)}"
            )

        for row_number, row in enumerate(reader, start=1):
            record = {}

            for column in table.columns:
                name = column.name

                if table.name in GENERATED_ID_TABLES and name == "id":
                    record["id"] = row_number
                    continue

                if name in row:
                    record[name] = convert_value(row[name], column)

            records.append(record)

    return records


def reset_generated_id_sequence(connection, table_name):
    """Move a generated-ID sequence beyond the IDs loaded from CSV."""
    statement = text(
        f"""
        SELECT setval(
            pg_get_serial_sequence('{table_name}', 'id'),
            (SELECT COALESCE(MAX(id), 1) FROM {table_name}),
            EXISTS (SELECT 1 FROM {table_name})
        )
        """
    )
    connection.execute(statement)


def main():
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    for _, table_name in IMPORTS:
        if table_name not in existing_tables:
            raise RuntimeError(
                f"Table {table_name} does not exist. Apply migrations first."
            )

    # All data loads commit together, or all roll back on failure.
    with engine.begin() as connection:
        for filename, table_name in IMPORTS:
            path = DATA_DIR / filename

            if not path.exists():
                raise FileNotFoundError(f"Missing prepared dataset: {path}")

            table = Base.metadata.tables[table_name]
            records = load_csv_records(path, table)

            if not records:
                raise ValueError(f"No records found in {filename}")

            inserted = 0

            for start in range(0, len(records), 500):
                batch = records[start : start + 500]

                statement = (
                    insert(table)
                    .values(batch)
                    .on_conflict_do_nothing()
                )

                result = connection.execute(
                    statement.returning(*table.primary_key.columns)
                )
                inserted += len(result.fetchall())

            if table_name in GENERATED_ID_TABLES:
                reset_generated_id_sequence(connection, table_name)

            total = connection.execute(
                text(f"SELECT COUNT(*) FROM {table_name}")
            ).scalar_one()

            print(
                f"{table_name}: CSV rows={len(records):,}, "
                f"new rows={inserted:,}, table rows={total:,}"
            )

    print("\nImport transaction committed successfully.")


if __name__ == "__main__":
    main()
