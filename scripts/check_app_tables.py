
from sqlalchemy import inspect, text

from app.db import engine


TABLES = ["customers", "orders", "returns", "payments", "tickets"]

inspector = inspect(engine)
existing_tables = set(inspector.get_table_names())

print("Existing application tables:")

with engine.connect() as connection:
    for table_name in TABLES:
        if table_name not in existing_tables:
            print(f"{table_name}: MISSING")
            continue

        count = connection.execute(
            text(f'SELECT COUNT(*) FROM "{table_name}"')
        ).scalar_one()

        print(f"{table_name}: {count} rows")
