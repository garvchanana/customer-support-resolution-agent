
from sqlalchemy import text

from app.db import engine


def main():
    with engine.connect() as connection:
        total = connection.execute(
            text("SELECT COUNT(*) FROM bitext_messages")
        ).scalar_one()

        distinct_categories = connection.execute(
            text("SELECT COUNT(DISTINCT category) FROM bitext_messages")
        ).scalar_one()

        distinct_intents = connection.execute(
            text("SELECT COUNT(DISTINCT intent) FROM bitext_messages")
        ).scalar_one()

        print("=== Bitext label overview ===")
        print(f"Total messages: {total:,}")
        print(f"Distinct categories: {distinct_categories}")
        print(f"Distinct intents: {distinct_intents}")

        print("\n=== Messages per category ===")
        rows = connection.execute(
            text("""
                SELECT category, COUNT(*) AS total
                FROM bitext_messages
                GROUP BY category
                ORDER BY total DESC
            """)
        ).all()

        for category, count in rows:
            print(f"{category}: {count:,}")

        print("\n=== Intent frequency summary ===")
        summary = connection.execute(
            text("""
                SELECT
                    MIN(intent_count) AS smallest_intent,
                    MAX(intent_count) AS largest_intent,
                    COUNT(*) FILTER (WHERE intent_count < 5) AS intents_under_5,
                    COUNT(*) FILTER (WHERE intent_count < 10) AS intents_under_10
                FROM (
                    SELECT intent, COUNT(*) AS intent_count
                    FROM bitext_messages
                    GROUP BY intent
                ) counts
            """)
        ).one()

        print(f"Smallest intent count: {summary.smallest_intent}")
        print(f"Largest intent count: {summary.largest_intent}")
        print(f"Intents with fewer than 5 examples: {summary.intents_under_5}")
        print(f"Intents with fewer than 10 examples: {summary.intents_under_10}")

        print("\n=== Least frequent intents (up to 15) ===")
        rows = connection.execute(
            text("""
                SELECT intent, COUNT(*) AS total
                FROM bitext_messages
                GROUP BY intent
                ORDER BY total ASC, intent ASC
                LIMIT 15
            """)
        ).all()

        for intent, count in rows:
            print(f"{intent}: {count:,}")


if __name__ == "__main__":
    main()
