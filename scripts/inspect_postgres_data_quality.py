
from sqlalchemy import text

from app.db import engine


CHECKS = {
    "Orders without a matching customer": """
        SELECT COUNT(*)
        FROM olist_orders o
        LEFT JOIN olist_customers c ON c.customer_id = o.customer_id
        WHERE c.customer_id IS NULL
    """,
    "Order items without a matching order": """
        SELECT COUNT(*)
        FROM olist_order_items i
        LEFT JOIN olist_orders o ON o.order_id = i.order_id
        WHERE o.order_id IS NULL
    """,
    "Order items without a matching product": """
        SELECT COUNT(*)
        FROM olist_order_items i
        LEFT JOIN olist_products p ON p.product_id = i.product_id
        WHERE p.product_id IS NULL
    """,
    "Order items without a matching seller": """
        SELECT COUNT(*)
        FROM olist_order_items i
        LEFT JOIN olist_sellers s ON s.seller_id = i.seller_id
        WHERE s.seller_id IS NULL
    """,
    "Payments without a matching order": """
        SELECT COUNT(*)
        FROM olist_order_payments p
        LEFT JOIN olist_orders o ON o.order_id = p.order_id
        WHERE o.order_id IS NULL
    """,
    "Reviews without a matching order": """
        SELECT COUNT(*)
        FROM olist_order_reviews r
        LEFT JOIN olist_orders o ON o.order_id = r.order_id
        WHERE o.order_id IS NULL
    """,
    "Orders with missing delivery timestamp": """
        SELECT COUNT(*)
        FROM olist_orders
        WHERE order_delivered_customer_date IS NULL
    """,
    "Orders with missing purchase timestamp": """
        SELECT COUNT(*)
        FROM olist_orders
        WHERE order_purchase_timestamp IS NULL
    """,
    "Order items with invalid negative amounts": """
        SELECT COUNT(*)
        FROM olist_order_items
        WHERE price < 0 OR freight_value < 0
    """,
    "Payments with invalid negative amounts": """
        SELECT COUNT(*)
        FROM olist_order_payments
        WHERE payment_value < 0
    """,
}


def main():
    with engine.connect() as connection:
        print("=== Relationship and value checks ===")

        for label, query in CHECKS.items():
            result = connection.execute(text(query)).scalar_one()
            print(f"{label}: {result:,}")

        print("\n=== Order status distribution ===")
        rows = connection.execute(
            text("""
                SELECT order_status, COUNT(*) AS total
                FROM olist_orders
                GROUP BY order_status
                ORDER BY total DESC
            """)
        ).all()

        for status, total in rows:
            print(f"{status}: {total:,}")

        print("\n=== Payment totals versus item totals ===")
        rows = connection.execute(
            text("""
                WITH item_totals AS (
                    SELECT
                        order_id,
                        SUM(price + freight_value) AS item_total
                    FROM olist_order_items
                    GROUP BY order_id
                ),
                payment_totals AS (
                    SELECT
                        order_id,
                        SUM(payment_value) AS payment_total
                    FROM olist_order_payments
                    GROUP BY order_id
                )
                SELECT
                    COUNT(*) AS compared_orders,
                    COUNT(*) FILTER (
                        WHERE ABS(i.item_total - p.payment_total) > 0.02
                    ) AS orders_with_difference,
                    ROUND(
                        AVG(ABS(i.item_total - p.payment_total))::numeric,
                        2
                    ) AS average_absolute_difference
                FROM item_totals i
                JOIN payment_totals p USING (order_id)
            """)
        ).one()

        print(f"Orders compared: {rows.compared_orders:,}")
        print(f"Orders with a difference over 0.02: {rows.orders_with_difference:,}")
        print(f"Average absolute difference: {rows.average_absolute_difference}")

        print("\n=== Most common Bitext intents ===")
        rows = connection.execute(
            text("""
                SELECT category, intent, COUNT(*) AS total
                FROM bitext_messages
                GROUP BY category, intent
                ORDER BY total DESC
                LIMIT 10
            """)
        ).all()

        for category, intent, total in rows:
            print(f"{category} / {intent}: {total:,}")

        print("\nInspection complete.")
        print(
            "Note: delivery-date gaps and payment/item differences are "
            "diagnostic findings, not automatically data errors."
        )


if __name__ == "__main__":
    main()
