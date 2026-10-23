
from sqlalchemy import text

from app.db import engine


def main():
    with engine.connect() as connection:
        print("=== Orders with payment/item differences over 0.02 ===")

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
                    o.order_id,
                    o.order_status,
                    ROUND(i.item_total, 2) AS item_total,
                    ROUND(p.payment_total, 2) AS payment_total,
                    ROUND(p.payment_total - i.item_total, 2) AS difference
                FROM olist_orders o
                JOIN item_totals i USING (order_id)
                JOIN payment_totals p USING (order_id)
                WHERE ABS(i.item_total - p.payment_total) > 0.02
                ORDER BY ABS(i.item_total - p.payment_total) DESC
                LIMIT 15
            """)
        ).all()

        if not rows:
            print("No mismatches found.")
        else:
            for row in rows:
                print(
                    f"order={row.order_id} | status={row.order_status} | "
                    f"items={row.item_total} | payments={row.payment_total} | "
                    f"difference={row.difference}"
                )

        print("\n=== Missing delivery timestamps by status ===")

        rows = connection.execute(
            text("""
                SELECT
                    order_status,
                    COUNT(*) AS total_orders,
                    COUNT(*) FILTER (
                        WHERE order_delivered_customer_date IS NULL
                    ) AS missing_delivery_timestamp
                FROM olist_orders
                GROUP BY order_status
                ORDER BY total_orders DESC
            """)
        ).all()

        for row in rows:
            print(
                f"{row.order_status}: {row.total_orders:,} orders, "
                f"{row.missing_delivery_timestamp:,} missing delivery timestamps"
            )

    print("\nInvestigation complete. No records were modified.")


if __name__ == "__main__":
    main()
