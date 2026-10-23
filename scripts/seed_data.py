
from datetime import date

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Customer, Order, ReturnRecord, Payment


def seed_data():
    with SessionLocal.begin() as session:
        customers = [
            Customer(
                id="cus_001",
                name="Aarav Sharma",
                email="aarav@example.test",
            ),
            Customer(
                id="cus_002",
                name="Meera Patel",
                email="meera@example.test",
            ),
        ]

        for customer in customers:
            existing = session.get(Customer, customer.id)
            if existing is None:
                session.add(customer)

        session.flush()

        orders = [
            Order(
                id="ord_1001",
                customer_id="cus_001",
                product="Wireless Headphones",
                status="delivered",
                paid_amount_minor=499900,
                currency="INR",
                delivered_at=date(2026, 10, 1),
            ),
            Order(
                id="ord_1002",
                customer_id="cus_001",
                product="USB-C Charger",
                status="delivered",
                paid_amount_minor=199900,
                currency="INR",
                delivered_at=date(2026, 10, 10),
            ),
            Order(
                id="ord_2001",
                customer_id="cus_002",
                product="Bluetooth Speaker",
                status="shipped",
                paid_amount_minor=299900,
                currency="INR",
                delivered_at=None,
            ),
        ]

        for order in orders:
            if session.get(Order, order.id) is None:
                session.add(order)

        session.flush()

        returns = [
            ReturnRecord(
                id="ret_1001",
                order_id="ord_1001",
                status="received",
                inspection_status="passed",
            ),
            ReturnRecord(
                id="ret_1002",
                order_id="ord_1002",
                status="requested",
                inspection_status="pending",
            ),
        ]

        for record in returns:
            if session.get(ReturnRecord, record.id) is None:
                session.add(record)

        payments = [
            Payment(
                order_id="ord_1001",
                status="paid",
                refunded_amount_minor=0,
            ),
            Payment(
                order_id="ord_1002",
                status="paid",
                refunded_amount_minor=0,
            ),
            Payment(
                order_id="ord_2001",
                status="paid",
                refunded_amount_minor=0,
            ),
        ]

        for payment in payments:
            if session.get(Payment, payment.order_id) is None:
                session.add(payment)

    print("Synthetic seed data inserted or already present.")


if __name__ == "__main__":
    seed_data()
