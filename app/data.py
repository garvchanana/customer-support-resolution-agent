
from datetime import date

CUSTOMERS = {
    "cus_001": {
        "id": "cus_001",
        "name": "Aarav Sharma",
        "email": "aarav@example.test",
    },
    "cus_002": {
        "id": "cus_002",
        "name": "Meera Patel",
        "email": "meera@example.test",
    },
}

ORDERS = {
    "ord_1001": {
        "id": "ord_1001",
        "customer_id": "cus_001",
        "product": "Wireless Headphones",
        "status": "delivered",
        "paid_amount_minor": 499900,
        "currency": "INR",
        "delivered_at": date(2026, 10, 1),
    },
    "ord_1002": {
        "id": "ord_1002",
        "customer_id": "cus_001",
        "product": "USB-C Charger",
        "status": "delivered",
        "paid_amount_minor": 199900,
        "currency": "INR",
        "delivered_at": date(2026, 10, 10),
    },
    "ord_2001": {
        "id": "ord_2001",
        "customer_id": "cus_002",
        "product": "Bluetooth Speaker",
        "status": "shipped",
        "paid_amount_minor": 299900,
        "currency": "INR",
        "delivered_at": None,
    },
}

RETURNS = {
    "ord_1001": {
        "status": "received",
        "inspection_status": "passed",
    },
    "ord_1002": {
        "status": "requested",
        "inspection_status": "pending",
    },
}

PAYMENTS = {
    "ord_1001": {
        "status": "paid",
        "refunded_amount_minor": 0,
    },
    "ord_1002": {
        "status": "paid",
        "refunded_amount_minor": 0,
    },
    "ord_2001": {
        "status": "paid",
        "refunded_amount_minor": 0,
    },
}

# Invented demonstration policy, not a real retailer's policy.
REFUND_POLICY = {
    "version": "demo-v1",
    "return_window_days": 30,
    "requires_received_return": True,
    "requires_passed_inspection": True,
    "requires_human_approval": True,
}
