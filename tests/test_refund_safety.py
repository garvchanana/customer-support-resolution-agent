from datetime import date, timedelta
from unittest.mock import Mock

import pytest

from app.data import REFUND_POLICY
from app.models import Order, Payment, ReturnRecord
from app.services.refunds import assess_refund


def make_order(delivered_at=None, paid_amount_minor=499900):
    return Order(
        id="test_order",
        customer_id="cus_test",
        product="Test product",
        status="delivered",
        paid_amount_minor=paid_amount_minor,
        currency="INR",
        delivered_at=delivered_at or date.today() - timedelta(days=10),
    )


def make_db(order, payment=None, return_record=None):
    db = Mock()

    def get(model, key):
        if model is Order:
            return order if order is not None and key == order.id else None
        if model is Payment:
            return payment
        return None

    db.get.side_effect = get
    db.query.return_value.filter_by.return_value.first.return_value = (
        return_record
    )
    return db


def make_payment(status="paid", refunded_amount_minor=0):
    return Payment(
        order_id="test_order",
        status=status,
        refunded_amount_minor=refunded_amount_minor,
    )


def make_return(status="received", inspection_status="passed"):
    return ReturnRecord(
        id="test_return",
        order_id="test_order",
        status=status,
        inspection_status=inspection_status,
    )


def test_missing_order_is_ineligible():
    db = make_db(None)

    result = assess_refund("missing_order", db)

    assert result.eligible is False
    assert result.reason == "ORDER_NOT_FOUND"
    assert result.requires_approval is True


def test_unpaid_payment_is_ineligible():
    order = make_order()
    db = make_db(order, payment=make_payment(status="pending"))

    result = assess_refund(order.id, db)

    assert result.eligible is False
    assert result.reason == "PAYMENT_NOT_PAID"


def test_missing_return_is_ineligible():
    order = make_order()
    db = make_db(order, payment=make_payment(), return_record=None)

    result = assess_refund(order.id, db)

    assert result.eligible is False
    assert result.reason == "RETURN_NOT_RECEIVED"


def test_return_not_received_is_ineligible():
    order = make_order()
    db = make_db(
        order,
        payment=make_payment(),
        return_record=make_return(status="in_transit"),
    )

    result = assess_refund(order.id, db)

    assert result.eligible is False
    assert result.reason == "RETURN_NOT_RECEIVED"


def test_inspection_not_passed_is_ineligible():
    order = make_order()
    db = make_db(
        order,
        payment=make_payment(),
        return_record=make_return(inspection_status="failed"),
    )

    result = assess_refund(order.id, db)

    assert result.eligible is False
    assert result.reason == "INSPECTION_NOT_PASSED"


def test_return_window_boundary_and_human_approval():
    order = make_order(delivered_at=date.today() - timedelta(days=30))
    db = make_db(
        order,
        payment=make_payment(),
        return_record=make_return(),
    )

    result = assess_refund(order.id, db)

    assert result.eligible is True
    assert result.amount_minor == 499900
    assert result.requires_approval is True
    assert result.reason == "ELIGIBLE_PENDING_HUMAN_APPROVAL"


def test_return_outside_window_is_ineligible():
    order = make_order(delivered_at=date.today() - timedelta(days=31))
    db = make_db(
        order,
        payment=make_payment(),
        return_record=make_return(),
    )

    result = assess_refund(order.id, db)

    assert result.eligible is False
    assert result.reason == "RETURN_WINDOW_EXPIRED"


def test_fully_refunded_order_has_no_balance():
    order = make_order(paid_amount_minor=499900)
    db = make_db(
        order,
        payment=make_payment(refunded_amount_minor=499900),
        return_record=make_return(),
    )

    result = assess_refund(order.id, db)

    assert result.eligible is False
    assert result.amount_minor == 0
    assert result.reason == "NO_REFUND_BALANCE"

