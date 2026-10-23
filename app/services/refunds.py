
from datetime import date

from sqlalchemy.orm import Session

from app.data import REFUND_POLICY
from app.models import Order, Payment, ReturnRecord
from app.schemas import RefundAssessment


def assess_refund(order_id: str, db: Session) -> RefundAssessment:
    order = db.get(Order, order_id)

    if order is None:
        return RefundAssessment(
            eligible=False,
            amount_minor=None,
            currency=None,
            reason="ORDER_NOT_FOUND",
            requires_approval=True,
        )

    payment = db.get(Payment, order_id)
    if payment is None or payment.status != "paid":
        return RefundAssessment(
            eligible=False,
            amount_minor=None,
            currency=order.currency,
            reason="PAYMENT_NOT_PAID",
            requires_approval=True,
        )

    if order.delivered_at is None:
        return RefundAssessment(
            eligible=False,
            amount_minor=None,
            currency=order.currency,
            reason="ORDER_NOT_DELIVERED",
            requires_approval=True,
        )

    age_days = (date.today() - order.delivered_at).days
    if age_days < 0 or age_days > REFUND_POLICY["return_window_days"]:
        return RefundAssessment(
            eligible=False,
            amount_minor=None,
            currency=order.currency,
            reason="RETURN_WINDOW_EXPIRED",
            requires_approval=True,
        )

    return_record = db.query(ReturnRecord).filter_by(
        order_id=order_id
    ).first()

    if (
        REFUND_POLICY["requires_received_return"]
        and (
            return_record is None
            or return_record.status != "received"
        )
    ):
        return RefundAssessment(
            eligible=False,
            amount_minor=None,
            currency=order.currency,
            reason="RETURN_NOT_RECEIVED",
            requires_approval=True,
        )

    if (
        REFUND_POLICY["requires_passed_inspection"]
        and (
            return_record is None
            or return_record.inspection_status != "passed"
        )
    ):
        return RefundAssessment(
            eligible=False,
            amount_minor=None,
            currency=order.currency,
            reason="INSPECTION_NOT_PASSED",
            requires_approval=True,
        )

    remaining_amount = (
        order.paid_amount_minor - payment.refunded_amount_minor
    )

    if remaining_amount <= 0:
        return RefundAssessment(
            eligible=False,
            amount_minor=0,
            currency=order.currency,
            reason="NO_REFUND_BALANCE",
            requires_approval=True,
        )

    return RefundAssessment(
        eligible=True,
        amount_minor=remaining_amount,
        currency=order.currency,
        reason="ELIGIBLE_PENDING_HUMAN_APPROVAL",
        requires_approval=REFUND_POLICY["requires_human_approval"],
    )
