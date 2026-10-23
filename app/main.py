
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import Customer, Order, Ticket

from app.agent.workflow import support_graph
from app.schemas import (
    CreateTicketRequest,
    RefundAssessment,
    SupportClassifyRequest,
    SupportClassifyResponse,
    SupportResolveRequest,
    SupportResolveResponse,
    TicketResponse,
)
from app.services.refunds import assess_refund
from app.agent.classification import classify_support_message


app = FastAPI(
    title="Customer Support Resolution Agent",
    version="0.2.0",
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_demo_customer(
    x_demo_customer_id: str = Header(alias="X-Demo-Customer-ID"),
    db: Session = Depends(get_db),
) -> Customer:
    customer = db.get(Customer, x_demo_customer_id)

    if customer is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid demo customer identity",
        )

    return customer


def classify_complaint(message: str) -> str:
    text = message.lower()

    if any(word in text for word in ["refund", "money back", "reimbursement"]):
        return "refund"

    if any(word in text for word in ["damaged", "broken", "cracked", "defective"]):
        return "damaged_item"

    if any(word in text for word in ["where is", "shipping", "delivery", "tracking"]):
        return "order_status"

    return "other"


def ticket_response(ticket: Ticket) -> dict:
    return {
        "ticket_id": ticket.id,
        "customer_id": ticket.customer_id,
        "order_id": ticket.order_id,
        "category": ticket.category,
        "status": ticket.status,
        "message": ticket.message,
        "created_at": ticket.created_at,
    }


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post(
    "/support/classify",
    response_model=SupportClassifyResponse,
)
def classify_support(
    request: SupportClassifyRequest,
    customer: Customer = Depends(get_demo_customer),
):
    try:
        return classify_support_message(request.message)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Classification models are unavailable. Train the models first.",
        ) from exc


@app.post(
    "/tickets",
    response_model=TicketResponse,
    status_code=201,
)
def create_ticket(
    request: CreateTicketRequest,
    customer: Customer = Depends(get_demo_customer),
    db: Session = Depends(get_db),
):
    if request.order_id is not None:
        order = db.get(Order, request.order_id)

        if order is None or order.customer_id != customer.id:
            raise HTTPException(
                status_code=404,
                detail="Order not found",
            )

    ticket = Ticket(
        id=f"tkt_{uuid4().hex[:12]}",
        customer_id=customer.id,
        order_id=request.order_id,
        category=classify_complaint(request.message),
        status="open",
        message=request.message,
        created_at=datetime.now(timezone.utc),
    )

    db.add(ticket)
    db.commit()
    db.refresh(ticket)

    return ticket_response(ticket)


@app.get(
    "/orders/{order_id}/refund-assessment",
    response_model=RefundAssessment,
)
def refund_assessment(
    order_id: str,
    customer: Customer = Depends(get_demo_customer),
    db: Session = Depends(get_db),
):
    order = db.get(Order, order_id)

    if order is None or order.customer_id != customer.id:
        raise HTTPException(
            status_code=404,
            detail="Order not found",
        )

    return assess_refund(order_id, db)


@app.get(
    "/tickets/{ticket_id}",
    response_model=TicketResponse,
)
def get_ticket(
    ticket_id: str,
    customer: Customer = Depends(get_demo_customer),
    db: Session = Depends(get_db),
):
    ticket = db.get(Ticket, ticket_id)

    if ticket is None or ticket.customer_id != customer.id:
        raise HTTPException(
            status_code=404,
            detail="Ticket not found",
        )

    return ticket_response(ticket)


@app.post(
    "/support/resolve",
    response_model=SupportResolveResponse,
)
def resolve_support_request(
    request: SupportResolveRequest,
    customer: Customer = Depends(get_demo_customer),
    db: Session = Depends(get_db),
):
    # Validate order ownership before running the workflow.
    if request.order_id is not None:
        order = db.get(Order, request.order_id)

        if order is None or order.customer_id != customer.id:
            raise HTTPException(
                status_code=404,
                detail="Order not found",
            )

    result = support_graph.invoke({
        "customer_id": customer.id,
        "order_id": request.order_id,
        "message": request.message,
    })

    return {
        "category": result["category"],
        "order_summary": result.get(
            "order_summary", "No order information available."
        ),
        "refund_assessment": result.get("refund_assessment", {}),
        "response": result["response"],
        "ml_intent_suggestion": result["ml_intent_suggestion"],
        "ml_intent_probability": result["ml_intent_probability"],
        "ml_intent_margin": result["ml_intent_margin"],
        "ml_category_suggestion": result["ml_category_suggestion"],
        "ml_category_probability": result["ml_category_probability"],
        "ml_category_margin": result["ml_category_margin"],
        "ml_routing_status": result["ml_routing_status"],
        "ml_review_required": result["ml_review_required"],
        "needs_clarification": result.get("needs_clarification", False),
        "clarification_question": result.get("clarification_question", ""),
        "outcome": result["outcome"],
    }
