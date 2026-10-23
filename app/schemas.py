
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class CreateTicketRequest(BaseModel):
    order_id: str | None = None
    message: str = Field(min_length=5, max_length=2000)


class TicketResponse(BaseModel):
    ticket_id: str
    customer_id: str
    order_id: str | None
    category: Literal[
        "order_status",
        "refund",
        "damaged_item",
        "other",
    ]
    status: str
    message: str
    created_at: datetime


class RefundAssessment(BaseModel):
    eligible: bool
    amount_minor: int | None
    currency: str | None
    reason: str
    requires_approval: bool

class SupportResolveRequest(BaseModel):
    order_id: str | None = None
    message: str = Field(min_length=5, max_length=2000)


class SupportResolveResponse(BaseModel):
    category: Literal[
        "order_status", "refund", "damaged_item", "other"
    ]
    order_summary: str
    refund_assessment: dict = Field(default_factory=dict)
    response: str

    ml_intent_suggestion: str
    ml_intent_probability: float
    ml_intent_margin: float
    ml_category_suggestion: str
    ml_category_probability: float
    ml_category_margin: float
    ml_routing_status: Literal["suggestion_only"]
    ml_review_required: bool

    needs_clarification: bool
    clarification_question: str

    outcome: Literal[
    "resolved",
    "clarification_required",
    "human_review_required",
    "unsupported",
    ]



class SupportClassifyRequest(BaseModel):
    message: str = Field(min_length=5, max_length=2000)


class SupportClassifyResponse(BaseModel):
    message: str
    intent_suggestion: str
    intent_probability: float
    intent_margin: float
    intent_alternatives: list[dict]
    category_suggestion: str
    category_probability: float
    category_margin: float
    category_alternatives: list[dict]
    routing_status: Literal["suggestion_only"]
    review_required: bool