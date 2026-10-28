
import logging
import os
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.db import SessionLocal
from app.models import Order
from app.services.refunds import assess_refund
from app.agent.retrieval import retrieve_policy
from app.agent.classification import classify_support_message
from app.agent.response_generation import generate_support_draft



logger = logging.getLogger(__name__)
class SupportState(TypedDict, total=False):
    customer_id: str
    order_id: str | None
    message: str

    # Existing operational classification
    category: str

    # ML predictions: suggestions only, not operational decisions
    ml_intent_suggestion: str
    ml_intent_probability: float
    ml_intent_margin: float
    ml_intent_alternatives: list[dict]

    ml_category_suggestion: str
    ml_category_probability: float
    ml_category_margin: float
    ml_category_alternatives: list[dict]

    ml_routing_status: str
    ml_review_required: bool

    order_summary: str
    refund_assessment: dict
    response: str
    policy_evidence: list[dict[str, str]]

    needs_clarification: bool
    clarification_question: str
    outcome: Literal[
    "resolved",
    "clarification_required",
    "human_review_required",
    "unsupported",
    ]
    damage_reported: bool
    llm_draft: str | None

def assess_clarification(state: SupportState) -> dict:
    """Determine whether the request needs more information or is unsupported."""
    category = state["category"]
    order_id = state.get("order_id")

    if category == "other":
        message = " ".join(
            state["message"].lower().strip(" .!?").split()
        )

        vague_phrases = {
            "can you help me",
            "help me",
            "i need help",
            "i need assistance",
            "i have a question",
            "i have an issue",
            "please help",
            "can you assist me",
            "i need support",
        }

        if message in vague_phrases:
            return {
                "outcome": "clarification_required",
                "needs_clarification": True,
                "clarification_question": (
                    "Could you tell me a little more about what you need "
                    "help with? For example, is it about an order, "
                    "a refund, or a damaged item?"
                ),
            }

        return {
            "outcome": "unsupported",
            "needs_clarification": False,
            "clarification_question": "",
        }

    if not order_id:
        questions = {
            "refund": (
                "Please provide a valid order ID so I can assess "
                "your refund request."
            ),
            "order_status": (
                "Please provide your order ID so I can check its status."
            ),
            "damaged_item": (
                "Please provide your order ID and describe the damage."
            ),
        }

        return {
            "outcome": "clarification_required",
            "needs_clarification": True,
            "clarification_question": questions[category],
        }

    return {
        "needs_clarification": False,
        "clarification_question": "",
    }


def classify_message(state: SupportState) -> dict:
    """Classify supported requests using conservative phrase rules."""
    message = " ".join(state["message"].lower().strip().split())
        # Refund tracking is not implemented as a supported workflow.
    refund_tracking_phrases = [
        "refund hasn't appeared",
        "refund has not appeared",
        "refund hasn't arrived",
        "refund has not arrived",
        "refund hasn't been credited",
        "refund has not been credited",
        "refund hasn't shown up",
        "refund has not shown up",
        "refund hasn't hit my account",
        "refund has not hit my account",
        "refund not received",
        "haven't received my refund",
        "have not received my refund",
        "still waiting for my refund",
        "where is my refund",
        "track my refund",
        "check my refund status",
        "refund status",
    ]

    if any(phrase in message for phrase in refund_tracking_phrases):
        return {
            "category": "other",
            "damage_reported": False,
        }

    damage_phrases = [
        "damaged",
        "arrived broken",
        "broken",
        "cracked",
        "defective",
        "not working",
        "stopped working",
    ]

    damage_reported = any(
        phrase in message for phrase in damage_phrases
    )

    refund_negations = [
        "don't want a refund",
        "do not want a refund",
        "don't need a refund",
        "do not need a refund",
        "not looking for a refund",
        "no refund",
    ]

    refund_phrases = [
        "refund",
        "money back",
        "money returned",
        "get my money back",
        "return my money",
        "reimbursement",
        "return my payment",
        "give me my payment back",
        "get my payment back",
    ]

    refund_negated = any(
        phrase in message for phrase in refund_negations
    )
    refund_requested = (
        not refund_negated
        and any(phrase in message for phrase in refund_phrases)
    )

    # Explicit refund requests take priority, but retain the damage context.
    if refund_requested:
        return {
            "category": "refund",
            "damage_reported": damage_reported,
        }

    if damage_reported:
        return {
            "category": "damaged_item",
            "damage_reported": True,
        }

    order_status_phrases = [
        "where is",
        "where my order",
        "shipping",
        "delivery",
        "tracking",
        "hasn't arrived",
        "has not arrived",
        "never arrived",
        "not arrived",
        "still hasn't come",
        "still has not come",
        "package is late",
        "order is late",
        "order is delayed",
        "delivery is delayed",
        "running late",
        "late delivery",
        "check when",
        "will arrive",
        "never showed up",
        "track my package",
        "track my order",
        "track my delivery",
        "status of this order",
        "status of my order",
        "check the status",
        "check its status",
    ]

    if any(
        phrase in message for phrase in order_status_phrases
    ):
        return {
            "category": "order_status",
            "damage_reported": False,
        }

    return {
        "category": "other",
        "damage_reported": False,
    }

def suggest_ml_classification(state: SupportState) -> dict:
    """Attach ML predictions without changing operational routing."""
    prediction = classify_support_message(state["message"])

    return {
        "ml_intent_suggestion": prediction["intent_suggestion"],
        "ml_intent_probability": prediction["intent_probability"],
        "ml_intent_margin": prediction["intent_margin"],
        "ml_intent_alternatives": prediction["intent_alternatives"],
        "ml_category_suggestion": prediction["category_suggestion"],
        "ml_category_probability": prediction["category_probability"],
        "ml_category_margin": prediction["category_margin"],
        "ml_category_alternatives": prediction["category_alternatives"],
        "ml_routing_status": prediction["routing_status"],
        "ml_review_required": prediction["review_required"],
    }

def retrieve_policy_node(state: SupportState) -> dict:
    query = f"{state['category']} {state['message']}"
    retrieval_mode = os.getenv(
        "SUPPORT_RETRIEVAL_MODE",
        "keyword",
    ).strip().lower()

    if retrieval_mode == "keyword":
        evidence = retrieve_policy(query, top_k=3)

    elif retrieval_mode == "semantic":
        from app.agent.semantic_retrieval import retrieve_policy_semantic

        evidence = retrieve_policy_semantic(query, top_k=3)

    else:
        raise ValueError(
            f"Unsupported SUPPORT_RETRIEVAL_MODE: {retrieval_mode!r}. "
            "Choose 'keyword' or 'semantic'."
        )
    return {"policy_evidence": evidence}


def draft_response_shadow_node(state: SupportState) -> dict:
    """Generate an optional LLM draft without changing the final response."""
    enabled = os.getenv(
        "SUPPORT_LLM_ENABLED",
        "false",
    ).strip().lower() in {"1", "true", "yes", "on"}

    if not enabled:
        logger.info("Groq shadow draft status=disabled")
        return {}

    try:
        draft = generate_support_draft(
            customer_message=state.get("message", ""),
            verified_facts={
                "category": state.get("category"),
                "outcome": state.get("outcome"),
                "order_summary": state.get("order_summary"),
                "refund_assessment": state.get("refund_assessment"),
                "deterministic_response": state.get("response"),
            },
            policy_evidence=state.get("policy_evidence", []),
        )
    except Exception as exc:
        logger.warning(
            "Groq shadow draft status=error error_type=%s",
            type(exc).__name__,
        )
        draft = None

    if draft is None:
        logger.info("Groq shadow draft status=unavailable")
    else:
        logger.info("Groq shadow draft status=accepted")

    # Deliberately do not update state["response"].
    return {"llm_draft": draft}


def investigate_order(state: SupportState) -> dict:
    """Look up the order and assess a refund when appropriate."""
    order_id = state.get("order_id")
    customer_id = state["customer_id"]

    if not order_id:
        return {
            "order_summary": "No order ID was provided.",
            "refund_assessment": {},
        }

    with SessionLocal() as db:
        order = db.get(Order, order_id)

        # Never expose another customer's order information.
        if order is None or order.customer_id != customer_id:
            return {
                "order_summary": "Order not found for this customer.",
                "refund_assessment": {},
            }

        summary = (
            f"Product: {order.product}; "
            f"status: {order.status}; "
            f"order ID: {order.id}."
        )

        updates = {"order_summary": summary}

        if state["category"] == "refund":
            assessment = assess_refund(order_id, db)
            updates["refund_assessment"] = assessment.model_dump()

        return updates


def formulate_response(state: SupportState) -> dict:
    """Produce a response based on verified information and set an explicit outcome."""
    category = state["category"]
    summary = state.get("order_summary", "")
    assessment = state.get("refund_assessment", {})
    evidence = state.get("policy_evidence", [])
    outcome = state.get("outcome")

    if outcome == "unsupported":
        return {
            "response": (
                "I can help with order status, refund assessments, "
                "and damaged-item support. I can't help with that request."
            ),
            "outcome": "unsupported",
            "needs_clarification": False,
            "clarification_question": "",
        }

    if state.get("needs_clarification", False):
        question = state.get("clarification_question") or (
            "Could you provide a little more information so I can help?"
        )
        return {
            "response": question,
            "outcome": "clarification_required",
            "needs_clarification": True,
            "clarification_question": question,
        }

    policy_note = ""
    if evidence:
        sections = ", ".join(
            f"{item['source']} — {item['section']}"
            for item in evidence
        )
        policy_note = f" Policy sections consulted: {sections}."

    final_needs_clarification = False
    final_clarification_question = ""

    if summary.startswith("Order not found"):
        response = (
            "I couldn't verify this order for your customer account. "
            "Please check the order ID."
        )
        final_outcome = "resolved"

    elif category == "refund":
        if not assessment:
            response = (
                "Please provide a valid order ID so I can assess your refund request."
            )
            final_outcome = "clarification_required"
            final_needs_clarification = True
            final_clarification_question = (
                "Please provide a valid order ID so I can assess your refund request."
            )
            response = final_clarification_question
        elif assessment.get("eligible"):
            response = (
                f"{summary} Your refund request meets the demo policy "
                f"checks for ₹{assessment['amount_minor'] / 100:,.2f}. "
                "Human approval is required before any refund is issued."
            )

            if state.get("damage_reported", False):
                response += (
                    " You also reported a damaged item. "
                    "The damage details should be reviewed alongside "
                    "the refund request."
                )

            final_outcome = "human_review_required"
        else:
            response = (
                f"{summary} The refund is not currently eligible. "
                f"Reason: {assessment.get('reason', 'UNKNOWN')}."
            )
            final_outcome = "resolved"

    elif category == "order_status":
        if summary and not summary.startswith("No order ID"):
            response = (
                f"{summary} This is the current status recorded in "
                "the demo database. Checking the status does not "
                "itself change or resolve a delivery delay."
            )
            final_outcome = "resolved"
        else:
            final_outcome = "clarification_required"
            final_needs_clarification = True
            final_clarification_question = (
                "Please provide your order ID so I can check its status."
            )
            response = final_clarification_question

    elif category == "damaged_item":
        if summary.startswith("No order ID"):
            final_outcome = "clarification_required"
            final_needs_clarification = True
            final_clarification_question = (
                "Please provide your order ID and describe the damage."
            )
            response = final_clarification_question
        else:
            response = (
                f"{summary} Please share details of the damage. "
                "The case can then be reviewed under the applicable policy."
            )
            final_outcome = "clarification_required"
            final_needs_clarification = True
            final_clarification_question = (
                "Please share details of the damage so the case can be reviewed."
            )

    else:
        response = (
            "I've classified your message for support review. "
            "Please provide any relevant order ID or additional details."
        )
        final_outcome = "clarification_required"
        final_needs_clarification = True
        final_clarification_question = (
            "Please provide any relevant order ID or additional details."
        )

    response += policy_note
    return {
        "response": response,
        "outcome": final_outcome,
        "needs_clarification": final_needs_clarification,
        "clarification_question": final_clarification_question,
    }

def route_after_clarification(state: SupportState) -> str:
    if state.get("outcome") in {
        "clarification_required",
        "unsupported",
    }:
        return "respond"

    return "retrieve_policy"



builder = StateGraph(SupportState)

builder.add_node("classify", classify_message)
builder.add_node("ml_classify", suggest_ml_classification)
builder.add_node("retrieve_policy", retrieve_policy_node)
builder.add_node("investigate", investigate_order)
builder.add_node("respond", formulate_response)
builder.add_node(
    "draft_response_shadow",
    draft_response_shadow_node,
)
builder.add_node("assess_clarification", assess_clarification)
builder.add_conditional_edges(
    "assess_clarification",
    route_after_clarification,
    {
        "respond": "respond",
        "retrieve_policy": "retrieve_policy",
    },
)


builder.add_edge(START, "classify")
builder.add_edge("classify", "ml_classify")
builder.add_edge("ml_classify", "assess_clarification")
builder.add_edge("retrieve_policy", "investigate")
builder.add_edge("investigate", "respond")
builder.add_edge("respond", "draft_response_shadow")
builder.add_edge("draft_response_shadow", END)

support_graph = builder.compile()
