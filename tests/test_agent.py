
from app.agent.workflow import support_graph

def test_delivery_question_with_refund_mention_is_order_status():
    result = support_graph.invoke(
        {
            "customer_id": "cus_001",
            "order_id": "ord_1001",
            "message": (
                "I don't need a refund; "
                "I just want to know where my order is."
            ),
        }
    )

    assert result["category"] == "order_status"
    assert result["needs_clarification"] is False


def test_refund_request_returns_eligible_assessment():
    result = support_graph.invoke({
        "customer_id": "cus_001",
        "order_id": "ord_1001",
        "message": "I want a refund for my headphones.",
    })

    assert result["category"] == "refund"
    assert result["refund_assessment"]["eligible"] is True
    assert result["refund_assessment"]["amount_minor"] == 499900
    assert result["refund_assessment"]["requires_approval"] is True
    assert "Human approval is required" in result["response"]


def test_unreceived_return_is_not_eligible():
    result = support_graph.invoke({
        "customer_id": "cus_001",
        "order_id": "ord_1002",
        "message": "Please refund my charger.",
    })

    assert result["refund_assessment"]["eligible"] is False
    assert result["refund_assessment"]["reason"] == "RETURN_NOT_RECEIVED"


def test_customer_cannot_access_another_customers_order():
    result = support_graph.invoke({
        "customer_id": "cus_001",
        "order_id": "ord_2001",
        "message": "Where is my order?",
    })

    assert result["order_summary"] == (
        "Order not found for this customer."
    )
    assert "ord_2001" not in result["response"]


def test_message_without_order_id_requests_more_information():
    result = support_graph.invoke({
        "customer_id": "cus_001",
        "order_id": None,
        "message": "I want a refund.",
    })

    assert result["category"] == "refund"
    assert "provide a valid order ID" in result["response"]


def test_damaged_item_is_classified_correctly():
    result = support_graph.invoke({
        "customer_id": "cus_001",
        "order_id": "ord_1001",
        "message": "My headphones arrived broken.",
    })

    assert result["category"] == "damaged_item"
    assert "damage" in result["response"].lower()

def test_ml_suggestions_do_not_override_operational_classification():
    result = support_graph.invoke({
        "customer_id": "cus_001",
        "order_id": None,
        "message": "I need to change my shipping address.",
    })

    # Existing deterministic classification remains unchanged.
    assert result["category"] == "order_status"

    # The trained model provides a more specific intent suggestion.
    assert result["ml_intent_suggestion"] == "change_shipping_address"
    assert result["ml_category_suggestion"] == "SHIPPING"

    # Suggestions are explicitly not authorized for automatic routing.
    assert result["ml_routing_status"] == "suggestion_only"
    assert result["ml_review_required"] is True

    assert 0 <= result["ml_intent_probability"] <= 1
    assert 0 <= result["ml_category_probability"] <= 1

def test_unclear_request_asks_for_clarification():
    result = support_graph.invoke(
        {
            "customer_id": "cus_001",
            "order_id": None,
            "message": "Can you help me?",
        }
    )

    assert result["needs_clarification"] is True
    assert result["clarification_question"]
    assert result["response"] == result["clarification_question"]

def test_clarification_skips_order_investigation():
    result = support_graph.invoke(
        {
            "customer_id": "cus_001",
            "order_id": "ord_1001",
            "message": "Can you help me?",
        }
    )

    assert result["needs_clarification"] is True
    assert result["response"] == result["clarification_question"]

    # The clarification branch should bypass order investigation.
    assert not result.get("order_summary")
    assert not result.get("refund_assessment")

def test_missing_package_is_classified_as_order_status():
    result = support_graph.invoke({
        "customer_id": "cus_001",
        "order_id": "ord_1001",
        "message": "My package still hasn't arrived.",
    })

    assert result["category"] == "order_status"
    assert result["outcome"] == "resolved"
    assert result["needs_clarification"] is False


def test_common_refund_wording_is_classified_correctly():
    result = support_graph.invoke({
        "customer_id": "cus_001",
        "order_id": "ord_1001",
        "message": "Can I get my money returned?",
    })

    assert result["category"] == "refund"
    assert result["outcome"] == "human_review_required"
    assert result["refund_assessment"]["requires_approval"] is True


def test_damaged_item_with_refund_request_requires_human_review():
    result = support_graph.invoke({
        "customer_id": "cus_001",
        "order_id": "ord_1001",
        "message": "My delivery arrived damaged and I want a refund.",
    })

    assert result["category"] == "refund"
    assert result["damage_reported"] is True
    assert result["refund_assessment"]["eligible"] is True
    assert result["refund_assessment"]["requires_approval"] is True
    assert result["outcome"] == "human_review_required"
    assert "damaged item" in result["response"].lower()
    assert "Human approval is required" in result["response"]

def test_order_status_response_does_not_claim_to_fix_delivery():
    result = support_graph.invoke({
        "customer_id": "cus_001",
        "order_id": "ord_1001",
        "message": "My order is delayed.",
    })

    assert result["category"] == "order_status"
    assert result["order_summary"]
    assert "current status recorded" in result["response"].lower()
    assert "does not itself change or resolve" in result["response"].lower()