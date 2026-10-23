
import pytest

from app.agent.classification import classify_support_message


def test_clear_shipping_request():
    result = classify_support_message(
        "Please change my shipping address."
    )

    assert result["intent_suggestion"] == "change_shipping_address"
    assert result["category_suggestion"] == "SHIPPING"
    assert result["routing_status"] == "suggestion_only"
    assert result["review_required"] is True


def test_returns_top_three_alternatives():
    result = classify_support_message(
        "I want to get a refund for my order."
    )

    assert len(result["intent_alternatives"]) == 3
    assert len(result["category_alternatives"]) == 3

    for prediction in (
        result["intent_alternatives"]
        + result["category_alternatives"]
    ):
        assert isinstance(prediction["label"], str)
        assert 0.0 <= prediction["probability"] <= 1.0


def test_empty_message_is_rejected():
    with pytest.raises(ValueError):
        classify_support_message("   ")


def test_out_of_scope_message_does_not_bypass_review():
    result = classify_support_message(
        "What will the weather be like tomorrow?"
    )

    # The classifier may assign an ordinary support label, but it
    # must never claim to have verified scope or approved routing.
    assert result["routing_status"] == "suggestion_only"
    assert result["review_required"] is True
