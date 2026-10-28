
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.db import SessionLocal
from app.main import app
from app.models import Ticket


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def demo_headers():
    return {"X-Demo-Customer-ID": "cus_001"}


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_eligible_refund_assessment(client, demo_headers):
    response = client.get(
        "/orders/ord_1001/refund-assessment",
        headers=demo_headers,
    )

    assert response.status_code == 200

    result = response.json()
    assert result["eligible"] is True
    assert result["amount_minor"] == 499900
    assert result["currency"] == "INR"
    assert result["requires_approval"] is True
    assert result["reason"] == "ELIGIBLE_PENDING_HUMAN_APPROVAL"


def test_refund_rejected_when_return_not_received(client, demo_headers):
    response = client.get(
        "/orders/ord_1002/refund-assessment",
        headers=demo_headers,
    )

    assert response.status_code == 200

    result = response.json()
    assert result["eligible"] is False
    assert result["reason"] == "RETURN_NOT_RECEIVED"


def test_customer_cannot_access_another_customers_order(client, demo_headers):
    response = client.get(
        "/orders/ord_2001/refund-assessment",
        headers=demo_headers,
    )

    assert response.status_code == 404


def test_create_retrieve_and_protect_ticket(client, demo_headers):
    response = client.post(
        "/tickets",
        headers=demo_headers,
        json={
            "order_id": "ord_1001",
            "message": f"Test refund request {uuid4().hex}",
        },
    )

    assert response.status_code == 201
    created = response.json()
    ticket_id = created["ticket_id"]

    try:
        assert created["customer_id"] == "cus_001"
        assert created["order_id"] == "ord_1001"
        assert created["category"] == "refund"
        assert created["status"] == "open"

        # The ticket can be retrieved by its owner.
        retrieve_response = client.get(
            f"/tickets/{ticket_id}",
            headers=demo_headers,
        )
        assert retrieve_response.status_code == 200
        assert retrieve_response.json()["ticket_id"] == ticket_id

        # Another customer must not be able to retrieve it.
        other_customer_headers = {
            "X-Demo-Customer-ID": "cus_002"
        }
        unauthorized_response = client.get(
            f"/tickets/{ticket_id}",
            headers=other_customer_headers,
        )
        assert unauthorized_response.status_code == 404

    finally:
        # Keep the demo database clean after this test.
        with SessionLocal.begin() as db:
            ticket = db.get(Ticket, ticket_id)
            if ticket is not None:
                db.delete(ticket)


def test_support_resolve_refund(client, demo_headers):
    response = client.post(
        "/support/resolve",
        headers=demo_headers,
        json={
            "order_id": "ord_1001",
            "message": "I would like a refund for my headphones.",
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["category"] == "refund"
    assert result["refund_assessment"]["eligible"] is True
    assert result["refund_assessment"]["requires_approval"] is True
    assert result["outcome"] == "human_review_required"
    assert "Human approval is required" in result["response"]

    assert result["ml_intent_suggestion"] == "get_refund"
    assert result["ml_category_suggestion"] == "REFUND"
    assert result["ml_routing_status"] == "suggestion_only"
    assert result["ml_review_required"] is True

    assert 0 <= result["ml_intent_probability"] <= 1
    assert 0 <= result["ml_category_probability"] <= 1

def test_support_resolve_asks_for_clarification(client, demo_headers):
    response = client.post(
        "/support/resolve",
        headers=demo_headers,
        json={
            "order_id": "ord_1001",
            "message": "Can you help me?",
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["needs_clarification"] is True
    assert data["outcome"] == "clarification_required"
    assert data["clarification_question"]
    assert data["response"] == data["clarification_question"]
    assert data["ml_routing_status"] == "suggestion_only"
    assert data["ml_review_required"] is True

def test_support_resolve_missing_order_id(client, demo_headers):
    response = client.post(
        "/support/resolve",
        headers=demo_headers,
        json={
            "order_id": None,
            "message": "Where is my package?",
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["category"] == "order_status"
    assert data["needs_clarification"] is True
    assert data["outcome"] == "clarification_required"
    assert data["clarification_question"]
    assert data["response"] == data["clarification_question"]


def test_support_resolve_unsupported_request(client, demo_headers):
    response = client.post(
        "/support/resolve",
        headers=demo_headers,
        json={
            "order_id": "ord_1001",
            "message": "Can you recommend a movie?",
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["category"] == "other"
    assert data["needs_clarification"] is False
    assert data["outcome"] == "unsupported"


def test_support_resolve_rejects_another_customers_order(
    client, demo_headers
):
    response = client.post(
        "/support/resolve",
        headers=demo_headers,
        json={
            "order_id": "ord_2001",
            "message": "Where is my order?",
        },
    )

    assert response.status_code == 404

def test_customer_cannot_access_nonexistent_order(client, demo_headers):
    response = client.get(
        "/orders/ord_does_not_exist/refund-assessment",
        headers=demo_headers,
    )

    assert response.status_code == 404

def test_support_classify_returns_suggestions(client, demo_headers):
    response = client.post(
        "/support/classify",
        headers=demo_headers,
        json={
            "message": "I need to change my shipping address."
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["intent_suggestion"] == "change_shipping_address"
    assert result["category_suggestion"] == "SHIPPING"
    assert result["routing_status"] == "suggestion_only"
    assert result["review_required"] is True

    assert 0 <= result["intent_probability"] <= 1
    assert 0 <= result["category_probability"] <= 1

    assert len(result["intent_alternatives"]) == 3
    assert len(result["category_alternatives"]) == 3


def test_support_classify_requires_demo_identity(client):
    response = client.post(
        "/support/classify",
        json={
            "message": "I need to change my shipping address."
        },
    )

    assert response.status_code == 422
