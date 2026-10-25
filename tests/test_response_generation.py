import json
from types import SimpleNamespace

from app.agent.response_generation import generate_support_draft
from app.agent.response_generation import _draft_has_unsupported_claims

def test_missing_api_key_returns_none(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    result = generate_support_draft(
        customer_message="I want a refund",
        verified_facts={"refund_eligible": False},
        policy_evidence=[{
            "source": "REFUND-DEMO-001",
            "section": "Refund Policy",
            "text": "Refunds require eligibility checks.",
        }],
    )

    assert result is None


def test_valid_model_response_returns_draft():
    expected_text = (
        "Your refund request needs to be reviewed "
        "against our refund policy."
    )

    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["model"] == "openai/gpt-oss-120b"
            assert kwargs["response_format"] == {
                "type": "json_object"
            }

            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps({
                                "response": expected_text
                            })
                        )
                    )
                ]
            )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=FakeCompletions()
        )
    )

    result = generate_support_draft(
        customer_message="I want a refund",
        verified_facts={"refund_eligible": False},
        policy_evidence=[{
            "source": "REFUND-DEMO-001",
            "section": "Refund Policy",
            "text": "Refunds require eligibility checks.",
        }],
        client=fake_client,
    )

    assert result == expected_text


def test_invalid_json_returns_none():
    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="not valid JSON"
                        )
                    )
                ]
            )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=FakeCompletions()
        )
    )

    result = generate_support_draft(
        customer_message="I want a refund",
        verified_facts={"refund_eligible": False},
        policy_evidence=[{
            "source": "REFUND-DEMO-001",
            "section": "Refund Policy",
            "text": "Refunds require eligibility checks.",
        }],
        client=fake_client,
    )

    assert result is None


def test_empty_evidence_returns_none():
    result = generate_support_draft(
        customer_message="I want a refund",
        verified_facts={"refund_eligible": False},
        policy_evidence=[],
        client=object(),
    )

    assert result is None


def test_unsupported_follow_up_promise_is_rejected():
    from app.agent.response_generation import (
        _draft_has_unsupported_claims,
    )

    assert _draft_has_unsupported_claims(
        "Our team will update you as soon as possible.",
        {"outcome": "human_review_required"},
        [{"text": "Human approval is required."}],
    )


def test_unconfirmed_recording_claim_is_rejected():
    from app.agent.response_generation import (
        _draft_has_unsupported_claims,
    )

    assert _draft_has_unsupported_claims(
        "We have recorded your refund request.",
        {"outcome": "human_review_required"},
        [{"text": "Human approval is required."}],
    )


def test_grounded_approval_language_is_allowed():
    from app.agent.response_generation import (
        _draft_has_unsupported_claims,
    )

    assert not _draft_has_unsupported_claims(
        "Human approval is required before a refund is issued.",
        {"outcome": "human_review_required"},
        [{"text": "Human approval is required before any refund is issued."}],
    )

def test_rejects_unsupported_follow_up_commitment():
    draft = (
        "Our team will review your refund request "
        "and will follow up with you."
    )

    assert _draft_has_unsupported_claims(
        draft,
        {"request_recorded": False},
        [],
    )

def test_rejects_refund_request_noted_without_confirmation():
    draft = "Your refund request has been noted."

    assert _draft_has_unsupported_claims(
        draft,
        {"request_recorded": False},
        [],
    )


def test_rejects_damage_report_recorded_without_confirmation():
    draft = "We've recorded that your headphones arrived damaged."

    assert _draft_has_unsupported_claims(
        draft,
        {"request_recorded": False},
        [],
    )


def test_accepts_safe_human_review_response():
    draft = "A support representative needs to review your case."

    assert not _draft_has_unsupported_claims(
        draft,
        {"request_recorded": False},
        [],
    )

def test_rejects_currency_minor_units_wording():
    draft = (
        "Your order is eligible for a refund of "
        "499900 INR minor units."
    )

    assert _draft_has_unsupported_claims(
        draft,
        {"request_recorded": False},
        [],
    )

def test_rejects_damaged_item_draft_with_unsupported_claims():
    draft = (
        "We've recorded that your wireless headphones arrived damaged. "
        "Your refund request for 4,999.00 INR has been recorded and "
        "will be reviewed by a team member before any refund is issued. "
        "To help us process the request, could you please share photos "
        "or a brief description of the damage if you have them? "
        "A support representative will review the case and follow up "
        "with next steps."
    )

    assert _draft_has_unsupported_claims(
        draft,
        {"request_recorded": False},
        [],
    )

def test_rejects_follow_up_with_next_steps():
    draft = (
        "A support representative will review the case "
        "and follow up with next steps."
    )

    assert _draft_has_unsupported_claims(
        draft,
        {"request_recorded": False},
        [],
    )

def test_rejects_follow_up_after_review_promise():
    draft = (
        "A support representative will follow up "
        "once the review is complete."
    )

    assert _draft_has_unsupported_claims(
        draft,
        {"request_recorded": False},
        [],
    )

from app.agent.response_generation import _unsupported_claim_reason


def test_rejection_reason_identifies_unsupported_follow_up():
    reason = _unsupported_claim_reason(
        response="A representative will follow up once the review is complete.",
        verified_facts={},
        policy_evidence=[],
    )

    assert reason == "unsupported_follow_up_or_processing_commitment"


def test_rejection_reason_identifies_unverified_recording_claim():
    reason = _unsupported_claim_reason(
        response="Your refund request has been recorded.",
        verified_facts={},
        policy_evidence=[],
    )

    assert reason == "unverified_recording_or_ticket_claim"


def test_recording_claim_allowed_when_explicitly_verified():
    reason = _unsupported_claim_reason(
        response="Your request has been recorded.",
        verified_facts={"request_recorded": True},
        policy_evidence=[],
    )

    assert reason is None


def test_rejection_reason_identifies_unsupported_photo_request():
    reason = _unsupported_claim_reason(
        response="Please provide a photo of the damaged item.",
        verified_facts={},
        policy_evidence=[],
    )

    assert reason == "unsupported_photo_or_image_request"


def test_rejection_reason_identifies_minor_units_wording():
    reason = _unsupported_claim_reason(
        response="The amount is expressed in minor units.",
        verified_facts={},
        policy_evidence=[],
    )

    assert reason == "currency_minor_units_wording"


def test_safe_draft_has_no_known_rejection_reason():
    reason = _unsupported_claim_reason(
        response="A support representative must approve the refund before it can be issued.",
        verified_facts={},
        policy_evidence=[],
    )

    assert reason is None

def test_rejects_internal_currency_field_names():
    for phrase in ("amount_minor", "amount_minor_units"):
        reason = _unsupported_claim_reason(
            response=f"The value is {phrase}.",
            verified_facts={},
            policy_evidence=[],
        )
        assert reason == "currency_minor_units_wording"


def test_accepts_customer_friendly_currency_amount():
    reason = _unsupported_claim_reason(
        response="The eligible refund amount is ₹4,999.00, subject to human approval.",
        verified_facts={},
        policy_evidence=[],
    )
    assert reason is None
