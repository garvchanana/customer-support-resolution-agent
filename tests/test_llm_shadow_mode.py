
import app.agent.workflow as workflow


def test_shadow_mode_disabled_does_not_call_llm(monkeypatch):
    monkeypatch.setenv("SUPPORT_LLM_ENABLED", "false")

    def unexpected_call(**kwargs):
        raise AssertionError("LLM should not be called")

    monkeypatch.setattr(
        workflow,
        "generate_support_draft",
        unexpected_call,
    )

    result = workflow.draft_response_shadow_node({
        "message": "I need a refund",
        "response": "Human approval is required.",
        "policy_evidence": [],
    })

    assert result == {}


def test_shadow_mode_returns_draft_without_replacing_response(
    monkeypatch,
):
    monkeypatch.setenv("SUPPORT_LLM_ENABLED", "true")

    def fake_generator(**kwargs):
        assert kwargs["customer_message"] == "I need a refund"
        assert (
            kwargs["verified_facts"]["deterministic_response"]
            == "Human approval is required."
        )
        return "Your request needs human review."

    monkeypatch.setattr(
        workflow,
        "generate_support_draft",
        fake_generator,
    )

    result = workflow.draft_response_shadow_node({
        "message": "I need a refund",
        "response": "Human approval is required.",
        "category": "refund",
        "outcome": "human_review_required",
        "policy_evidence": [{
            "source": "REFUND-DEMO-001",
            "section": "Human Approval",
            "text": "Refunds require human approval.",
            "score": "2",
        }],
    })

    assert result["llm_draft"] == "Your request needs human review."
    assert "response" not in result
    assert "outcome" not in result


def test_shadow_mode_failure_does_not_break_workflow(monkeypatch):
    monkeypatch.setenv("SUPPORT_LLM_ENABLED", "true")

    def failing_generator(**kwargs):
        raise RuntimeError("Provider unavailable")

    monkeypatch.setattr(
        workflow,
        "generate_support_draft",
        failing_generator,
    )

    result = workflow.draft_response_shadow_node({
        "message": "I need a refund",
        "response": "Human approval is required.",
        "policy_evidence": [],
    })

    assert result == {"llm_draft": None}
