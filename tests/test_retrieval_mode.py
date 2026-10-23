
import pytest

import app.agent.workflow as workflow


def test_keyword_mode_uses_keyword_retriever(monkeypatch):
    expected = [
        {
            "source": "REFUND-DEMO-001",
            "section": "Refund Policy",
            "text": "Refund evidence",
            "score": "2",
        }
    ]

    def fake_retriever(query, top_k=3):
        return expected

    monkeypatch.setenv("SUPPORT_RETRIEVAL_MODE", "keyword")
    monkeypatch.setattr(workflow, "retrieve_policy", fake_retriever)

    result = workflow.retrieve_policy_node({
        "message": "I need a refund",
        "category": "refund",
    })

    assert result["policy_evidence"] == expected


def test_semantic_mode_uses_semantic_retriever(monkeypatch):
    from app.agent import semantic_retrieval

    expected = [
        {
            "source": "POLICY-DAMAGED_ITEM_POLICY",
            "section": "Damage Report",
            "text": "Damage evidence",
            "score": "0.82",
        }
    ]

    def fake_retriever(query, top_k=3):
        return expected

    monkeypatch.setenv("SUPPORT_RETRIEVAL_MODE", "semantic")
    monkeypatch.setattr(
        semantic_retrieval,
        "retrieve_policy_semantic",
        fake_retriever,
    )

    result = workflow.retrieve_policy_node({
        "message": "My item arrived broken",
        "category": "damaged_item",
    })

    assert result["policy_evidence"] == expected


def test_invalid_retrieval_mode_fails_clearly(monkeypatch):
    monkeypatch.setenv("SUPPORT_RETRIEVAL_MODE", "unknown")

    with pytest.raises(ValueError, match="Unsupported SUPPORT_RETRIEVAL_MODE"):
        workflow.retrieve_policy_node({
            "message": "I need help",
            "category": "other",
        })
