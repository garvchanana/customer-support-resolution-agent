
from app.agent.retrieval import retrieve_policy
from app.agent.semantic_retrieval import retrieve_policy_semantic


def test_refund_query_retrieves_policy_evidence():
    results = retrieve_policy(
        "I want a refund and need human approval",
        top_k=5,
    )

    assert results
    assert any(
        "Human approval" in result["text"]
        for result in results
    )


def test_return_query_retrieves_return_policy():
    results = retrieve_policy(
        "My returned item has not been received",
        top_k=5,
    )

    assert results
    assert any(
        "Return receipt" in result["text"]
        for result in results
    )


def test_retrieved_sections_have_source_metadata():
    results = retrieve_policy("refund payment", top_k=3)

    assert results
    assert all(
        result["source"] == "REFUND-DEMO-001"
        for result in results
    )


def test_damaged_item_query_retrieves_damage_policy():
    results = retrieve_policy(
        "My item arrived damaged and needs human review",
        top_k=5,
    )

    assert results
    assert any(
        result["source"] == "POLICY-DAMAGED_ITEM_POLICY"
        for result in results
    )
    assert any(
        "Human Review" in result["text"]
        for result in results
    )


def test_unrelated_query_returns_no_policy_evidence():
    results = retrieve_policy(
        "Explain quantum physics and black holes",
        top_k=3,
    )

    assert results == []


def test_semantic_retrieval_finds_damage_policy():
    results = retrieve_policy_semantic(
        "My package arrived broken and I need help",
        top_k=5,
    )

    assert results
    assert any(
        result["source"] == "POLICY-DAMAGED_ITEM_POLICY"
        for result in results
    )


def test_semantic_retrieval_finds_refund_policy():
    results = retrieve_policy_semantic(
        "I want my money returned",
        top_k=5,
    )

    assert results
    assert any(
        result["source"] == "REFUND-DEMO-001"
        for result in results
    )


def test_semantic_retrieval_rejects_empty_query():
    assert retrieve_policy_semantic("") == []


def test_keyword_retrieval_returns_both_policies_for_mixed_request():
    results = retrieve_policy(
        "My item arrived damaged and I want a refund",
        top_k=3,
    )

    sources = {result["source"] for result in results}

    assert "REFUND-DEMO-001" in sources
    assert "POLICY-DAMAGED_ITEM_POLICY" in sources


def test_semantic_retrieval_returns_both_policies_for_mixed_request():
    results = retrieve_policy_semantic(
        "My item arrived damaged and I want a refund",
        top_k=3,
    )

    sources = {result["source"] for result in results}

    assert "REFUND-DEMO-001" in sources
    assert "POLICY-DAMAGED_ITEM_POLICY" in sources
