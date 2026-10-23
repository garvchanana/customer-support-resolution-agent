
from app.agent.retrieval import retrieve_policy
from app.agent.semantic_retrieval import retrieve_policy_semantic


REFUND = "REFUND-DEMO-001"
DAMAGE = "POLICY-DAMAGED_ITEM_POLICY"

EVALUATION_CASES = [
    {
        "id": "ret_001",
        "query": "I want a refund and need human approval",
        "expected_sources": [REFUND],
    },
    {
        "id": "ret_002",
        "query": "My returned item has not been received",
        "expected_sources": [REFUND],
    },
    {
        "id": "ret_003",
        "query": "My package arrived broken",
        "expected_sources": [DAMAGE],
    },
    {
        "id": "ret_004",
        "query": "The product is defective and needs human review",
        "expected_sources": [DAMAGE],
    },
    {
        "id": "ret_005",
        "query": "My item arrived damaged and I want a refund",
        "expected_sources": [REFUND, DAMAGE],
    },
    {
        "id": "ret_006",
        "query": "Explain quantum physics and black holes",
        "expected_sources": [],
    },
    {
        "id": "ret_007",
        "query": "Tell me a joke about the weather",
        "expected_sources": [],
    },
]


def evaluate_retriever(name, retriever):
    supported = [
        case for case in EVALUATION_CASES
        if case["expected_sources"]
    ]
    unsupported = [
        case for case in EVALUATION_CASES
        if not case["expected_sources"]
    ]

    source_hits = 0
    expected_source_total = sum(
        len(case["expected_sources"]) for case in supported
    )
    irrelevant_results = 0
    total_results_for_supported = 0
    unsupported_false_positives = 0

    print(f"\n=== {name} ===")

    for case in EVALUATION_CASES:
        results = retriever(case["query"], top_k=3)
        sources = [item["source"] for item in results]
        expected = case["expected_sources"]

        matched = sum(
            1 for source in set(expected)
            if source in sources
        )
        source_hits += matched

        if expected:
            passed = all(source in sources for source in expected)
            irrelevant = sum(
                1 for source in sources if source not in expected
            )
            irrelevant_results += irrelevant
            total_results_for_supported += len(sources)
        else:
            passed = not results
            if results:
                unsupported_false_positives += 1

        print(
            f'{case["id"]}: {"PASS" if passed else "FAIL"}'
            f" | expected={expected or None}"
            f" | retrieved={sources}"
        )

    hit_rate = source_hits / expected_source_total
    false_positive_rate = (
        unsupported_false_positives / len(unsupported)
        if unsupported else 0.0
    )
    irrelevant_rate = (
        irrelevant_results / total_results_for_supported
        if total_results_for_supported else 0.0
    )

    print(f"Expected-source hit rate: {hit_rate:.1%}")
    print(f"Unsupported-query false-positive rate: {false_positive_rate:.1%}")
    print(f"Irrelevant-result rate on supported queries: {irrelevant_rate:.1%}")

    return {
        "hit_rate": hit_rate,
        "false_positive_rate": false_positive_rate,
        "irrelevant_rate": irrelevant_rate,
    }


def main():
    results = {
        "Keyword": evaluate_retriever(
            "Keyword retrieval",
            retrieve_policy,
        ),
        "Semantic": evaluate_retriever(
            "Semantic retrieval",
            retrieve_policy_semantic,
        ),
    }

    print("\n=== Comparison ===")
    for name, metrics in results.items():
        print(
            f'{name}: hit={metrics["hit_rate"]:.1%}, '
            f'unsupported false positives={metrics["false_positive_rate"]:.1%}, '
            f'irrelevant results={metrics["irrelevant_rate"]:.1%}'
        )


if __name__ == "__main__":
    main()
