
from pathlib import Path

import joblib
import numpy as np
from app.agent.workflow import support_graph


INTENT_MODEL_PATH = (
    Path("artifacts")
    / "grouped_baselines"
    / "intent"
    / "model.joblib"
)

CATEGORY_MODEL_PATH = (
    Path("artifacts")
    / "grouped_baselines"
    / "category"
    / "model.joblib"
)


# Hand-authored diagnostic cases. These are not training data.
CASES = [
    {
        "id": "clear_01",
        "message": "I want to get a refund for my order.",
        "expected": "classify",
        "intent": "get_refund",
        "category": "REFUND",
    },
    {
        "id": "clear_02",
        "message": "Please change my shipping address.",
        "expected": "classify",
        "intent": "change_shipping_address",
        "category": "SHIPPING",
    },
    {
        "id": "clear_03",
        "message": "I forgot my password and need to recover it.",
        "expected": "classify",
        "intent": "recover_password",
        "category": "ACCOUNT",
    },
    {
        "id": "clear_04",
        "message": "How can I check the status of my invoice?",
        "expected": "classify",
        "intent": "check_invoice",
        "category": "INVOICE",
    },
    {
        "id": "clear_05",
        "message": "My payment was declined.",
        "expected": "classify",
        "intent": "payment_issue",
        "category": "PAYMENT",
    },
    {
        "id": "clear_06",
        "message": "I would like to cancel my order.",
        "expected": "classify",
        "intent": "cancel_order",
        "category": "CANCEL",
    },
    {
        "id": "clear_07",
        "message": "Can you tell me where my package is?",
        "expected": "classify",
        "intent": "track_order",
        "category": "ORDER",
    },
    {
        "id": "clear_08",
        "message": "I want to delete my account.",
        "expected": "classify",
        "intent": "delete_account",
        "category": "ACCOUNT",
    },
    {
        "id": "clarify_01",
        "message": "I have a problem with my order.",
        "expected": "clarify",
        "intent": None,
        "category": None,
    },
    {
        "id": "clarify_02",
        "message": "I need help with a payment and a refund.",
        "expected": "clarify",
        "intent": None,
        "category": None,
    },
    {
        "id": "clarify_03",
        "message": "Can you help me with something?",
        "expected": "clarify",
        "intent": None,
        "category": None,
    },
    {
        "id": "clarify_04",
        "message": "My package is late and I need to change its address.",
        "expected": "clarify",
        "intent": None,
        "category": None,
    },
    {
        "id": "clarify_05",
        "message": "Something went wrong with my account.",
        "expected": "clarify",
        "intent": None,
        "category": None,
    },
    {
        "id": "clarify_06",
        "message": "I paid but I am not sure what happened to my order.",
        "expected": "clarify",
        "intent": None,
        "category": None,
    },
    {
        "id": "out_01",
        "message": "What will the weather be like tomorrow?",
        "expected": "out_of_scope",
        "intent": None,
        "category": None,
    },
    {
        "id": "out_02",
        "message": "What is the capital of France?",
        "expected": "out_of_scope",
        "intent": None,
        "category": None,
    },
    {
        "id": "out_03",
        "message": "Help me debug a Python function.",
        "expected": "out_of_scope",
        "intent": None,
        "category": None,
    },
    {
        "id": "out_04",
        "message": "Write me a poem about the ocean.",
        "expected": "out_of_scope",
        "intent": None,
        "category": None,
    },
    {
        "id": "out_05",
        "message": "Recommend a movie to watch tonight.",
        "expected": "out_of_scope",
        "intent": None,
        "category": None,
    },
    {
        "id": "out_06",
        "message": "Calculate the square root of 144.",
        "expected": "out_of_scope",
        "intent": None,
        "category": None,
    },
]


def top_prediction(model, message):
    probabilities = model.predict_proba([message])[0]
    classes = model.named_steps["classifier"].classes_
    order = np.argsort(probabilities)[::-1]

    best = int(order[0])
    second = int(order[1])

    return {
        "label": str(classes[best]),
        "probability": float(probabilities[best]),
        "margin": float(probabilities[best] - probabilities[second]),
    }


def main():
    intent_bundle = joblib.load(INTENT_MODEL_PATH)
    category_bundle = joblib.load(CATEGORY_MODEL_PATH)

    intent_model = intent_bundle["model"]
    category_model = category_bundle["model"]

    results = []

    print("=== Routing safety diagnostic ===")
    print(f"Cases: {len(CASES)}")
    print("No thresholds are being selected from this small sample.")

    for case in CASES:
        intent = top_prediction(intent_model, case["message"])
        category = top_prediction(category_model, case["message"])

        # Run the actual LangGraph workflow without an order ID.
        # This evaluates classification behavior without investigating an order.
        workflow_result = support_graph.invoke({
            "customer_id": "cus_001",
            "order_id": None,
            "message": case["message"],
        })

        intent_correct = (
            case["intent"] is not None
            and intent["label"] == case["intent"]
        )
        category_correct = (
            case["category"] is not None
            and category["label"] == case["category"]
        )

        result = {
            **case,
            "predicted_intent": intent["label"],
            "intent_probability": intent["probability"],
            "intent_margin": intent["margin"],
            "predicted_category": category["label"],
            "category_probability": category["probability"],
            "category_margin": category["margin"],
            "intent_correct": intent_correct,
            "category_correct": category_correct,
            "operational_category": workflow_result["category"],
            "workflow_ml_intent": workflow_result["ml_intent_suggestion"],
            "workflow_ml_category": workflow_result["ml_category_suggestion"],
            "workflow_review_required": workflow_result["ml_review_required"],
            "workflow_routing_status": workflow_result["ml_routing_status"],
        }
        results.append(result)

        print(f"\n[{case['id']}] Expected behavior: {case['expected']}")
        print(f"  Message: {case['message']}")
        print(
            f"  Intent: {intent['label']} "
            f"(p={intent['probability']:.3f}, "
            f"margin={intent['margin']:.3f})"
        )
        print(
            f"  Category: {category['label']} "
            f"(p={category['probability']:.3f}, "
            f"margin={category['margin']:.3f})"
        )

        print(
            f"  Operational category: "
            f"{workflow_result['category']}"
        )
        print(
            f"  Workflow ML intent: "
            f"{workflow_result['ml_intent_suggestion']}"
        )
        print(
            f"  Workflow ML category: "
            f"{workflow_result['ml_category_suggestion']}"
        )
        print(
            f"  Review required: "
            f"{workflow_result['ml_review_required']}"
        )

    clear_cases = [r for r in results if r["expected"] == "classify"]
    clarify_cases = [r for r in results if r["expected"] == "clarify"]
    out_cases = [r for r in results if r["expected"] == "out_of_scope"]

    print("\n=== Summary ===")
    print(f"Clear classification cases: {len(clear_cases)}")
    print(
        "Intent accuracy on clear cases: "
        f"{sum(r['intent_correct'] for r in clear_cases)}/{len(clear_cases)}"
    )
    print(
        "Category accuracy on clear cases: "
        f"{sum(r['category_correct'] for r in clear_cases)}/{len(clear_cases)}"
    )

    for name, group in [
        ("Clear", clear_cases),
        ("Clarify", clarify_cases),
        ("Out of scope", out_cases),
    ]:
        print(f"\n{name} cases: {len(group)}")
        print(
            "  Mean intent top probability: "
            f"{np.mean([r['intent_probability'] for r in group]):.3f}"
        )
        print(
            "  Mean category top probability: "
            f"{np.mean([r['category_probability'] for r in group]):.3f}"
        )

    print("\n=== Diagnostic threshold comparison ===")
    print("These are illustrative only—not recommended production thresholds.")

    for threshold in (0.20, 0.30, 0.40, 0.50):
        accepted_out = sum(
            r["intent_probability"] >= threshold for r in out_cases
        )
        accepted_clarify = sum(
            r["intent_probability"] >= threshold for r in clarify_cases
        )
        accepted_clear = sum(
            r["intent_probability"] >= threshold for r in clear_cases
        )

        print(
            f"Threshold {threshold:.2f}: "
            f"clear accepted {accepted_clear}/{len(clear_cases)}, "
            f"ambiguous accepted {accepted_clarify}/{len(clarify_cases)}, "
            f"out-of-scope accepted {accepted_out}/{len(out_cases)}"
        )

    print("\n=== Integrated workflow summary ===")

    operational_categories = sorted({
        r["operational_category"] for r in results
    })

    for category_name in operational_categories:
        count = sum(
            r["operational_category"] == category_name
            for r in results
        )
        print(f"Operational category '{category_name}': {count}/20")

    review_count = sum(
        r["workflow_review_required"] for r in results
    )
    print(f"Cases requiring review: {review_count}/20")

    suggestion_only_count = sum(
        r["workflow_routing_status"] == "suggestion_only"
        for r in results
    )
    print(f"Suggestion-only predictions: {suggestion_only_count}/20")

if __name__ == "__main__":
    main()
