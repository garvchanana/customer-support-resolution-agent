
import os

from dotenv import load_dotenv

load_dotenv()

from app.agent.workflow import support_graph


CASES = [
    {
        "id": "llm_001",
        "message": "I want a refund for order ord_1001",
    },
    {
        "id": "llm_002",
        "message": (
            "My order ord_1001 arrived damaged "
            "and I want a refund."
        ),
    },
    {
        "id": "llm_003",
        "message": "Can you book a hotel for me?",
    },
]


def main():
    if not os.getenv("GROQ_API_KEY", "").strip():
        raise SystemExit(
            "GROQ_API_KEY is missing. Add it to your local .env file."
        )

    original_flag = os.environ.get("SUPPORT_LLM_ENABLED")

    try:
        os.environ["SUPPORT_LLM_ENABLED"] = "true"

        for case in CASES:
            request = {
                "customer_id": "cus_001",
                "order_id": "ord_1001",
                "message": case["message"],
            }

            # Establish the deterministic baseline first.
            os.environ["SUPPORT_LLM_ENABLED"] = "false"
            baseline = support_graph.invoke(request)

            # Run shadow mode on the same request.
            os.environ["SUPPORT_LLM_ENABLED"] = "true"
            shadow = support_graph.invoke(request)

            baseline_response = baseline.get("response")
            shadow_response = shadow.get("response")
            draft = shadow.get("llm_draft")

            response_unchanged = (
                baseline_response == shadow_response
            )
            outcome_unchanged = (
                baseline.get("outcome") == shadow.get("outcome")
            )

            print(f"\n=== {case['id']} ===")
            print(f"Message: {case['message']}")
            print(f"Baseline outcome: {baseline.get('outcome')}")
            print(f"Shadow outcome:   {shadow.get('outcome')}")
            print(f"LLM draft available: {bool(draft)}")
            print(f"Response unchanged: {response_unchanged}")
            print(f"Outcome unchanged:  {outcome_unchanged}")

            if draft:
                print(f"LLM draft: {draft}")
            else:
                print(
                    "No draft returned. Check evidence availability, "
                    "configuration, and provider errors."
                )

            if not response_unchanged or not outcome_unchanged:
                raise AssertionError(
                    "Shadow mode changed the deterministic result."
                )

    finally:
        if original_flag is None:
            os.environ.pop("SUPPORT_LLM_ENABLED", None)
        else:
            os.environ["SUPPORT_LLM_ENABLED"] = original_flag


if __name__ == "__main__":
    main()
