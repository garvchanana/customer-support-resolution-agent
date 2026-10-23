
import json
from pathlib import Path

from app.agent.workflow import support_graph


DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "support_challenge.json"
)


def evaluate():
    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    total = 0
    category_correct = 0
    eligibility_checked = 0
    eligibility_correct = 0
    reason_checked = 0
    reason_correct = 0
    clarification_checked = 0
    clarification_correct = 0

    investigation_checked = 0
    investigation_correct = 0

    approval_checked = 0
    approval_correct = 0

    outcome_checked = 0
    outcome_correct = 0

    for case in cases:
        total += 1

        result = support_graph.invoke({
            "customer_id": case["customer_id"],
            "order_id": case["order_id"],
            "message": case["message"],
        })

        category_ok = (
            result["category"] == case["expected_category"]
        )
        category_correct += int(category_ok)

        expected_eligible = case["expected_eligible"]
        assessment = result.get("refund_assessment", {})

        if expected_eligible is not None:
            eligibility_checked += 1
            eligibility_ok = (
                assessment.get("eligible") == expected_eligible
            )
            eligibility_correct += int(eligibility_ok)
        else:
            eligibility_ok = None

        # Verify human approval requirements when specified.
        expected_requires_approval = case.get(
            "expected_requires_approval"
        )

        if expected_requires_approval is not None:
            approval_checked += 1
            approval_ok = (
                assessment.get("requires_approval")
                == expected_requires_approval
            )
            approval_correct += int(approval_ok)
        else:
            approval_ok = None

        expected_reason = case["expected_reason"]
        actual_reason = assessment.get("reason")

        if expected_reason == "ORDER_NOT_ACCESSIBLE":
            reason_ok = result.get("order_summary") == (
                "Order not found for this customer."
            )
        elif expected_reason == "ORDER_ID_REQUIRED":
            reason_ok = not assessment and (
                "order ID" in result["response"]
            )
        elif expected_reason is not None:
            reason_ok = actual_reason == expected_reason
        else:
            reason_ok = None

        if expected_reason is not None:
            reason_checked += 1
            reason_correct += int(bool(reason_ok))

        # Check clarification behavior when the case specifies it.
        if "expected_needs_clarification" in case:
            clarification_checked += 1

            clarification_ok = (
                result.get("needs_clarification")
                == case["expected_needs_clarification"]
            )
            clarification_correct += int(clarification_ok)
        else:
            clarification_ok = None

        # Verify whether order investigation occurred.
        if "expected_order_investigated" in case:
            investigation_checked += 1

            order_was_investigated = bool(
                result.get("order_summary")
                or result.get("refund_assessment")
            )

            investigation_ok = (
                order_was_investigated
                == case["expected_order_investigated"]
            )
            investigation_correct += int(investigation_ok)

        # Verify the explicit resolution outcome.
        expected_outcome = case.get("expected_outcome")

        if expected_outcome is not None:
            outcome_checked += 1
            actual_outcome = result.get("outcome")
            outcome_ok = actual_outcome == expected_outcome
            outcome_correct += int(outcome_ok)
        else:
            actual_outcome = None
            outcome_ok = None

        print(
            f"{case['case_id']}: "
            f"category={'PASS' if category_ok else 'FAIL'}; "
            f"eligibility={eligibility_ok}; "
            f"approval={approval_ok}; "
            f"reason={reason_ok}; "
            f"outcome={actual_outcome} "
            f"({'PASS' if outcome_ok else 'FAIL' if outcome_ok is not None else 'N/A'})"
        )

    print("\n--- Evaluation summary ---")
    print(f"Cases: {total}")
    print(
        f"Category accuracy: {category_correct}/{total} "
        f"({category_correct / total:.0%})"
    )

    if eligibility_checked:
        print(
            f"Eligibility accuracy: {eligibility_correct}/"
            f"{eligibility_checked} "
            f"({eligibility_correct / eligibility_checked:.0%})"
        )

    if reason_checked:
        print(
            f"Reason/security accuracy: {reason_correct}/"
            f"{reason_checked} "
            f"({reason_correct / reason_checked:.0%})"
        )

    if clarification_checked:
        print(
            f"Clarification accuracy: {clarification_correct}/"
            f"{clarification_checked} "
            f"({clarification_correct / clarification_checked:.0%})"
        )

    if investigation_checked:
        print(
            f"Investigation behavior: {investigation_correct}/"
            f"{investigation_checked} "
            f"({investigation_correct / investigation_checked:.0%})"
        )

    if approval_checked:
        print(
            f"Refund approval checks: {approval_correct}/"
            f"{approval_checked} "
            f"({approval_correct / approval_checked:.0%})"
        )

    if outcome_checked:
        print(
            f"Outcome accuracy: {outcome_correct}/"
            f"{outcome_checked} "
            f"({outcome_correct / outcome_checked:.0%})"
        )


    # Fail the evaluation process if any measured check fails.
    checks = [
        ("Category accuracy", category_correct == total),
    ]

    if eligibility_checked:
        checks.append(
            ("Refund eligibility", eligibility_correct == eligibility_checked)
        )

    if reason_checked:
        checks.append(
            ("Reason/security", reason_correct == reason_checked)
        )

    if clarification_checked:
        checks.append(
            (
                "Clarification behavior",
                clarification_correct == clarification_checked,
            )
        )

    if investigation_checked:
        checks.append(
            (
                "Investigation behavior",
                investigation_correct == investigation_checked,
            )
        )

    if approval_checked:
        checks.append(
            ("Refund approval", approval_correct == approval_checked)
        )

    if outcome_checked:
        checks.append(
            ("Resolution outcomes", outcome_correct == outcome_checked)
        )

    failed_checks = [name for name, passed in checks if not passed]

    if failed_checks:
        print("\nOVERALL RESULT: FAIL")
        print("Failed checks: " + ", ".join(failed_checks))
        raise SystemExit(1)

    print("\nOVERALL RESULT: PASS")


if __name__ == "__main__":
    evaluate()
