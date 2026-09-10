from src.escalation import decide_escalation


def test_high_confidence_specific_case_does_not_escalate():
    result = decide_escalation(
        intent="battery_charging",
        classifier_margin=0.9,
        top_similarity=0.9,
        customer_message=(
            "My iPhone battery is draining very quickly."
        ),
    )

    assert result["escalate"] is False


def test_low_confidence_case_escalates():
    result = decide_escalation(
        intent="battery_charging",
        classifier_margin=0.1,
        top_similarity=0.9,
        customer_message=(
            "My battery is draining."
        ),
    )

    assert result["escalate"] is True
    assert (
        "classifier" in result["escalation_reason"].lower()
    )


def test_vague_message_escalates():
    result = decide_escalation(
        intent="apple_services",
        classifier_margin=0.9,
        top_similarity=0.9,
        customer_message="Having the same issue",
    )

    assert result["escalate"] is True


def test_unclear_intent_escalates():
    result = decide_escalation(
        intent="other_unclear",
        classifier_margin=0.9,
        top_similarity=0.9,
        customer_message=(
            "Something is wrong with my phone."
        ),
    )

    assert result["escalate"] is True


def test_account_issues_escalate():
    result = decide_escalation(
        intent="account_authentication",
        classifier_margin=0.9,
        top_similarity=0.9,
        customer_message=(
            "I cannot access my Apple ID."
        ),
    )

    assert result["escalate"] is True