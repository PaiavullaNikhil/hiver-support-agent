from .config import (
    INTENT_MARGIN_THRESHOLD,
    RETRIEVAL_SIMILARITY_THRESHOLD,
)


VAGUE_PATTERNS = [
    "same issue",
    "same problem",
    "having the same",
    "please help",
    "help me",
    "what is going on",
    "what's going on",
]


ACCOUNT_SPECIFIC_INTENTS = {
    "account_authentication",
    "device_hardware",
}


def decide_escalation(
    intent: str,
    classifier_margin: float,
    top_similarity: float,
    customer_message: str,
) -> dict:
    """
    Decide whether a customer request should be escalated.

    Escalation is deterministic rather than delegated to the LLM.
    """

    customer_message = str(customer_message).strip().lower()

    reasons = []

    # 1. Very short or vague requests
    is_vague = any(
        pattern in customer_message
        for pattern in VAGUE_PATTERNS
    )

    if is_vague:
        reasons.append(
            "The customer message does not provide enough specific information."
        )

    # 2. Classifier could not identify a specific intent
    if intent == "other_unclear":
        reasons.append(
            "The customer's issue could not be confidently mapped "
            "to a specific support intent."
        )

    # 3. Low classifier confidence
    if classifier_margin < INTENT_MARGIN_THRESHOLD:
        reasons.append(
            "The intent classifier is not sufficiently confident."
        )

    # 4. Weak historical evidence
    if top_similarity < RETRIEVAL_SIMILARITY_THRESHOLD:
        reasons.append(
            "No sufficiently similar historical support case was retrieved."
        )

    # 5. Issues likely to require customer-specific investigation
    if intent in ACCOUNT_SPECIFIC_INTENTS:
        reasons.append(
            "The issue may require account- or device-specific investigation."
        )

    escalate = len(reasons) > 0

    if escalate:
        escalation_reason = " ".join(reasons)
    else:
        escalation_reason = (
            "The issue has sufficient intent confidence and relevant "
            "historical support evidence for automated handling."
        )

    return {
        "escalate": escalate,
        "escalation_reason": escalation_reason,
    }