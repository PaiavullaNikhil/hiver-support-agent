import json

from google import genai

from .config import GEMINI_API_KEY, GEMINI_MODEL
from .escalation import decide_escalation
from .intent_classifier import (
    predict_intent_with_confidence,
    train_final_model,
)
from .retrieval import build_retriever


class SupportAgent:
    """
    AppleSupport AI support agent.

    Pipeline:
        customer message
            ↓
        intent classification
            ↓
        historical retrieval
            ↓
        grounded LLM response
            ↓
        deterministic escalation
    """

    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not configured. "
                "Add it to the project's .env file."
            )

        self.intent_model = None
        self.retriever = None

        self.gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

    def initialize(self):
        """Initialize models and retrieval index once."""

        if self.intent_model is None:
            print("Training intent classifier...")
            self.intent_model = train_final_model()

        if self.retriever is None:
            print("Loading historical retrieval index...")
            self.retriever = build_retriever()

        print("Agent initialized.")

    def build_prompt(
        self,
        customer_message: str,
        previous_message: str,
        intent: str,
        retrieved_cases: list[dict],
    ) -> str:
        """Build the grounded LLM prompt."""

        evidence = []

        for case in retrieved_cases:
            evidence.append(
                {
                    "similarity": round(
                        case["similarity"],
                        4,
                    ),
                    "customer_message": case[
                        "customer_message"
                    ],
                    "brand_response": case[
                        "brand_response"
                    ],
                }
            )

        evidence_text = json.dumps(
            evidence,
            ensure_ascii=False,
            indent=2,
        )

        return f"""
You are an AI support agent for AppleSupport.

Draft a concise and helpful customer-support reply.

Use only:
1. The customer's message.
2. The previous conversation message, if provided.
3. The predicted intent.
4. Patterns supported by the historical AppleSupport responses.

Do not invent:
- company policies
- refunds
- guarantees
- troubleshooting steps unsupported by the evidence
- product capabilities
- URLs
- account actions

Do not copy usernames, tweet IDs, or social-media metadata.

Synthesize the historical responses rather than copying them verbatim.

If historical responses request information such as the device model,
iOS version, or moving the conversation to DM, you may make the same
request when relevant.

Keep the reply concise and natural.

CUSTOMER MESSAGE:
{customer_message}

PREVIOUS MESSAGE:
{previous_message}

PREDICTED INTENT:
{intent}

HISTORICAL EVIDENCE:
{evidence_text}

Return valid JSON with exactly:

{{
    "draft_reply": "string",
    "evidence_used": ["string"]
}}
""".strip()

    def generate_reply(
        self,
        customer_message: str,
        previous_message: str,
        intent: str,
        retrieved_cases: list[dict],
    ) -> dict:
        """Generate a grounded reply using Gemini."""

        prompt = self.build_prompt(
            customer_message,
            previous_message,
            intent,
            retrieved_cases,
        )

        response = self.gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
            },
        )

        try:
            result = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Gemini returned invalid JSON."
            ) from exc

        if "draft_reply" not in result:
            raise ValueError(
                "Gemini response does not contain draft_reply."
            )

        result.setdefault("evidence_used", [])

        return result

    def run(
        self,
        customer_message: str,
        previous_message: str = "",
    ) -> dict:
        """Run the complete support-agent pipeline."""

        customer_message = str(
            customer_message
        ).strip()

        previous_message = str(
            previous_message
        ).strip()

        if not customer_message:
            raise ValueError(
                "customer_message cannot be empty."
            )

        self.initialize()

        # -----------------------------------------
        # 1. Intent classification
        # -----------------------------------------

        prediction = predict_intent_with_confidence(
            self.intent_model,
            customer_message,
            previous_message,
        )

        intent = prediction["intent"]
        classifier_margin = prediction["margin"]

        # -----------------------------------------
        # 2. Historical retrieval
        # -----------------------------------------

        retrieved_cases = self.retriever.retrieve(
            customer_message,
            top_k=5,
        )

        top_similarity = (
            retrieved_cases[0]["similarity"]
            if retrieved_cases
            else 0.0
        )

        # -----------------------------------------
        # 3. Generate grounded reply
        # -----------------------------------------

        llm_result = self.generate_reply(
            customer_message,
            previous_message,
            intent,
            retrieved_cases,
        )

        # -----------------------------------------
        # 4. Deterministic escalation
        # -----------------------------------------

        escalation = decide_escalation(
            intent=intent,
            classifier_margin=classifier_margin,
            top_similarity=top_similarity,
            customer_message=customer_message,
        )

        # -----------------------------------------
        # 5. Return final result
        # -----------------------------------------

        return {
            "intent": intent,
            "classifier_margin": round(
                classifier_margin,
                4,
            ),
            "top_similarity": round(
                top_similarity,
                4,
            ),
            "draft_reply": llm_result[
                "draft_reply"
            ],
            "evidence_used": llm_result.get(
                "evidence_used",
                [],
            ),
            "escalate": escalation[
                "escalate"
            ],
            "escalation_reason": escalation[
                "escalation_reason"
            ],
            "retrieved_cases": retrieved_cases,
        }


def create_agent() -> SupportAgent:
    """Create a configured support agent."""

    return SupportAgent()