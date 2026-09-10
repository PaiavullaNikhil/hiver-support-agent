import json
from pathlib import Path

import pandas as pd
from google import genai

from .config import GEMINI_API_KEY, GEMINI_MODEL


JUDGE_RUBRIC = {
    "relevance": "Does the reply directly address the customer's issue?",
    "grounding": "Is the reply supported by the historical support evidence provided?",
    "helpfulness": "Would the reply be useful to the customer as a next step?",
    "overall_quality": "Considering relevance, grounding, and helpfulness, how good is the reply overall?",
    "unsupported_claim": "Does the reply contain an unsupported factual claim, policy, guarantee, or capability?",
}


def build_judge_prompt(
    customer_message: str,
    previous_message: str,
    predicted_intent: str,
    draft_reply: str,
    historical_cases: list[dict],
) -> str:
    evidence_text = "\n\n".join(
        [
            (
                f"Historical case {i + 1}:\n"
                f"Customer: {case['customer_message']}\n"
                f"AppleSupport response: {case['brand_response']}"
            )
            for i, case in enumerate(historical_cases)
        ]
    )

    return f"""
You are evaluating an AI customer-support reply for AppleSupport.

Do not infer the true intent from any hidden label.
Evaluate only the information provided below.

Customer message:
{customer_message}

Previous message:
{previous_message or "(none)"}

Predicted intent:
{predicted_intent}

Draft reply:
{draft_reply}

Historical support evidence:
{evidence_text}

Score the reply using this rubric.

Relevance (1-5):
{JUDGE_RUBRIC["relevance"]}

Grounding (1-5):
{JUDGE_RUBRIC["grounding"]}

Helpfulness (1-5):
{JUDGE_RUBRIC["helpfulness"]}

Overall quality (1-5):
{JUDGE_RUBRIC["overall_quality"]}

Unsupported claim (0 or 1):
{JUDGE_RUBRIC["unsupported_claim"]}

Return ONLY valid JSON:

{{
  "relevance": 1,
  "grounding": 1,
  "helpfulness": 1,
  "overall_quality": 1,
  "unsupported_claim": 0,
  "short_reason": "brief explanation",
  "major_issue": "none or the main issue"
}}
""".strip()


def judge_response(
    client: genai.Client,
    customer_message: str,
    previous_message: str,
    predicted_intent: str,
    draft_reply: str,
    historical_cases: list[dict],
) -> dict:
    prompt = build_judge_prompt(
        customer_message=customer_message,
        previous_message=previous_message,
        predicted_intent=predicted_intent,
        draft_reply=draft_reply,
        historical_cases=historical_cases,
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "temperature": 0.0,
            "response_mime_type": "application/json",
        },
    )

    result = json.loads(response.text)

    required_fields = {
        "relevance",
        "grounding",
        "helpfulness",
        "overall_quality",
        "unsupported_claim",
        "short_reason",
        "major_issue",
    }

    missing = required_fields - set(result)

    if missing:
        raise ValueError(
            f"Judge response is missing fields: {sorted(missing)}"
        )

    return result


def evaluate_existing_judgments(
    judge_path: Path,
    human_path: Path,
) -> dict:
    """
    Compare existing LLM-judge results with the independent human review.
    No new LLM calls are made.
    """

    judge_df = pd.read_csv(judge_path)
    human_df = pd.read_csv(human_path)

    merged = human_df.merge(
        judge_df,
        on="evaluation_id",
        how="inner",
        suffixes=("_human", "_judge"),
    )

    if merged.empty:
        raise ValueError(
            "No matching evaluation IDs found between human and judge files."
        )

    dimensions = [
        ("relevance", "human_relevance", "judge_relevance"),
        ("grounding", "human_grounding", "judge_grounding"),
        ("helpfulness", "human_helpfulness", "judge_helpfulness"),
        ("overall_quality", "human_overall_quality", "judge_overall_quality"),
        (
            "unsupported_claim",
            "human_unsupported_claim",
            "judge_unsupported_claim",
        ),
    ]

    results = {}

    for name, human_col, judge_col in dimensions:
        exact_match = (
            merged[human_col].astype(float)
            == merged[judge_col].astype(float)
        )

        results[name] = {
            "examples": int(len(merged)),
            "exact_agreement": float(exact_match.mean()),
            "mean_absolute_difference": float(
                (merged[human_col].astype(float)
                 - merged[judge_col].astype(float))
                .abs()
                .mean()
            ),
        }

    return {
        "examples": int(len(merged)),
        "agreement": results,
    }


def summarize_judgments(judge_path: Path) -> dict:
    judge_df = pd.read_csv(judge_path)

    required_columns = {
        "judge_relevance",
        "judge_grounding",
        "judge_helpfulness",
        "judge_overall_quality",
        "judge_unsupported_claim",
    }

    missing = required_columns - set(judge_df.columns)

    if missing:
        raise ValueError(
            f"Judge file is missing columns: {sorted(missing)}"
        )

    return {
        "examples": int(len(judge_df)),
        "relevance_mean": float(judge_df["judge_relevance"].mean()),
        "grounding_mean": float(judge_df["judge_grounding"].mean()),
        "helpfulness_mean": float(judge_df["judge_helpfulness"].mean()),
        "overall_quality_mean": float(
            judge_df["judge_overall_quality"].mean()
        ),
        "unsupported_claim_rate": float(
            judge_df["judge_unsupported_claim"].mean()
        ),
    }