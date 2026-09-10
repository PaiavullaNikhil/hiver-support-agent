import json

from .config import (
    LLM_EVALUATION_PATH,
    HUMAN_REVIEW_PATH,
)


def main():
    from .llm_judge import (
        summarize_judgments,
        evaluate_existing_judgments,
    )

    print("=" * 70)
    print("LLM-AS-JUDGE EVALUATION")
    print("=" * 70)

    judge_summary = summarize_judgments(
        LLM_EVALUATION_PATH
    )

    agreement = evaluate_existing_judgments(
        LLM_EVALUATION_PATH,
        HUMAN_REVIEW_PATH,
    )

    print("\nLLM JUDGE SUMMARY")
    print("-" * 70)

    print(
        f"Examples              : {judge_summary['examples']}"
    )
    print(
        f"Mean relevance        : "
        f"{judge_summary['relevance_mean']:.3f}/5"
    )
    print(
        f"Mean grounding        : "
        f"{judge_summary['grounding_mean']:.3f}/5"
    )
    print(
        f"Mean helpfulness      : "
        f"{judge_summary['helpfulness_mean']:.3f}/5"
    )
    print(
        f"Mean overall quality  : "
        f"{judge_summary['overall_quality_mean']:.3f}/5"
    )
    print(
        f"Unsupported claim rate: "
        f"{judge_summary['unsupported_claim_rate']:.1%}"
    )

    print("\nHUMAN VS LLM-JUDGE AGREEMENT")
    print("-" * 70)

    for dimension, values in agreement["agreement"].items():
        print(
            f"{dimension:20s} "
            f"exact={values['exact_agreement']:.1%} "
            f"MAD={values['mean_absolute_difference']:.2f}"
        )

    output = {
        "judge_summary": judge_summary,
        "human_judge_agreement": agreement,
    }

    output_path = (
        LLM_EVALUATION_PATH.parent
        / "llm_judge_evaluation_results.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(
        f"\nResults saved to: {output_path}"
    )


if __name__ == "__main__":
    main()