import json

from .config import GOLDEN_SET_PATH
from .evaluation import (
    evaluate_baselines,
    evaluate_per_intent,
    get_confusion_pairs,
    get_oof_predictions,
    prepare_evaluation_data,
    build_word_char_svm,
)
from .intent_classifier import load_labeled_data


def main():
    print("=" * 70)
    print("APPLE SUPPORT AGENT - EVALUATION")
    print("=" * 70)

    if not GOLDEN_SET_PATH.exists():
        raise FileNotFoundError(
            f"Golden evaluation set not found: {GOLDEN_SET_PATH}"
        )

    data = load_labeled_data(GOLDEN_SET_PATH)

    print(f"\nGolden examples : {len(data)}")
    print(f"Intents         : {data['intent'].nunique()}")

    X, y = prepare_evaluation_data(data)

    # ---------------------------------------------------------
    # Baseline comparison
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("BASELINE COMPARISON")
    print("-" * 70)

    baseline_results = evaluate_baselines(data)

    print(
        baseline_results[
            [
                "model",
                "accuracy_mean",
                "macro_f1_mean",
                "weighted_f1_mean",
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # Final classifier
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("FINAL CLASSIFIER")
    print("-" * 70)

    final_model = build_word_char_svm()
    y_pred = get_oof_predictions(final_model, X, y)

    per_intent, _ = evaluate_per_intent(
        final_model,
        data,
    )


    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------
    from sklearn.metrics import accuracy_score, f1_score

    accuracy = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro")
    weighted_f1 = f1_score(y, y_pred, average="weighted")

    print(f"\nAccuracy   : {accuracy:.4f}")
    print(f"Macro F1   : {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    # ---------------------------------------------------------
    # Per-intent performance
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("PER-INTENT PERFORMANCE")
    print("-" * 70)

    print(per_intent.to_string(index=False))

    # ---------------------------------------------------------
    # Confusion pairs
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("TOP CONFUSION PAIRS")
    print("-" * 70)

    confusion_pairs = get_confusion_pairs(
        y,
        y_pred,
        top_n=10,
    )

    print(confusion_pairs.to_string(index=False))

    # ---------------------------------------------------------
    # Save machine-readable results
    # ---------------------------------------------------------
    output = {
        "golden_examples": int(len(data)),
        "num_intents": int(data["intent"].nunique()),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "baselines": baseline_results.to_dict(orient="records"),
        "confusion_pairs": confusion_pairs.to_dict(orient="records"),
    }

    output_path = (
        GOLDEN_SET_PATH.parent / "evaluation_results.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()