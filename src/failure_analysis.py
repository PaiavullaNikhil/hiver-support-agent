import pandas as pd

from .config import GOLDEN_SET_PATH
from .evaluation import (
    build_word_char_svm,
    get_oof_predictions,
    prepare_evaluation_data,
    get_confusion_pairs,
)
from .intent_classifier import load_labeled_data

def generate_failure_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate out-of-fold predictions and decision margins while
    preserving the original row alignment.
    """

    X, y = prepare_evaluation_data(data)

    model = build_word_char_svm()

    predictions = [None] * len(data)
    margins = [None] * len(data)

    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=2026,
    )

    for train_idx, test_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]

        model.fit(X_train, y_train)

        fold_predictions = model.predict(X_test)
        scores = model.decision_function(X_test)

        for local_idx, original_idx in enumerate(test_idx):
            sorted_scores = sorted(
                scores[local_idx],
                reverse=True,
            )

            predictions[original_idx] = fold_predictions[local_idx]
            margins[original_idx] = (
                sorted_scores[0] - sorted_scores[1]
            )

    result = data.copy()

    result["predicted_intent"] = predictions
    result["margin"] = margins

    failures = result[
        result["intent"] != result["predicted_intent"]
    ].copy()

    return failures.sort_values("margin")

def save_failure_examples(
    failures: pd.DataFrame,
    output_path,
    n: int = 20,
):
    columns = [
        "customer_tweet_id",
        "customer_message",
        "previous_message",
        "intent",
        "predicted_intent",
        "margin",
    ]

    failures[columns].head(n).to_csv(
        output_path,
        index=False,
    )

def main():
    print("=" * 70)
    print("FAILURE ANALYSIS")
    print("=" * 70)

    data = load_labeled_data(GOLDEN_SET_PATH)

    failures = generate_failure_analysis(data)

    print(f"\nTotal examples       : {len(data)}")
    print(f"Misclassified        : {len(failures)}")
    print(
        f"Classification error : "
        f"{len(failures) / len(data):.1%}"
    )

    print("\nTOP FAILURE EXAMPLES")
    print("-" * 70)

    preview_columns = [
        "customer_tweet_id",
        "customer_message",
        "intent",
        "predicted_intent",
        "margin",
    ]

    print(
        failures[preview_columns]
        .head(10)
        .to_string(index=False)
    )

    output_path = (
        GOLDEN_SET_PATH.parent
        / "failure_examples.csv"
    )

    save_failure_examples(
        failures,
        output_path,
        n=20,
    )

    print(
        f"\nFailure examples saved to: {output_path}"
    )

if __name__ == "__main__":
    main()