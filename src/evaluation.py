import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from .config import RANDOM_STATE
from .intent_classifier import build_model_text


def prepare_evaluation_data(data: pd.DataFrame):
    """
    Prepare model text and labels for evaluation.
    """

    data = data.copy()

    data["model_text"] = data.apply(
        lambda row: build_model_text(
            row["customer_message"],
            row["previous_message"],
        ),
        axis=1,
    )

    return data["model_text"], data["intent"]


def build_word_char_svm():
    """
    Word + character TF-IDF Linear SVM.

    This is the strongest classifier identified during development.
    """

    word_features = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        max_features=15000,
    )

    char_features = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=20000,
    )

    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("word", word_features),
                        ("char", char_features),
                    ]
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    class_weight="balanced",
                    C=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_tfidf_logistic():
    """TF-IDF + Logistic Regression baseline."""

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_tfidf_svm():
    """Word TF-IDF + Linear SVM baseline."""

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    class_weight="balanced",
                    C=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_complement_nb():
    """TF-IDF + Complement Naive Bayes baseline."""

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                ComplementNB(),
            ),
        ]
    )


def evaluate_model(
    model,
    X,
    y,
    cv=None,
):
    """
    Evaluate a classifier using stratified cross-validation.
    """

    if cv is None:
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=[
            "accuracy",
            "f1_macro",
            "f1_weighted",
        ],
        n_jobs=-1,
    )

    return {
        "accuracy_mean": scores[
            "test_accuracy"
        ].mean(),

        "accuracy_std": scores[
            "test_accuracy"
        ].std(),

        "macro_f1_mean": scores[
            "test_f1_macro"
        ].mean(),

        "macro_f1_std": scores[
            "test_f1_macro"
        ].std(),

        "weighted_f1_mean": scores[
            "test_f1_weighted"
        ].mean(),

        "weighted_f1_std": scores[
            "test_f1_weighted"
        ].std(),
    }


def evaluate_baselines(data: pd.DataFrame):
    """
    Compare the main classifier against simple baselines.
    """

    X, y = prepare_evaluation_data(data)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = {
        "TF-IDF + Linear SVM": build_tfidf_svm(),
        "TF-IDF + Logistic Regression": build_tfidf_logistic(),
        "TF-IDF + Complement NB": build_complement_nb(),
        "Word + Character TF-IDF + Linear SVM":
            build_word_char_svm(),
    }

    results = []

    for name, model in models.items():

        metrics = evaluate_model(
            model,
            X,
            y,
            cv,
        )

        results.append(
            {
                "model": name,
                **metrics,
            }
        )

    # Majority baseline
    majority_accuracy = (
        y.value_counts(normalize=True).iloc[0]
    )

    results.append(
        {
            "model": "Majority Baseline",
            "accuracy_mean": majority_accuracy,
            "accuracy_std": 0.0,
            "macro_f1_mean": np.nan,
            "macro_f1_std": np.nan,
            "weighted_f1_mean": np.nan,
            "weighted_f1_std": np.nan,
        }
    )

    return pd.DataFrame(results)


def get_oof_predictions(
    model,
    X,
    y,
):
    """
    Generate out-of-fold predictions for detailed
    per-intent evaluation.
    """

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    predictions = np.empty(
        len(y),
        dtype=object,
    )

    for train_idx, test_idx in cv.split(X, y):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]

        model.fit(
            X_train,
            y_train,
        )

        predictions[test_idx] = model.predict(
            X_test
        )

    return predictions


def evaluate_per_intent(
    model,
    data: pd.DataFrame,
):
    """
    Generate per-intent precision, recall and F1
    using out-of-fold predictions.
    """

    X, y = prepare_evaluation_data(data)

    predictions = get_oof_predictions(
        model,
        X,
        y,
    )

    report = classification_report(
        y,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    per_intent = []

    for intent in sorted(y.unique()):

        metrics = report[intent]

        per_intent.append(
            {
                "intent": intent,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1-score"],
                "support": metrics["support"],
            }
        )

    return (
        pd.DataFrame(per_intent),
        predictions,
    )


def get_confusion_pairs(
    y_true,
    y_pred,
    top_n=15,
):
    """
    Return the most common misclassification pairs.
    """

    confusion = pd.crosstab(
        pd.Series(
            y_true,
            name="true_intent",
        ),
        pd.Series(
            y_pred,
            name="predicted_intent",
        ),
    )

    pairs = []

    for true_intent in confusion.index:

        for predicted_intent in confusion.columns:

            if true_intent == predicted_intent:
                continue

            count = confusion.loc[
                true_intent,
                predicted_intent,
            ]

            if count > 0:
                pairs.append(
                    {
                        "true_intent": true_intent,
                        "predicted_intent": predicted_intent,
                        "count": int(count),
                    }
                )

    return (
        pd.DataFrame(pairs)
        .sort_values(
            "count",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )