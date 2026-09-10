from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from .config import (
    GOLDEN_SET_PATH,
    RANDOM_STATE,
)


# ---------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------

INTENT_DEFINITIONS = {
    "ios_update": (
        "Problems or questions specifically about iOS updates, "
        "update failures, update-related bugs, or iOS version behavior."
    ),

    "battery_charging": (
        "Battery draining, battery health, charging, power consumption, "
        "or unexpected battery percentage changes."
    ),

    "connectivity": (
        "Wi-Fi, cellular/mobile data, Bluetooth, network, "
        "or connection problems."
    ),

    "account_authentication": (
        "Apple ID, login, password, authentication, account access, "
        "subscriptions/account management."
    ),

    "icloud_backup": (
        "iCloud storage, iCloud backup, missing/syncing iCloud data, "
        "or iCloud-related problems."
    ),

    "apps_app_store": (
        "Problems with apps, installing/updating apps, App Store, "
        "app crashes, or application behavior."
    ),

    "audio": (
        "Problems with speakers, microphone, calls audio, sound, "
        "headphones, AirPods, or other audio functionality."
    ),

    "camera_photos": (
        "Camera, taking photos/videos, photo library, photo syncing, "
        "or photo-related problems."
    ),

    "display_input": (
        "Screen/display, touchscreen, keyboard, typing, text input, "
        "or visual/input interaction problems."
    ),

    "apple_services": (
        "Problems with Apple services such as Apple Music or other "
        "Apple-provided services that do not fit another intent."
    ),

    "device_hardware": (
        "Physical/device-level problems that cannot be better classified "
        "into a more specific intent."
    ),

    "other_unclear": (
        "Too vague, insufficient context, unrelated, or impossible "
        "to confidently assign to another intent."
    ),
}


VALID_INTENTS = list(INTENT_DEFINITIONS.keys())


# ---------------------------------------------------------
# Build classifier input
# ---------------------------------------------------------

def build_model_text(
    customer_message: str,
    previous_message: str = "",
) -> str:
    """
    Build the text representation used by the classifier.

    Previous conversational context is included only when available.
    """

    customer_message = str(customer_message).strip()
    previous_message = str(previous_message).strip()

    if previous_message:
        return (
            f"Previous message: {previous_message} "
            f"Customer message: {customer_message}"
        )

    return customer_message


# ---------------------------------------------------------
# Build final classifier
# ---------------------------------------------------------

def build_intent_classifier() -> Pipeline:
    """
    Create the final word + character TF-IDF LinearSVC model.
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

    classifier = Pipeline([
        (
            "features",
            FeatureUnion([
                ("word", word_features),
                ("char", char_features),
            ]),
        ),
        (
            "classifier",
            LinearSVC(
                class_weight="balanced",
                C=1.0,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    return classifier


# ---------------------------------------------------------
# Load labeled golden set
# ---------------------------------------------------------

def load_labeled_data(
    path: Path = GOLDEN_SET_PATH,
) -> pd.DataFrame:
    """
    Load the 200 reviewed examples used to train the final classifier.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Golden dataset not found: {path}"
        )

    data = pd.read_csv(path)

    required_columns = {
        "customer_tweet_id",
        "customer_message",
        "previous_message",
        "has_context",
        "intent",
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return data


# ---------------------------------------------------------
# Train final model
# ---------------------------------------------------------

def train_intent_classifier(
    data: pd.DataFrame | None = None,
) -> Pipeline:
    """
    Train the final intent classifier on the reviewed examples.

    If data is not supplied, the golden evaluation set is loaded.
    """

    if data is None:
        data = load_labeled_data()

    data = data.copy()

    data["model_text"] = data.apply(
        lambda row: build_model_text(
            row["customer_message"],
            row["previous_message"],
        ),
        axis=1,
    )

    if data["intent"].isna().any():
        raise ValueError(
            "Training data contains missing intent labels."
        )

    invalid_intents = set(data["intent"]) - set(VALID_INTENTS)

    if invalid_intents:
        raise ValueError(
            f"Unknown intent labels: {sorted(invalid_intents)}"
        )

    model = build_intent_classifier()

    model.fit(
        data["model_text"],
        data["intent"],
    )

    return model


# ---------------------------------------------------------
# Predict intent
# ---------------------------------------------------------

def predict_intent(
    model: Pipeline,
    customer_message: str,
    previous_message: str = "",
) -> str:
    """
    Predict the customer's primary support intent.
    """

    model_text = build_model_text(
        customer_message,
        previous_message,
    )

    return model.predict([model_text])[0]


# ---------------------------------------------------------
# Predict intent + confidence
# ---------------------------------------------------------

def predict_intent_with_confidence(
    model: Pipeline,
    customer_message: str,
    previous_message: str = "",
) -> dict:
    """
    Predict intent and return the LinearSVC decision margin.

    Margin = top decision score - second-highest decision score.

    A larger margin indicates stronger separation between the
    predicted intent and the next-best intent.
    """

    model_text = build_model_text(
        customer_message,
        previous_message,
    )

    predicted_intent = model.predict(
        [model_text]
    )[0]

    decision_scores = model.decision_function(
        [model_text]
    )[0]

    sorted_scores = np.sort(
        decision_scores
    )[::-1]

    top_score = float(sorted_scores[0])
    second_score = float(sorted_scores[1])

    margin = top_score - second_score

    return {
        "intent": predicted_intent,
        "margin": float(margin),
        "top_score": top_score,
        "second_score": second_score,
    }


# ---------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------

def train_final_model() -> Pipeline:
    """
    Load the reviewed dataset and train the final classifier.
    """

    return train_intent_classifier()