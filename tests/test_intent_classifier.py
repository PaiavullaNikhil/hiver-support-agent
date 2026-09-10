import pandas as pd

from src.intent_classifier import (
    build_model_text,
    build_intent_classifier,
    predict_intent,
)


def test_build_model_text_without_context():
    result = build_model_text(
        "Battery is draining",
        "",
    )

    assert result == "Battery is draining"


def test_build_model_text_with_context():
    result = build_model_text(
        "Still happening",
        "My battery is draining",
    )

    assert "Previous message:" in result
    assert "Customer message:" in result
    assert "My battery is draining" in result
    assert "Still happening" in result


def test_classifier_can_train_and_predict():
    data = pd.DataFrame(
        {
            "customer_message": [
                "battery draining quickly",
                "battery dies very fast",
                "wifi is not connecting",
                "bluetooth is not working",
                "apple id password problem",
                "cannot login to apple id",
            ],
            "previous_message": [
                "",
                "",
                "",
                "",
                "",
                "",
            ],
            "intent": [
                "battery_charging",
                "battery_charging",
                "connectivity",
                "connectivity",
                "account_authentication",
                "account_authentication",
            ],
        }
    )

    data["model_text"] = data.apply(
        lambda row: build_model_text(
            row["customer_message"],
            row["previous_message"],
        ),
        axis=1,
    )

    model = build_intent_classifier()

    model.fit(
        data["model_text"],
        data["intent"],
    )

    prediction = predict_intent(
        model,
        "my battery is draining",
    )

    assert prediction in {
        "battery_charging",
        "connectivity",
        "account_authentication",
    }