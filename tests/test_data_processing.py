import pandas as pd

from src.data_processing import build_apple_support_pairs


def test_build_apple_support_pairs():
    data = pd.DataFrame(
        {
            "tweet_id": [1, 2, 3],
            "author_id": [
                "customer",
                "AppleSupport",
                "customer",
            ],
            "inbound": [
                True,
                False,
                True,
            ],
            "text": [
                "My battery is draining",
                "Please DM us",
                "Thanks",
            ],
            "in_response_to_tweet_id": [
                None,
                1,
                2,
            ],
        }
    )

    result = build_apple_support_pairs(data)

    assert len(result) == 1
    assert result.iloc[0]["customer_message"] == (
        "My battery is draining"
    )
    assert result.iloc[0]["brand_response"] == (
        "Please DM us"
    )