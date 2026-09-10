import re
from pathlib import Path

import pandas as pd

from .config import (
    RAW_DATA_PATH,
    CLEAN_APPLE_PATH,
)


# ---------------------------------------------------------
# Low-information message detection
# ---------------------------------------------------------

LOW_INFO_PATTERN = re.compile(
    r"^\s*("
    r"thanks?(\s+(you|a lot|so much))?"
    r"|thank you(\s+(so much|very much))?"
    r"|you'?re welcome"
    r"|yes"
    r"|yeah"
    r"|yep"
    r"|no"
    r"|nope"
    r"|ok"
    r"|okay"
    r"|alright"
    r"|sure"
    r"|same"
    r"|same for me"
    r"|me too"
    r"|got it"
    r"|that worked"
    r"|it worked"
    r"|works"
    r"|please check dm"
    r"|check dm"
    r"|check your dm"
    r"|dm me"
    r"|hello"
    r"|hi"
    r"|hey"
    r"|help"
    r"|i need your help"
    r"|@AppleSupport"
    r")\s*[.!?]*\s*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------
# Raw dataset loading
# ---------------------------------------------------------

def load_raw_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the Customer Support on Twitter dataset.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return pd.read_csv(path)


# ---------------------------------------------------------
# Build AppleSupport conversation pairs
# ---------------------------------------------------------

def build_apple_support_pairs(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build customer -> AppleSupport response pairs.

    Each row represents:
        customer_message -> brand_response

    Previous conversational context is retained when available.
    """

    required_columns = {
        "tweet_id",
        "author_id",
        "inbound",
        "text",
        "in_response_to_tweet_id",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Lookup tables from the full dataset.
    tweet_text = df.set_index("tweet_id")["text"]

    parent_lookup = (
        df.set_index("tweet_id")["in_response_to_tweet_id"]
    )

    author_lookup = (
        df.set_index("tweet_id")["author_id"]
    )

    # AppleSupport replies to customer tweets.
    apple = df[
        (df["author_id"] == "AppleSupport")
        & (df["inbound"] == False)
        & (df["in_response_to_tweet_id"].notna())
    ].copy()

    # The tweet being replied to is the customer message.
    apple["customer_tweet_id"] = (
        apple["in_response_to_tweet_id"]
    )

    apple["customer_message"] = (
        apple["customer_tweet_id"].map(tweet_text)
    )

    # Apple's actual response.
    apple["brand_response"] = apple["text"]

    # Find the customer's parent tweet.
    apple["parent_tweet_id"] = (
        apple["customer_tweet_id"].map(parent_lookup)
    )

    # Previous message, when available.
    apple["previous_message"] = (
        apple["parent_tweet_id"]
        .map(tweet_text)
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Author of the previous message.
    apple["parent_author"] = (
        apple["parent_tweet_id"].map(author_lookup)
    )

    # -----------------------------------------------------
    # Remove low-information messages
    # -----------------------------------------------------

    apple["is_low_info"] = (
        apple["customer_message"]
        .fillna("")
        .str.strip()
        .apply(
            lambda text: bool(
                LOW_INFO_PATTERN.match(text)
            )
        )
    )

    apple = apple[
        ~apple["is_low_info"]
    ].copy()

    # -----------------------------------------------------
    # Remove missing/empty interactions
    # -----------------------------------------------------

    apple = apple[
        apple["customer_message"].notna()
        & apple["customer_message"].str.strip().ne("")
        & apple["brand_response"].notna()
        & apple["brand_response"].str.strip().ne("")
    ].copy()

    # -----------------------------------------------------
    # Remove exact duplicate interactions
    # -----------------------------------------------------

    apple = apple.drop_duplicates(
        subset=[
            "customer_tweet_id",
            "customer_message",
            "brand_response",
        ]
    ).copy()

    # -----------------------------------------------------
    # Context flag
    # -----------------------------------------------------

    apple["has_context"] = (
        apple["previous_message"].ne("")
    )

    # -----------------------------------------------------
    # Keep only fields required downstream
    # -----------------------------------------------------

    apple = apple[
        [
            "customer_tweet_id",
            "customer_message",
            "previous_message",
            "has_context",
            "parent_author",
            "brand_response",
        ]
    ].reset_index(drop=True)

    return apple


# ---------------------------------------------------------
# Save cleaned AppleSupport dataset
# ---------------------------------------------------------

def save_clean_dataset(
    clean_data: pd.DataFrame,
    path: Path = CLEAN_APPLE_PATH,
) -> None:
    """
    Save cleaned AppleSupport interactions.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_data.to_csv(
        path,
        index=False,
    )


# ---------------------------------------------------------
# Load already processed dataset
# ---------------------------------------------------------

def load_clean_dataset(
    path: Path = CLEAN_APPLE_PATH,
) -> pd.DataFrame:
    """
    Load the processed AppleSupport dataset.

    This avoids repeatedly loading the 0.48 GB raw dataset
    during normal application execution.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {path}"
        )

    return pd.read_csv(path)


# ---------------------------------------------------------
# Convenience function
# ---------------------------------------------------------

def prepare_clean_dataset(
    raw_path: Path = RAW_DATA_PATH,
    output_path: Path = CLEAN_APPLE_PATH,
) -> pd.DataFrame:
    """
    Build and save the cleaned AppleSupport dataset.
    """

    df = load_raw_dataset(raw_path)

    clean_data = build_apple_support_pairs(df)

    save_clean_dataset(
        clean_data,
        output_path,
    )

    return clean_data