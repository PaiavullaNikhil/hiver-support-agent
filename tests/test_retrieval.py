import pandas as pd

from src.retrieval import HistoricalRetriever


def test_retriever_returns_top_k():
    corpus = pd.DataFrame(
        {
            "customer_tweet_id": [
                "1",
                "2",
                "3",
            ],
            "customer_message": [
                "My battery is draining very fast",
                "WiFi is not connecting",
                "My Apple ID password does not work",
            ],
            "brand_response": [
                "Please check your battery settings.",
                "Please troubleshoot your WiFi connection.",
                "Please recover your Apple ID.",
            ],
        }
    )

    retriever = HistoricalRetriever(corpus)

    retriever.build_index()

    results = retriever.retrieve(
        "My battery drains quickly",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["similarity"] >= results[1]["similarity"]
    assert "customer_message" in results[0]
    assert "brand_response" in results[0]