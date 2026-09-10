from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .config import (
    CLEAN_APPLE_PATH,
    GOLDEN_SET_PATH,
    RETRIEVAL_MODEL_NAME,
    RETRIEVAL_TOP_K,
)


# Cached retrieval artifacts
RETRIEVAL_DIR = (
    CLEAN_APPLE_PATH.parent / "retrieval_index"
)

INDEX_PATH = RETRIEVAL_DIR / "apple_support.faiss"
CORPUS_PATH = RETRIEVAL_DIR / "retrieval_corpus.csv"


class HistoricalRetriever:
    """
    Retrieve historically similar AppleSupport interactions.

    Embeddings are normalized, so FAISS inner-product search
    is equivalent to cosine similarity.
    """

    def __init__(
        self,
        corpus: pd.DataFrame,
        model_name: str = RETRIEVAL_MODEL_NAME,
        index=None,
    ):
        required_columns = {
            "customer_tweet_id",
            "customer_message",
            "brand_response",
        }

        missing = required_columns - set(corpus.columns)

        if missing:
            raise ValueError(
                f"Missing required retrieval columns: {sorted(missing)}"
            )

        self.corpus = corpus.reset_index(drop=True).copy()

        self.model = SentenceTransformer(model_name)

        self.index = index

    def build_index(self):
        """
        Generate embeddings and build the FAISS index.

        This is only called when a cached index does not exist.
        """

        texts = (
            self.corpus["customer_message"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        print(
            f"Encoding {len(texts):,} historical messages..."
        )

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32",
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        return self.index

    def retrieve(
        self,
        customer_message: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[dict]:
        """
        Retrieve the most similar historical support cases.
        """

        customer_message = str(
            customer_message
        ).strip()

        if not customer_message:
            return []

        if self.index is None:
            raise RuntimeError(
                "Retrieval index has not been initialized."
            )

        query_embedding = self.model.encode(
            [customer_message],
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for rank, (score, corpus_index) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):
            if corpus_index < 0:
                continue

            row = self.corpus.iloc[int(corpus_index)]

            results.append(
                {
                    "rank": rank,
                    "similarity": float(score),
                    "customer_message": row[
                        "customer_message"
                    ],
                    "brand_response": row[
                        "brand_response"
                    ],
                }
            )

        return results


def build_retrieval_corpus(
    clean_data: pd.DataFrame,
    golden_path: Path = GOLDEN_SET_PATH,
) -> pd.DataFrame:
    """
    Remove golden-set examples from the retrieval corpus.

    This prevents evaluation examples from retrieving themselves.
    """

    if not golden_path.exists():
        raise FileNotFoundError(
            f"Golden set not found: {golden_path}"
        )

    golden = pd.read_csv(golden_path)

    golden_ids = set(
        golden["customer_tweet_id"]
        .astype(str)
    )

    corpus = clean_data[
        ~clean_data["customer_tweet_id"]
        .astype(str)
        .isin(golden_ids)
    ].copy()

    return corpus.reset_index(drop=True)


def build_retriever(
    clean_path: Path = CLEAN_APPLE_PATH,
) -> HistoricalRetriever:
    """
    Load or build the historical retrieval index.

    First run:
        Build embeddings → save FAISS index.

    Later runs:
        Load the saved FAISS index directly.
    """

    RETRIEVAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------
    # Load cached index if available
    # ---------------------------------------------

    if INDEX_PATH.exists() and CORPUS_PATH.exists():

        print("Loading cached retrieval index...")

        corpus = pd.read_csv(
            CORPUS_PATH
        )

        index = faiss.read_index(
            str(INDEX_PATH)
        )

        print(
            f"Loaded retrieval index with "
            f"{index.ntotal:,} cases."
        )

        return HistoricalRetriever(
            corpus=corpus,
            index=index,
        )

    # ---------------------------------------------
    # Otherwise build the index
    # ---------------------------------------------

    if not clean_path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {clean_path}"
        )

    clean_data = pd.read_csv(
        clean_path
    )

    retrieval_corpus = build_retrieval_corpus(
        clean_data
    )

    print(
        f"Retrieval corpus size: "
        f"{len(retrieval_corpus):,}"
    )

    retriever = HistoricalRetriever(
        corpus=retrieval_corpus
    )

    retriever.build_index()

    # Save artifacts
    print("Saving retrieval index...")

    faiss.write_index(
        retriever.index,
        str(INDEX_PATH),
    )

    retrieval_corpus.to_csv(
        CORPUS_PATH,
        index=False,
    )

    print(
        f"Saved index to: {INDEX_PATH}"
    )

    return retriever