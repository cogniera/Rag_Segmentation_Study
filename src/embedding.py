import os
from typing import List

import numpy as np
import cohere


# Cohere client
def _get_client() -> cohere.Client:
    api_key = os.environ.get("COHERE_API_KEY")
    if api_key is None:
        raise RuntimeError("COHERE_API_KEY not set")

    return cohere.Client(api_key)


# Document embeddings
def embed_documents(
    texts: List[str],
    model: str = "embed-english-v3.0",
    batch_size: int = 96,
) -> np.ndarray:
    """
    Embed document chunks for retrieval.

    This is batched to stay within API limits and
    to keep memory usage predictable.
    """
    if not texts:
        # Return a consistent empty array to avoid downstream shape issues
        return np.zeros((0, 0), dtype="float32")

    client = _get_client()
    embeddings: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        response = client.embed(
            model=model,
            texts=batch,
            input_type="search_document",
        )

        if len(response.embeddings) != len(batch):
            raise RuntimeError("Embedding count mismatch")

        embeddings.extend(response.embeddings)

    return np.asarray(embeddings, dtype="float32")


# Query embedding
def embed_query(
    query: str,
    model: str = "embed-english-v3.0",
) -> np.ndarray:
    """
    Embed a single query for retrieval.
    """
    if not query.strip():
        raise ValueError("Query must be non-empty")

    client = _get_client()

    response = client.embed(
        model=model,
        texts=[query],
        input_type="search_query",
    )

    if not response.embeddings:
        raise RuntimeError("No embedding returned for query")

    return np.asarray(response.embeddings[0], dtype="float32")