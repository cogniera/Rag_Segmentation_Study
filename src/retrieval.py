from typing import Tuple

import numpy as np
import faiss


# Index construction
def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index for cosine similarity search.
    """
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")

    if embeddings.shape[0] == 0:
        raise ValueError("Cannot build FAISS index with no embeddings")

    # FAISS expects float32 and normalized vectors for inner product similarity
    vectors = embeddings.astype("float32")
    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    return index


# Retrieval
def retrieve_top_k(
    index: faiss.Index,
    query_embedding: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve the top-k most similar embeddings from the index.
    """
    if query_embedding.ndim != 1:
        raise ValueError("Query embedding must be a 1D array")

    if k <= 0:
        raise ValueError("k must be positive")

    query = query_embedding.astype("float32").reshape(1, -1)
    faiss.normalize_L2(query)

    # Avoid asking FAISS for more items than exist
    k = min(k, index.ntotal)

    scores, indices = index.search(query, k)

    return indices[0], scores[0]