import os
import cohere
from typing import List, Tuple

from dotenv import load_dotenv
load_dotenv()

co = cohere.Client(os.environ["COHERE_API_KEY"])


def rerank(
    query: str,
    documents: List[str],
    model: str = "rerank-english-v3.0",
    top_n: int = 5,
) -> List[Tuple[str, float, int]]:
    """
    Rerank candidate documents using Cohere cross-encoder.

    Returns:
        List of (document_text, relevance_score)
    """

    if len(documents) == 0:
        return []

    resp = co.rerank(
        model=model,
        query=query,
        documents=documents,
        top_n=min(top_n, len(documents)),
    )

    return [
        (r.index,  r.relevance_score ,documents[r.index])
        for r in resp.results
    ]