import os
import time
import re
from typing import List, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# Helpers
def _estimate_tokens(text: str) -> int:
    """
    Rough token estimate.
    This does not need to be exact since it is only
    used to keep prompts reasonably sized.
    """
    return int(1.2 * len(text.split()))


def _query_llm_boundary(
    client: OpenAI,
    prompt: str,
    max_retries: int = 3,
    sleep_seconds: int = 10,
) -> Optional[int]:
    """
    Ask the LLM to identify a paragraph boundary.
    Returns the paragraph index if found, otherwise None.
    """
    system_prompt = (
        "You will receive a document with paragraphs labeled as "
        "'ID X: <text>'. Identify the first paragraph (not the first one) "
        "where the topic clearly shifts. Respond with: Answer: ID X"
    )

    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )

            content = response.choices[0].message.content
            match = re.search(r"Answer:\s*ID\s*(\d+)", content)

            if match:
                return int(match.group(1))

            return None

        except Exception:
            time.sleep(sleep_seconds)

    return None


# Query-fitted chunking
def query_fitted_splitter(
    text: str,
    query: str,
    max_chunk_tokens: int = 600,
    max_llm_calls: int = 20,
) -> List[str]:
    """
    Split a document into chunks using an LLM to suggest
    semantic boundary points.

    This is intentionally conservative and limited in scope.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    # Split document into paragraphs first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if len(paragraphs) < 3:
        return ["\n\n".join(paragraphs)]

    labeled_paragraphs = [
        f"ID {i}: {p}" for i, p in enumerate(paragraphs)
    ]

    boundaries: List[int] = []
    pointer = 0
    llm_calls = 0

    # Iteratively search for topic shifts
    while pointer < len(paragraphs) - 2 and llm_calls < max_llm_calls:
        window: List[str] = []
        token_count = 0
        idx = pointer

        while idx < len(labeled_paragraphs) and token_count < max_chunk_tokens:
            window.append(labeled_paragraphs[idx])
            token_count = _estimate_tokens("\n".join(window))
            idx += 1

        prompt = "\n".join(window)
        boundary = _query_llm_boundary(client, prompt)

        llm_calls += 1

        if boundary is None or boundary <= pointer:
            pointer += 1
        else:
            boundaries.append(boundary)
            pointer = boundary

    boundaries.append(len(paragraphs))

    # Assemble final chunks
    chunks: List[str] = []
    start = 0

    for end in boundaries:
        chunk = "\n\n".join(paragraphs[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end

    return chunks


"""

@misc{duarte2024lumberchunker,
      title={LumberChunker: Long-Form Narrative Document Segmentation}, 
      author={André V. Duarte and João Marques and Miguel Graça and Miguel Freire and Lei Li and Arlindo L. Oliveira},
      year={2024},
      eprint={2406.17526},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.17526}, 
}

"""