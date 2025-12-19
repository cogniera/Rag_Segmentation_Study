from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Static chunking
def static_text_splitter(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Split text into fixed chunks using a recursive splitter.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    raw_chunks = splitter.split_text(text)

    # Clean up edge cases from the splitter
    chunks: List[str] = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

    return chunks