
import json
from pathlib import Path

from src.embedding import embed_documents, embed_query
from src.retrieval import build_faiss_index, retrieve_top_k
from src.rerank import rerank
from src.chunking.query_fitted import query_fitted_splitter

from dotenv import load_dotenv
load_dotenv()

# Paths
DATA_DIR = Path("data")
RESULTS_DIR = Path("results/query_fitted")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Query-fitted pipeline
def run_query_fitted_pipeline(
    document_path: Path,
    query: str,
    top_k: int = 20,
):
    """
    RAG pipeline where chunk boundaries are influenced by the query.
    """

    # Load document
    text = document_path.read_text(encoding="utf-8")

    # Chunk document using the query
    chunks = query_fitted_splitter(text, query)

    # Embed chunks and build index
    doc_embeddings = embed_documents(chunks)
    index = build_faiss_index(doc_embeddings)

    # Embed query and retrieve candidates
    query_embedding = embed_query(query)
    retrieved_ids, scores = retrieve_top_k(index, query_embedding, top_k)

    retrieved_pairs = [(int(i), chunks[int(i)]) for i in retrieved_ids]
    retrieved_texts = [text for _, text in retrieved_pairs]

    # Rerank retrieved chunks
    reranked = rerank(query, retrieved_texts, top_n=top_k)

    reranked_chunk_ids = [
        retrieved_pairs[idx][0]
        for idx, _, _ in reranked
    ]

    # Record results for analysis
    record = {
        "query": query,
        "pipeline": "query_fitted",
        "num_chunks": len(chunks),
        "retrieved_chunk_ids": retrieved_ids.tolist(),
        "reranked_chunk_ids" : reranked_chunk_ids,
        "reranked_chunks": [
            {"text": text, "score": score}
            for _ , score, text in reranked
        ],
        # Same gold reference as baseline for fair comparison
        "gold_chunk_id": 7,
    }

    return record


if __name__ == "__main__":
    document_path = DATA_DIR / "raw/document.txt"
    query = "How does averaging over photon directions and polarizations in thermal radiation modify the role of the dipole matrix element, and why does this averaging fundamentally change the structure of the final transition rate?"

    result = run_query_fitted_pipeline(
        document_path=document_path,
        query=query,
    )

    output_path = RESULTS_DIR / "result.json"
    
    dump = False;

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    

    print(f"Query-fitted results written to {output_path}")