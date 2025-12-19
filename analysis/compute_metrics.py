import json
import math
from pathlib import Path
from typing import Dict, List


# Recal at k for a pipeline 
def recall_at_k(record: Dict, k: int, key: str) -> int:
    gold_id = record.get("gold_chunk_id")
    if gold_id is None:
        return 0

    ids: List[int] = record.get(key, [])
    return int(gold_id in ids[:k])

#dcg for a pipeline 
def dcg_at_k(record: Dict, k: int, key: str) -> float:
    gold_id = record.get("gold_chunk_id")
    if gold_id is None:
        return 0.0

    ids: List[int] = record.get(key, [])[:k]
    if gold_id not in ids:
        return 0.0

    rank = ids.index(gold_id)
    return 1.0 / math.log2(rank + 2)

#ndcg for a pipeline 
def ndcg_at_k(record: Dict, k: int, key: str) -> float:
    # single relevant item → ideal DCG = 1
    return dcg_at_k(record, k, key)

if __name__ == "__main__":

    KS = [1, 5, 10, 20]

    PIPELINES = {
        "baseline": Path("results/baseline"),
        "querry_fitted": Path("results/query_fitted"),
    }

    for pipeline_name, base_dir in PIPELINES.items():

        result_path = base_dir / "result.json"
        output_path = base_dir / "metrics.json"

        with open(result_path, "r", encoding="utf-8") as f:
            record = json.load(f)

        metrics = {}

        for k in KS:
            metrics[k] = {
                # Retrieval-only (pre-rerank)
                "retrieval_recall": recall_at_k(
                    record, k, key="retrieved_chunk_ids"
                ),
                "retrieval_ndcg": ndcg_at_k(
                    record, k, key="retrieved_chunk_ids"
                ),

                # End-to-end (post-rerank)
                "reranked_recall": recall_at_k(
                    record, k, key="reranked_chunk_ids"
                ),
                "reranked_ndcg": ndcg_at_k(
                    record, k, key="reranked_chunk_ids"
                ),
            }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"{pipeline_name} metrics written to {output_path}")