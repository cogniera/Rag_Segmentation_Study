from typing import List, Dict
import json
from pathlib import Path


def recall_ceiling(records: List[Dict], k: int, key: str) -> float:
    """
    Upper bound on recall@k assuming perfect reranking.

    Must be computed over retrieved IDs since reranking
    cannot recover missing chunks.
    """
    if not records:
        return 0.0

    hits = 0
    valid = 0

    for record in records:
        gold_id = record.get("gold_chunk_id")
        if gold_id is None:
            continue

        ids = record.get(key, [])
        valid += 1

        if gold_id in ids[:k]:
            hits += 1

    return hits / valid if valid > 0 else 0.0


if __name__ == "__main__":

    # Configuration
    KS = [1, 5, 10, 20]
    ID_KEY = "retrieved_chunk_ids"

    PIPELINES = {
        "baseline": Path("results/baseline"),
        "querry_fitted": Path("results/query_fitted"),
    }

    for pipeline_name, base_dir in PIPELINES.items():

        result_path = base_dir / "result.json"
        output_path = base_dir / "recall_curve.json"

        if not result_path.exists():
            print(f"[SKIP] {pipeline_name}: result.json not found")
            continue

        # Load records
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Allow both single-record and list-of-records formats
        if isinstance(data, dict):
            records = [data]
        else:
            records = data

        # Compute recall ceiling curve
        curve = {}

        for k in KS:
            curve[k] = recall_ceiling(
                records=records,
                k=k,
                key=ID_KEY,
            )

        # Save results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(curve, f, indent=2)

        print(f"{pipeline_name} recall ceiling written to {output_path}")