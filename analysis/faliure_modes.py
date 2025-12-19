import json
from pathlib import Path


# Failure mode analysis
def analyze_failure_modes():
    """
    Identify retrieval vs reranking failures.

    Retrieval failure: gold chunk not retrieved at all.
    Reranking failure: gold chunk retrieved but ranked poorly after reranking.
    """

    baseline_path = Path("results/baseline/result.json")
    fitted_path = Path("results/querry_fitted/result.json")

    if not baseline_path.exists() or not fitted_path.exists():
        return

    with open(baseline_path) as f:
        baseline = json.load(f)

    with open(fitted_path) as f:
        fitted = json.load(f)

    failures = []

    for pipeline_name, record in [
        ("baseline", baseline),
        ("querry_fitted", fitted),
    ]:
        gold_id = record.get("gold_chunk_id")

        retrieved_ids = record.get("retrieved_chunk_ids", [])
        reranked_ids = record.get("reranked_chunk_ids", [])

        if gold_id is None:
            continue

        # Case 1: retrieval failure
        if gold_id not in retrieved_ids:
            failures.append({
                "pipeline": pipeline_name,
                "failure_type": "retrieval_failure",
                "reason": "gold chunk not retrieved",
            })
            continue

        retrieval_rank = retrieved_ids.index(gold_id)

        # Case 2: reranking failure
        if gold_id not in reranked_ids:
            failures.append({
                "pipeline": pipeline_name,
                "failure_type": "reranking_failure",
                "reason": "gold chunk retrieved but lost during reranking",
                "retrieval_rank": retrieval_rank,
            })
            continue

        reranked_rank = reranked_ids.index(gold_id)

        # Case 3: retrieved but ranked low
        if reranked_rank > 0:
            failures.append({
                "pipeline": pipeline_name,
                "failure_type": "reranking_failure",
                "reason": "gold chunk retrieved but not ranked first",
                "retrieval_rank": retrieval_rank,
                "reranked_rank": reranked_rank,
            })

    output_path = Path("results/failure_modes.json")
    with open(output_path, "w") as f:
        json.dump(failures, f, indent=2)

    print(f"Failure mode analysis written to {output_path}")


if __name__ == "__main__":
    analyze_failure_modes()