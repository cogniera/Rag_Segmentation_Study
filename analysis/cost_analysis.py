import json
from pathlib import Path


# Cost analysis
def estimate_costs():
    """
    Rough cost comparison between baseline and query-fitted pipelines.
    """

    baseline_path = Path("results/baseline/metrics.json")
    fitted_path = Path("results/querry_fitted/metrics.json")

    if not baseline_path.exists() or not fitted_path.exists():
        return

    with open(baseline_path) as f:
        baseline_metrics = json.load(f)

    with open(fitted_path) as f:
        fitted_metrics = json.load(f)

    summary = {
        "baseline": {},
        "querry_fitted": {},
    }

    for k, metrics in baseline_metrics.items():
        summary["baseline"][k] = {
            "num_chunks": metrics.get("num_chunks", 0),
            "recall": metrics.get("recall", 0.0),
        }

    for k, metrics in fitted_metrics.items():
        summary["querry_fitted"][k] = {
            "num_chunks": metrics.get("num_chunks", 0),
            "recall": metrics.get("recall", 0.0),
        }

    output_path = Path("results/cost_summary.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Cost summary written to {output_path}")


if __name__ == "__main__":
    estimate_costs()