import json
from pathlib import Path


# Sensitivity analysis
def run_sensitivity_analysis():
    """
    Check how recall changes as k increases.
    """

    baseline_path = Path("results/baseline/metrics.json")
    fitted_path = Path("results/querry_fitted/metrics.json")

    if not baseline_path.exists() or not fitted_path.exists():
        return

    with open(baseline_path) as f:
        baseline_metrics = json.load(f)

    with open(fitted_path) as f:
        fitted_metrics = json.load(f)

    ks = sorted(baseline_metrics.keys(), key=int)

    deltas = {}

    for k in ks:
        baseline_recall = baseline_metrics[k]["recall"]
        fitted_recall = fitted_metrics[k]["recall"]

        deltas[k] = fitted_recall - baseline_recall

    output_path = Path("results/sensitivity.json")
    with open(output_path, "w") as f:
        json.dump(deltas, f, indent=2)

    print(f"Sensitivity results written to {output_path}")


if __name__ == "__main__":
    run_sensitivity_analysis()