"""Check benchmark results against regression thresholds.

Usage: python tools/scripts/check_regression.py results.json
Exits 0 when within thresholds, 1 when a regression is detected.
"""

import json
import sys
from pathlib import Path

THRESHOLDS = {
    "latency": 10,      # allow 10% latency regression
    "throughput": 5,    # allow 5% throughput regression
    "power": 15,        # allow 15% power regression
}


def check_regression(report_path: str, thresholds: dict) -> None:
    """Compare baseline/candidate metrics in the report against thresholds."""
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    if "baseline" not in report or "candidate" not in report:
        # Single-run report (no comparison): nothing to check.
        print("No baseline/candidate comparison found; skipping regression check.")
        sys.exit(0)

    violations = []
    for metric, threshold in thresholds.items():
        baseline = report["baseline"].get(metric)
        candidate = report["candidate"].get(metric)
        if baseline is None or candidate is None:
            continue

        if metric in ("latency", "power"):
            # Lower is better
            regression = (candidate - baseline) / baseline * 100
            if regression > threshold:
                violations.append({
                    "metric": metric,
                    "regression_pct": regression,
                    "threshold_pct": threshold,
                    "baseline": baseline,
                    "candidate": candidate,
                })
        else:  # throughput: higher is better
            regression = (baseline - candidate) / baseline * 100
            if regression > threshold:
                violations.append({
                    "metric": metric,
                    "regression_pct": regression,
                    "threshold_pct": threshold,
                    "baseline": baseline,
                    "candidate": candidate,
                })

    if violations:
        print(f"Found {len(violations)} regression violations:")
        for v in violations:
            print(f"  - {v['metric']}: {v['regression_pct']:.1f}% regression "
                  f"(baseline: {v['baseline']}, candidate: {v['candidate']})")
        sys.exit(1)
    print("No regression detected")
    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    check_regression(sys.argv[1], THRESHOLDS)
