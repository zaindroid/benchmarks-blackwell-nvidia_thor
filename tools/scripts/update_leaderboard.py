"""Update the leaderboard from benchmark result files.

Aggregates results/*.json (or a single file passed as argv[1]) into
leaderboard.json with the best score per model/workload/metric.

Usage:
    python tools/scripts/update_leaderboard.py results.json
    python tools/scripts/update_leaderboard.py          # scans results/
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
LEADERBOARD = ROOT / "leaderboard.json"


def _best(existing, metric_key, new_value, lower_is_better):
    if new_value is None:
        return existing
    if existing is None:
        return new_value
    if lower_is_better:
        return min(existing, new_value)
    return max(existing, new_value)


def update(files) -> None:
    entries = {}
    for path in files:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model = data.get("model", {})
        workload = data.get("workload", {})
        results = data.get("results", {})
        key = (model.get("name"), workload.get("type"), model.get("precision", "fp16"))

        latency = results.get("latency", {}).get("p50_ms")
        throughput = results.get("throughput", {}).get("samples_per_second")
        power = results.get("power", {}).get("average_watts")
        memory = results.get("memory", {}).get("peak_mb")

        entry = entries.setdefault(key, {
            "model": key[0], "workload": key[1], "precision": key[2],
            "best_latency_p50_ms": None, "best_throughput_sps": None,
            "best_power_watts": None, "best_memory_peak_mb": None, "runs": 0,
        })
        entry["best_latency_p50_ms"] = _best(
            entry["best_latency_p50_ms"], latency, latency, lower_is_better=True)
        entry["best_throughput_sps"] = _best(
            entry["best_throughput_sps"], throughput, throughput, lower_is_better=False)
        entry["best_power_watts"] = _best(
            entry["best_power_watts"], power, power, lower_is_better=True)
        entry["best_memory_peak_mb"] = _best(
            entry["best_memory_peak_mb"], memory, memory, lower_is_better=True)
        entry["runs"] += 1

    payload = {
        "entries": sorted(
            entries.values(), key=lambda e: (e["best_latency_p50_ms"] is None,
                                             e["best_latency_p50_ms"])
        ),
        "count": len(entries),
    }
    LEADERBOARD.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Leaderboard updated: {len(entries)} entries -> {LEADERBOARD}")


def main() -> None:
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = [str(p) for p in sorted(RESULTS_DIR.glob("*.json"))]
    if not files:
        print("No result files found.")
        sys.exit(1)
    update(files)


if __name__ == "__main__":
    main()
