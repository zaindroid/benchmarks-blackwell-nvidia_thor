"""Quickstart: run a benchmark programmatically.

Synthetic (simulate) mode needs no GPU — real runs require a Thor device
with torch/ultralytics installed.
"""

from thor_benchmark.runner import BenchmarkRunner

runner = BenchmarkRunner()

results = runner.run(
    model_id="ultralytics/yolov8n",
    workload_type="vision",
    precision="fp16",
    batch_sizes=[1, 4, 8],
    iterations=100,
    simulate=True,  # set False on a real Thor device
)

print(f"run_id:      {results.run_id}")
print(f"latency p50: {results.results['latency']['p50_ms']:.2f} ms")
print(f"throughput:  {results.results['throughput']['samples_per_second']:.0f} samples/s")
power = results.results["power"]
print(f"power:       {power.get('average_watts', 'n/a')} W")
