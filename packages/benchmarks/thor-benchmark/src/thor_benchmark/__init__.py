"""thor-benchmark — benchmark orchestrator and workloads for NVIDIA Thor.

Entry points:

* :class:`thor_benchmark.runner.BenchmarkRunner` — programmatic API
* ``thor-benchmark`` CLI — ``run`` / ``list-workloads`` / ``hardware``

Workloads load models lazily (torch/ultralytics/transformers) so the
package imports and tests cleanly on machines without a GPU. Pass
``simulate=True`` to the runner to run deterministic synthetic
benchmarks for development and CI.
"""

__version__ = "0.1.0"
