"""thor-sense: reference sensor fusion / BEV perception pipeline (scaffold).

Phase 3 of the platform roadmap. This directory will contain:

- configs/cameras.yaml, lidar.yaml, fusion.yaml — sensor configuration
- models/bevformer.py, encoder.py — BEV perception models
- pipeline/sensor_fusion.py, tracker.py — fusion and tracking

The benchmark side already exists: see
packages/benchmarks/thor-benchmark/configs/bevformer.yaml and the
SegmentationBenchmark workload.
"""

__version__ = "0.1.0"
