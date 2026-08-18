"""Tests for thor_core.metrics."""

import statistics

import pytest

from thor_core.metrics import MetricCollector, percentile, summarize_latency


def test_percentile_matches_numpy_default():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 50) == 3.0
    assert percentile(values, 100) == 5.0
    # linear interpolation: index 4 * 0.9 = 3.6 -> 4 + 0.6*(5-4) = 4.6
    assert percentile(values, 90) == pytest.approx(4.6)


def test_percentile_empty():
    assert percentile([], 50) == 0.0


def test_summarize_latency():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    summary = summarize_latency(values)
    assert summary["p50_ms"] == 30.0
    assert summary["min_ms"] == 10.0
    assert summary["max_ms"] == 50.0
    assert summary["std_ms"] == pytest.approx(statistics.stdev(values), rel=1e-3)
    assert summary["count"] == 5


def test_metric_collector_to_dict():
    collector = MetricCollector()
    collector.add_latency([10.0, 20.0, 30.0])
    collector.add_throughput(samples_per_second=100.0, batch_size=4, total_samples=12)
    collector.add_power({"available": True, "average_watts": 250.0}, elapsed_s=1.0)
    collector.add_memory({"available": True, "peak_mb": 2048.0, "average_mb": 1024.0})
    collector.add_thermal({"available": True, "peak_temp_c": 75.0})

    result = collector.to_dict()
    assert result["latency"]["p50_ms"] == 20.0
    assert result["throughput"]["max_batch_size"] == 4
    assert result["power"]["average_watts"] == 250.0
    # 250 W * 1 s / 12 samples
    assert result["power"]["joules_per_sample"] == pytest.approx(250.0 / 12, rel=1e-3)
    assert result["memory"]["peak_mb"] == 2048.0
    assert result["thermal"]["peak_temp_c"] == 75.0


def test_metric_collector_empty_is_valid():
    result = MetricCollector().to_dict()
    assert result["latency"]["p50_ms"] == 0.0
    assert result["power"]["average_watts"] is None
    assert result["memory"]["available"] is False
    assert result["thermal"]["available"] is False
