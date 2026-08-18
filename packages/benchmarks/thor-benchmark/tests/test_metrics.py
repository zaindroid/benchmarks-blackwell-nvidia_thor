"""Tests for thor_benchmark.metrics helpers."""

import pytest

from thor_benchmark.metrics import (
    energy_joules,
    joules_per_sample,
    memory_summary,
    power_summary,
    samples_per_second,
    tokens_per_second,
)


def test_samples_per_second():
    assert samples_per_second(100, 2.0) == 50.0
    assert samples_per_second(100, 0.0) == 0.0


def test_tokens_per_second():
    assert tokens_per_second(256, 8.0) == 32.0


def test_power_summary():
    summary = power_summary([100.0, 200.0, 300.0])
    assert summary["average_watts"] == 200.0
    assert summary["peak_watts"] == 300.0
    assert summary["idle_watts"] == 100.0
    assert summary["available"] is True


def test_power_summary_empty():
    assert power_summary([])["available"] is False


def test_energy():
    assert energy_joules(250.0, 2.0) == 500.0
    assert joules_per_sample(250.0, 2.0, 100) == 5.0
    assert joules_per_sample(250.0, 2.0, 0) == 0.0


def test_memory_summary():
    summary = memory_summary([100.0, 200.0, 300.0])
    assert summary["peak_mb"] == 300.0
    assert summary["average_mb"] == 200.0
    assert summary["available"] is True
    assert len(summary["allocation_pattern"]) == 3
