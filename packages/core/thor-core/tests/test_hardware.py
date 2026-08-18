"""Tests for thor_core.hardware (no GPU required)."""

import time

from thor_core.hardware import HardwareMonitor, detect_hardware


def test_detect_hardware_returns_info():
    info = detect_hardware(cuda_device=0)
    data = info.to_dict()
    assert data["device"] == "NVIDIA DRIVE Thor"
    assert "gpu_available" in data
    # Must not raise and always returns the full schema
    for key in ("gpu_name", "driver_version", "cuda_version", "tensorrt_version",
                "compute_capability", "memory_total_mb"):
        assert key in data


def test_hardware_monitor_start_stop():
    monitor = HardwareMonitor(sample_interval=0.01)
    monitor.start()
    time.sleep(0.05)
    monitor.stop()

    stats = monitor.get_stats()
    assert "hardware" in stats
    assert "power" in stats
    assert "memory" in stats
    assert "thermal" in stats
    # On GPU-less machines these sections are marked unavailable; on a real
    # Thor they carry values. Either way the schema holds.
    for section in ("power", "memory", "thermal"):
        assert "available" in stats[section]
    assert stats["elapsed_s"] >= 0.0


def test_hardware_monitor_snapshots_captured():
    monitor = HardwareMonitor(sample_interval=0.005)
    monitor.start()
    time.sleep(0.05)
    monitor.stop()
    assert len(monitor.get_snapshots()) >= 1
