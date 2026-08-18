"""Tests for thor_sdk (no GPU required)."""

from thor_sdk.device import ThorDevice
from thor_sdk.power import PowerMonitor
from thor_sdk.telemetry import TelemetryCollector


def test_device_local_status():
    device = ThorDevice()
    device.connect()
    assert device.connected is True
    status = device.status()
    data = status.to_dict()
    assert data["connected"] is True
    assert data["device"] == "NVIDIA DRIVE Thor"
    assert "gpu" in data and "cpu" in data and "software" in data
    device.close()
    assert device.connected is False


def test_power_monitor_lifecycle():
    monitor = PowerMonitor()
    monitor.start()
    monitor.stop()
    stats = monitor.get_power_stats()
    assert "available" in stats


def test_telemetry_collector():
    telemetry = TelemetryCollector().collect()
    assert "cpu" in telemetry
    assert "memory" in telemetry
    assert "disk" in telemetry
    assert "network" in telemetry
