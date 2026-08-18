"""Power monitoring for Thor devices (pynvml-backed)."""

from __future__ import annotations

from typing import Any, Dict

from thor_core.hardware import HardwareMonitor


class PowerMonitor:
    """Tracks GPU power draw while a workload runs.

    Usage::

        monitor = PowerMonitor()
        monitor.start()
        ... workload ...
        monitor.stop()
        stats = monitor.get_power_stats()   # average/peak/idle watts
    """

    def __init__(self, cuda_device: int = 0):
        self._hw = HardwareMonitor(cuda_device=cuda_device)

    def start(self) -> None:
        self._hw.start()

    def stop(self) -> None:
        self._hw.stop()

    def get_power_stats(self) -> Dict[str, Any]:
        return self._hw.get_power_stats()

    def average_watts(self) -> float:
        return self.get_power_stats().get("average_watts", 0.0) or 0.0

    def peak_watts(self) -> float:
        return self.get_power_stats().get("peak_watts", 0.0) or 0.0
