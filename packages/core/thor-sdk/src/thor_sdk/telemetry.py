"""System telemetry collection (CPU, disk, network, host memory)."""

from __future__ import annotations

from typing import Any, Dict

try:
    import psutil

    _PSUTIL_AVAILABLE = True
except Exception:  # pragma: no cover
    _PSUTIL_AVAILABLE = False


class TelemetryCollector:
    """One-shot and streaming host telemetry via psutil."""

    def collect(self) -> Dict[str, Any]:
        """Return a full system telemetry snapshot."""
        if not _PSUTIL_AVAILABLE:  # pragma: no cover
            return {"cpu": {}, "memory": {}, "disk": {}, "network": {}, "available": False}

        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        net = psutil.net_io_counters()

        return {
            "cpu": {
                "utilization_pct": round(cpu, 1),
                "cores": psutil.cpu_count(),
                "load_avg": list(psutil.getloadavg()) if hasattr(psutil, "getloadavg") else None,
            },
            "memory": {
                "used_mb": round(mem.used / (1024 ** 2), 1),
                "total_mb": round(mem.total / (1024 ** 2), 1),
                "utilization_pct": round(mem.percent, 1),
            },
            "disk": {
                "used_mb": round(disk.used / (1024 ** 2), 1),
                "total_mb": round(disk.total / (1024 ** 2), 1),
                "utilization_pct": round(disk.percent, 1),
            },
            "network": {
                "bytes_sent": net.bytes_sent,
                "bytes_recv": net.bytes_recv,
            },
            "available": True,
        }
