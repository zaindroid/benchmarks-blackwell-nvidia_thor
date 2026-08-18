"""Device communication for NVIDIA DRIVE Thor.

The Thor device is reached locally (same host) in the MVP. Remote
transport (SSH) is scaffolded and can be enabled with the ``ssh`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from thor_core.config import HardwareConfig
from thor_core.hardware import HardwareMonitor, detect_hardware
from thor_core.logging import get_logger

from thor_sdk.telemetry import TelemetryCollector

logger = get_logger(__name__)


@dataclass
class DeviceStatus:
    """Snapshot of device state for the hardware.status tool."""

    connected: bool
    device: str
    gpu_available: bool
    gpu_name: Optional[str]
    gpu_utilization_pct: Optional[float]
    gpu_temp_c: Optional[float]
    power_watts: Optional[float]
    memory_used_mb: Optional[float]
    memory_total_mb: Optional[float]
    cpu_utilization_pct: float
    driver_version: Optional[str]
    cuda_version: Optional[str]
    tensorrt_version: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "connected": self.connected,
            "device": self.device,
            "gpu": {
                "available": self.gpu_available,
                "name": self.gpu_name,
                "utilization_pct": self.gpu_utilization_pct,
                "temp_c": self.gpu_temp_c,
                "power_watts": self.power_watts,
                "memory_used_mb": self.memory_used_mb,
                "memory_total_mb": self.memory_total_mb,
            },
            "cpu": {"utilization_pct": self.cpu_utilization_pct},
            "software": {
                "driver_version": self.driver_version,
                "cuda_version": self.cuda_version,
                "tensorrt_version": self.tensorrt_version,
            },
        }


class ThorDevice:
    """Handle to a Thor device (local in the MVP)."""

    def __init__(self, config: Optional[HardwareConfig] = None,
                 host: Optional[str] = None, username: Optional[str] = None):
        self.config = config or HardwareConfig()
        self.host = host or self.config.device_ip
        self.username = username
        self._transport: Optional[Any] = None
        self.connected = False
        self._monitor = HardwareMonitor(cuda_device=self.config.cuda_device)

    def connect(self) -> bool:
        """Connect to the device.

        Local mode always succeeds. Remote SSH mode is enabled when
        ``username`` is provided and the ``ssh`` extra is installed.
        """
        if self.username:
            self._transport = self._connect_ssh()
        else:
            # Local device: hardware probe acts as the connection check.
            info = detect_hardware(self.config.cuda_device)
            self.connected = True
            logger.info("connected to local Thor device", gpu_available=info.gpu_available)
        return self.connected

    def _connect_ssh(self) -> Any:  # pragma: no cover - needs optional dep + remote
        try:
            import paramiko  # type: ignore

            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(self.host, username=self.username)
            self.connected = True
            return client
        except ImportError:
            raise RuntimeError(
                "Remote SSH transport requires the 'ssh' extra: "
                "pip install thor-sdk[ssh]"
            ) from None

    def status(self) -> DeviceStatus:
        """Return a one-shot status snapshot of the device."""
        if not self.connected:
            self.connect()
        info = detect_hardware(self.config.cuda_device)
        snapshot = self._monitor.read_snapshot()
        telemetry = TelemetryCollector().collect()
        return DeviceStatus(
            connected=self.connected,
            device=self.config.device,
            gpu_available=info.gpu_available,
            gpu_name=info.gpu_name,
            gpu_utilization_pct=info.gpu_utilization_pct,
            gpu_temp_c=info.gpu_temp_c,
            power_watts=snapshot.get("power_watts"),
            memory_used_mb=snapshot.get("memory_used_mb"),
            memory_total_mb=info.memory_total_mb,
            cpu_utilization_pct=telemetry["cpu"]["utilization_pct"],
            driver_version=info.driver_version,
            cuda_version=info.cuda_version,
            tensorrt_version=info.tensorrt_version,
        )

    def close(self) -> None:
        if self._transport is not None:
            try:
                self._transport.close()
            except Exception:  # pragma: no cover
                pass
            self._transport = None
        self.connected = False
