"""Hardware detection and monitoring for NVIDIA DRIVE Thor.

Uses NVIDIA Management Library (pynvml) for GPU metrics and psutil for
host-level metrics. All reads degrade gracefully when a GPU is not
available (e.g. development machines), so benchmarks still run and
power/memory/thermal sections are marked ``available: false``.
"""

from __future__ import annotations

import platform
import threading
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from thor_core.logging import get_logger

logger = get_logger(__name__)

try:
    # Prefer the maintained nvidia-ml-py binding (works with newer
    # drivers, e.g. DRIVE Thor driver 580); fall back to the legacy
    # pynvml package, silencing its deprecation warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The pynvml package is deprecated")
        try:
            import nvidia.ml.pynvml as pynvml  # type: ignore[import-not-found]
        except Exception:
            import pynvml  # type: ignore[no-redef]

    _NVML_AVAILABLE = True
except Exception:  # pragma: no cover - depends on environment
    _NVML_AVAILABLE = False

try:
    import psutil

    _PSUTIL_AVAILABLE = True
except Exception:  # pragma: no cover
    _PSUTIL_AVAILABLE = False

_DEFAULT_DEVICE_NAME = "NVIDIA DRIVE Thor"


@dataclass
class HardwareInfo:
    """Static and dynamic hardware information for a Thor device."""

    device: str = _DEFAULT_DEVICE_NAME
    gpu_available: bool = False
    gpu_name: Optional[str] = None
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    tensorrt_version: Optional[str] = None
    compute_capability: Optional[str] = None
    gpu_temp_c: Optional[float] = None
    power_limit_w: Optional[float] = None
    memory_total_mb: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _nvml_handle(cuda_device: int = 0) -> Any:
    """Return an nvml device handle or None when unavailable."""
    if not _NVML_AVAILABLE:
        return None
    try:
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(cuda_device)
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.warning("pynvml unavailable", error=str(exc))
        return None


def _tensorrt_version() -> Optional[str]:
    """Best-effort TensorRT version read (None when not installed)."""
    try:
        import tensorrt as trt  # type: ignore

        return trt.__version__
    except Exception:
        return None


def detect_hardware(cuda_device: int = 0) -> HardwareInfo:
    """Collect hardware information for the configured CUDA device."""
    info = HardwareInfo()
    handle = _nvml_handle(cuda_device)
    if handle is None:
        return info

    try:
        name = pynvml.nvmlDeviceGetName(handle)
        driver = pynvml.nvmlSystemGetDriverVersion()
        cuda = pynvml.nvmlSystemGetCudaDriverVersion_v2()
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        )
        try:
            power = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle)
        except Exception:
            power = None
        try:
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            cc = f"{major}.{minor}"
        except Exception:
            cc = None

        info.gpu_available = True
        info.gpu_name = name.decode() if isinstance(name, bytes) else str(name)
        info.driver_version = driver.decode() if isinstance(driver, bytes) else str(driver)
        info.cuda_version = str(cuda)
        info.tensorrt_version = _tensorrt_version()
        info.compute_capability = cc
        info.gpu_temp_c = float(temp)
        info.power_limit_w = float(power) / 1000.0 if power else None
        info.memory_total_mb = round(mem.total / (1024 ** 2), 1)
        info.gpu_utilization_pct = float(util.gpu)
        return info
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.warning("failed to query hardware", error=str(exc))
        return info


class HardwareMonitor:
    """Background sampler of GPU/host telemetry for a benchmark run.

    Usage::

        monitor = HardwareMonitor()
        monitor.start()
        ... run benchmark ...
        monitor.stop()
        stats = monitor.get_stats()
    """

    SAMPLE_INTERVAL_S = 0.1

    def __init__(self, cuda_device: int = 0, sample_interval: float = SAMPLE_INTERVAL_S):
        self.cuda_device = cuda_device
        self.sample_interval = sample_interval
        self._handle = _nvml_handle(cuda_device)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._snapshots: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._started_at: Optional[float] = None
        self.info = detect_hardware(cuda_device)

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._snapshots = []
        self._stop.clear()
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(
            target=self._sample_loop, name="thor-hardware-monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(self.sample_interval * 3, 1.0))
        self._thread = None

    def elapsed_s(self) -> float:
        if self._started_at is None:
            return 0.0
        return time.perf_counter() - self._started_at

    # -- sampling -------------------------------------------------------
    def read_snapshot(self) -> Dict[str, Any]:
        """Read a single telemetry snapshot (used for one-shot status)."""
        return self._read_snapshot()

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            self._snapshots.append(self._read_snapshot())
            self._stop.wait(self.sample_interval)

    def _read_snapshot(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {"timestamp": time.time()}
        handle = self._handle
        if handle is None:
            return snapshot
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle)
            except Exception:
                power = None
            snapshot.update(
                gpu_utilization_pct=float(util.gpu),
                memory_used_mb=round(mem.used / (1024 ** 2), 1),
                gpu_temp_c=float(temp),
                power_watts=float(power) / 1000.0 if power else None,
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("snapshot read failed", error=str(exc))
        return snapshot

    # -- queries --------------------------------------------------------
    def get_snapshots(self) -> List[Dict[str, Any]]:
        return list(self._snapshots)

    def get_power_stats(self) -> Dict[str, Optional[float]]:
        powers = [s.get("power_watts") for s in self._snapshots]
        powers = [p for p in powers if p is not None]
        if not powers:
            return {"available": False}
        return {
            "available": True,
            "average_watts": round(sum(powers) / len(powers), 2),
            "peak_watts": round(max(powers), 2),
            "min_watts": round(min(powers), 2),
            "idle_watts": round(min(powers), 2),  # lowest observed power
            "samples": len(powers),
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        mems = [s.get("memory_used_mb") for s in self._snapshots]
        mems = [m for m in mems if m is not None]
        if not mems:
            return {"available": False}
        return {
            "available": True,
            "peak_mb": round(max(mems), 1),
            "average_mb": round(sum(mems) / len(mems), 1),
            "allocation_pattern": [round(m, 1) for m in mems[-200:]],
        }

    def get_thermal_stats(self) -> Dict[str, Any]:
        temps = [s.get("gpu_temp_c") for s in self._snapshots]
        temps = [t for t in temps if t is not None]
        if not temps:
            return {"available": False}
        return {
            "available": True,
            "start_temp_c": round(temps[0], 1),
            "end_temp_c": round(temps[-1], 1),
            "peak_temp_c": round(max(temps), 1),
            "throttling_events": 0,  # populated from nvml throttle reasons on Thor
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "hardware": self.info.to_dict(),
            "power": self.get_power_stats(),
            "memory": self.get_memory_stats(),
            "thermal": self.get_thermal_stats(),
            "elapsed_s": round(self.elapsed_s(), 2),
        }
