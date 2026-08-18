"""InfluxDB time-series writer (schema-compliant telemetry).

Implements the platform telemetry schema (see
website/database/influxdb_schema.md):

* ``hardware_metrics``  — per run: power, temps, memory, utilization
* ``inference_metrics`` — per run/batch: latency, tokens/s, samples/s
* ``system_metrics``    — per device: cpu/disk/network utilization

Optional dependency: ``pip install thor-core[timeseries]``
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from thor_core.config import InfluxDBConfig
from thor_core.logging import get_logger

logger = get_logger(__name__)

try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS

    _INFLUX_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _INFLUX_AVAILABLE = False


def influx_available() -> bool:
    return _INFLUX_AVAILABLE


def run_to_points(run: Dict[str, Any]) -> List["Point"]:
    """Convert a benchmark result (platform schema) into InfluxDB points."""
    if not _INFLUX_AVAILABLE:
        raise RuntimeError(
            "influxdb-client is not installed; install thor-core[timeseries]"
        )

    model = run.get("model", {})
    hardware = run.get("hardware", {})
    results = run.get("results", {})
    workload = run.get("workload", {})
    run_id = run.get("run_id", "run-unknown")
    device_id = hardware.get("gpu_name") or hardware.get("device", "thor")

    points: List[Point] = []
    timestamp = run.get("timestamp")

    # hardware_metrics (one point per run)
    power = results.get("power", {}) or {}
    memory = results.get("memory", {}) or {}
    thermal = results.get("thermal", {}) or {}
    hw_point = Point("hardware_metrics").tag("device_id", device_id) \
        .tag("model_id", model.get("name", "unknown")) \
        .tag("run_id", run_id)
    if power.get("average_watts") is not None:
        hw_point = hw_point.field("power_watts", float(power["average_watts"]))
    if thermal.get("peak_temp_c") is not None:
        hw_point = hw_point.field("gpu_temp_c", float(thermal["peak_temp_c"]))
    if memory.get("average_mb") is not None:
        hw_point = hw_point.field("memory_used_mb", float(memory["average_mb"]))
    if hardware.get("gpu_utilization_pct") is not None:
        hw_point = hw_point.field("gpu_utilization_pct", float(hardware["gpu_utilization_pct"]))
    if timestamp:
        hw_point = hw_point.time(timestamp)
    points.append(hw_point)

    # inference_metrics (one point per batch size)
    latency = results.get("latency", {}) or {}
    throughput = results.get("throughput", {}) or {}
    for batch_size in workload.get("batch_sizes", [1]):
        point = Point("inference_metrics") \
            .tag("run_id", run_id) \
            .tag("batch_size", str(batch_size)) \
            .tag("precision", model.get("precision", "fp16"))
        if latency.get("p50_ms") is not None:
            point = point.field("latency_ms", float(latency["p50_ms"]))
        if throughput.get("samples_per_second") is not None:
            point = point.field("samples_per_second", float(throughput["samples_per_second"]))
        if throughput.get("tokens_per_second"):
            point = point.field("tokens_per_second", float(throughput["tokens_per_second"]))
        if timestamp:
            point = point.time(timestamp)
        points.append(point)

    # system_metrics (host-level, minimal by default)
    sys_point = Point("system_metrics").tag("device_id", device_id)
    if timestamp:
        sys_point = sys_point.time(timestamp)
    points.append(sys_point)

    return points


class TimeSeriesWriter:
    """Writes benchmark telemetry to InfluxDB."""

    def __init__(self, url: str, token: str, org: str, bucket: str):
        if not _INFLUX_AVAILABLE:
            raise RuntimeError(
                "influxdb-client is not installed; install thor-core[timeseries]"
            )
        self.url = url
        self.org = org
        self.bucket = bucket
        self._client = InfluxDBClient(url=url, token=token, org=org)
        self._write_api = self._client.write_api(write_options=SYNCHRONOUS)

    @classmethod
    def from_config(cls, config: InfluxDBConfig | None = None) -> "TimeSeriesWriter":
        cfg = config or InfluxDBConfig(
            url=os.getenv("INFLUXDB_URL", "http://localhost:8086"),
            token=os.getenv("INFLUXDB_TOKEN", ""),
            org=os.getenv("INFLUXDB_ORG", "thor-org"),
            bucket=os.getenv("INFLUXDB_BUCKET", "thor-bucket"),
        )
        return cls(url=cfg.url, token=cfg.token, org=cfg.org, bucket=cfg.bucket)

    def write_run(self, run: Dict[str, Any]) -> int:
        """Write a benchmark run; returns number of points written."""
        points = run_to_points(run)
        if points:
            self._write_api.write(bucket=self.bucket, org=self.org, record=points)
        logger.info("timeseries written", bucket=self.bucket, points=len(points))
        return len(points)

    def close(self) -> None:
        self._client.close()
