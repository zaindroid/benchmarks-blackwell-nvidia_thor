"""Hardware telemetry resource."""

from __future__ import annotations

from typing import Any, Dict

from thor_sdk.telemetry import TelemetryCollector


def get_hardware_telemetry(uri: str, ctx: Any) -> Dict[str, Any]:
    status = ctx.device.status().to_dict()
    host = TelemetryCollector().collect()
    return {
        "uri": uri,
        "device": status,
        "host": host,
    }
