"""Hardware status tool."""

from __future__ import annotations

from typing import Any, Dict, List

SPECS: List[Dict[str, Any]] = [
    {
        "name": "hardware_status",
        "description": "Get current hardware status of the Thor device",
        "properties": {},
        "required": [],
    },
]

HANDLERS: Dict[str, Any] = {}


async def hardware_status(args: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    status = ctx.device.status()
    data = status.to_dict()
    data["device_ip"] = ctx.config.hardware.device_ip
    if not status.gpu_available:
        data["note"] = (
            "No GPU detected. Benchmarks will run in simulate mode "
            "(pass custom_config.simulate=true to benchmark_run)."
        )
    return {"status": "ok", **data}


HANDLERS.update({"hardware_status": hardware_status})
