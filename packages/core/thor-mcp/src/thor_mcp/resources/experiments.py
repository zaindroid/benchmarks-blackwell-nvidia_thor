"""Experiment history resource."""

from __future__ import annotations

from typing import Any, Dict
from urllib.parse import parse_qs, urlparse


def get_experiment_history(uri: str, ctx: Any) -> Dict[str, Any]:
    query = parse_qs(urlparse(uri).query)
    status = query.get("status", [None])[0]
    experiments = ctx.experiments.list(status=status)
    return {"uri": uri, "count": len(experiments), "experiments": experiments}
