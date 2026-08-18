"""Model registry resource."""

from __future__ import annotations

from typing import Any, Dict
from urllib.parse import parse_qs, urlparse


def get_model_registry(uri: str, ctx: Any) -> Dict[str, Any]:
    query = parse_qs(urlparse(uri).query)
    models = ctx.registry.list(
        architecture=_first(query, "architecture"),
        optimized=_bool(query, "optimized"),
    )
    return {"uri": uri, "count": len(models), "models": models}


def _first(query: Dict[str, Any], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


def _bool(query: Dict[str, Any], key: str) -> bool | None:
    value = _first(query, key)
    if value is None:
        return None
    return value.lower() in ("1", "true", "yes")
