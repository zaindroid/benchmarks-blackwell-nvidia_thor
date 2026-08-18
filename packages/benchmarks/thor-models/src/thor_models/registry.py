"""Model registry — in-memory by default, mirroring the `models` table."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from thor_core.logging import get_logger

logger = get_logger(__name__)

_UTC = timezone.utc


class ModelRegistry:
    """Registers models and tracks their best benchmark metrics."""

    def __init__(self, initial: Optional[Dict[str, Dict[str, Any]]] = None):
        self._models: Dict[str, Dict[str, Any]] = dict(initial or {})

    def register(
        self,
        model_id: str,
        name: Optional[str] = None,
        architecture: Optional[str] = None,
        parameters: Optional[int] = None,
        source: Optional[str] = None,
        license: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register a model. Existing entries are updated in place."""
        if model_id in self._models:
            entry = self._models[model_id]
            entry.update(
                name=name or entry.get("name"),
                architecture=architecture or entry.get("architecture"),
                parameters=parameters or entry.get("parameters"),
                source=source or entry.get("source"),
                license=license or entry.get("license"),
                metadata=metadata or entry.get("metadata"),
            )
            return entry

        entry = {
            "model_id": model_id,
            "name": name or model_id,
            "architecture": architecture,
            "parameters": parameters,
            "source": source or "custom",
            "license": license,
            "last_benchmarked": None,
            "best_metrics": {},
            "metadata": metadata or {},
            "created_at": datetime.now(_UTC).isoformat(),
        }
        self._models[model_id] = entry
        logger.info("model registered", model_id=model_id)
        return entry

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self._models.get(model_id)

    def list(self, architecture: Optional[str] = None,
             optimized: Optional[bool] = None) -> List[Dict[str, Any]]:
        items = list(self._models.values())
        if architecture:
            items = [m for m in items if m.get("architecture") == architecture]
        if optimized is not None:
            items = [
                m for m in items
                if bool(m.get("metadata", {}).get("optimized")) == optimized
            ]
        return sorted(items, key=lambda m: m["created_at"])

    def update_best_metrics(self, model_id: str, metrics: Dict[str, Any]) -> None:
        entry = self.get(model_id)
        if entry is None:
            return
        entry["best_metrics"] = metrics
        entry["last_benchmarked"] = datetime.now(_UTC).isoformat()

    def seed_zoo(self, zoo: Dict[str, Dict[str, Any]]) -> int:
        """Register built-in zoo models; returns number registered."""
        count = 0
        for model_id, spec in zoo.items():
            if model_id not in self._models:
                self.register(
                    model_id=model_id,
                    name=spec.get("name", model_id),
                    architecture=spec.get("architecture"),
                    parameters=spec.get("parameters"),
                    source=spec.get("source", "huggingface"),
                    license=spec.get("license"),
                    metadata=spec,
                )
                count += 1
        return count

    def to_dict(self) -> Dict[str, Any]:
        return {"models": list(self._models.values()), "count": len(self._models)}
