"""Experiment tracking.

Default backend is an in-memory store (optionally persisted to JSON),
so research workflows work without any external services. When the
``tracking`` extra is installed (wandb/mlflow), results can also be
mirrored to those services.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from thor_core.logging import get_logger

logger = get_logger(__name__)

_UTC = timezone.utc


def _now() -> str:
    return datetime.now(_UTC).isoformat()


class ExperimentStore:
    """Minimal experiment store (in-memory + optional JSON persistence)."""

    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path else None
        self._experiments: Dict[str, Dict[str, Any]] = {}
        if self.path and self.path.exists():
            self._experiments = json.loads(self.path.read_text(encoding="utf-8"))

    def create(self, name: str, hypothesis: str = "", description: str = "",
               config: Optional[Dict[str, Any]] = None,
               tags: Optional[List[str]] = None) -> Dict[str, Any]:
        experiment = {
            "experiment_id": str(uuid.uuid4()),
            "name": name,
            "description": description,
            "hypothesis": hypothesis,
            "config": config or {},
            "results": {},
            "metrics": {},
            "status": "pending",
            "tags": tags or [],
            "created_at": _now(),
            "updated_at": _now(),
        }
        self._experiments[experiment["experiment_id"]] = experiment
        self._persist()
        return experiment

    def update(self, experiment_id: str, **changes: Any) -> Optional[Dict[str, Any]]:
        experiment = self._experiments.get(experiment_id)
        if experiment is None:
            return None
        for key in ("results", "metrics", "status", "description", "hypothesis", "config", "tags"):
            if key in changes:
                experiment[key] = changes[key]
        experiment["updated_at"] = _now()
        self._persist()
        return experiment

    def get(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        return self._experiments.get(experiment_id)

    def list(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        items = list(self._experiments.values())
        if status:
            items = [e for e in items if e["status"] == status]
        return sorted(items, key=lambda e: e["created_at"], reverse=True)

    def _persist(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._experiments, indent=2), encoding="utf-8"
            )


class ExperimentTracker:
    """High-level tracker used by MCP tools and research workflows."""

    def __init__(self, store: Optional[ExperimentStore] = None,
                 wandb: bool = False, mlflow: bool = False):
        self.store = store or ExperimentStore()
        self._wandb = wandb
        self._mlflow = mlflow

    def track(self, name: str, hypothesis: str = "", description: str = "",
              config: Optional[Dict[str, Any]] = None,
              results: Optional[Dict[str, Any]] = None,
              metrics: Optional[Dict[str, Any]] = None,
              tags: Optional[List[str]] = None) -> Dict[str, Any]:
        experiment = self.store.create(
            name=name,
            hypothesis=hypothesis,
            description=description,
            config=config,
            tags=tags,
        )
        if results or metrics:
            experiment = self.store.update(
                experiment["experiment_id"],
                results=results or {},
                metrics=metrics or {},
                status="completed",
            ) or experiment
        self._mirror_external(experiment)
        return experiment

    def update(self, experiment_id: str, **changes: Any) -> Optional[Dict[str, Any]]:
        return self.store.update(experiment_id, **changes)

    def get(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        return self.store.get(experiment_id)

    def list(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.store.list(status=status)

    def _mirror_external(self, experiment: Dict[str, Any]) -> None:
        """Best-effort mirror to wandb/mlflow when enabled and installed."""
        if self._wandb:
            try:
                import wandb  # type: ignore

                wandb.init(project="thor-ai", config=experiment.get("config", {}))
                wandb.log(experiment.get("metrics", {}))
                wandb.finish()
            except Exception as exc:  # pragma: no cover
                logger.warning("wandb mirror failed", error=str(exc))
        if self._mlflow:
            try:
                import mlflow  # type: ignore

                mlflow.log_params(experiment.get("config", {}))
                mlflow.log_metrics(experiment.get("metrics", {}))
            except Exception as exc:  # pragma: no cover
                logger.warning("mlflow mirror failed", error=str(exc))
