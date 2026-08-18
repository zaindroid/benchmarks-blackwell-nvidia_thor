"""Benchmark/dataset/report storage.

Uses PostgreSQL when configured and reachable (``postgres`` extra),
otherwise an in-memory backend. Every call falls back gracefully, so
the server runs on a laptop without any database.

The ``benchmark_runs`` table stores our text run ids (``run-<hex>``);
see website/database/migrations/001_initial_schema.sql.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from thor_core.config import ThorConfig
from thor_core.logging import get_logger

logger = get_logger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _MemoryBackend:
    """Plain in-memory storage used by default and as fallback."""

    def __init__(self) -> None:
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._datasets: Dict[str, Dict[str, Any]] = {}
        self._reports: Dict[str, Dict[str, Any]] = {}
        self._experiments: Dict[str, Dict[str, Any]] = {}

    async def save_run(self, run: Dict[str, Any]) -> str:
        self._runs[run["run_id"]] = run
        return run["run_id"]

    async def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self._runs.get(run_id)

    async def list_runs(self, model_id: Optional[str] = None,
                        workload_type: Optional[str] = None,
                        limit: int = 100, since: Optional[str] = None,
                        offset: int = 0) -> List[Dict[str, Any]]:
        runs = list(self._runs.values())
        if model_id:
            runs = [r for r in runs if r.get("model", {}).get("name") == model_id]
        if workload_type:
            runs = [r for r in runs if r.get("workload", {}).get("type") == workload_type]
        if since:
            runs = [r for r in runs if r.get("timestamp", "") >= since]
        runs.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        return runs[offset:offset + limit]

    async def register_dataset(self, dataset_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        entry = {"dataset_id": dataset_id, "created_at": _now(), **metadata}
        self._datasets[dataset_id] = entry
        return entry

    async def list_datasets(self) -> List[Dict[str, Any]]:
        return list(self._datasets.values())

    async def save_report(self, report: Dict[str, Any]) -> str:
        report_id = report.get("report_id") or f"report-{uuid.uuid4().hex[:12]}"
        report["report_id"] = report_id
        self._reports[report_id] = report
        return report_id

    async def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        return self._reports.get(report_id)

    async def save_experiment(self, experiment_id: str, config: Dict[str, Any],
                              results: Dict[str, Any],
                              name: Optional[str] = None,
                              status: str = "completed") -> str:
        entry = self._experiments.get(experiment_id, {
            "experiment_id": experiment_id,
            "created_at": _now(),
        })
        entry.update({
            "name": name or entry.get("name") or experiment_id,
            "config": config,
            "results": results,
            "status": status,
            "updated_at": _now(),
        })
        self._experiments[experiment_id] = entry
        return experiment_id

    async def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        return self._experiments.get(experiment_id)


class _PostgresBackend:
    """PostgreSQL-backed storage for benchmark runs (asyncpg, pooled)."""

    def __init__(self, config: Any):
        from thor_core.config import PostgresConfig

        self._pool: Any = None
        self._dsn = config.dsn if isinstance(config, PostgresConfig) else str(config)

    async def connect(self) -> None:
        import asyncpg

        self._pool = await asyncpg.create_pool(
            self._dsn, min_size=1, max_size=5, timeout=3, command_timeout=5,
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def _pg(self) -> Any:
        return self._pool

    async def save_run(self, run: Dict[str, Any]) -> str:
        await self._pg.execute(
            """
            INSERT INTO benchmark_runs
                (run_id, hardware_info, model_info, workload_info, results, git_commit)
            VALUES ($1, $2::jsonb, $3::jsonb, $4::jsonb, $5::jsonb, $6)
            """,
            run["run_id"],
            json.dumps(run.get("hardware", {})),
            json.dumps(run.get("model", {})),
            json.dumps(run.get("workload", {})),
            json.dumps(run.get("results", {})),
            run.get("git_commit"),
        )
        return run["run_id"]

    async def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = await self._pg.fetchrow(
            "SELECT * FROM benchmark_runs WHERE run_id = $1", run_id
        )
        return self._row_to_run(row) if row else None

    async def list_runs(self, model_id: Optional[str] = None,
                        workload_type: Optional[str] = None,
                        limit: int = 100, since: Optional[str] = None,
                        offset: int = 0) -> List[Dict[str, Any]]:
        conditions, params = [], []
        if model_id:
            params.append(model_id)
            conditions.append(f"model_info->>'name' = ${len(params)}")
        if workload_type:
            params.append(workload_type)
            conditions.append(f"workload_info->>'type' = ${len(params)}")
        if since:
            params.append(datetime.fromisoformat(since))
            conditions.append(f"timestamp >= ${len(params)}")
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)
        limit_idx = len(params)
        params.append(offset)
        offset_idx = len(params)
        rows = await self._pg.fetch(
            f"SELECT * FROM benchmark_runs {where} ORDER BY timestamp DESC "
            f"LIMIT ${limit_idx} OFFSET ${offset_idx}",
            *params,
        )
        return [self._row_to_run(r) for r in rows]

    @staticmethod
    def _row_to_run(row: Any) -> Dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "timestamp": row["timestamp"].isoformat(),
            "hardware": row["hardware_info"],
            "model": row["model_info"],
            "workload": row["workload_info"],
            "results": row["results"],
            "git_commit": row["git_commit"],
        }

    async def save_experiment(self, experiment_id: str, config: Dict[str, Any],
                              results: Dict[str, Any],
                              name: Optional[str] = None,
                              status: str = "completed") -> str:
        await self._pg.execute(
            """
            INSERT INTO experiments (experiment_id, name, config, results, status)
            VALUES ($1, $2, $3::jsonb, $4::jsonb, $5)
            ON CONFLICT (experiment_id) DO UPDATE SET
                config = EXCLUDED.config,
                results = EXCLUDED.results,
                status = EXCLUDED.status,
                updated_at = NOW()
            """,
            experiment_id,
            name or experiment_id,
            json.dumps(config),
            json.dumps(results),
            status,
        )
        return experiment_id

    async def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        row = await self._pg.fetchrow(
            "SELECT * FROM experiments WHERE experiment_id = $1", experiment_id
        )
        if row is None:
            return None
        return {
            "experiment_id": row["experiment_id"],
            "name": row["name"],
            "config": row["config"],
            "results": row["results"],
            "status": row["status"],
            "created_at": row["created_at"].isoformat(),
            "updated_at": row["updated_at"].isoformat(),
        }


class BenchmarkStore:
    """Storage facade: PostgreSQL when available, in-memory otherwise."""

    def __init__(self, config: Optional[ThorConfig] = None,
                 force_memory: bool = False):
        self._memory = _MemoryBackend()
        self._pg: Optional[_PostgresBackend] = None
        self._pg_checked = force_memory or config is None
        if not self._pg_checked:
            try:
                import os

                import asyncpg  # noqa: F401

                # DATABASE_URL (provided by deployment platforms such as
                # zorc) takes precedence over the config file.
                dsn = os.getenv("DATABASE_URL") or config.database.postgres  # type: ignore[union-attr]
                self._pg = _PostgresBackend(dsn)
            except Exception:  # pragma: no cover
                self._pg = None
                self._pg_checked = True

    async def _backend(self) -> Any:
        """Return the postgres backend if usable, else the memory backend."""
        if self._pg is not None and not self._pg_checked:
            self._pg_checked = True
            try:
                await self._pg.connect()
            except Exception as exc:
                logger.warning("postgres unavailable; using in-memory store",
                               error=str(exc))
                self._pg = None
        return self._pg if self._pg is not None else self._memory

    async def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        backend = await self._backend()
        if isinstance(backend, _PostgresBackend):
            try:
                return await getattr(backend, method)(*args, **kwargs)
            except Exception as exc:
                logger.warning("postgres call failed; falling back to memory",
                               method=method, error=str(exc))
                backend = self._memory
        return await getattr(backend, method)(*args, **kwargs)

    async def save_run(self, run: Dict[str, Any]) -> str:
        return await self._call("save_run", run)

    async def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        return await self._call("get_run", run_id)

    async def list_runs(self, model_id: Optional[str] = None,
                        workload_type: Optional[str] = None,
                        limit: int = 100, since: Optional[str] = None,
                        offset: int = 0) -> List[Dict[str, Any]]:
        return await self._call("list_runs", model_id, workload_type, limit,
                                since=since, offset=offset)

    async def register_dataset(self, dataset_id: str,
                               metadata: Dict[str, Any]) -> Dict[str, Any]:
        return await self._call("register_dataset", dataset_id, metadata)

    async def list_datasets(self) -> List[Dict[str, Any]]:
        return await self._call("list_datasets")

    async def save_report(self, report: Dict[str, Any]) -> str:
        return await self._call("save_report", report)

    async def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        return await self._call("get_report", report_id)

    async def save_experiment(self, experiment_id: str, config: Dict[str, Any],
                              results: Dict[str, Any],
                              name: Optional[str] = None,
                              status: str = "completed") -> str:
        return await self._call("save_experiment", experiment_id, config, results,
                                name=name, status=status)

    async def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        return await self._call("get_experiment", experiment_id)

    async def is_persisted(self) -> bool:
        """Whether the active backend is PostgreSQL (vs. in-memory fallback)."""
        return isinstance(await self._backend(), _PostgresBackend)
