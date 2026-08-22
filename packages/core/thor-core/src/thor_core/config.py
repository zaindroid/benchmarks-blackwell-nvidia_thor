"""Configuration management (pydantic models + YAML loading)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class RateLimitConfig(BaseModel):
    enabled: bool = True
    requests_per_minute: int = 10


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 3000
    mode: str = "http"  # http | stdio
    secret_key: str = "dev-secret-change-me"
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)


class HardwareConfig(BaseModel):
    device: str = "NVIDIA DRIVE Thor"
    device_ip: str = "192.168.1.100"
    sdk_path: str = "/opt/nvidia/thor"
    cuda_device: int = 0


class PostgresConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    user: str = "thor"
    password: str = "thor"
    database: str = "thorbench"

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class InfluxDBConfig(BaseModel):
    url: str = "http://localhost:8086"
    token: str = ""
    org: str = "thor-org"
    bucket: str = "thor-bucket"


class RedisConfig(BaseModel):
    url: str = "redis://localhost:6379"

    @property
    def dsn(self) -> str:
        return self.url


class DatabaseConfig(BaseModel):
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    influxdb: InfluxDBConfig = Field(default_factory=InfluxDBConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)


class BenchmarkConfig(BaseModel):
    default_iterations: int = 100
    default_warmup: int = 10
    timeout_seconds: int = 3600
    collect_power: bool = True
    collect_memory: bool = True
    collect_thermal: bool = True


class ModelsConfig(BaseModel):
    registry_path: str = "/data/models"
    cache_dir: str = "/data/cache"
    default_precision: str = "fp16"


class WandbConfig(BaseModel):
    enabled: bool = False
    project: str = "thor-ai"
    entity: str = ""


class MlflowConfig(BaseModel):
    enabled: bool = False
    tracking_uri: str = "http://localhost:5000"


class ExperimentsConfig(BaseModel):
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    mlflow: MlflowConfig = Field(default_factory=MlflowConfig)


class RemoteDeviceConfig(BaseModel):
    """Configuration for a remote benchmark worker (e.g. a DRIVE Thor
    device reachable over the network or tailnet).

    When ``enabled``, real (non-simulated) ``benchmark_run`` calls are
    dispatched to this MCP endpoint instead of running locally.
    """

    enabled: bool = False
    url: str = ""      # MCP streamable-HTTP endpoint, e.g. http://100.64.0.10:8000/mcp
    token: str = ""    # bearer token for the worker endpoint


class ThorConfig(BaseModel):
    """Top-level platform configuration (mirrors thor-config.yaml)."""

    model_config = ConfigDict(extra="ignore")

    server: ServerConfig = Field(default_factory=ServerConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    benchmarks: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    experiments: ExperimentsConfig = Field(default_factory=ExperimentsConfig)
    remote_device: RemoteDeviceConfig = Field(default_factory=RemoteDeviceConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ThorConfig":
        """Load config from a YAML file (or defaults when path is None)."""
        if path is None:
            cfg = cls()
        else:
            data: Dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            cfg = cls(**data)
        # Environment overrides (deployment platforms provide these).
        if os.getenv("THOR_DEVICE_URL"):
            cfg.remote_device.url = os.environ["THOR_DEVICE_URL"]
            cfg.remote_device.enabled = True
        if os.getenv("THOR_DEVICE_TOKEN"):
            cfg.remote_device.token = os.environ["THOR_DEVICE_TOKEN"]
        return cfg

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )

    @property
    def postgres_dsn(self) -> str:
        return self.database.postgres.dsn

    @property
    def rate_limit_rpm(self) -> int:
        if not self.server.rate_limit.enabled:
            return 0  # unlimited
        return self.server.rate_limit.requests_per_minute
