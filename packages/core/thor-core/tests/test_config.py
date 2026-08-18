"""Tests for thor_core.config."""

from pathlib import Path

from thor_core.config import ThorConfig

EXAMPLE = Path(__file__).resolve().parents[4] / "thor-config.example.yaml"


def test_load_example_config():
    cfg = ThorConfig.load(EXAMPLE)
    assert cfg.server.host == "0.0.0.0"
    assert cfg.server.rate_limit.requests_per_minute == 10
    assert cfg.hardware.device == "NVIDIA DRIVE Thor"
    assert cfg.database.postgres.database == "thorbench"
    assert cfg.benchmarks.default_iterations == 100
    assert cfg.experiments.wandb.enabled is True


def test_default_config():
    cfg = ThorConfig()
    assert cfg.server.port == 3000
    assert cfg.rate_limit_rpm == 10
    assert cfg.postgres_dsn.startswith("postgresql://")


def test_rate_limit_disabled():
    cfg = ThorConfig()
    cfg.server.rate_limit.enabled = False
    assert cfg.rate_limit_rpm == 0


def test_roundtrip(tmp_path):
    cfg = ThorConfig()
    path = tmp_path / "config.yaml"
    cfg.save(path)
    loaded = ThorConfig.load(path)
    assert loaded.server.secret_key == cfg.server.secret_key
