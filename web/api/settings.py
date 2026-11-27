"""Web-layer settings, read from ``CODEPLAG_WEB_*`` environment variables.

The detector itself keeps using ``configs/default.yaml`` and ``CODEPLAG_*`` variables
(see :mod:`codeplag.core.config`).
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class WebSettings(BaseSettings):
    """Limits and connection settings for the API and the worker."""

    model_config = SettingsConfigDict(env_prefix="CODEPLAG_WEB_", env_file=".env", extra="ignore")

    redis_url: str = "redis://localhost:6379/0"
    jobs_dir: str = "data/jobs"  # shared volume between api and worker (/data/jobs in Docker)
    max_files: int = 200
    max_file_bytes: int = 512 * 1024
    max_total_bytes: int = 20 * 1024 * 1024
    job_ttl_seconds: int = 24 * 3600  # student code is deleted after a day
    cors_origins: list[str] = ["http://localhost:3000"]
    neural_batch: int = 32  # pairs per neural batch before results are emitted
    heartbeat_ms: int = 15_000


settings = WebSettings()
