from __future__ import annotations

import os
from dataclasses import dataclass


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class CloudConfig:
    enabled: bool
    region: str
    s3_bucket: str
    dynamodb_table: str
    aws_profile: str
    access_key_id: str
    secret_access_key: str

    @property
    def is_configured(self) -> bool:
        return bool(self.region and self.s3_bucket and self.dynamodb_table)

    @property
    def bucket(self) -> str:
        return self.s3_bucket


def get_cloud_config() -> CloudConfig:
    return CloudConfig(
        enabled=_as_bool(os.getenv("CLOUD_SYNC_ENABLED"), default=False),
        region=os.getenv("AWS_REGION", "").strip(),
        s3_bucket=os.getenv("AWS_S3_BUCKET", "").strip(),
        dynamodb_table=os.getenv("AWS_DYNAMODB_TABLE", "").strip(),
        aws_profile=os.getenv("AWS_PROFILE", "").strip(),
        access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "").strip(),
        secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "").strip(),
    )


def require_cloud_config() -> CloudConfig:
    cfg = get_cloud_config()
    if not cfg.enabled:
        raise RuntimeError("Cloud sync is disabled. Set CLOUD_SYNC_ENABLED=true to enable it.")
    if not cfg.is_configured:
        raise RuntimeError(
            "Cloud sync is enabled but AWS_REGION, AWS_S3_BUCKET, and AWS_DYNAMODB_TABLE are not fully configured."
        )
    return cfg
