from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.cloud_config import CloudConfig, require_cloud_config


SYNCABLE_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
SYNCABLE_DATA_SUFFIXES = {".json", ".csv", ".md", ".html"}


@dataclass
class SyncedArtifact:
    local_path: str
    s3_key: str
    category: str


def _load_boto3():
    try:
        import boto3  # type: ignore
    except ImportError as exc:
        raise RuntimeError("boto3 is required for cloud sync. Install it in the active environment.") from exc
    return boto3


def _session_kwargs(cfg: CloudConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"region_name": cfg.region}
    if cfg.aws_profile:
        kwargs["profile_name"] = cfg.aws_profile
    elif cfg.access_key_id and cfg.secret_access_key:
        kwargs["aws_access_key_id"] = cfg.access_key_id
        kwargs["aws_secret_access_key"] = cfg.secret_access_key
    return kwargs


def _boto3_session(cfg: CloudConfig):
    boto3 = _load_boto3()
    return boto3.Session(**_session_kwargs(cfg))


def _caller_identity(cfg: CloudConfig) -> dict[str, str]:
    session = _boto3_session(cfg)
    identity = session.client("sts").get_caller_identity()
    return {
        "account_id": identity.get("Account", ""),
        "arn": identity.get("Arn", ""),
        "user_id": identity.get("UserId", ""),
    }


def infer_stage_from_output_dir(output_dir: str | Path) -> str:
    parts = [part.lower() for part in Path(output_dir).resolve().parts]
    if "1_grid_search" in parts:
        return "grid_search"
    if "2_peak_finding" in parts:
        return "peak_finding"
    if "3_strain_map" in parts:
        return "strain_map"
    if "4_clustering" in parts:
        return "clustering"
    if "5_defects" in parts:
        return "defect_region_screening"
    if "compare_runs_report" in parts:
        return "llm_review"
    return "unknown"


def _categorize_artifact(path: Path) -> str | None:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name == "run_manifest.json":
        return "manifests"
    if suffix == ".json":
        return "stats"
    if suffix == ".csv":
        return "csv"
    if suffix in {".md", ".html"}:
        return "reports"
    if suffix in SYNCABLE_IMAGE_SUFFIXES:
        return "previews"
    return None


def collect_syncable_artifacts(output_dir: str | Path, max_preview_images: int = 2) -> list[tuple[Path, str]]:
    root = Path(output_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Run output directory does not exist: {root}")

    preview_count = 0
    collected: list[tuple[Path, str]] = []
    for path in sorted(p for p in root.iterdir() if p.is_file()):
        category = _categorize_artifact(path)
        if not category:
            continue
        if category == "previews":
            if preview_count >= max_preview_images:
                continue
            preview_count += 1
        collected.append((path, category))
    return collected


def _s3_key(project: str, stage: str, run_id: str, category: str, filename: str) -> str:
    safe_project = project.strip().replace("\\", "/").strip("/") or "unassigned"
    return f"projects/{safe_project}/runs/{run_id}/{category}/{filename}"


def upload_file_to_s3(local_path: str | Path, s3_key: str, cfg: CloudConfig | None = None) -> dict[str, str]:
    cfg = cfg or require_cloud_config()
    session = _boto3_session(cfg)
    s3_client = session.client("s3")
    local_file = Path(local_path).resolve()
    s3_client.upload_file(str(local_file), cfg.s3_bucket, s3_key)
    return {
        "bucket": cfg.s3_bucket,
        "key": s3_key,
        "uri": f"s3://{cfg.s3_bucket}/{s3_key}",
    }


def write_run_index_record(record: dict[str, Any], cfg: CloudConfig | None = None) -> dict[str, Any]:
    cfg = cfg or require_cloud_config()
    session = _boto3_session(cfg)
    table = session.resource("dynamodb").Table(cfg.dynamodb_table)
    table.put_item(Item=record)
    return record


def sync_run_directory_to_cloud(
    project: str,
    output_dir: str | Path,
    stage: str | None = None,
    run_id: str | None = None,
    sample_name: str = "",
    include_previews: bool = True,
) -> dict[str, Any]:
    cfg = require_cloud_config()
    output_root = Path(output_dir).resolve()
    stage_name = stage or infer_stage_from_output_dir(output_root)
    effective_run_id = run_id or output_root.name
    syncables = collect_syncable_artifacts(output_root, max_preview_images=2 if include_previews else 0)

    uploaded: list[SyncedArtifact] = []
    artifact_index: dict[str, list[str]] = {
        "manifests": [],
        "stats": [],
        "csv": [],
        "reports": [],
        "previews": [],
    }

    manifest_payload: dict[str, Any] = {}
    manifest_path = output_root / "run_manifest.json"
    if manifest_path.exists():
        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest_payload = {}

    for local_file, category in syncables:
        s3_key = _s3_key(project, stage_name, effective_run_id, category, local_file.name)
        upload_file_to_s3(local_file, s3_key, cfg=cfg)
        uploaded.append(SyncedArtifact(local_path=str(local_file), s3_key=s3_key, category=category))
        artifact_index.setdefault(category, []).append(s3_key)

    timestamp = datetime.now(timezone.utc).isoformat()
    caller = _caller_identity(cfg)
    record = {
        "project_id": project or "unassigned",
        "run_stage": f"{effective_run_id}#{stage_name}",
        "run_id": effective_run_id,
        "stage": stage_name,
        "sample_name": sample_name or manifest_payload.get("parameters", {}).get("sample_name", ""),
        "tool": manifest_payload.get("tool", ""),
        "server": manifest_payload.get("server", ""),
        "created_at": timestamp,
        "local_output_dir": str(output_root),
        "manifest_s3_key": artifact_index["manifests"][0] if artifact_index["manifests"] else "",
        "stats_s3_keys": artifact_index["stats"],
        "csv_s3_keys": artifact_index["csv"],
        "report_s3_keys": artifact_index["reports"],
        "preview_s3_keys": artifact_index["previews"],
        "screening_outcome": manifest_payload.get("metrics", {}).get("screening_outcome", ""),
        "aws_account_id": caller["account_id"],
        "synced_by": caller["arn"],
    }
    write_run_index_record(record, cfg=cfg)

    manifest_path = output_root / "run_manifest.json"
    if manifest_path.exists():
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest_data = {}
        manifest_data["cloud_sync"] = {
            "synced": True,
            "synced_at": timestamp,
            "bucket": cfg.s3_bucket,
            "project": project,
            "run_id": effective_run_id,
            "stage": stage_name,
            "aws_account_id": caller["account_id"],
            "synced_by": caller["arn"],
            "manifest_s3_key": artifact_index["manifests"][0] if artifact_index["manifests"] else "",
            "stats_s3_keys": artifact_index["stats"],
            "csv_s3_keys": artifact_index["csv"],
            "report_s3_keys": artifact_index["reports"],
            "preview_s3_keys": artifact_index["previews"],
        }
        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    return {
        "project": project,
        "stage": stage_name,
        "run_id": effective_run_id,
        "output_dir": str(output_root),
        "bucket": cfg.s3_bucket,
        "aws_account_id": caller["account_id"],
        "synced_by": caller["arn"],
        "synced_at": timestamp,
        "uploaded_count": len(uploaded),
        "artifacts": [artifact.__dict__ for artifact in uploaded],
        "ddb_record": record,
    }
