from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

from .models import Artifact, ToolRunManifest

ROOT = Path(__file__).resolve().parent.parent
MCP_RESULTS_DIR = ROOT / "results" / "_mcp"


def new_run_id() -> str:
    return uuid.uuid4().hex[:8]


def ensure_file(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")
    if not p.is_file():
        raise ValueError(f"Expected a file path, got: {p}")
    return p


def ensure_directory(path: str | None, server: str, tool: str, run_id: str) -> Path:
    if path:
        out_dir = Path(path).expanduser().resolve()
    else:
        out_dir = MCP_RESULTS_DIR / server / tool / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def classify_artifact(path: Path) -> tuple[str, str | None]:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return "image", f"image/{suffix.lstrip('.')}" if suffix not in {".jpg"} else "image/jpeg"
    if suffix == ".csv":
        return "csv", "text/csv"
    if suffix == ".json":
        return "json", "application/json"
    if path.is_dir():
        return "directory", None
    if suffix in {".txt", ".md"}:
        return "text", "text/plain"
    return "other", None


def label_for_path(path: Path) -> str:
    name = path.name.lower()
    label_map = [
        ("summary_contact_sheet", "Grid Search Contact Sheet"),
        ("best_parameters", "Best Parameters"),
        ("grid_search_results", "Grid Search Results"),
        ("grid_search_live", "Grid Search Live Results"),
        ("_visual", "Atom Fitting Image"),
        ("_sublattice.png", "Lattice Mode Image"),
        ("_sublattice_0", "Sublattice 0 CSV"),
        ("_sublattice_1", "Sublattice 1 CSV"),
        ("_atoms", "Atom Coordinates"),
        ("_strain.csv", "Strain CSV"),
        ("_cluster_map", "Cluster Map"),
        ("_clustered", "Clustered CSV"),
        ("_defect_map", "Defect Map"),
        ("_defect_score_hist", "Defect Score Histogram"),
        ("_defects", "Defect CSV"),
        ("run_manifest", "Run Manifest"),
    ]
    for needle, label in label_map:
        if needle in name:
            return label
    return path.name


def build_artifacts(paths: list[str | Path]) -> list[Artifact]:
    artifacts: list[Artifact] = []
    seen: set[str] = set()
    for raw_path in paths:
        p = Path(raw_path).resolve()
        key = str(p)
        if key in seen or not p.exists():
            continue
        seen.add(key)
        kind, media_type = classify_artifact(p)
        artifacts.append(
            Artifact(
                label=label_for_path(p),
                path=str(p),
                kind=kind,
                media_type=media_type,
            )
        )
    artifacts.sort(key=lambda a: a.path)
    return artifacts


def persist_manifest(manifest: ToolRunManifest) -> Path:
    out_dir = Path(manifest.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "run_manifest.json"
    manifest_artifact = Artifact(
        label="Run Manifest",
        path=str(manifest_path),
        kind="json",
        media_type="application/json",
    )
    manifest.artifacts = [a for a in manifest.artifacts if Path(a.path).name != "run_manifest.json"]
    manifest.artifacts.append(manifest_artifact)
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest_path


@contextmanager
def staged_single_input(image_path: str):
    src = ensure_file(image_path)
    with tempfile.TemporaryDirectory(prefix="stem_mcp_") as tmp_dir:
        staged = Path(tmp_dir) / src.name
        shutil.copy2(src, staged)
        yield staged


def write_json(path: str | Path, payload: object) -> Path:
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def summarize_directory(root: str, limit: int = 200) -> dict[str, object]:
    folder = Path(root).expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    files = []
    for index, path in enumerate(sorted(folder.rglob("*"))):
        if index >= limit:
            break
        rel = path.relative_to(folder).as_posix()
        files.append(
            {
                "path": rel or ".",
                "type": "directory" if path.is_dir() else "file",
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
        )

    return {
        "root": str(folder),
        "items": files,
        "truncated": len(files) >= limit,
    }
