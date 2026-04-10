from __future__ import annotations

from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP

from ml.defect_detection import detect_defects
from ml.feature_engineering import run_clustering

from .models import ToolRunManifest
from .utils import build_artifacts, ensure_directory, ensure_file, new_run_id, persist_manifest

mcp = FastMCP("STEM ML Server", json_response=True, streamable_http_path="/")


@mcp.resource("stem://ml/overview")
def ml_overview() -> str:
    return (
        "ML tools cluster atomic environments and screen abnormal regions from clustered "
        "strain tables. They return structured run manifests so downstream agent workflows can "
        "use result artifacts directly."
    )


@mcp.prompt()
def clustering_review_prompt(sample_name: str, scientific_question: str = "") -> str:
    question = scientific_question or "identify meaningful structural environments"
    return (
        f"Review the clustering outputs for '{sample_name}'. Focus on {question}. "
        "Discuss whether cluster boundaries align with microstructural regions or whether "
        "the result appears to be dominated by noise, parameter sensitivity, or artifacts."
    )


@mcp.prompt()
def defect_review_prompt(sample_name: str) -> str:
    return (
        f"Review the defect region screening outputs for '{sample_name}'. Explain which "
        "abnormal regions appear robust, which may be edge or interpolation artifacts, "
        "and what follow-up analysis should be run."
    )


@mcp.tool()
def cluster_atomic_environments(
    csv_path: str,
    image_path: str = "",
    output_dir: str = "",
    min_cluster_size: int = 15,
    gmm_components: int = 4,
    flip_h: bool = False,
    flip_v: bool = False,
) -> ToolRunManifest:
    csv_file = ensure_file(csv_path)
    image_file = ensure_file(image_path) if image_path else None
    run_id = new_run_id()
    out_dir = ensure_directory(output_dir or None, "ml", "cluster_atomic_environments", run_id)

    out_csv = run_clustering(
        csv_path=str(csv_file),
        image_path=str(image_file) if image_file else None,
        out_dir=str(out_dir),
        min_cluster_size=min_cluster_size,
        gmm_components=gmm_components,
        flip_h=flip_h,
        flip_v=flip_v,
    )

    df = pd.read_csv(out_csv)
    artifact_paths = [str(p) for p in sorted(out_dir.glob("*")) if p.is_file()]
    manifest = ToolRunManifest(
        run_id=run_id,
        server="ml",
        tool="cluster_atomic_environments",
        output_dir=str(out_dir),
        inputs=[str(csv_file)] + ([str(image_file)] if image_file else []),
        parameters={
            "min_cluster_size": min_cluster_size,
            "gmm_components": gmm_components,
            "flip_h": flip_h,
            "flip_v": flip_v,
        },
        metrics={
            "n_rows": int(len(df)),
            "n_hdbscan_clusters": int(df["cluster_hdbscan"].dropna().nunique()) if "cluster_hdbscan" in df.columns else None,
            "mean_defect_probability": float(df["defect_prob"].dropna().mean()) if "defect_prob" in df.columns and not df["defect_prob"].dropna().empty else None,
        },
        artifacts=build_artifacts(artifact_paths),
        notes=["Cluster overlays use image-space axes even when no raw background image is supplied."],
    )
    persist_manifest(manifest)
    return manifest


@mcp.tool()
def detect_structural_defects(
    csv_path: str,
    image_path: str = "",
    output_dir: str = "",
    peak_csv_path: str = "",
    strain_csv_path: str = "",
    peak_manifest_path: str = "",
    strain_manifest_path: str = "",
    region_threshold: float = 0.45,
    blur_sigma: float = 2.0,
    min_region_atoms: int = 12,
    peak_weight: float = 0.45,
    cluster_weight: float = 0.30,
    strain_weight: float = 0.25,
    flip_h: bool = False,
    flip_v: bool = False,
) -> ToolRunManifest:
    csv_file = ensure_file(csv_path)
    image_file = ensure_file(image_path) if image_path else None
    peak_csv_file = ensure_file(peak_csv_path) if peak_csv_path else None
    strain_csv_file = ensure_file(strain_csv_path) if strain_csv_path else None
    peak_manifest_file = ensure_file(peak_manifest_path) if peak_manifest_path else None
    strain_manifest_file = ensure_file(strain_manifest_path) if strain_manifest_path else None
    run_id = new_run_id()
    out_dir = ensure_directory(output_dir or None, "ml", "detect_structural_defects", run_id)

    out_csv = detect_defects(
        csv_path=str(csv_file),
        image_path=str(image_file) if image_file else None,
        out_dir=str(out_dir),
        peak_csv_path=str(peak_csv_file) if peak_csv_file else None,
        strain_csv_path=str(strain_csv_file) if strain_csv_file else None,
        peak_manifest_path=str(peak_manifest_file) if peak_manifest_file else None,
        strain_manifest_path=str(strain_manifest_file) if strain_manifest_file else None,
        region_threshold=region_threshold,
        blur_sigma=blur_sigma,
        min_region_atoms=min_region_atoms,
        peak_weight=peak_weight,
        cluster_weight=cluster_weight,
        strain_weight=strain_weight,
        flip_h=flip_h,
        flip_v=flip_v,
    )

    df = pd.read_csv(out_csv)
    artifact_paths = [str(p) for p in sorted(out_dir.glob("*")) if p.is_file()]
    manifest = ToolRunManifest(
        run_id=run_id,
        server="ml",
        tool="detect_structural_defects",
        output_dir=str(out_dir),
        inputs=(
            [str(csv_file)]
            + ([str(image_file)] if image_file else [])
            + ([str(peak_csv_file)] if peak_csv_file else [])
            + ([str(strain_csv_file)] if strain_csv_file else [])
            + ([str(peak_manifest_file)] if peak_manifest_file else [])
            + ([str(strain_manifest_file)] if strain_manifest_file else [])
        ),
        parameters={
            "peak_csv_path": str(peak_csv_file) if peak_csv_file else "",
            "strain_csv_path": str(strain_csv_file) if strain_csv_file else "",
            "peak_manifest_path": str(peak_manifest_file) if peak_manifest_file else "",
            "strain_manifest_path": str(strain_manifest_file) if strain_manifest_file else "",
            "region_threshold": region_threshold,
            "blur_sigma": blur_sigma,
            "min_region_atoms": min_region_atoms,
            "peak_weight": peak_weight,
            "cluster_weight": cluster_weight,
            "strain_weight": strain_weight,
            "flip_h": flip_h,
            "flip_v": flip_v,
        },
        metrics={
            "n_rows": int(len(df)),
            "n_region_atoms": int(df["is_region_atom"].sum()) if "is_region_atom" in df.columns else None,
            "mean_region_score": float(df["region_disorder_score"].mean()) if "region_disorder_score" in df.columns else None,
        },
        artifacts=build_artifacts(artifact_paths),
        notes=["Defect region screening fuses peak, strain, and clustering signals into abnormal region maps for review and triage."],
    )
    persist_manifest(manifest)
    return manifest


def main():
    mcp.run(transport="streamable-http")
