from __future__ import annotations

from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP

from core.strain_analysis import generate_strain_maps

from .models import ToolRunManifest
from .utils import build_artifacts, ensure_directory, ensure_file, new_run_id, persist_manifest

mcp = FastMCP("STEM Strain Server", json_response=True, streamable_http_path="/")


@mcp.resource("stem://strain/overview")
def strain_overview() -> str:
    return (
        "Strain tools compute local lattice distortions from atom coordinate CSVs and "
        "export strain maps, contour plots, line profiles, and enriched CSV tables."
    )


@mcp.prompt()
def strain_review_prompt(sample_name: str, reference_note: str = "") -> str:
    ref_text = reference_note or "the default local reference lattice"
    return (
        f"Review the strain outputs for '{sample_name}' using {ref_text}. "
        "Focus on whether the spatial strain pattern is physically plausible, "
        "whether the contour maps contain interpolation artifacts, and whether "
        "the line profiles support the same interpretation."
    )


@mcp.tool()
def compute_strain_map(
    csv_path: str,
    image_path: str,
    output_dir: str = "",
    prune_thresh: float = 0.0,
    axis_tolerance_scale: float = 0.7,
    min_axis_neighbors: int = 2,
    flip_h: bool = False,
    flip_v: bool = False,
) -> ToolRunManifest:
    csv_file = ensure_file(csv_path)
    image_file = ensure_file(image_path)
    run_id = new_run_id()
    out_dir = ensure_directory(output_dir or None, "strain", "compute_strain_map", run_id)

    out_csv = generate_strain_maps(
        csv_path=str(csv_file),
        image_path=str(image_file),
        out_dir=str(out_dir),
        prune_thresh=prune_thresh,
        axis_tolerance_scale=axis_tolerance_scale,
        min_axis_neighbors=min_axis_neighbors,
        flip_h=flip_h,
        flip_v=flip_v,
    )

    df = pd.read_csv(out_csv)
    artifact_paths = [str(p) for p in sorted(out_dir.glob("*")) if p.is_file()]
    manifest = ToolRunManifest(
        run_id=run_id,
        server="strain",
        tool="compute_strain_map",
        output_dir=str(out_dir),
        inputs=[str(csv_file), str(image_file)],
        parameters={
            "prune_thresh": prune_thresh,
            "axis_tolerance_scale": axis_tolerance_scale,
            "min_axis_neighbors": min_axis_neighbors,
            "flip_h": flip_h,
            "flip_v": flip_v,
        },
        metrics={
            "n_atoms_in_strain_csv": int(len(df)),
            "n_nan_rows": int(df[["e_xx", "e_yy"]].isna().any(axis=1).sum()) if {"e_xx", "e_yy"}.issubset(df.columns) else None,
        },
        artifacts=build_artifacts(artifact_paths),
        notes=["Strain analysis preserves CSV coordinates and uses flip flags only for figure orientation."],
    )
    persist_manifest(manifest)
    return manifest


def main():
    mcp.run(transport="streamable-http")
