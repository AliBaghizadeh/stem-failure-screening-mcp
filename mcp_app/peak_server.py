from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP

from core.batch_processor import BatchProcessor
from core.grid_search import GridSearchEngine

from .models import ToolRunManifest
from .utils import (
    build_artifacts,
    ensure_directory,
    ensure_file,
    new_run_id,
    persist_manifest,
    staged_single_input,
    write_json,
)

mcp = FastMCP("STEM Peak Server", json_response=True, streamable_http_path="/")


def _peak_assessment(n_atoms: int, sub0: int, sub1: int, paired_count: int, sublattice_mode: str = "two") -> dict:
    total = max(n_atoms, 1)
    paired_fraction = paired_count / total
    balance = min(sub0, sub1) / max(max(sub0, sub1), 1)
    single_mode = str(sublattice_mode).strip().lower() == "single"

    if n_atoms >= 2000:
        density_flag = "High Detection Coverage"
    elif n_atoms >= 500:
        density_flag = "Moderate Detection Coverage"
    else:
        density_flag = "Sparse Detection Coverage"

    if single_mode:
        sublattice_flag = "Single-Sublattice Mode"
    elif balance >= 0.9:
        sublattice_flag = "Well-Balanced Sublattices"
    elif balance >= 0.8:
        sublattice_flag = "Balanced Sublattices"
    elif balance >= 0.5:
        sublattice_flag = "Moderately Balanced"
    else:
        sublattice_flag = "Imbalanced Sublattices"

    if single_mode:
        pairing_flag = "Pairing Not Applied"
    elif paired_fraction >= 0.9:
        pairing_flag = "Very Strong Pairing"
    elif paired_fraction >= 0.6:
        pairing_flag = "Strong Pairing"
    elif paired_fraction >= 0.3:
        pairing_flag = "Partial Pairing"
    else:
        pairing_flag = "Weak Pairing"

    if single_mode:
        material_flag = "Single-Species Column Map"
    elif balance >= 0.9 and paired_fraction >= 0.9:
        material_flag = "Cleaner Lattice Signature"
    elif balance >= 0.75 and paired_fraction >= 0.7:
        material_flag = "Moderate Lattice Regularity"
    else:
        material_flag = "Disturbed Lattice Signature"

    if density_flag == "Sparse Detection Coverage" or sublattice_flag == "Imbalanced Sublattices":
        confidence_flag = "Medium Confidence"
    else:
        confidence_flag = "High Confidence"

    return {
        "overall_flag": density_flag,
        "confidence_flag": confidence_flag,
        "sublattice_flag": sublattice_flag,
        "pairing_flag": pairing_flag,
        "material_flag": material_flag,
        "summary": (
            f"{density_flag}; {material_flag.lower()} without binary sublattice splitting."
            if single_mode
            else f"{density_flag}; {material_flag.lower()} with {paired_fraction:.1%} paired atoms and balance ratio {balance:.3f}."
        ),
        "warning": (
            "Heuristic guidance only. High detection coverage is not the same as sample quality. "
            "For single-sublattice HAADF cases, do not interpret missing pair metrics as poor quality."
        ),
    }


def _build_peak_stats(out_dir: Path, result, sublattice_mode: str, sublattice_method: str,
                      sublattice_pair_max_dist: float | None,
                      sublattice_pair_min_dist: float) -> str | None:
    atoms_csv = next(iter(sorted(out_dir.glob("*_atoms.csv"))), None)
    if atoms_csv is None:
        return None

    df = pd.read_csv(atoms_csv)
    n_atoms = int(len(df))
    sub_counts = {}
    if "sublattice" in df.columns:
        sub_counts = {
            int(k): int(v)
            for k, v in df["sublattice"].dropna().astype(int).value_counts().to_dict().items()
        }
    sub0 = int(sub_counts.get(0, 0))
    sub1 = int(sub_counts.get(1, 0))
    paired = df["pair_distance"].dropna() if "pair_distance" in df.columns else pd.Series(dtype=float)
    pair_angles = df["pair_angle"].dropna() if "pair_angle" in df.columns else pd.Series(dtype=float)
    paired_count = int(len(paired))

    stats = {
        "input_atom_count": n_atoms,
        "sublattice_0_count": sub0,
        "sublattice_1_count": sub1,
        "identified_sublattice_count": int((sub0 > 0) + (sub1 > 0)),
        "sublattice_balance_ratio": float(min(sub0, sub1) / max(max(sub0, sub1), 1)),
        "paired_atom_count": paired_count,
        "paired_atom_fraction": float(paired_count / max(n_atoms, 1)),
        "pair_distance_mean": float(paired.mean()) if len(paired) else 0.0,
        "pair_distance_median": float(paired.median()) if len(paired) else 0.0,
        "pair_distance_std": float(paired.std()) if len(paired) else 0.0,
        "pair_angle_mean": float(pair_angles.mean()) if len(pair_angles) else 0.0,
        "processing_time_seconds": round(result.processing_time, 3),
        "sublattice_mode": sublattice_mode,
        "sublattice_split_applied": bool(str(sublattice_mode).strip().lower() != "single"),
        "sublattice_method": sublattice_method,
        "sublattice_pair_max_dist": sublattice_pair_max_dist,
        "sublattice_pair_min_dist": sublattice_pair_min_dist,
        "assessment": _peak_assessment(n_atoms, sub0, sub1, paired_count, sublattice_mode=sublattice_mode),
    }
    stats_path = out_dir / f"{atoms_csv.stem.replace('_atoms', '')}_peak_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return str(stats_path)


@mcp.resource("stem://peak/overview")
def peak_overview() -> str:
    return (
        "Peak server tools expose the microscopy pipeline stages for parameter search "
        "and atom finding. Outputs are written to structured run folders with a "
        "run_manifest.json file for downstream agents."
    )


@mcp.prompt()
def peak_qc_prompt(sample_name: str, scientific_goal: str = "") -> str:
    goal = scientific_goal or "verify atom detection quality and identify next actions"
    return (
        f"Review the peak-finding outputs for '{sample_name}'. Focus on {goal}. "
        "Check whether the atom fitting overlay, sublattice split, and coordinate CSV "
        "look scientifically coherent before proceeding to strain analysis."
    )


@mcp.tool()
def grid_search_image(
    image_path: str,
    output_dir: str = "",
    sep_min: int = 4,
    sep_max: int = 10,
    sep_step: int = 2,
    thresh_min: float = 0.3,
    thresh_max: float = 0.6,
    thresh_step: float = 0.1,
    hardware: str = "cpu",
    workers: int = 4,
    flip_h: bool = False,
    flip_v: bool = False,
) -> ToolRunManifest:
    src = ensure_file(image_path)
    run_id = new_run_id()
    out_dir = ensure_directory(output_dir or None, "peak", "grid_search", run_id)

    sep_range = list(range(sep_min, sep_max + 1, sep_step))
    thresh_range = [
        round(thresh_min + i * thresh_step, 4)
        for i in range(int((thresh_max - thresh_min) / thresh_step + 0.001) + 1)
    ]

    engine = GridSearchEngine(max_workers=max(1, workers))
    best_params, results = engine.search(
        image_path=str(src),
        config={"separations": sep_range, "thresholds": thresh_range, "pca_options": [True]},
        plot_dir=str(out_dir),
        hardware=hardware,
        flip_h=flip_h,
        flip_v=flip_v,
    )

    best_path = write_json(out_dir / "best_parameters.json", best_params)
    results_path = out_dir / "grid_search_results.json"
    results_path.write_text(
        json.dumps([result.to_dict() for result in results], indent=2),
        encoding="utf-8",
    )
    results_csv = out_dir / "grid_search_results.csv"
    pd.DataFrame([result.to_dict() for result in results]).to_csv(results_csv, index=False)

    artifact_paths = [best_path, results_path, results_csv] + [str(p) for p in sorted(out_dir.glob("*.png"))]
    manifest = ToolRunManifest(
        run_id=run_id,
        server="peak",
        tool="grid_search_image",
        output_dir=str(out_dir),
        inputs=[str(src)],
        parameters={
            "sep_min": sep_min,
            "sep_max": sep_max,
            "sep_step": sep_step,
            "thresh_min": thresh_min,
            "thresh_max": thresh_max,
            "thresh_step": thresh_step,
            "hardware": hardware,
            "workers": workers,
            "flip_h": flip_h,
            "flip_v": flip_v,
        },
        metrics={
            "n_trials": len(results),
            "best_separation": best_params["separation"],
            "best_threshold": best_params["threshold"],
            "best_pca": best_params["pca"],
        },
        artifacts=build_artifacts(artifact_paths),
        notes=["Grid search completed with structured outputs for downstream review."],
    )
    persist_manifest(manifest)
    return manifest


@mcp.tool()
def peak_find_atoms(
    image_path: str,
    output_dir: str = "",
    separation: int = 6,
    threshold: float = 0.5,
    sublattice_mode: str = "two",
    sublattice_method: str = "local_pairing",
    sublattice_pair_max_dist: float | None = None,
    sublattice_pair_min_dist: float = 0.0,
    hardware: str = "cpu",
    workers: int = 4,
    flip_h: bool = False,
    flip_v: bool = False,
) -> ToolRunManifest:
    src = ensure_file(image_path)
    run_id = new_run_id()
    out_dir = ensure_directory(output_dir or None, "peak", "peak_find", run_id)

    with staged_single_input(str(src)) as staged_path:
        processor = BatchProcessor(max_workers=1 if hardware.lower() == "gpu" else max(1, workers))
        summary = processor.run(
            input_folder=str(staged_path.parent),
            output_folder=str(out_dir),
            config={
                "mode": "full",
                "peak_detection": {
                    "separation": separation,
                    "threshold": threshold,
                    "pca_enabled": True,
                },
                "sublattice": {
                    "mode": sublattice_mode,
                    "method": sublattice_method,
                    "pair_max_dist": sublattice_pair_max_dist,
                    "pair_min_dist": sublattice_pair_min_dist,
                },
            },
            flip_h=flip_h,
            flip_v=flip_v,
        )

    if summary.failed > 0 or not summary.results:
        raise RuntimeError("Peak finding failed; inspect logs and source image.")

    result = summary.results[0]
    peak_stats_path = _build_peak_stats(
        out_dir,
        result,
        sublattice_mode=sublattice_mode,
        sublattice_method=sublattice_method,
        sublattice_pair_max_dist=sublattice_pair_max_dist,
        sublattice_pair_min_dist=sublattice_pair_min_dist,
    )
    artifact_paths = list(result.output_files)
    if peak_stats_path:
        artifact_paths.append(peak_stats_path)
    manifest = ToolRunManifest(
        run_id=run_id,
        server="peak",
        tool="peak_find_atoms",
        output_dir=str(out_dir),
        inputs=[str(src)],
        parameters={
            "separation": separation,
            "threshold": threshold,
            "sublattice_mode": sublattice_mode,
            "sublattice_method": sublattice_method,
            "sublattice_pair_max_dist": sublattice_pair_max_dist,
            "sublattice_pair_min_dist": sublattice_pair_min_dist,
            "hardware": hardware,
            "workers": workers,
            "flip_h": flip_h,
            "flip_v": flip_v,
        },
        metrics={
            "n_atoms": result.n_atoms,
            "image_status": result.status,
            "processing_time_seconds": round(result.processing_time, 3),
            "success_rate": summary.success_rate,
        },
        artifacts=build_artifacts(artifact_paths),
        notes=[
            "Outputs include atom fitting and lattice overlays plus CSV exports.",
            "Sublattice mode supports both two-sublattice and single-sublattice workflows.",
            "Sublattice determination supports both local_pairing and gmm modes when two-sublattice mode is selected.",
            "This tool stages a single input image to avoid unintentionally batch-processing a folder.",
        ],
    )
    persist_manifest(manifest)
    return manifest


def main():
    mcp.run(transport="streamable-http")
