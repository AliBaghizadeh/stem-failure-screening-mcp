from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import ollama
import pandas as pd

from .models import ToolRunManifest

OLLAMA_MODEL = "qwen2.5:14b"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _safe_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _resolve_manifest_path(path: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        candidate = candidate / "run_manifest.json"
    if not candidate.exists():
        raise FileNotFoundError(f"Run manifest not found: {candidate}")
    if candidate.name != "run_manifest.json":
        raise ValueError(f"Expected a run directory or run_manifest.json, got: {candidate}")
    return candidate


def _resolve_run_input(path: str) -> tuple[Path, str]:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        manifest = candidate / "run_manifest.json"
        if manifest.exists():
            return manifest, "manifest"
        return candidate, "legacy_dir"
    if not candidate.exists():
        raise FileNotFoundError(f"Run path not found: {candidate}")
    if candidate.name == "run_manifest.json":
        return candidate, "manifest"
    raise ValueError(f"Expected a run directory or run_manifest.json, got: {candidate}")


def _load_manifest(path: str) -> ToolRunManifest:
    manifest_path = _resolve_manifest_path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return ToolRunManifest.model_validate(payload)


def _artifact_path(manifest: ToolRunManifest, suffix: str) -> Path | None:
    suffix = suffix.lower()
    for artifact in manifest.artifacts:
        path = Path(artifact.path)
        if path.name.lower().endswith(suffix):
            return path
    return None


def _artifact_json(manifest: ToolRunManifest, suffix: str) -> dict[str, Any] | None:
    path = _artifact_path(manifest, suffix)
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _stats_from_series(series: pd.Series) -> dict[str, float | None]:
    clean = series.dropna()
    if clean.empty:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": _safe_float(clean.mean()),
        "std": _safe_float(clean.std()),
        "min": _safe_float(clean.min()),
        "max": _safe_float(clean.max()),
    }


def _summarize_atoms_csv(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    frame = pd.read_csv(path)
    summary: dict[str, Any] = {
        "path": str(path),
        "n_rows": int(len(frame)),
    }
    if "sublattice" in frame.columns:
        counts = frame["sublattice"].fillna(-1).astype(int).value_counts().sort_index()
        summary["sublattice_counts"] = {str(key): int(value) for key, value in counts.items()}
    if "intensity" in frame.columns:
        summary["intensity"] = _stats_from_series(frame["intensity"])
    if "pair_distance" in frame.columns:
        summary["pair_distance"] = _stats_from_series(frame["pair_distance"])
        summary["paired_atoms"] = int(frame["pair_distance"].notna().sum())
    if "pair_angle" in frame.columns:
        summary["pair_angle"] = _stats_from_series(frame["pair_angle"])
    return summary


def _summarize_strain_csv(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    frame = pd.read_csv(path)
    summary: dict[str, Any] = {"path": str(path), "n_rows": int(len(frame))}
    for column in ("e_xx", "e_yy", "a1_len", "a2_len"):
        if column in frame.columns:
            summary[column] = _stats_from_series(frame[column])
    return summary


def _summarize_cluster_csv(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    frame = pd.read_csv(path)
    summary: dict[str, Any] = {"path": str(path), "n_rows": int(len(frame))}
    if "cluster_hdbscan" in frame.columns:
        summary["cluster_hdbscan_count"] = int(frame["cluster_hdbscan"].dropna().nunique())
    if "cluster_gmm" in frame.columns:
        summary["cluster_gmm_count"] = int(frame["cluster_gmm"].dropna().nunique())
    if "defect_prob" in frame.columns and not frame["defect_prob"].dropna().empty:
        summary["defect_prob"] = _stats_from_series(frame["defect_prob"])
    return summary


def _summarize_defect_csv(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    frame = pd.read_csv(path)
    summary: dict[str, Any] = {"path": str(path), "n_rows": int(len(frame))}
    if "is_defect" in frame.columns:
        summary["n_defects"] = int(frame["is_defect"].fillna(0).astype(int).sum())
    if "defect_score" in frame.columns:
        summary["defect_score"] = _stats_from_series(frame["defect_score"])
    if "defect_region_id" in frame.columns:
        positive = frame["defect_region_id"].fillna(0).astype(int)
        summary["n_regions"] = int(positive[positive > 0].nunique())
        summary["region_atoms"] = int((positive > 0).sum())
    if "region_disorder_score" in frame.columns:
        summary["region_disorder_score"] = _stats_from_series(frame["region_disorder_score"])
    return summary


def _artifact_by_glob(folder: Path, pattern: str) -> Path | None:
    return next(iter(sorted(folder.glob(pattern))), None)


def _infer_stage(folder: Path) -> str:
    name = folder.as_posix().lower()
    if "/2_peak_finding/" in name or any(folder.glob("*_atoms.csv")):
        return "peak"
    if "/3_strain_map/" in name or any(folder.glob("*_strain.csv")):
        return "strain"
    if "/4_clustering/" in name or any(folder.glob("*_clustered.csv")):
        return "cluster"
    if "/5_defects/" in name or any(folder.glob("*_defects.csv")) or any(folder.glob("*_defect_regions.csv")):
        return "defects"
    return "unknown"


def _load_legacy_stats(folder: Path, suffix: str) -> dict[str, Any] | None:
    path = _artifact_by_glob(folder, suffix)
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_stage_stats(tool: str, stats: dict[str, Any] | None) -> dict[str, Any] | None:
    if not stats:
        return stats

    stats = dict(stats)
    assessment = dict(stats.get("assessment") or {})

    if tool in {"peak", "peak_find_atoms"}:
        n_atoms = int(stats.get("input_atom_count") or 0)
        sub0 = int(stats.get("sublattice_0_count") or 0)
        sub1 = int(stats.get("sublattice_1_count") or 0)
        paired_fraction = float(stats.get("paired_atom_fraction") or 0.0)
        balance = float(stats.get("sublattice_balance_ratio") or 0.0)
        single_mode = str(stats.get("sublattice_mode") or "").strip().lower() == "single"
        if n_atoms >= 2000:
            overall_flag = "High Detection Coverage"
        elif n_atoms >= 500:
            overall_flag = "Moderate Detection Coverage"
        else:
            overall_flag = "Sparse Detection Coverage"
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
        material_flag = (
            "Single-Species Column Map"
            if single_mode
            else "Cleaner Lattice Signature"
            if balance >= 0.9 and paired_fraction >= 0.9
            else "Moderate Lattice Regularity"
            if balance >= 0.75 and paired_fraction >= 0.7
            else "Disturbed Lattice Signature"
        )
        confidence_flag = "Medium Confidence" if overall_flag == "Sparse Detection Coverage" or sublattice_flag == "Imbalanced Sublattices" else "High Confidence"
        assessment.update({
            "overall_flag": overall_flag,
            "confidence_flag": confidence_flag,
            "sublattice_flag": sublattice_flag,
            "pairing_flag": pairing_flag,
            "material_flag": material_flag,
            "summary": (
                f"{overall_flag}; {material_flag.lower()} without binary sublattice splitting."
                if single_mode
                else f"{overall_flag}; {material_flag.lower()} with {paired_fraction:.1%} paired atoms and balance ratio {balance:.3f}."
            ),
            "warning": "Heuristic guidance only. High detection coverage is not the same as sample quality. For single-sublattice HAADF cases, do not interpret missing pair metrics as poor quality.",
        })

    if tool in {"strain", "compute_strain_map"}:
        valid_fraction = float(stats.get("valid_strain_fraction") or 0.0)
        pruned_fraction = float((stats.get("pruned_atom_count") or 0) / max(int(stats.get("input_atom_count") or 1), 1))
        axis_tolerance_scale = float(stats.get("axis_tolerance_scale") or 0.0)
        min_axis_neighbors = int(stats.get("min_axis_neighbors") or 0)
        if valid_fraction >= 0.6:
            overall_flag = "High Strain Coverage"
        elif valid_fraction >= 0.35:
            overall_flag = "Moderate Strain Coverage"
        elif valid_fraction >= 0.2:
            overall_flag = "Localized Strain Coverage"
        else:
            overall_flag = "Sparse Strain Coverage"
        material_flag = (
            "Widespread Lattice Distortion Signal"
            if valid_fraction >= 0.8
            else "Moderate Lattice Distortion Signal"
            if valid_fraction >= 0.5
            else "Localized Lattice Distortion Signal"
            if valid_fraction >= 0.2
            else "Weak / Incomplete Distortion Signal"
        )
        if min_axis_neighbors <= 0:
            confidence_flag = "Low Confidence"
            parameter_regime = "Very Permissive"
        elif min_axis_neighbors == 1 and axis_tolerance_scale >= 1.5:
            confidence_flag = "Medium Confidence"
            parameter_regime = "Coverage-Optimized"
        elif min_axis_neighbors >= 2:
            confidence_flag = "Medium Confidence"
            parameter_regime = "Balanced"
        else:
            confidence_flag = "Medium Confidence"
            parameter_regime = "Intermediate"
        pruning_flag = "Aggressive Pruning" if pruned_fraction > 0.05 else "Light Pruning" if pruned_fraction > 0 else "No Pruning"
        assessment.update({
            "overall_flag": overall_flag,
            "confidence_flag": confidence_flag,
            "parameter_regime": parameter_regime,
            "pruning_flag": pruning_flag,
            "material_flag": material_flag,
            "summary": f"{overall_flag}; {material_flag.lower()} captured at {valid_fraction:.1%} analysis coverage.",
            "warning": "Heuristic guidance only. High strain coverage means the pipeline can map a broad strain field; in this project it should be treated as stronger measurable distortion, not cleaner material.",
        })

    if tool in {"cluster", "cluster_atomic_environments"}:
        clustered_fraction = float(stats.get("clustered_fraction") or 0.0)
        hdb_noise_count = int(stats.get("hdb_noise_count") or 0)
        clustered_atom_count = int(stats.get("clustered_atom_count") or 0)
        defect_prob_max = float(stats.get("defect_prob_max") or 0.0)
        noise_fraction = hdb_noise_count / max(clustered_atom_count, 1)
        coverage_flag = "High Clustering Coverage" if clustered_fraction >= 0.5 else "Partial Clustering Coverage" if clustered_fraction >= 0.2 else "Sparse Clustering Coverage"
        overall_flag = "Low Cluster Noise" if noise_fraction < 0.05 else "Moderate Cluster Noise" if noise_fraction < 0.12 else "High Cluster Noise"
        confidence_flag = "High Confidence" if defect_prob_max < 0.1 else "Medium Confidence" if defect_prob_max <= 0.4 else "Low Confidence"
        material_flag = (
            "Heterogeneous Atomic Environments"
            if defect_prob_max >= 0.4 or noise_fraction >= 0.12
            else "Mixed Atomic Environments"
            if defect_prob_max >= 0.1 or noise_fraction >= 0.05
            else "More Uniform Atomic Environments"
        )
        assessment.update({
            "overall_flag": overall_flag,
            "confidence_flag": confidence_flag,
            "coverage_flag": coverage_flag,
            "material_flag": material_flag,
            "summary": f"{coverage_flag}; {material_flag.lower()} with {noise_fraction:.1%} HDB noise among clustered atoms.",
            "warning": "Heuristic guidance only. High clustering coverage means more atoms can be grouped in feature space; it does not mean the sample is cleaner.",
        })

    if tool in {"defects", "detect_structural_defects"}:
        region_atom_fraction = float(stats.get("region_atom_fraction") or stats.get("defect_fraction") or 0.0)
        max_region_severity = float(stats.get("max_region_severity") or stats.get("defect_score_max") or 0.0)
        region_count = int(stats.get("region_count") or stats.get("defect_count") or 0)
        dominant_region_source = str(stats.get("dominant_region_source") or "unknown")
        overall_flag = "Low Region Burden" if region_atom_fraction < 0.01 and region_count <= 1 else "Localized Disorder Regions" if region_atom_fraction < 0.08 and region_count <= 4 else "Broad Disorder Regions"
        if max_region_severity >= 0.75 or region_atom_fraction >= 0.12:
            confidence_flag = "High Confidence"
            evidence_flag = "Strong Evidence"
        elif max_region_severity >= 0.5:
            confidence_flag = "Medium Confidence"
            evidence_flag = "Moderate Evidence"
        else:
            confidence_flag = "Medium Confidence"
            evidence_flag = "Tentative Evidence"
        material_flag = (
            "Cleaner Lattice Signature"
            if region_atom_fraction < 0.01 and region_count <= 1
            else "Localized Disorder Signal"
            if region_atom_fraction < 0.08
            else "Broad Defect Signal"
        )
        source_flag = {
            "peak": "Peak-Finding-Led",
            "cluster": "Clustering-Led",
            "strain": "Strain-Led",
        }.get(dominant_region_source, "Mixed Evidence")
        assessment.update({
            "overall_flag": overall_flag,
            "confidence_flag": confidence_flag,
            "evidence_flag": evidence_flag,
            "material_flag": material_flag,
            "source_flag": source_flag,
            "summary": f"{overall_flag}; {material_flag.lower()} with {region_count} abnormal regions covering {region_atom_fraction:.1%} of atoms.",
            "warning": "Heuristic guidance only. Region screening highlights abnormal zones, not confirmed physical defect types.",
        })
        peak_context = stats.get("peak_context") or {}
        strain_context = stats.get("strain_context") or {}
        if peak_context.get("peak_csv_used") or peak_context.get("peak_manifest_used"):
            assessment["peak_context_flag"] = "Peak Context Used"
        if strain_context.get("strain_csv_used") or strain_context.get("strain_manifest_used"):
            assessment["strain_context_flag"] = "Strain Context Used"

    stats["assessment"] = assessment
    return stats


def _build_legacy_summary(folder: Path) -> dict[str, Any]:
    stage = _infer_stage(folder)
    atoms_path = _artifact_by_glob(folder, "*_atoms.csv")
    strain_path = _artifact_by_glob(folder, "*_strain.csv")
    cluster_path = _artifact_by_glob(folder, "*_clustered.csv")
    defects_path = _artifact_by_glob(folder, "*_defects.csv") or _artifact_by_glob(folder, "*_defect_regions.csv")

    return {
        "run_manifest": None,
        "server": "legacy",
        "tool": stage,
        "output_dir": str(folder),
        "inputs": [],
        "parameters": {},
        "metrics": {},
        "atoms": _summarize_atoms_csv(atoms_path),
        "strain": _summarize_strain_csv(strain_path),
        "cluster": _summarize_cluster_csv(cluster_path),
        "defects": _summarize_defect_csv(defects_path),
        "stage_stats": _normalize_stage_stats(
            stage,
            (
                _load_legacy_stats(folder, "*_peak_stats.json")
                or _load_legacy_stats(folder, "*_strain_stats.json")
                or _load_legacy_stats(folder, "*_cluster_stats.json")
                or _load_legacy_stats(folder, "*_defect_stats.json")
                or _load_legacy_stats(folder, "*_defect_region_stats.json")
            ),
        ),
    }


def build_run_summary(path: str) -> dict[str, Any]:
    resolved, mode = _resolve_run_input(path)
    if mode == "manifest":
        manifest = _load_manifest(str(resolved))
        stage_stats = (
            _artifact_json(manifest, "_peak_stats.json")
            or _artifact_json(manifest, "_strain_stats.json")
            or _artifact_json(manifest, "_cluster_stats.json")
            or _artifact_json(manifest, "_defect_stats.json")
            or _artifact_json(manifest, "_defect_region_stats.json")
        )
        tool = manifest.tool
        return {
            "run_manifest": str(resolved),
            "server": manifest.server,
            "tool": tool,
            "output_dir": manifest.output_dir,
            "inputs": manifest.inputs,
            "parameters": manifest.parameters,
            "metrics": manifest.metrics,
            "atoms": _summarize_atoms_csv(_artifact_path(manifest, "_atoms.csv")),
            "strain": _summarize_strain_csv(_artifact_path(manifest, "_strain.csv")),
            "cluster": _summarize_cluster_csv(_artifact_path(manifest, "_clustered.csv")),
            "defects": _summarize_defect_csv(
                _artifact_path(manifest, "_defects.csv") or _artifact_path(manifest, "_defect_regions.csv")
            ),
            "stage_stats": _normalize_stage_stats(tool, stage_stats),
        }
    return _build_legacy_summary(resolved)


def _compute_screening_outcome(run_a: dict[str, Any], run_b: dict[str, Any]) -> dict[str, Any]:
    decision = "Accept"
    reasons: list[str] = []

    for label, run in [("Run A", run_a), ("Run B", run_b)]:
        stats = run.get("stage_stats") or {}
        tool = str(run.get("tool", ""))
        assessment = stats.get("assessment") or {}
        confidence_flag = str(assessment.get("confidence_flag", ""))

        if tool in {"strain", "compute_strain_map"}:
            valid_fraction = float(stats.get("valid_strain_fraction") or 0.0)
            min_neighbors = int(stats.get("min_axis_neighbors") or 0)
            if valid_fraction < 0.2:
                decision = "Escalate"
                reasons.append(f"{label}: strain coverage is very sparse")
            elif valid_fraction < 0.5 and decision != "Escalate":
                decision = "Review"
                reasons.append(f"{label}: strain coverage is only partial")
            if min_neighbors <= 0 and decision != "Escalate":
                decision = "Review"
                reasons.append(f"{label}: strain settings are very permissive")

        if tool in {"cluster", "cluster_atomic_environments"}:
            clustered_fraction = float(stats.get("clustered_fraction") or 0.0)
            defect_prob_max = float(stats.get("defect_prob_max") or 0.0)
            hdb_noise_count = int(stats.get("hdb_noise_count") or 0)
            clustered_atom_count = int(stats.get("clustered_atom_count") or 0)
            noise_fraction = hdb_noise_count / max(clustered_atom_count, 1)
            if clustered_fraction < 0.2:
                decision = "Escalate"
                reasons.append(f"{label}: clustering coverage is too low")
            elif clustered_fraction < 0.5 and decision != "Escalate":
                decision = "Review"
                reasons.append(f"{label}: clustering coverage is incomplete")
            if defect_prob_max > 0.4 and decision != "Escalate":
                decision = "Review"
                reasons.append(f"{label}: clustering uncertainty is elevated")
            if noise_fraction > 0.12 and decision != "Escalate":
                decision = "Review"
                reasons.append(f"{label}: clustering has high HDB noise")

        if tool in {"defects", "detect_structural_defects"}:
            region_atom_fraction = float(stats.get("region_atom_fraction") or stats.get("defect_fraction") or 0.0)
            max_region_severity = float(stats.get("max_region_severity") or stats.get("defect_score_max") or 0.0)
            region_count = int(stats.get("region_count") or stats.get("defect_count") or 0)
            if region_atom_fraction > 0.08 or max_region_severity >= 0.75:
                decision = "Escalate"
                reasons.append(f"{label}: broad abnormal regions are present")
            elif region_atom_fraction > 0.01 or region_count > 1 and decision != "Escalate":
                decision = "Review"
                reasons.append(f"{label}: localized abnormal regions need review")

        if confidence_flag == "Low Confidence" and decision == "Accept":
            decision = "Review"
            reasons.append(f"{label}: low-confidence stage assessment")

    if not reasons:
        reasons.append("No major QC issues exceeded review thresholds.")

    return {"decision": decision, "reasons": reasons}


def build_comparison_context(
    run_a: str,
    run_b: str,
    question: str = "",
    report_id: str = "",
    prepared_by: str = "",
    approved_by: str = "",
) -> dict[str, Any]:
    run_a_summary = build_run_summary(run_a)
    run_b_summary = build_run_summary(run_b)
    screening = _compute_screening_outcome(run_a_summary, run_b_summary)
    return {
        "question": question.strip(),
        "report_meta": {
            "report_id": report_id.strip(),
            "prepared_by": prepared_by.strip(),
            "approved_by": approved_by.strip(),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        "screening_outcome": screening,
        "run_a": run_a_summary,
        "run_b": run_b_summary,
    }


def build_review_prompt(context: dict[str, Any]) -> str:
    question = context.get("question") or (
        "Compare these two semiconductor metrology runs and write a concise client-facing report."
    )
    payload = json.dumps(context, indent=2)
    return (
        "You are a semiconductor metrology reporting assistant. "
        "Use only the structured run data provided below. "
        "Do not invent image observations, raw microscope conditions, or process causes not supported by the data. "
        "Important terminology rule: material quality means a cleaner lattice with lower defect evidence and stronger atomic regularity. "
        "Do not equate high strain coverage or high clustering coverage with better sample quality. "
        "In this pipeline, high strain or clustering coverage means the analysis captured a broader measurable distortion or variation field. "
        "When discussing which sample is cleaner, prioritize peak regularity metrics such as paired fraction and sublattice balance for two-sublattice runs only, "
        "plus defect-region evidence such as region count, region atom fraction, dominant source, GMM flags, and neighbor deficiency. "
        "Return valid JSON with exactly these keys: "
        "executive_summary (string), key_differences (array of strings), qc_flags (array of strings), "
        "client_interpretation (string), recommended_next_steps (array of strings), limitations (array of strings). "
        "Keep each item concise and professional. "
        f"Scientific question: {question}\n\n"
        f"Structured run comparison data:\n{payload}"
    )


def generate_report_with_ollama(context: dict[str, Any]) -> dict[str, Any]:
    client = ollama.Client(host=OLLAMA_BASE_URL)
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise semiconductor metrology analyst. "
                    "Base every conclusion on supplied structured run data only."
                ),
            },
            {"role": "user", "content": build_review_prompt(context)},
        ],
        options={"temperature": 0.2},
    )
    text = response["message"]["content"].strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:].lstrip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    return {
        "executive_summary": text,
        "key_differences": [],
        "qc_flags": [],
        "client_interpretation": "",
        "recommended_next_steps": [],
        "limitations": ["The LLM response did not return the requested structured JSON format."],
    }


def _fmt_num(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if abs(value) >= 1:
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _stage_summary_line(run: dict[str, Any]) -> str:
    stats = run.get("stage_stats") or {}
    tool = str(run.get("tool", "unknown"))
    assessment = stats.get("assessment") or {}
    material_flag = assessment.get("material_flag")
    if tool in {"strain", "compute_strain_map"}:
        return (
            f"strain analysis coverage {_fmt_num(stats.get('valid_strain_fraction'))}, "
            f"material signal {material_flag or '-'}"
        )
    if tool in {"cluster", "cluster_atomic_environments"}:
        return (
            f"clustering analysis coverage {_fmt_num(stats.get('clustered_fraction'))}, "
            f"material signal {material_flag or '-'}"
        )
    if tool in {"defects", "detect_structural_defects"}:
        return (
            f"abnormal regions {stats.get('region_count', stats.get('defect_count', '-'))}, "
            f"material signal {material_flag or '-'}"
        )
    if tool in {"peak", "peak_find_atoms"}:
        return (
            f"pairing coverage {_fmt_num(stats.get('paired_atom_fraction'))}, "
            f"material signal {material_flag or '-'}"
        )
    return "stage summary unavailable"


def _run_rows(context: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [("Run A", context["run_a"]), ("Run B", context["run_b"])]


def _markdown_run_table(context: dict[str, Any]) -> str:
    lines = [
        "| Run | Stage | Source | Key Summary |",
        "|---|---|---|---|",
    ]
    for label, run in _run_rows(context):
        lines.append(
            f"| {label} | {run.get('tool', '-')} | `{run.get('output_dir', '-')}` | {_stage_summary_line(run)} |"
        )
    return "\n".join(lines)


def _markdown_stats(run: dict[str, Any]) -> list[str]:
    stats = run.get("stage_stats") or {}
    if not stats:
        return ["- No stage statistics available."]
    lines: list[str] = []
    for key, value in stats.items():
        if key == "assessment":
            continue
        pretty = key.replace("_", " ")
        lines.append(f"- {pretty}: `{_fmt_num(value)}`")
    assessment = stats.get("assessment") or {}
    for key in ("overall_flag", "material_flag", "confidence_flag", "sublattice_flag", "pairing_flag", "parameter_regime", "pruning_flag", "coverage_flag", "evidence_flag"):
        if assessment.get(key):
            pretty = key.replace("_", " ")
            lines.append(f"- {pretty}: {assessment[key]}")
    if assessment.get("summary"):
        lines.append(f"- assessment: {assessment['summary']}")
    if assessment.get("warning"):
        lines.append(f"- warning: {assessment['warning']}")
    return lines


def render_professional_markdown(context: dict[str, Any], narrative: dict[str, Any]) -> str:
    question = context.get("question") or "Comparison of two microscopy analysis runs."
    meta = context.get("report_meta") or {}
    date_str = meta.get("generated_at") or datetime.now().strftime("%Y-%m-%d %H:%M")
    screening = context.get("screening_outcome") or {}
    run_a = context["run_a"]
    run_b = context["run_b"]

    sections = [
        "# Semiconductor Metrology Comparison Report",
        "",
        "## Report Header",
        f"- Report ID: `{meta.get('report_id') or 'AUTO'}`",
        f"- Report date: `{date_str}`",
        f"- Model: `{OLLAMA_MODEL}`",
        f"- Prepared by: `{meta.get('prepared_by') or 'Not specified'}`",
        f"- Approved by: `{meta.get('approved_by') or 'Not specified'}`",
        f"- Screening Outcome: `{screening.get('decision', 'Review')}`",
        f"- Scope: compare `{run_a.get('tool', '-')}` and `{run_b.get('tool', '-')}` results",
        f"- Scientific question: {question}",
        "",
        "## Terminology Note",
        (
            "Detection, strain, and clustering coverage measure analysis completeness or the extent of measurable "
            "variation. They do not by themselves mean the sample is cleaner. In this project, cleaner material is "
            "inferred mainly from stronger atomic regularity and lower defect evidence."
        ),
        "",
        "## Runs Compared",
        _markdown_run_table(context),
        "",
        "## Screening Decision",
    ]
    sections.extend([f"- {item}" for item in (screening.get("reasons") or [])])
    sections.extend([
        "",
        "## Executive Summary",
        narrative.get("executive_summary", "").strip() or "No executive summary generated.",
        "",
        "## Key Differences",
    ])
    key_diffs = narrative.get("key_differences") or []
    sections.extend([f"- {item}" for item in key_diffs] or ["- No key differences generated."])
    sections.extend([
        "",
        "## Quantitative Summary",
        "### Run A",
        *(_markdown_stats(run_a)),
        "",
        "### Run B",
        *(_markdown_stats(run_b)),
        "",
        "## QC Flags",
    ])
    qc_flags = narrative.get("qc_flags") or []
    sections.extend([f"- {item}" for item in qc_flags] or ["- No QC flags generated."])
    sections.extend([
        "",
        "## Client Interpretation",
        narrative.get("client_interpretation", "").strip() or "No client interpretation generated.",
        "",
        "## Recommended Next Steps",
    ])
    next_steps = narrative.get("recommended_next_steps") or []
    sections.extend([f"- {item}" for item in next_steps] or ["- No next steps generated."])
    sections.extend([
        "",
        "## Limitations",
    ])
    limits = narrative.get("limitations") or []
    sections.extend([f"- {item}" for item in limits] or ["- This report is based on structured analysis outputs only."])
    sections.extend([
        "",
        "## Traceability",
        f"- Run A folder: `{run_a.get('output_dir', '-')}`",
        f"- Run B folder: `{run_b.get('output_dir', '-')}`",
        f"- Run A manifest: `{run_a.get('run_manifest') or 'legacy folder input'}`",
        f"- Run B manifest: `{run_b.get('run_manifest') or 'legacy folder input'}`",
        "",
        "> Guidance only: this report standardizes presentation and interpretation, but it does not replace scientific review.",
        "",
    ])
    return "\n".join(sections)


def render_professional_html(context: dict[str, Any], narrative: dict[str, Any]) -> str:
    question = context.get("question") or "Comparison of two microscopy analysis runs."
    meta = context.get("report_meta") or {}
    date_str = meta.get("generated_at") or datetime.now().strftime("%Y-%m-%d %H:%M")
    screening = context.get("screening_outcome") or {}
    run_a = context["run_a"]
    run_b = context["run_b"]

    def ul(items: list[str]) -> str:
        if not items:
            items = ["No content generated."]
        return "<ul>" + "".join(f"<li>{escape(str(item))}</li>" for item in items) + "</ul>"

    def stats_block(run: dict[str, Any]) -> str:
        lines = _markdown_stats(run)
        return ul([line[2:] if line.startswith("- ") else line for line in lines])

    body = f"""
    <h1>Semiconductor Metrology Comparison Report</h1>
    <h2>Report Header</h2>
    {ul([
        f"Report ID: {meta.get('report_id') or 'AUTO'}",
        f"Report date: {date_str}",
        f"Model: {OLLAMA_MODEL}",
        f"Prepared by: {meta.get('prepared_by') or 'Not specified'}",
        f"Approved by: {meta.get('approved_by') or 'Not specified'}",
        f"Screening Outcome: {screening.get('decision', 'Review')}",
        f"Scope: compare {run_a.get('tool', '-')} and {run_b.get('tool', '-')}",
        f"Scientific question: {question}",
    ])}
    <h2>Terminology Note</h2>
    <p>Detection, strain, and clustering coverage measure analysis completeness or the extent of measurable variation. They do not by themselves mean the sample is cleaner. In this project, cleaner material is inferred mainly from stronger atomic regularity and lower defect evidence.</p>
    <h2>Runs Compared</h2>
    <table>
      <tr><th>Run</th><th>Stage</th><th>Source</th><th>Key Summary</th></tr>
      <tr><td>Run A</td><td>{escape(str(run_a.get('tool', '-')))}</td><td><code>{escape(str(run_a.get('output_dir', '-')))}</code></td><td>{escape(_stage_summary_line(run_a))}</td></tr>
      <tr><td>Run B</td><td>{escape(str(run_b.get('tool', '-')))}</td><td><code>{escape(str(run_b.get('output_dir', '-')))}</code></td><td>{escape(_stage_summary_line(run_b))}</td></tr>
    </table>
    <h2>Screening Decision</h2>
    {ul(screening.get('reasons') or [])}
    <h2>Executive Summary</h2>
    <p>{escape(narrative.get('executive_summary', '') or 'No executive summary generated.')}</p>
    <h2>Key Differences</h2>
    {ul(narrative.get('key_differences') or [])}
    <h2>Quantitative Summary</h2>
    <h3>Run A</h3>
    {stats_block(run_a)}
    <h3>Run B</h3>
    {stats_block(run_b)}
    <h2>QC Flags</h2>
    {ul(narrative.get('qc_flags') or [])}
    <h2>Client Interpretation</h2>
    <p>{escape(narrative.get('client_interpretation', '') or 'No client interpretation generated.')}</p>
    <h2>Recommended Next Steps</h2>
    {ul(narrative.get('recommended_next_steps') or [])}
    <h2>Limitations</h2>
    {ul(narrative.get('limitations') or ['This report is based on structured analysis outputs only.'])}
    <h2>Traceability</h2>
    {ul([
        f"Run A folder: {run_a.get('output_dir', '-')}",
        f"Run B folder: {run_b.get('output_dir', '-')}",
        f"Run A manifest: {run_a.get('run_manifest') or 'legacy folder input'}",
        f"Run B manifest: {run_b.get('run_manifest') or 'legacy folder input'}",
    ])}
    <blockquote>Guidance only: this report standardizes presentation and interpretation, but it does not replace scientific review.</blockquote>
    """
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Semiconductor Metrology Comparison Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; color: #1f2937; line-height: 1.55; }}
    h1 {{ font-size: 28px; margin-bottom: 8px; }}
    h2 {{ margin-top: 28px; font-size: 20px; border-bottom: 1px solid #d1d5db; padding-bottom: 6px; }}
    h3 {{ margin-top: 18px; font-size: 16px; }}
    p, li, blockquote {{ font-size: 14px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; margin-top: 8px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    blockquote {{ border-left: 4px solid #9ca3af; margin-left: 0; padding-left: 12px; color: #4b5563; }}
    code {{ background: #f3f4f6; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
{body}
</body>
</html>"""
