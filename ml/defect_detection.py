"""
ml/defect_detection.py
Defect region screening from clustered strain data.
Input: _clustered.csv (output of ml/feature_engineering.py)
Output: per-atom region scores, region summary CSV, heatmap PNGs, stats JSON
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, label
from scipy.spatial import KDTree

from core.image_utils import normalize_hyperspy_load, signal_to_2d_float_array

logger = logging.getLogger("stem-atom-finder")

HDBSCAN_NOISE_LABEL = -1
GMM_DEFECT_PROB_THRESHOLD = 0.40


def _load_signal(image_path: str):
    import hyperspy.api as hs

    return normalize_hyperspy_load(hs.load(image_path), image_path)


def _robust_scale(values: pd.Series | np.ndarray, clip: float = 4.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.zeros(len(arr), dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return out
    clean = arr[finite]
    med = np.median(clean)
    mad = np.median(np.abs(clean - med))
    scale = 1.4826 * mad if mad > 1e-9 else np.std(clean)
    scale = float(scale) if scale and np.isfinite(scale) and scale > 1e-9 else 1.0
    z = np.abs((clean - med) / scale)
    out[finite] = np.clip(z / clip, 0.0, 1.0)
    return out


def _circular_irregularity(series: pd.Series | np.ndarray, clip_deg: float = 30.0) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    out = np.zeros(len(arr), dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return out
    clean = arr[finite]
    median_angle = float(np.median(clean))
    diff = np.abs(((clean - median_angle + 90.0) % 180.0) - 90.0)
    out[finite] = np.clip(diff / clip_deg, 0.0, 1.0)
    return out


def _dominant_cluster(series: pd.Series) -> float | None:
    clean = series.dropna()
    clean = clean[clean != HDBSCAN_NOISE_LABEL]
    if clean.empty:
        return None
    return float(clean.value_counts().idxmax())


def _local_mean(values: np.ndarray, coords: np.ndarray, radius: float) -> np.ndarray:
    if len(coords) == 0:
        return np.zeros(0, dtype=float)
    tree = KDTree(coords)
    out = np.zeros(len(coords), dtype=float)
    for i, coord in enumerate(coords):
        idx = tree.query_ball_point(coord, r=radius)
        out[i] = float(np.mean(values[idx])) if idx else float(values[i])
    return out


def _safe_column(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _build_atom_signals(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    coords = df[["x", "y"]].to_numpy(dtype=float)
    sub_mode = ""
    if "sublattice_mode" in df.columns and not df["sublattice_mode"].dropna().empty:
        sub_mode = str(df["sublattice_mode"].dropna().iloc[0]).strip().lower()
    split_applied = True
    if "sublattice_split_applied" in df.columns and not df["sublattice_split_applied"].dropna().empty:
        split_applied = bool(int(pd.to_numeric(df["sublattice_split_applied"], errors="coerce").fillna(1).iloc[0]))
    if sub_mode == "single":
        split_applied = False

    pair_distance = _safe_column(df, "pair_distance")
    pair_angle = _safe_column(df, "pair_angle")
    pair_missing = pair_distance.isna().astype(float).to_numpy()
    pair_distance_irreg = _robust_scale(pair_distance)
    pair_angle_irreg = _circular_irregularity(pair_angle)

    if "ref_dx" in df.columns:
        spacing_ref = float(pd.to_numeric(df["ref_dx"], errors="coerce").dropna().median())
    elif pair_distance.notna().any():
        spacing_ref = float(pair_distance.dropna().median())
    else:
        tree = KDTree(coords)
        dists, _ = tree.query(coords, k=2)
        spacing_ref = float(np.median(dists[:, 1]))
    spacing_ref = spacing_ref if np.isfinite(spacing_ref) and spacing_ref > 1e-6 else 8.0
    local_radius = spacing_ref * 2.0

    if split_applied:
        peak_base = (
            0.45 * pair_missing +
            0.35 * pair_distance_irreg +
            0.20 * pair_angle_irreg
        )
        peak_signal = 0.7 * peak_base + 0.3 * _local_mean(peak_base, coords, local_radius)
    else:
        peak_signal = np.zeros(len(df), dtype=float)

    e_xx = _safe_column(df, "e_xx")
    e_yy = _safe_column(df, "e_yy")
    e_xy = _safe_column(df, "e_xy")
    strain_mag = np.sqrt(np.nan_to_num(e_xx, nan=0.0) ** 2 + np.nan_to_num(e_yy, nan=0.0) ** 2 + np.nan_to_num(e_xy, nan=0.0) ** 2)
    strain_signal = _robust_scale(strain_mag) if np.isfinite(strain_mag).any() else np.zeros(len(df), dtype=float)

    cluster_hdb = _safe_column(df, "cluster_hdbscan")
    defect_prob = _safe_column(df, "defect_prob")
    dominant = _dominant_cluster(cluster_hdb)
    nonbackground = np.zeros(len(df), dtype=float)
    if dominant is not None:
        nonbackground = ((cluster_hdb.notna()) & (cluster_hdb != dominant) & (cluster_hdb != HDBSCAN_NOISE_LABEL)).astype(float).to_numpy()
    hdb_noise = (cluster_hdb == HDBSCAN_NOISE_LABEL).astype(float).to_numpy()
    gmm_anom = np.clip(np.nan_to_num(defect_prob, nan=0.0) / GMM_DEFECT_PROB_THRESHOLD, 0.0, 1.0)
    cluster_signal = 0.45 * nonbackground + 0.35 * hdb_noise + 0.20 * gmm_anom
    cluster_signal = 0.75 * cluster_signal + 0.25 * _local_mean(cluster_signal, coords, local_radius)

    df = df.copy()
    df["peak_irregularity"] = np.clip(peak_signal, 0.0, 1.0)
    df["strain_distortion"] = np.clip(strain_signal, 0.0, 1.0)
    df["cluster_abnormality"] = np.clip(cluster_signal, 0.0, 1.0)
    df["peak_signal_enabled"] = int(split_applied)
    return df, spacing_ref


def _compute_disorder_score(df: pd.DataFrame, peak_weight: float, cluster_weight: float, strain_weight: float) -> pd.DataFrame:
    total_weight = max(peak_weight + cluster_weight + strain_weight, 1e-9)
    w_peak = peak_weight / total_weight
    w_cluster = cluster_weight / total_weight
    w_strain = strain_weight / total_weight
    df = df.copy()
    df["region_disorder_score"] = (
        w_peak * df["peak_irregularity"] +
        w_cluster * df["cluster_abnormality"] +
        w_strain * df["strain_distortion"]
    )
    return df


def _load_background(image_path: str | None):
    if not image_path or not os.path.exists(image_path):
        return None, None
    try:
        os.environ["HYPERSPY_DISABLE_PLUGIN_SCAN"] = "1"
        s = _load_signal(image_path)
        bg = signal_to_2d_float_array(s)
        bg = np.asarray(bg, dtype=float)
        lo, hi = np.percentile(bg, [1, 99])
        bg = np.clip((bg - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        h, w = bg.shape[:2]
        return bg, (0.0, float(w), float(h), 0.0)
    except Exception as exc:
        logger.warning("  Defect region background load failed: %s", exc)
        return None, None


def _build_heatmap(
    df: pd.DataFrame,
    spacing_ref: float,
    region_threshold: float,
    blur_sigma: float,
    min_region_atoms: int,
    image_shape: tuple[int, int] | None = None,
):
    xs = df["x"].to_numpy(dtype=float)
    ys = df["y"].to_numpy(dtype=float)
    scores = df["region_disorder_score"].to_numpy(dtype=float)

    if image_shape is not None:
        height, width = image_shape
        x_min, x_max = 0.0, float(width)
        y_min, y_max = 0.0, float(height)
    else:
        x_min, x_max = float(np.nanmin(xs)), float(np.nanmax(xs))
        y_min, y_max = float(np.nanmin(ys)), float(np.nanmax(ys))

    bin_size = max(spacing_ref / 1.5, 2.0)
    width_bins = max(16, int(np.ceil((x_max - x_min) / bin_size)))
    height_bins = max(16, int(np.ceil((y_max - y_min) / bin_size)))

    x_edges = np.linspace(x_min, x_max, width_bins + 1)
    y_edges = np.linspace(y_min, y_max, height_bins + 1)
    score_sum, _, _ = np.histogram2d(ys, xs, bins=[y_edges, x_edges], weights=scores)
    count_map, _, _ = np.histogram2d(ys, xs, bins=[y_edges, x_edges])
    smooth_scores = gaussian_filter(score_sum, sigma=blur_sigma)
    smooth_counts = gaussian_filter(count_map, sigma=blur_sigma)
    heatmap = np.divide(smooth_scores, np.maximum(smooth_counts, 1e-6))

    region_mask = heatmap >= float(region_threshold)
    labels, _ = label(region_mask)

    atom_x_idx = np.clip(np.digitize(xs, x_edges) - 1, 0, width_bins - 1)
    atom_y_idx = np.clip(np.digitize(ys, y_edges) - 1, 0, height_bins - 1)
    atom_region_ids = labels[atom_y_idx, atom_x_idx].astype(int)

    df = df.copy()
    df["defect_region_id"] = atom_region_ids

    region_rows: list[dict[str, float | int]] = []
    keep_regions: set[int] = set()
    for region_id in sorted(int(v) for v in np.unique(atom_region_ids) if v > 0):
        sub = df[df["defect_region_id"] == region_id]
        if len(sub) < min_region_atoms:
            df.loc[df["defect_region_id"] == region_id, "defect_region_id"] = 0
            labels[labels == region_id] = 0
            continue
        keep_regions.add(region_id)
        peak_mean = float(sub["peak_irregularity"].mean())
        cluster_mean = float(sub["cluster_abnormality"].mean())
        strain_mean = float(sub["strain_distortion"].mean())
        dominant_source = max(
            [("peak", peak_mean), ("cluster", cluster_mean), ("strain", strain_mean)],
            key=lambda item: item[1],
        )[0]
        region_rows.append(
            {
                "region_id": region_id,
                "atom_count": int(len(sub)),
                "mean_severity": float(sub["region_disorder_score"].mean()),
                "max_severity": float(sub["region_disorder_score"].max()),
                "x_center": float(sub["x"].mean()),
                "y_center": float(sub["y"].mean()),
                "dominant_source": dominant_source,
            }
        )

    if keep_regions:
        remap = {old: new for new, old in enumerate(sorted(keep_regions), start=1)}
        df["defect_region_id"] = df["defect_region_id"].map(lambda v: remap.get(int(v), 0))
        relabeled = np.zeros_like(labels)
        for old, new in remap.items():
            relabeled[labels == old] = new
        labels = relabeled
        for row in region_rows:
            row["region_id"] = remap[int(row["region_id"])]
    else:
        df["defect_region_id"] = 0
        labels[:] = 0
        region_rows = []

    df["is_region_atom"] = (df["defect_region_id"] > 0).astype(int)
    region_table = pd.DataFrame(region_rows)
    return df, heatmap, labels, region_table, (x_min, x_max, y_min, y_max), bin_size


def _region_assessment(region_count: int, region_atom_fraction: float, max_region_severity: float,
                       dominant_source: str, mean_region_severity: float) -> dict:
    if region_atom_fraction < 0.01 and region_count <= 1:
        overall_flag = "Low Region Burden"
        material_flag = "Cleaner Lattice Signature"
    elif region_atom_fraction < 0.08 and region_count <= 4:
        overall_flag = "Localized Disorder Regions"
        material_flag = "Localized Disorder Signal"
    else:
        overall_flag = "Broad Disorder Regions"
        material_flag = "Broad Defect Signal"

    if max_region_severity >= 0.75 or region_atom_fraction >= 0.12:
        confidence_flag = "High Confidence"
    elif max_region_severity >= 0.5:
        confidence_flag = "Medium Confidence"
    else:
        confidence_flag = "Tentative"

    evidence_flag = {
        "peak": "Peak-Finding-Led",
        "cluster": "Clustering-Led",
        "strain": "Strain-Led",
    }.get(dominant_source, "Mixed Evidence")

    return {
        "overall_flag": overall_flag,
        "confidence_flag": confidence_flag,
        "evidence_flag": evidence_flag,
        "material_flag": material_flag,
        "summary": (
            f"{overall_flag}; {material_flag.lower()} with {region_count} abnormal regions "
            f"covering {region_atom_fraction:.1%} of atoms."
        ),
        "warning": (
            "Heuristic guidance only. Region screening fuses peak finding, clustering, and strain signals "
            "to highlight abnormal zones rather than confirm physical defect types."
        ),
    }


def detect_defects(
    csv_path: str,
    image_path: str = None,
    out_dir: str = None,
    peak_csv_path: str | None = None,
    strain_csv_path: str | None = None,
    peak_manifest_path: str | None = None,
    strain_manifest_path: str | None = None,
    flip_h: bool = False,
    flip_v: bool = False,
    region_threshold: float = 0.45,
    blur_sigma: float = 2.0,
    min_region_atoms: int = 12,
    peak_weight: float = 0.45,
    cluster_weight: float = 0.30,
    strain_weight: float = 0.25,
):
    """Fuse peak, strain, and clustering signals into defect-region screening outputs."""
    if not out_dir:
        out_dir = os.path.dirname(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["x", "y"]).copy()
    if df.empty:
        raise ValueError("Clustered CSV contains no valid x/y coordinates.")

    peak_meta: dict = {}
    strain_meta: dict = {}
    peak_csv_file = Path(peak_csv_path).resolve() if peak_csv_path else None
    strain_csv_file = Path(strain_csv_path).resolve() if strain_csv_path else None
    peak_manifest_file = Path(peak_manifest_path).resolve() if peak_manifest_path else None
    strain_manifest_file = Path(strain_manifest_path).resolve() if strain_manifest_path else None

    if peak_csv_file and peak_csv_file.exists():
        peak_df = pd.read_csv(peak_csv_file).dropna(subset=["x", "y"]).copy()
        if {"pair_distance", "pair_angle"} - set(df.columns):
            join_cols = [c for c in ["pair_distance", "pair_angle", "sublattice", "intensity"] if c in peak_df.columns and c not in df.columns]
            if join_cols:
                peak_key = peak_df[["x", "y"]].round(3).astype(str).agg("|".join, axis=1)
                base_key = df[["x", "y"]].round(3).astype(str).agg("|".join, axis=1)
                peak_lookup = peak_df.assign(_key=peak_key).drop_duplicates("_key").set_index("_key")[join_cols]
                merged = base_key.to_frame(name="_key").join(peak_lookup, on="_key")
                for col in join_cols:
                    df[col] = merged[col].values
        peak_meta["source"] = str(peak_csv_file)

    if strain_csv_file and strain_csv_file.exists():
        strain_df = pd.read_csv(strain_csv_file).dropna(subset=["x", "y"]).copy()
        join_cols = [c for c in ["e_xx", "e_yy", "e_xy", "a1_len", "a2_len", "ref_dx", "ref_dy", "strain_valid"] if c in strain_df.columns and c not in df.columns]
        if join_cols:
            strain_key = strain_df[["x", "y"]].round(3).astype(str).agg("|".join, axis=1)
            base_key = df[["x", "y"]].round(3).astype(str).agg("|".join, axis=1)
            strain_lookup = strain_df.assign(_key=strain_key).drop_duplicates("_key").set_index("_key")[join_cols]
            merged = base_key.to_frame(name="_key").join(strain_lookup, on="_key")
            for col in join_cols:
                df[col] = merged[col].values
        strain_meta["source"] = str(strain_csv_file)

    def _load_manifest_stats(path_obj: Path | None, suffixes: tuple[str, ...]) -> dict:
        if not path_obj or not path_obj.exists():
            return {}
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
        artifacts = payload.get("artifacts") or []
        for artifact in artifacts:
            apath = artifact.get("path", "")
            if any(apath.endswith(suf) for suf in suffixes) and Path(apath).exists():
                return json.loads(Path(apath).read_text(encoding="utf-8"))
        return {}

    peak_stats = _load_manifest_stats(peak_manifest_file, ("_peak_stats.json",))
    strain_stats = _load_manifest_stats(strain_manifest_file, ("_strain_stats.json",))

    bg, extent = _load_background(image_path)
    image_shape = bg.shape[:2] if bg is not None else None

    df, spacing_ref = _build_atom_signals(df)
    df = _compute_disorder_score(df, peak_weight=peak_weight, cluster_weight=cluster_weight, strain_weight=strain_weight)
    df, heatmap, labels, region_table, bounds, bin_size = _build_heatmap(
        df,
        spacing_ref=spacing_ref,
        region_threshold=region_threshold,
        blur_sigma=blur_sigma,
        min_region_atoms=min_region_atoms,
        image_shape=image_shape,
    )

    base = os.path.basename(csv_path).replace(".csv", "")
    atom_csv = os.path.join(out_dir, f"{base}_defect_regions.csv")
    region_csv = os.path.join(out_dir, f"{base}_defect_region_summary.csv")
    stats_path = os.path.join(out_dir, f"{base}_defect_region_stats.json")
    heatmap_png = os.path.join(out_dir, f"{base}_defect_region_heatmap.png")
    overlay_png = os.path.join(out_dir, f"{base}_defect_region_overlay.png")

    df.to_csv(atom_csv, index=False)
    region_table.to_csv(region_csv, index=False)

    region_count = int(len(region_table))
    region_atom_count = int((df["defect_region_id"] > 0).sum())
    dominant_source = (
        str(region_table["dominant_source"].value_counts().idxmax())
        if not region_table.empty else "none"
    )
    largest_region_atoms = int(region_table["atom_count"].max()) if not region_table.empty else 0
    mean_region_severity = float(region_table["mean_severity"].mean()) if not region_table.empty else 0.0
    max_region_severity = float(region_table["max_severity"].max()) if not region_table.empty else 0.0

    stats = {
        "input_atom_count": int(len(df)),
        "region_count": region_count,
        "region_atom_count": region_atom_count,
        "region_atom_fraction": float(region_atom_count / max(len(df), 1)),
        "largest_region_atoms": largest_region_atoms,
        "mean_region_severity": mean_region_severity,
        "max_region_severity": max_region_severity,
        "peak_signal_mean": float(df["peak_irregularity"].mean()),
        "cluster_signal_mean": float(df["cluster_abnormality"].mean()),
        "strain_signal_mean": float(df["strain_distortion"].mean()),
        "dominant_region_source": dominant_source,
        "region_threshold": float(region_threshold),
        "blur_sigma": float(blur_sigma),
        "min_region_atoms": int(min_region_atoms),
        "peak_weight": float(peak_weight),
        "cluster_weight": float(cluster_weight),
        "strain_weight": float(strain_weight),
        "peak_context": {
            "peak_csv_used": bool(peak_csv_file and peak_csv_file.exists()),
            "peak_manifest_used": bool(peak_manifest_file and peak_manifest_file.exists()),
            "paired_atom_fraction": peak_stats.get("paired_atom_fraction"),
            "sublattice_balance_ratio": peak_stats.get("sublattice_balance_ratio"),
        },
        "strain_context": {
            "strain_csv_used": bool(strain_csv_file and strain_csv_file.exists()),
            "strain_manifest_used": bool(strain_manifest_file and strain_manifest_file.exists()),
            "valid_strain_fraction": strain_stats.get("valid_strain_fraction"),
            "parameter_regime": (strain_stats.get("assessment") or {}).get("parameter_regime"),
        },
        "assessment": _region_assessment(
            region_count=region_count,
            region_atom_fraction=float(region_atom_count / max(len(df), 1)),
            max_region_severity=max_region_severity,
            dominant_source=dominant_source,
            mean_region_severity=mean_region_severity,
        ),
    }
    if stats["peak_context"]["peak_csv_used"] or stats["peak_context"]["peak_manifest_used"]:
        stats["assessment"]["peak_context_flag"] = "Peak Context Used"
    if stats["strain_context"]["strain_csv_used"] or stats["strain_context"]["strain_manifest_used"]:
        stats["assessment"]["strain_context_flag"] = "Strain Context Used"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    x_min, x_max, y_min, y_max = bounds
    fig, ax = plt.subplots(figsize=(10, 8))
    if bg is not None and extent is not None:
        ax.imshow(bg, cmap="gray", interpolation="nearest", origin="upper", extent=[extent[0], extent[1], extent[2], extent[3]])
    else:
        ax.set_facecolor("black")
    hm = ax.imshow(
        heatmap,
        cmap="hot",
        alpha=0.75 if bg is not None else 0.95,
        origin="upper",
        extent=[x_min, x_max, y_max, y_min],
    )
    plt.colorbar(hm, ax=ax, fraction=0.03, pad=0.04, label="Region disorder score")
    ax.set_title("Defect Region Heatmap")
    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    if flip_h:
        ax.invert_xaxis()
    if flip_v:
        ax.invert_yaxis()
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(heatmap_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    if bg is not None and extent is not None:
        ax2.imshow(bg, cmap="gray", interpolation="nearest", origin="upper", extent=[extent[0], extent[1], extent[2], extent[3]])
    else:
        ax2.set_facecolor("black")
    if heatmap.max() > 0:
        ax2.imshow(
            heatmap,
            cmap="hot",
            alpha=0.45,
            origin="upper",
            extent=[x_min, x_max, y_max, y_min],
        )
    if np.any(labels > 0):
        yy = np.linspace(y_min, y_max, labels.shape[0])
        xx = np.linspace(x_min, x_max, labels.shape[1])
        XX, YY = np.meshgrid(xx, yy)
        ax2.contour(XX, YY, labels, levels=np.unique(labels[labels > 0]), colors="cyan", linewidths=1.2)
    region_atoms = df[df["defect_region_id"] > 0]
    if not region_atoms.empty:
        sc = ax2.scatter(
            region_atoms["x"],
            region_atoms["y"],
            c=region_atoms["region_disorder_score"],
            cmap="hot",
            s=10,
            alpha=0.8,
            vmin=0,
            vmax=1,
        )
        plt.colorbar(sc, ax=ax2, fraction=0.03, pad=0.04, label="Atom disorder score")
    for _, row in region_table.iterrows():
        ax2.text(row["x_center"], row["y_center"], f"R{int(row['region_id'])}", color="white", fontsize=9, ha="center", va="center")
    ax2.set_title("Defect Region Overlay")
    ax2.set_aspect("equal")
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_max, y_min)
    if flip_h:
        ax2.invert_xaxis()
    if flip_v:
        ax2.invert_yaxis()
    ax2.axis("off")
    plt.tight_layout()
    fig2.savefig(overlay_png, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    logger.info(
        "  Defect region screening: %d regions, %d / %d atoms inside regions (%.1f%%)",
        region_count,
        region_atom_count,
        len(df),
        100.0 * region_atom_count / max(len(df), 1),
    )
    return atom_csv
