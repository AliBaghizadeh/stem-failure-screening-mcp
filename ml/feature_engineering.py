"""
ml/feature_engineering.py
Per-atom feature extraction and HDBSCAN/GMM clustering.
Input: strain CSV (output of core/strain_analysis.py)
Output: same CSV enriched with cluster label + defect probability
"""
import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.image_utils import normalize_hyperspy_load, signal_to_2d_float_array

logger = logging.getLogger("stem-atom-finder")


def _load_signal(image_path: str):
    import hyperspy.api as hs

    return normalize_hyperspy_load(hs.load(image_path), image_path)


def _clustering_assessment(clustered_fraction: float, hdb_noise_count: int,
                           clustered_atom_count: int, defect_prob_max: float) -> dict:
    noise_fraction = hdb_noise_count / max(clustered_atom_count, 1)

    if clustered_fraction >= 0.5:
        coverage_flag = "High Clustering Coverage"
    elif clustered_fraction >= 0.2:
        coverage_flag = "Partial Clustering Coverage"
    else:
        coverage_flag = "Sparse Clustering Coverage"

    if noise_fraction < 0.05:
        stability_flag = "Low Cluster Noise"
    elif noise_fraction < 0.12:
        stability_flag = "Moderate Cluster Noise"
    else:
        stability_flag = "High Cluster Noise"

    if defect_prob_max < 0.1:
        confidence_flag = "High Confidence"
    elif defect_prob_max <= 0.4:
        confidence_flag = "Medium Confidence"
    else:
        confidence_flag = "Low Confidence"

    if defect_prob_max >= 0.4 or noise_fraction >= 0.12:
        material_flag = "Heterogeneous Atomic Environments"
    elif defect_prob_max >= 0.1 or noise_fraction >= 0.05:
        material_flag = "Mixed Atomic Environments"
    else:
        material_flag = "More Uniform Atomic Environments"

    return {
        "overall_flag": stability_flag,
        "confidence_flag": confidence_flag,
        "coverage_flag": coverage_flag,
        "material_flag": material_flag,
        "summary": (
            f"{coverage_flag}; {material_flag.lower()} with "
            f"{noise_fraction:.1%} HDB noise among clustered atoms."
        ),
        "warning": (
            "Heuristic guidance only. High clustering coverage means more atoms can be grouped in feature space; "
            "it does not mean the sample is cleaner."
        ),
    }


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build normalised per-atom feature matrix from strain CSV columns."""
    feature_cols = []
    candidates = ["e_xx", "e_yy", "e_xy", "a1_len", "a2_len", "ref_dx", "ref_dy"]
    for c in candidates:
        if c in df.columns:
            feature_cols.append(c)

    sub = df[feature_cols].copy()
    sub = sub.dropna()

    # Z-score normalise per feature
    mu = sub.mean()
    sd = sub.std().replace(0, 1)
    X = ((sub - mu) / sd).values
    return X, feature_cols, sub.index


def run_hdbscan(X: np.ndarray, min_cluster_size: int = 15) -> np.ndarray:
    """
    Run clustering. HDBSCAN has DLL issues on some Windows systems.
    Falling back to stable SKLearn DBSCAN by default for robustness.
    """
    try:
        from sklearn.cluster import DBSCAN
        # eps=0.5 is a safe normalized starting point; min_samples = user's cluster size
        clusterer = DBSCAN(eps=0.5, min_samples=min_cluster_size)
        labels = clusterer.fit_predict(X)
        logger.info(f"  Clustering (DBSCAN): {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
        return labels
    except Exception as e:
        logger.error(f"  Clustering failed: {e}")
        return np.zeros(len(X), dtype=int)


def run_gmm(X: np.ndarray, n_components: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Run GMM clustering. Returns (labels, defect_probability)."""
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=n_components, covariance_type="full",
                          random_state=42, n_init=3)
    gmm.fit(X)
    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)

    # Defect probability = 1 - max(prob) → how uncertain the assignment is
    defect_prob = 1.0 - probs.max(axis=1)
    logger.info(f"  GMM: {n_components} components fitted, "
                f"{(defect_prob > 0.4).sum()} uncertain atoms (prob>0.4)")
    return labels, defect_prob


def run_clustering(csv_path: str, image_path: str = None, out_dir: str = None,
                   min_cluster_size: int = 15, gmm_components: int = 4,
                   flip_h: bool = False, flip_v: bool = False):
    """Main entry: load strain CSV, cluster, save enriched CSV and overlay plot."""
    if not out_dir:
        out_dir = os.path.dirname(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    X, feature_cols, valid_idx = build_feature_matrix(df)
    logger.info(f"  Feature matrix: {X.shape[0]} atoms × {X.shape[1]} features "
                f"({', '.join(feature_cols)})")

    # HDBSCAN
    hdb_labels = run_hdbscan(X, min_cluster_size=min_cluster_size)
    df.loc[valid_idx, "cluster_hdbscan"] = hdb_labels

    # GMM
    gmm_labels, defect_prob = run_gmm(X, n_components=gmm_components)
    df.loc[valid_idx, "cluster_gmm"] = gmm_labels
    df.loc[valid_idx, "defect_prob"] = defect_prob

    # Save enriched CSV
    base = os.path.basename(csv_path).replace(".csv", "")
    out_csv = os.path.join(out_dir, f"{base}_clustered.csv")
    df.to_csv(out_csv, index=False)
    logger.info(f"  Saved: {out_csv}")

    hdb_series = df.loc[valid_idx, "cluster_hdbscan"].dropna()
    gmm_series = df.loc[valid_idx, "cluster_gmm"].dropna()
    defect_series = df.loc[valid_idx, "defect_prob"].dropna()
    stats = {
        "input_atom_count": int(len(df)),
        "clustered_atom_count": int(len(valid_idx)),
        "clustered_fraction": float(len(valid_idx) / max(len(df), 1)),
        "feature_columns": feature_cols,
        "min_cluster_size": int(min_cluster_size),
        "gmm_components": int(gmm_components),
        "hdb_noise_count": int((hdb_series == -1).sum()) if len(hdb_series) else 0,
        "hdb_cluster_count": int(len(set(hdb_series.astype(int))) - (1 if (hdb_series == -1).any() else 0)) if len(hdb_series) else 0,
        "gmm_cluster_count": int(len(set(gmm_series.astype(int)))) if len(gmm_series) else 0,
        "defect_prob_mean": float(defect_series.mean()) if len(defect_series) else 0.0,
        "defect_prob_max": float(defect_series.max()) if len(defect_series) else 0.0,
        "assessment": _clustering_assessment(
            clustered_fraction=float(len(valid_idx) / max(len(df), 1)),
            hdb_noise_count=int((hdb_series == -1).sum()) if len(hdb_series) else 0,
            clustered_atom_count=int(len(valid_idx)),
            defect_prob_max=float(defect_series.max()) if len(defect_series) else 0.0,
        ),
    }
    stats_path = os.path.join(out_dir, f"{base}_cluster_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  Saved: {stats_path}")

    # ── Cluster overlay plot ──────────────────────────────────────────────────
    valid = df.loc[valid_idx].copy()
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"Atom Clustering — {base}", fontsize=13, fontweight="bold")

    x_min = float(valid["x"].min()) if not valid.empty else 0.0
    x_max = float(valid["x"].max()) if not valid.empty else 1.0
    y_min = float(valid["y"].min()) if not valid.empty else 0.0
    y_max = float(valid["y"].max()) if not valid.empty else 1.0

    if image_path and os.path.exists(image_path):
        try:
            import sys
            # Use same fast-load trick as batch_processor for consistency
            os.environ["HYPERSPY_DISABLE_PLUGIN_SCAN"] = "1"
            import hyperspy.api as hs
            bg = signal_to_2d_float_array(_load_signal(image_path))
            
            from skimage import exposure
            bg_display = exposure.rescale_intensity(bg, out_range=(0, 0.4))
            
            h, w = bg.shape[:2]
            extent = [0, w, h, 0] 
            x_min, x_max = 0.0, float(w)
            y_min, y_max = 0.0, float(h)
            
            axes[0].imshow(bg_display, cmap="gray", interpolation="nearest", origin='upper', extent=extent)
            axes[1].imshow(bg_display, cmap="gray", interpolation="nearest", origin='upper', extent=extent)
        except Exception as e:
            logger.warning(f"  Background image load failed: {e}")
            pass

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)
        if flip_h:
            ax.invert_xaxis()
        if flip_v:
            ax.invert_yaxis()

    # HDBSCAN panel
    sc0 = axes[0].scatter(valid["x"], valid["y"], c=valid["cluster_hdbscan"],
                          cmap="tab10", s=8, alpha=0.85, vmin=-1, vmax=9)
    axes[0].set_title("HDBSCAN Clusters\n(-1 = noise/defect candidate)", fontsize=10)
    axes[0].axis("off")
    plt.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04, label="Cluster ID")

    # GMM defect probability panel
    sc1 = axes[1].scatter(valid["x"], valid["y"], c=valid["defect_prob"],
                          cmap="hot_r", s=8, alpha=0.85, vmin=0, vmax=1)
    axes[1].set_title("GMM Defect Probability\n(high = structurally anomalous)", fontsize=10)
    axes[1].axis("off")
    plt.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04, label="P(defect)")

    plt.tight_layout()
    out_png = os.path.join(out_dir, f"{base}_cluster_map.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_png}")

    return out_csv
