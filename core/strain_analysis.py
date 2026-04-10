import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
from scipy.spatial import KDTree
from skimage import exposure

from core.image_utils import normalize_hyperspy_load, signal_to_2d_float_array

logger = logging.getLogger("stem-atom-finder")


def _load_signal(image_path: str):
    import hyperspy.api as hs

    return normalize_hyperspy_load(hs.load(image_path), image_path)


def _strain_assessment(valid_fraction: float, pruned_fraction: float,
                       axis_tolerance_scale: float, min_axis_neighbors: int) -> dict:
    if valid_fraction >= 0.6:
        overall_flag = "High Strain Coverage"
    elif valid_fraction >= 0.35:
        overall_flag = "Moderate Strain Coverage"
    elif valid_fraction >= 0.2:
        overall_flag = "Localized Strain Coverage"
    else:
        overall_flag = "Sparse Strain Coverage"

    if valid_fraction >= 0.8:
        material_flag = "Widespread Lattice Distortion Signal"
    elif valid_fraction >= 0.5:
        material_flag = "Moderate Lattice Distortion Signal"
    elif valid_fraction >= 0.2:
        material_flag = "Localized Lattice Distortion Signal"
    else:
        material_flag = "Weak / Incomplete Distortion Signal"

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

    if pruned_fraction > 0.05:
        pruning_flag = "Aggressive Pruning"
    elif pruned_fraction > 0:
        pruning_flag = "Light Pruning"
    else:
        pruning_flag = "No Pruning"

    return {
        "overall_flag": overall_flag,
        "confidence_flag": confidence_flag,
        "parameter_regime": parameter_regime,
        "pruning_flag": pruning_flag,
        "material_flag": material_flag,
        "summary": (
            f"{overall_flag}; {material_flag.lower()} captured at "
            f"{valid_fraction:.1%} analysis coverage."
        ),
        "warning": (
            "Heuristic guidance only. High strain coverage means the pipeline can map a broad strain field; "
            "in this project it should be treated as stronger measurable distortion, not cleaner material."
        ),
    }

def calculate_local_strain(
    coords: np.ndarray,
    ref_vectors: np.ndarray = None,
    axis_tolerance_scale: float = 0.7,
    min_axis_neighbors: int = 2,
):
    """
    Calculate local strain tensor for each atom based on its neighbors.
    Reference dx is the median of the row, dy is the median of the column.
    """
    tree = KDTree(coords)
    # Find 8 nearest neighbors to ensure we get the local grid
    dists, indices = tree.query(coords, k=9)
    
    # Estimate global average basis if not provided
    if ref_vectors is None:
        all_diffs = []
        diff_sources = []
        for i in range(len(coords)):
            diffs = coords[indices[i, 1:]] - coords[i]
            all_diffs.append(diffs)
            diff_sources.extend([i] * len(diffs))
        all_diffs = np.vstack(all_diffs)
        diff_sources = np.array(diff_sources)
        
        h_mask = np.abs(all_diffs[:, 0]) > np.abs(all_diffs[:, 1])
        v_mask = np.abs(all_diffs[:, 1]) > np.abs(all_diffs[:, 0])
        
        global_dx = np.median(np.abs(all_diffs[h_mask, 0]))
        global_dy = np.median(np.abs(all_diffs[v_mask, 1]))
        
        logger.info("  Estimating local reference lattice (row/col wise)...")
        local_refs = np.zeros((len(coords), 2, 2))
        source_coords = coords[diff_sources]
        
        h_diffs = np.abs(all_diffs[h_mask, 0])
        h_sources = source_coords[h_mask]
        
        v_diffs = np.abs(all_diffs[v_mask, 1])
        v_sources = source_coords[v_mask]
        
        for i in range(len(coords)):
            x_i, y_i = coords[i]
            # Row median for dx
            row_mask = np.abs(h_sources[:, 1] - y_i) < (global_dy * axis_tolerance_scale)
            dx_local = np.median(h_diffs[row_mask]) if np.sum(row_mask) > 0 else global_dx
            # Col median for dy
            col_mask = np.abs(v_sources[:, 0] - x_i) < (global_dx * axis_tolerance_scale)
            dy_local = np.median(v_diffs[col_mask]) if np.sum(col_mask) > 0 else global_dy
            
            local_refs[i] = np.array([[dx_local, 0], [0, dy_local]])
        ref_vectors_out = local_refs
    else:
        ref_vectors_out = ref_vectors

    strains = []
    for i in range(len(coords)):
        if ref_vectors is None:
            rv = ref_vectors_out[i]
        else:
            rv = ref_vectors
            
        ref_inv = np.linalg.inv(rv.T) 
        neighbors = coords[indices[i, 1:]]
        y = (neighbors - coords[i]).T 
        projs = ref_inv @ y 
        X_ref = np.round(projs)
        mask = (np.abs(X_ref).sum(axis=0) == 1)
        if np.sum(mask) < min_axis_neighbors:
            strains.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            continue
            
        y_valid = y[:, mask]
        X_valid = (rv.T @ X_ref[:, mask])
        try:
            J = y_valid @ np.linalg.pinv(X_valid)
            eps = 0.5 * (J + J.T) - np.eye(2)
            local_basis = J @ rv.T
            a1_len = np.linalg.norm(local_basis[:, 0])
            a2_len = np.linalg.norm(local_basis[:, 1])
            strains.append([eps[0, 0], eps[1, 1], eps[0, 1], a1_len, a2_len, rv[0, 0], rv[1, 1]])
        except:
            strains.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            
    return np.array(strains), ref_vectors_out

def generate_line_profiles(strain_csv: str, image_name: str, n_bins: int = 30):
    """
    Generate spatial line-scan profiles of inter-dumbbell spacing (a1, a2)
    expressed as % deviation from the image-wide mean.
    """
    df = pd.read_csv(strain_csv)
    has_abs = {"a1_len", "a2_len"}.issubset(df.columns)
    if has_abs:
        col_x, col_y = "a1_len", "a2_len"
        label_x, label_y = "Horizontal spacing a1", "Vertical spacing a2"
    else:
        col_x, col_y = "e_xx", "e_yy"
        label_x, label_y = "exx", "eyy"

    df = df.dropna(subset=[col_x, col_y, "x", "y"])
    if df.empty: return

    mean_x, mean_y = df[col_x].mean(), df[col_y].mean()
    df = df.copy()
    df["pct_x"] = (df[col_x] - df["ref_dx"]) / df["ref_dx"] * 100.0
    df["pct_y"] = (df[col_y] - df["ref_dy"]) / df["ref_dy"] * 100.0

    def bin_profile(pos_col, val_col, n):
        bins = pd.cut(df[pos_col], bins=n)
        grp = df.groupby(bins, observed=True)[val_col]
        centers = np.array([iv.mid for iv in grp.mean().index])
        means, stds = grp.mean().values, grp.std().values
        cmin, cmax = centers.min(), centers.max()
        if cmax > cmin: centers = (centers - cmin) / (cmax - cmin) * 100.0
        return centers, means, stds

    cx_h, mu_h, sd_h = bin_profile("x", "pct_x", n_bins)
    cy_v, mu_v, sd_v = bin_profile("y", "pct_y", n_bins)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"Spatial Line-Scan Profiles — {image_name}", fontsize=12, fontweight="bold")

    for ax, centers, means, stds, xlabel, title, color in [
        (axes[0], cx_h, mu_h, sd_h, "X position (%)", f"{label_x}: % deviation (mean={mean_x:.2f})", "#1f77b4"),
        (axes[1], cy_v, mu_v, sd_v, "Y position (%)", f"{label_y}: % deviation (mean={mean_y:.2f})", "#d62728"),
    ]:
        valid = ~np.isnan(means)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.plot(centers[valid], means[valid], "-o", color=color, ms=4, lw=1.5)
        ax.fill_between(centers[valid], means[valid]-stds[valid], means[valid]+stds[valid], alpha=0.2, color=color)
        ax.set_xlabel(xlabel); ax.set_ylabel("Delta (%)"); ax.set_title(title, fontsize=10); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(os.path.dirname(strain_csv), f"{image_name}_line_profiles.png")
    fig.savefig(out_png, dpi=150); plt.close(fig)
    return out_png

def generate_strain_maps(
    csv_path: str,
    image_path: str,
    out_dir: str = None,
    flip_h: bool = False,
    flip_v: bool = False,
    prune_thresh: float = 0.0,
    axis_tolerance_scale: float = 0.7,
    min_axis_neighbors: int = 2,
):
    """Compute strain using local row/col medians and save maps."""
    if not out_dir: out_dir = os.path.dirname(csv_path)
    os.makedirs(out_dir, exist_ok=True)
    
    # ── 1. Load and Prune (Dumbbell Handling) ──────────────────────────
    df = pd.read_csv(csv_path)
    if df.empty: return
    
    # Automatic Sublattice/Dumbbell filtering for GaAs-like structures
    # If atoms are extremly close, they confuse the grid neighbor search
    coords = df[["x", "y"]].values
    tree = KDTree(coords)
    dists, _ = tree.query(coords, k=2)
    near_dist = dists[:, 1]
    med_near = np.median(near_dist)
    
    # Threshold: If distance is < 40% of standard unit cell, it's likely a dumbbell member
    # Standard unit cell is usually the 4th-8th neighbor, but k=2 is the sibling.
    # We use a heuristic: if k=2 dist is < 3.5 px (typical for GaAs at this mag), prune.
    to_keep = np.ones(len(df), dtype=bool)
    
    # Efficient pruning: iterate and mark
    if prune_thresh and prune_thresh > 0:
        for i in range(len(coords)):
            if not to_keep[i]:
                continue
            near_idx = tree.query_ball_point(coords[i], prune_thresh)
            for nid in near_idx:
                if nid != i:
                    to_keep[nid] = False
            
    df_pruned = df[to_keep].copy()
    n_rem = len(df) - len(df_pruned)
    if n_rem > 0:
        logger.info(f"  Dumbbell Pruning: Collapsed {n_rem} nearby peaks for cleaner strain mapping.")
    
    coords_p = df_pruned[["x", "y"]].values
    strains, ref_out = calculate_local_strain(
        coords_p,
        axis_tolerance_scale=axis_tolerance_scale,
        min_axis_neighbors=min_axis_neighbors,
    )
    
    # Enriched CSV with strain components
    cols = ["e_xx", "e_yy", "e_xy", "a1_len", "a2_len", "ref_dx", "ref_dy"]
    for i, col in enumerate(cols):
        df_pruned[col] = strains[:, i]
    strain_valid_mask = np.isfinite(strains).all(axis=1)
    df_pruned["strain_valid"] = strain_valid_mask.astype(int)
    
    out_csv = os.path.join(out_dir, os.path.basename(csv_path).replace(".csv", "_strain.csv"))
    df_pruned.to_csv(out_csv, index=False)

    stats = {
        "input_atom_count": int(len(df)),
        "post_prune_atom_count": int(len(df_pruned)),
        "pruned_atom_count": int(n_rem),
        "valid_strain_atom_count": int(strain_valid_mask.sum()),
        "valid_strain_fraction": float(strain_valid_mask.mean()) if len(df_pruned) else 0.0,
        "prune_thresh": float(prune_thresh or 0.0),
        "axis_tolerance_scale": float(axis_tolerance_scale),
        "min_axis_neighbors": int(min_axis_neighbors),
        "assessment": _strain_assessment(
            valid_fraction=float(strain_valid_mask.mean()) if len(df_pruned) else 0.0,
            pruned_fraction=float(n_rem / max(len(df), 1)),
            axis_tolerance_scale=float(axis_tolerance_scale),
            min_axis_neighbors=int(min_axis_neighbors),
        ),
    }
    stats_path = os.path.join(out_dir, os.path.basename(csv_path).replace(".csv", "_strain_stats.json"))
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    try:
        # Use hyperspy for consistency with batch_processor and clustering
        os.environ["HYPERSPY_DISABLE_PLUGIN_SCAN"] = "1"
        s_img = _load_signal(image_path)
        img_data = signal_to_2d_float_array(s_img)
        bg = exposure.rescale_intensity(img_data, out_range=(0, 0.4))
        
        for comp, label, cmap, vmin, vmax in [
            ("e_xx", "Horizontal Strain (exx)", "RdBu_r", -0.1, 0.1),
            ("e_yy", "Vertical Strain (eyy)", "RdBu_r", -0.1, 0.1),
            ("a1_len", "Horizontal Spacing (a1)", "viridis", None, None),
            ("a2_len", "Vertical Spacing (a2)", "viridis", None, None)
        ]:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(bg, cmap="gray", origin='upper', interpolation='nearest')
            valid = df_pruned.dropna(subset=[comp])
            if valid.empty: continue
            if vmin is None: 
                mean_val = valid[comp].mean()
                vmin, vmax = mean_val * 0.95, mean_val * 1.05
            s = ax.scatter(valid["x"], valid["y"], c=valid[comp], s=12, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.9)
            plt.colorbar(s, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Inter-Dumbbell: {label}"); ax.axis("off")
            if flip_h: ax.invert_xaxis()
            if flip_v: ax.invert_yaxis()
            out_png = os.path.join(out_dir, os.path.basename(image_path).rsplit(".", 1)[0] + f"_strain_{comp}.png")
            fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)
    except Exception as e: logger.warning(f"Map error: {e}")

    # --- Line Scan Profiles ---
    try:
        generate_line_profiles(out_csv, os.path.basename(image_path).rsplit(".", 1)[0])
    except Exception as e: logger.warning(f"Profile error: {e}")

    # ── 2D Contour Spacing Maps (Smooth Surface) ──────────────────────────
    try:
        from scipy.interpolate import griddata
        for col, label in [("a1_len", "Horizontal Spacing (a1)"), ("a2_len", "Vertical Spacing (a2)")]:
            valid = df_pruned.dropna(subset=[col, "x", "y"])
            if valid.empty: continue
            
            # Robust Filtering: Drop outliers (> 15% deviation)
            valid = valid.copy()
            ref_col = "ref_dx" if col == "a1_len" else "ref_dy"
            valid["dev"] = (valid[col] - valid[ref_col]) / valid[ref_col] * 100.0
            valid = valid[np.abs(valid["dev"]) < 15.0]
            if valid.empty: continue

            # Grid Interpolation
            gx, gy = np.mgrid[valid.x.min():valid.x.max():250j, valid.y.min():valid.y.max():150j]
            gz = griddata(valid[["x", "y"]].values, valid["dev"].values, (gx, gy), method='cubic')
            
            # Power Mask: Don't interpolate into empty regions
            med_spacing = valid[col].median()
            tree = KDTree(valid[["x", "y"]].values)
            grid_points = np.c_[gx.ravel(), gy.ravel()]
            dists, _ = tree.query(grid_points, k=1)
            dists = dists.reshape(gx.shape)
            gz[dists > (med_spacing * 2.5)] = np.nan 
            
            fig, ax = plt.subplots(figsize=(10, 8))
            v_lim = 7.5
            cp = ax.contourf(gx, gy, gz, levels=np.linspace(-v_lim, v_lim, 51), cmap="RdBu_r", extend="both")
            fig.colorbar(cp, ax=ax, label="% Deviation from Local Ref")
            ax.set_title(f"Local-Reference Strain Map: {label}\n(Median Spacing = {med_spacing:.2f} px)")
            ax.set_aspect('equal') 
            if flip_h: ax.invert_xaxis()
            if flip_v: ax.invert_yaxis()
            ax.axis('off') 
            out_png = os.path.join(out_dir, os.path.basename(image_path).rsplit(".", 1)[0] + f"_contour_{col}.png")
            fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)

        # ── Spacing Histogram (Verification) ──
        fig, ax = plt.subplots(figsize=(8, 5))
        for col, color, lbl in [("a1_len", "#1f77b4", "a1"), ("a2_len", "#d62728", "a2")]:
            vals = df_pruned[col].dropna()
            ax.hist(vals, bins=50, alpha=0.5, color=color, label=f"{lbl} (avg={vals.mean():.1f})")
        ax.set_title("Distribution of Inter-Dumbbell Spacings (Direct Check)"); ax.set_xlabel("Pixels"); ax.legend()
        fig.savefig(os.path.join(out_dir, os.path.basename(image_path).rsplit(".", 1)[0] + "_hist.png"), dpi=120); plt.close(fig)
    except Exception as e: logger.warning(f"Contour error: {e}")

    return out_csv
