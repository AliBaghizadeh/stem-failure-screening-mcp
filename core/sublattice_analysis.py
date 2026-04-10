import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import exposure
import logging

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

logger = logging.getLogger("stem-atom-finder")


def _auto_max_dist(coords: np.ndarray) -> float:
    """Estimate the dumbbell pairing distance from the NN distance distribution.
    Dumbbells are the SHORTEST NN distance (the intra-dumbbell spacing).
    We use the 20th percentile of NN distances * 1.5 as the cutoff.
    """
    from scipy.spatial import KDTree
    tree = KDTree(coords)
    dists, _ = tree.query(coords, k=2)
    nn = dists[:, 1]
    cutoff = np.percentile(nn, 20) * 1.5
    logger.info("  Auto max_dist = %.2f px (p20_nn=%.2f * 1.5)", cutoff, np.percentile(nn, 20))
    return cutoff


def separate_sublattices(
    csv_path: str,
    image_path: str,
    out_dir: str = None,
    max_dist: float = None,   # None = auto-detect
    min_dist: float = 0.0,    # Optional floor to ignore artifacts
):
    """
    Classify atoms from CSV into Ga (dim) and As (bright) sublattices.

    Strategy — Local A/B Pairing:
    1. For each atom, find its nearest neighbor within ``max_dist``.
    2. The two atoms in each dumbbell are compared by intensity.
       Brighter → As,  Dimmer → Ga.
    3. Singletons (no partner within max_dist or < min_dist) fall back to the global median.

    Parameters
    ----------
    max_dist : float or None
        Maximum pixel distance to consider two atoms a dumbbell pair.
        If None, auto-detected from the 20th-percentile nearest-neighbor distance.
    """
    from scipy.spatial import KDTree

    if not out_dir:
        out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "intensity" not in df.columns:
        raise ValueError(f"CSV {csv_path} has no 'intensity' column. Re-run batch first.")

    coords = df[["x", "y"]].values
    intensities = df["intensity"].values.astype(float)

    if max_dist is None:
        max_dist = _auto_max_dist(coords)

    # ── Pairing ────────────────────────────────────────────────────────────────
    tree = KDTree(coords)
    dists, indices = tree.query(coords, k=2)
    nn_dists = dists[:, 1]
    nn_indices = indices[:, 1]

    species = np.full(len(df), "Unknown", dtype=object)
    processed = np.zeros(len(df), dtype=bool)
    pair_distances = np.full(len(df), np.nan)
    pair_angles = np.full(len(df), np.nan)

    # Sort by ascending NN distance so we pair the closest atoms first
    order = np.argsort(nn_dists)

    for i in order:
        if processed[i]:
            continue
        
        d = nn_dists[i]
        if d > max_dist or d < min_dist:
            continue                          # Under minimum or over maximum: singleton

        j = nn_indices[i]
        if processed[j]:
            continue                          # partner already taken

        # Assign species by relative intensity
        if intensities[i] >= intensities[j]:
            species[i], species[j] = "As", "Ga"
        else:
            species[i], species[j] = "Ga", "As"
            
        # Store pair geometry
        dy = coords[j, 1] - coords[i, 1]
        dx = coords[j, 0] - coords[i, 0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        pair_distances[i] = pair_distances[j] = d
        pair_angles[i] = pair_angles[j] = angle
        
        processed[i] = processed[j] = True

    # ── Fallback for singletons ────────────────────────────────────────────────
    singletons_mask = ~processed
    if np.any(singletons_mask):
        threshold = np.median(intensities)
        species[singletons_mask] = np.where(
            intensities[singletons_mask] >= threshold, "As", "Ga"
        )

    df["species"] = species
    df["pair_distance"] = pair_distances
    df["pair_angle"] = pair_angles

    # ── Output CSVs ────────────────────────────────────────────────────────────
    name_base = os.path.basename(csv_path).replace("_atoms.csv", "")
    ga_path = os.path.join(out_dir, f"{name_base}_Ga.csv")
    as_path = os.path.join(out_dir, f"{name_base}_As.csv")
    paired_path = os.path.join(out_dir, f"{name_base}_paired.csv")
    dumbbells_path = os.path.join(out_dir, f"{name_base}_dumbbells.csv")
    
    df[df["species"] == "Ga"].to_csv(ga_path, index=False)
    df[df["species"] == "As"].to_csv(as_path, index=False)
    
    # Save a specific CSV with ONLY paired atoms and their distances
    paired_df = df[processed].copy()
    paired_df.to_csv(paired_path, index=False)
    
    # [NEW] Save center points of checkboxes to completely bypass Ga/As species noise in strain maps
    dumbbells_list = []
    # Note: we only want one row per pair. 'processed' is True for both i and j.
    # We can iterate through 'order' and track added pairs to avoid duplicates.
    added_to_dumbbells = np.zeros(len(df), dtype=bool)
    for i in order:
        if processed[i] and not added_to_dumbbells[i]:
            j = nn_indices[i]
            cx = (coords[i, 0] + coords[j, 0]) / 2.0
            cy = (coords[i, 1] + coords[j, 1]) / 2.0
            dumbbells_list.append({"x": cx, "y": cy, "distance": nn_dists[i], "angle": pair_angles[i]})
            added_to_dumbbells[i] = added_to_dumbbells[j] = True
            
    pd.DataFrame(dumbbells_list).to_csv(dumbbells_path, index=False)

    n_paired = np.sum(processed)
    n_single = int(np.sum(singletons_mask))
    logger.info("Sublattice separation (Local Pairing, max_dist=%.1f px):", max_dist)
    logger.info("  Paired atoms:    %d  (%d dumbbells)", n_paired, n_paired // 2)
    logger.info("  Singletons:      %d  (median fallback)", n_single)
    
    if n_paired > 0:
        valid_d = pair_distances[~np.isnan(pair_distances)]
        logger.info("Dumbbell Distance Statistics (px):")
        logger.info("  Mean:   %.3f", np.mean(valid_d))
        logger.info("  Std:    %.3f", np.std(valid_d))
        logger.info("  Min/Max: %.2f / %.2f", np.min(valid_d), np.max(valid_d))
        
    logger.info("  Ga (dim):        %d → %s", len(df[df["species"] == "Ga"]), ga_path)
    logger.info("  As (bright):     %d → %s", len(df[df["species"] == "As"]), as_path)
    logger.info("  Paired Data:     %s", paired_path)

    # ── Visual Overlay ─────────────────────────────────────────────────────────
    try:
        import tifffile
        data = tifffile.imread(image_path)
        if data.ndim > 2:
            data = data.mean(axis=-1)
        img = exposure.rescale_intensity(data, out_range=(0, 1))

        # --- Plot 1: Sublattice Classification ---
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ga_df = df[df["species"] == "Ga"]
        as_df = df[df["species"] == "As"]
        ax.scatter(ga_df["x"], ga_df["y"], s=5, c="cyan",   alpha=0.9, label=f"Ga ({len(ga_df)})")
        ax.scatter(as_df["x"], as_df["y"], s=5, c="yellow", alpha=0.9, label=f"As ({len(as_df)})")
        ax.set_title(
            f"Sublattice Separation  (max_dist={max_dist:.1f}px)\n"
            f"Cyan=Ga  Yellow=As   Pairs={n_paired//2}  Singletons={n_single}",
            fontsize=9,
        )
        ax.legend(loc="lower right", fontsize=8)
        ax.axis("off")
        png_path = os.path.join(out_dir, f"{name_base}_sublattice.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Visual overlay:  %s", png_path)
        
        # --- Plot 2: Dumbbell Spacing Heatmap ---
        if n_paired > 0:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(img, cmap="gray", alpha=0.5)
            # Use only Ga atoms to represent the dumbbell position (one point per pair)
            ga_paired = df[(df["species"] == "Ga") & processed]
            sc = ax.scatter(ga_paired["x"], ga_paired["y"], c=ga_paired["pair_distance"], 
                           s=15, cmap="viridis", alpha=0.9)
            plt.colorbar(sc, ax=ax, label="Dumbbell Spacing (px)")
            ax.set_title(f"Dumbbell Spacing Distribution\n{name_base}")
            ax.axis("off")
            dist_png = os.path.join(out_dir, f"{name_base}_spacing_map.png")
            fig.savefig(dist_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("  Spacing map:     %s", dist_png)

    except Exception as exc:
        logger.warning("Could not generate overlay: %s", exc)

    return ga_path, as_path


# ── Stand-alone entry point (for quick tests) ──────────────────────────────────
if __name__ == "__main__":
    import argparse, traceback

    parser = argparse.ArgumentParser(
        description="Separate Ga/As sublattices from an atoms CSV by local intensity pairing."
    )
    parser.add_argument("--csv",      required=True,  help="Path to *_atoms.csv")
    parser.add_argument("--image",    required=True,  help="Path to original TIFF")
    parser.add_argument("--out",      default=None,   help="Output directory")
    parser.add_argument(
        "--max-dist",
        type=float,
        default=None,
        dest="max_dist",
        help=(
            "Max pixel distance to consider two atoms a dumbbell pair. "
            "Default: auto-detect from NN distance distribution. "
            "Increase to catch wider dumbbells; decrease to avoid false pairs."
        ),
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.0,
        dest="min_dist",
        help="Minimum pixel distance to consider a pair valid (default 0.0).",
    )
    args = parser.parse_args()
    try:
        separate_sublattices(args.csv, args.image, args.out, max_dist=args.max_dist, min_dist=args.min_dist)
    except Exception:
        traceback.print_exc()
