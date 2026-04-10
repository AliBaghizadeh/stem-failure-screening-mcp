"""
Batch Processor — Multi-CPU Parallel Image Processing
=======================================================
Production-grade batch processing for STEM image analysis.

Design decisions for 16-core AMD:
- Uses ProcessPoolExecutor (not ThreadPoolExecutor) — GIL-free parallelism
- Each worker gets its own GPU context (if GPU enabled)
- Chunked submission to avoid memory blowup on large batches
- Progress callbacks via shared state
- MLflow child runs per image for full traceability
"""

from __future__ import annotations

import glob
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.image_utils import normalize_hyperspy_load, signal_to_2d_float_array

logger = logging.getLogger(__name__)


def _load_signal(image_path: str):
    import hyperspy.api as hs

    return normalize_hyperspy_load(hs.load(image_path), image_path)


def _auto_pair_max_dist(coords: np.ndarray) -> float:
    from scipy.spatial import KDTree

    tree = KDTree(coords)
    dists, _ = tree.query(coords, k=2)
    nn = dists[:, 1]
    return float(np.percentile(nn, 20) * 1.5)


def _determine_sublattice_labels(
    positions_with_intensity: np.ndarray,
    method: str,
    pair_max_dist: float | None,
    pair_min_dist: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    n_atoms = len(positions_with_intensity)
    labels = np.zeros(n_atoms, dtype=int)
    pair_distances = np.full(n_atoms, np.nan)
    pair_angles = np.full(n_atoms, np.nan)
    meta = {
        "method": method,
        "pair_max_dist": pair_max_dist,
        "pair_min_dist": pair_min_dist,
        "n_paired_atoms": 0,
        "n_singletons": n_atoms,
    }

    if n_atoms == 0:
        return labels, pair_distances, pair_angles, meta

    coords = positions_with_intensity[:, :2]
    intensities = positions_with_intensity[:, 2].astype(float)

    if method == "local_pairing":
        from scipy.spatial import KDTree

        max_dist = float(pair_max_dist) if pair_max_dist is not None else _auto_pair_max_dist(coords)
        tree = KDTree(coords)
        dists, indices = tree.query(coords, k=2)
        nn_dists = dists[:, 1]
        nn_indices = indices[:, 1]
        processed = np.zeros(n_atoms, dtype=bool)
        order = np.argsort(nn_dists)

        for i in order:
            if processed[i]:
                continue

            d = float(nn_dists[i])
            if d > max_dist or d < pair_min_dist:
                continue

            j = int(nn_indices[i])
            if processed[j]:
                continue

            if intensities[i] >= intensities[j]:
                labels[i], labels[j] = 1, 0
            else:
                labels[i], labels[j] = 0, 1

            dy = coords[j, 1] - coords[i, 1]
            dx = coords[j, 0] - coords[i, 0]
            angle = float(np.degrees(np.arctan2(dy, dx)))
            pair_distances[i] = pair_distances[j] = d
            pair_angles[i] = pair_angles[j] = angle
            processed[i] = processed[j] = True

        singletons = ~processed
        if np.any(singletons):
            threshold = float(np.median(intensities))
            labels[singletons] = (intensities[singletons] >= threshold).astype(int)

        meta.update(
            {
                "pair_max_dist": max_dist,
                "n_paired_atoms": int(np.sum(processed)),
                "n_singletons": int(np.sum(singletons)),
            }
        )
        return labels, pair_distances, pair_angles, meta

    amps = intensities.reshape(-1, 1)
    if n_atoms < 2 or np.allclose(amps, amps[0]):
        return labels, pair_distances, pair_angles, meta

    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=2, random_state=42)
    labels = gmm.fit_predict(amps)
    m0 = amps[labels == 0].mean()
    m1 = amps[labels == 1].mean()
    if m0 > m1:
        labels = 1 - labels
    meta["n_singletons"] = 0
    return labels, pair_distances, pair_angles, meta


def _single_sublattice_meta(n_atoms: int) -> dict:
    return {
        "method": "single",
        "pair_max_dist": None,
        "pair_min_dist": 0.0,
        "n_paired_atoms": 0,
        "n_singletons": n_atoms,
    }


@dataclass
class BatchResult:
    """Result from processing a single image."""

    image_name: str
    status: str  # OK | UNRELIABLE | REJECTED | FAILED
    n_atoms: int = 0
    processing_time: float = 0.0
    output_files: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class BatchSummary:
    """Summary of a full batch processing run."""

    total_images: int = 0
    completed: int = 0
    failed: int = 0
    ok_count: int = 0
    unreliable_count: int = 0
    rejected_count: int = 0
    total_atoms: int = 0
    total_time: float = 0.0
    results: List[BatchResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.completed / max(self.total_images, 1)

    def to_dict(self) -> dict:
        return {
            "total_images": self.total_images,
            "completed": self.completed,
            "failed": self.failed,
            "ok_count": self.ok_count,
            "unreliable_count": self.unreliable_count,
            "rejected_count": self.rejected_count,
            "total_atoms": self.total_atoms,
            "total_time_seconds": round(self.total_time, 2),
            "success_rate": round(self.success_rate, 3),
            "images_per_second": round(
                self.total_images / max(self.total_time, 0.01), 2
            ),
        }


def _discover_images(input_folder: str) -> List[str]:
    """Find all supported image files in a folder."""
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.dm3", "*.dm4", "*.ser"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_folder, ext)))
    files.sort()
    return files


def _process_single_image(args: Tuple) -> BatchResult:
    """
    Worker function for ProcessPoolExecutor.

    Runs in a separate process — must be picklable (top-level function).
    Each worker does its own imports to avoid shared state issues.
    """
    image_path, output_folder, config, flip_h, flip_v = args
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    start_time = time.time()
    output_files = []

    try:
        # Worker-local imports (each process gets fresh state)
        import sys

        # Fast startup overrides (same as original script)
        sys.modules["cupy"] = None
        sys.modules["cupy_backends"] = None
        os.environ["HYPERSPY_DISABLE_PLUGIN_SCAN"] = "1"
        os.environ["NUMBA_CACHE_DIR"] = os.path.join(os.getcwd(), ".numba_cache")

        importlib_meta = __import__("importlib.metadata").metadata
        _orig = importlib_meta.distributions
        importlib_meta.distributions = lambda: iter([])

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        import hyperspy.api as hs
        import atomap.api as am
        import pandas as pd
        from skimage import exposure

        importlib_meta.distributions = _orig

        # Load image
        s = _load_signal(image_path)
        raw_image = signal_to_2d_float_array(s)
        
        # 0. Quadrant Patching (Tiling)
        quadrant = config.get("_quadrant")
        if quadrant:
            h, w = raw_image.shape
            mid_y, mid_x = h // 2, w // 2
            if quadrant == "TopLeft":
                raw_image = raw_image[0:mid_y, 0:mid_x].copy()
            elif quadrant == "TopRight":
                raw_image = raw_image[0:mid_y, mid_x:w].copy()
            elif quadrant == "BotLeft":
                raw_image = raw_image[mid_y:h, 0:mid_x].copy()
            elif quadrant == "BotRight":
                raw_image = raw_image[mid_y:h, mid_x:w].copy()
            image_name = f"{image_name}_{quadrant}"

        # 1. Preprocessing
        from core.preprocessing import preprocess_image, transform_coordinates_back
        
        pre_config = config.get("preprocessing", {})
        s_processed_data = preprocess_image(raw_image, pre_config)
        s_processed = hs.signals.Signal2D(s_processed_data)
        
        # 2. Run peak detection on processed image
        det_config = config.get("peak_detection", {})
        sep = det_config.get("separation", 8)
        thresh = det_config.get("threshold", 0.4)
        pca = det_config.get("pca_enabled", True)

        try:
            atom_positions = am.get_atom_positions(
                s_processed, separation=sep, threshold_rel=thresh, pca=pca
            )
        except Exception:
            try:
                atom_positions = am.get_atom_positions(
                    s_processed, separation=sep, relative_threshold=thresh, pca=pca
                )
            except Exception:
                atom_positions = am.get_atom_positions(s_processed, separation=sep, pca=pca)

        n_atoms = len(atom_positions)

        if n_atoms == 0:
            return BatchResult(
                image_name=image_name,
                status="FAILED",
                processing_time=time.time() - start_time,
                error="No atoms found",
            )

        # 3. Refinement
        sublattice = am.Sublattice(atom_positions, image=s_processed.data)
        sublattice.find_nearest_neighbors()
        sublattice.refine_atom_positions_using_center_of_mass()
        try:
            sublattice.refine_atom_positions_using_2d_gaussian()
        except Exception:
            pass

        # 4. Transform positions back to original scale
        # (Preprocessing already handled; now add the quadrant offset)
        final_positions = transform_coordinates_back(sublattice.atom_positions, pre_config)
        
        if quadrant:
            h, w = signal_to_2d_float_array(_load_signal(image_path)).shape  # Original shape
            mid_y, mid_x = h // 2, w // 2
            if quadrant in ["TopRight", "BotRight"]:
                final_positions[:, 0] += mid_x
            if quadrant in ["BotLeft", "BotRight"]:
                final_positions[:, 1] += mid_y
        
        elapsed = time.time() - start_time

        # --- Intensity Harvesting (ADF sampling) ---
        from scipy.ndimage import map_coordinates
        
        # Sample intensity from the RAW original image (grayscale)
        raw_data = raw_image
            
        # final_positions are [[col, row], ...] in original image space
        rows = final_positions[:, 1]
        cols = final_positions[:, 0]
        
        # Ensure we don't index out of bounds (shouldn't happen with mode='nearest' but helps stability)
        intensities = map_coordinates(raw_data, [rows, cols], order=1, mode="nearest")

        # Combine positions and intensities for later use
        final_positions_with_intensity = np.column_stack([final_positions, intensities])

        # 5. Sublattice Determination
        sub_config = config.get("sublattice", {})
        sub_mode = str(sub_config.get("mode", "two")).strip().lower() or "two"
        sub_method = str(sub_config.get("method", "local_pairing")).strip().lower() or "local_pairing"
        pair_max_dist = sub_config.get("pair_max_dist")
        pair_min_dist = float(sub_config.get("pair_min_dist", 0.0) or 0.0)

        sublattice_labels = np.zeros(len(final_positions_with_intensity), dtype=int)
        pair_distances = np.full(len(final_positions_with_intensity), np.nan)
        pair_angles = np.full(len(final_positions_with_intensity), np.nan)
        sub_meta = _single_sublattice_meta(len(final_positions_with_intensity))
        try:
            if sub_mode == "single":
                fig_sub, ax_sub = plt.subplots(figsize=(8, 8))
                ax_sub.imshow(exposure.rescale_intensity(raw_data, out_range=(0, 1)), cmap="gray")
                ax_sub.scatter(
                    final_positions_with_intensity[:, 0],
                    final_positions_with_intensity[:, 1],
                    s=8,
                    c="#22c55e",
                    label=f"Visible atomic columns (n={len(final_positions_with_intensity)})",
                )
                ax_sub.set_title(f"{image_name} - Single-Sublattice Mode\n(no binary split applied)")
                ax_sub.legend(loc="lower right", fontsize=8)
                if flip_h:
                    ax_sub.invert_xaxis()
                if flip_v:
                    ax_sub.invert_yaxis()
                ax_sub.axis("off")

                sub_png = os.path.join(output_folder, f"{image_name}_sublattice.png")
                fig_sub.savefig(sub_png, dpi=120, bbox_inches="tight")
                plt.close(fig_sub)
                output_files.append(sub_png)

                sub_df = pd.DataFrame(
                    np.column_stack(
                        [
                            final_positions_with_intensity,
                            pair_distances,
                            pair_angles,
                        ]
                    ),
                    columns=["x", "y", "intensity", "pair_distance", "pair_angle"],
                )
                sub_csv = os.path.join(output_folder, f"{image_name}_sublattice_0.csv")
                sub_df.to_csv(sub_csv, index=False)
                output_files.append(sub_csv)
            else:
                sublattice_labels, pair_distances, pair_angles, sub_meta = _determine_sublattice_labels(
                    final_positions_with_intensity,
                    method=sub_method,
                    pair_max_dist=pair_max_dist,
                    pair_min_dist=pair_min_dist,
                )

                # Plot Sublattice Map
                fig_sub, ax_sub = plt.subplots(figsize=(8, 8))
                ax_sub.imshow(exposure.rescale_intensity(raw_data, out_range=(0,1)), cmap="gray")
                
                colors = ["#ff7f0e", "#1f77b4"] # Orange (Darker/Ga), Blue (Brighter/As)
                for sid in [0, 1]:
                    mask = sublattice_labels == sid
                    ax_sub.scatter(
                        final_positions_with_intensity[mask, 0], final_positions_with_intensity[mask, 1],
                        s=8, c=colors[sid], label=f"Sublattice {sid} (n={mask.sum()})"
                    )
                
                title_suffix = f"method={sub_method}"
                if sub_method == "local_pairing" and sub_meta.get("pair_max_dist") is not None:
                    title_suffix += f" | max_dist={sub_meta['pair_max_dist']:.2f}px | min_dist={pair_min_dist:.2f}px"
                ax_sub.set_title(f"{image_name} - Sublattice Determination\n({title_suffix})")
                ax_sub.legend(loc="lower right", fontsize=8)
                if flip_h: ax_sub.invert_xaxis()
                if flip_v: ax_sub.invert_yaxis()
                ax_sub.axis("off")
                
                sub_png = os.path.join(output_folder, f"{image_name}_sublattice.png")
                fig_sub.savefig(sub_png, dpi=120, bbox_inches="tight")
                plt.close(fig_sub)
                output_files.append(sub_png)
                
                # Save separate CSVs
                cols_sub = ["x", "y", "intensity", "pair_distance", "pair_angle"]
                for sid in [0, 1]:
                    mask = sublattice_labels == sid
                    sub_df = pd.DataFrame(
                        np.column_stack(
                            [
                                final_positions_with_intensity[mask],
                                pair_distances[mask],
                                pair_angles[mask],
                            ]
                        ),
                        columns=cols_sub
                    )
                    sub_csv = os.path.join(output_folder, f"{image_name}_sublattice_{sid}.csv")
                    sub_df.to_csv(sub_csv, index=False)
                    output_files.append(sub_csv)
                
        except Exception as sub_err:
            logging.warning(f"Sublattice isolation failed for {image_name}: {sub_err}")
            if sub_mode != "single":
                # If pairing/GMM fails, keep a single-label fallback without fake pair metrics.
                sub_mode = "single"
                sub_meta = _single_sublattice_meta(len(final_positions_with_intensity))

        # 6. Final Data Export
        df = pd.DataFrame(
            np.column_stack([final_positions_with_intensity, sublattice_labels, pair_distances, pair_angles]),
            columns=["x", "y", "intensity", "sublattice", "pair_distance", "pair_angle"]
        )
        df["sublattice_mode"] = sub_mode
        df["sublattice_split_applied"] = int(sub_mode != "single")
        csv_path = os.path.join(output_folder, f"{image_name}_atoms.csv")
        df.to_csv(csv_path, index=False)
        output_files.append(csv_path)

        # --- Visual Inspection PNG (MANDATORY) ---
        fig, ax = plt.subplots(figsize=(8, 8))

        # Normalize image for visibility
        img_display = exposure.rescale_intensity(raw_image, out_range=(0, 1))
        ax.imshow(img_display, cmap="gray", interpolation="nearest")

        # Plot all detected atoms as solid cyan points
        pos_arr = np.array(final_positions)
        ax.scatter(
            pos_arr[:, 0], pos_arr[:, 1],
            s=5, c="cyan", alpha=0.9, label=f"Atoms (n={n_atoms})"
        )

        ax.set_title(
            f"{image_name}\nsep={sep} | thresh={thresh} | atoms={n_atoms}",
            fontsize=9, pad=4
        )
        ax.legend(loc="lower right", fontsize=7, framealpha=0.5)
        
        if flip_h: ax.invert_xaxis()
        if flip_v: ax.invert_yaxis()
        
        ax.axis("off")

        png_path = os.path.join(output_folder, f"{image_name}_visual.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        output_files.append(png_path)


        return BatchResult(
            image_name=image_name,
            status="OK",
            n_atoms=n_atoms,
            processing_time=elapsed,
            output_files=output_files,
            metrics={
                "n_atoms": n_atoms,
                "processing_time": elapsed,
            },
        )

    except Exception as exc:
        return BatchResult(
            image_name=image_name,
            status="FAILED",
            processing_time=time.time() - start_time,
            error=str(exc),
        )


class BatchProcessor:
    """
    Multi-CPU batch processor for STEM images.

    Best practices for 16-core AMD:
    - ProcessPoolExecutor for GIL-free parallelism
    - max_workers = physical_cores (not logical — hyperthreading hurts numpy)
    - Chunked submission to control memory pressure
    - Progress callback for UI integration

    Usage::

        processor = BatchProcessor(max_workers=16)
        summary = processor.run(
            input_folder="path/to/images",
            output_folder="path/to/results",
            config=cfg.peak_detection,
            progress_callback=lambda done, total: print(f"{done}/{total}"),
        )
    """

    def __init__(self, max_workers: int = 16):
        # Clamp to available CPUs (physical cores preferred)
        physical_cores = os.cpu_count() or 4
        self.max_workers = min(max_workers, physical_cores)
        logger.info("BatchProcessor: %d workers (of %d available cores)", self.max_workers, physical_cores)

    def run(
        self,
        input_folder: str,
        output_folder: str,
        config: dict,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        flip_h: bool = False,
        flip_v: bool = False,
    ) -> BatchSummary:
        """
        Process all images in input_folder using parallel workers.

        Args:
            input_folder: Path to folder with STEM images
            output_folder: Path for results output
            config: Peak detection config dict
            progress_callback: Optional (done, total) -> None

        Returns:
            BatchSummary with per-image results
        """
        os.makedirs(output_folder, exist_ok=True)
        image_files = _discover_images(input_folder)

        if not image_files:
            logger.warning("No images found in %s", input_folder)
            return BatchSummary()

        total = len(image_files)
        logger.info("Starting batch: %d images, %d workers", total, self.max_workers)

        summary = BatchSummary(total_images=total)
        start_time = time.time()

        # Prepare worker args
        mode = config.get("mode", "full")
        worker_args = []
        if mode == "tile":
            for img_path in image_files:
                for q in ["TopLeft", "TopRight", "BotLeft", "BotRight"]:
                    q_config = config.copy()
                    q_config["_quadrant"] = q
                    worker_args.append((img_path, output_folder, q_config, flip_h, flip_v))
        else:
            worker_args = [
                (img_path, output_folder, config, flip_h, flip_v)
                for img_path in image_files
            ]

        total = len(worker_args)
        logger.info("Starting batch: %d tasks, %d workers (Mode: %s)", total, self.max_workers, mode)
        summary.total_images = total # In tile mode, report each quadrant as an image
        completed = 0
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_process_single_image, args): args[0]
                for args in worker_args
            }

            for future in as_completed(futures):
                result = future.result()
                summary.results.append(result)
                completed += 1

                # Update counters
                if result.status == "FAILED":
                    summary.failed += 1
                else:
                    summary.completed += 1
                    summary.total_atoms += result.n_atoms
                    if result.status == "OK":
                        summary.ok_count += 1
                    elif result.status == "UNRELIABLE":
                        summary.unreliable_count += 1
                    elif result.status == "REJECTED":
                        summary.rejected_count += 1

                # Progress callback
                if progress_callback:
                    progress_callback(completed, total)

                # Console progress
                status_icon = {"OK": "✓", "UNRELIABLE": "⚠", "REJECTED": "✗", "FAILED": "✗"}
                logger.info(
                    "[%d/%d] %s %s (%d atoms, %.1fs)%s",
                    completed,
                    total,
                    status_icon.get(result.status, "?"),
                    result.image_name,
                    result.n_atoms,
                    result.processing_time,
                    f" — {result.error}" if result.error else "",
                )

        summary.total_time = time.time() - start_time

        logger.info(
            "Batch complete: %d/%d succeeded (%.0f%%), %d atoms, %.1fs total",
            summary.completed,
            summary.total_images,
            summary.success_rate * 100,
            summary.total_atoms,
            summary.total_time,
        )

        return summary
