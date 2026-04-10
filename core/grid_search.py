"""
Grid Search Engine — Automated Parameter Optimization
=======================================================
Finds optimal (separation, threshold, pca) parameters for a single image
using multi-CPU parallelism and MLflow tracking.

Scoring strategy:
- Maximize high-confidence atoms (we want to find as many real atoms as possible)
- Penalise high UNK fraction (too many uncertain → bad parameters)
- Penalise too few atoms (missed detections) and too many (noise included)

Each parameter combination is logged as an MLflow child run.
"""

from __future__ import annotations

import itertools
import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings(
    "ignore",
    message=r"Importing `.*Signal1D` from `hyperspy\._signals\..*` is deprecated.*",
    category=Warning,
    module=r"hyperspy\.misc\._utils",
)

import hyperspy.api as hs
import tempfile
import atomap.api as am

logger = logging.getLogger(__name__)


def _signal_to_2d_float_array(signal) -> np.ndarray:
    """Normalize HyperSpy image payloads to a plain 2D float array."""
    data = np.asarray(signal.data)

    # HyperSpy can return structured RGBA arrays for some PNG/TIFF inputs.
    if data.dtype.fields:
        field_names = list(data.dtype.fields.keys())
        channels = [data[name].astype(np.float32, copy=False) for name in field_names[:4]]
        data = np.stack(channels, axis=-1)

    if data.ndim >= 3:
        data = data[..., :3].mean(axis=-1)

    return np.asarray(data, dtype=np.float32)


def _load_signal2d(image_path: str):
    """Normalize HyperSpy loads that may return a list of signals."""
    loaded = hs.load(image_path)
    if isinstance(loaded, list):
        if not loaded:
            raise ValueError(f"No signals were loaded from {image_path}")
        loaded = loaded[0]
    return loaded


@dataclass
class GridSearchResult:
    """Result from a single parameter combination."""

    separation: int
    threshold: float
    pca: bool
    n_atoms: int = 0
    n_high_conf: int = 0
    unk_fraction: float = 1.0
    mean_confidence: float = 0.0
    score: float = 0.0
    time_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "sep": self.separation,
            "thresh": self.threshold,
            "pca": self.pca,
            "n_atoms": self.n_atoms,
            "n_high_conf": self.n_high_conf,
            "unk_fraction": round(self.unk_fraction, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "score": round(self.score, 4),
            "time_seconds": round(self.time_seconds, 3),
            "error": self.error,
        }


def _evaluate_params(args: Tuple) -> GridSearchResult:
    """
    Worker: evaluate one (sep, thresh, pca) combination.
    """
    image_path, sep, thresh, pca_enabled, confidence_threshold, pre_config, plot_dir, ref_cfg, hardware, flip_h, flip_v = args
    start = time.time()

    result = GridSearchResult(separation=sep, threshold=thresh, pca=pca_enabled)
    pid = os.getpid()
    core_count = os.cpu_count() or 1
    core_info = f"PID:{pid} (Cores:{core_count})"
    s = None
    positions = None

    try:
        import sys
        import multiprocessing
        try:
            import psutil
            core_count = psutil.cpu_count(logical=False) or psutil.cpu_count() or core_count
        except Exception:
            core_count = core_count
        core_info = f"PID:{pid} (Cores:{core_count})"
        
        # Enforce strict single-threading per worker to prevent CPU thrashing
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

        if hardware == "cpu":
            sys.modules["cupy"] = None
            sys.modules["cupy_backends"] = None
            os.environ["HYPERSPY_DISABLE_PLUGIN_SCAN"] = "1"
        else:
            os.environ["HYPERSPY_DISABLE_PLUGIN_SCAN"] = "1"
            # Attempt to verify GPU presence
            try:
                import cupy
                if cupy.cuda.is_available():
                    core_info += f" [GPU:{cupy.cuda.runtime.getDeviceProperties(0)['name'].decode()}]"
                else:
                    core_info += " [GPU missing CUDA]"
            except Exception:
                core_info += " [GPU module absent]"

        importlib_meta = __import__("importlib.metadata").metadata
        _orig = importlib_meta.distributions
        importlib_meta.distributions = lambda: iter([])

        import matplotlib
        matplotlib.use("Agg")

        import hyperspy.api as hs
        import atomap.api as am

        importlib_meta.distributions = _orig

        # 1. Preprocessing
        from core.preprocessing import preprocess_image
        
        s = _load_signal2d(image_path)
        s_processed_data = preprocess_image(_signal_to_2d_float_array(s), pre_config)
        s_processed = hs.signals.Signal2D(s_processed_data)

        # 2. Peak detection
        try:
            positions = am.get_atom_positions(
                s_processed, separation=sep, threshold_rel=thresh, pca=pca_enabled
            )
        except Exception:
            try:
                positions = am.get_atom_positions(
                    s_processed, separation=sep, relative_threshold=thresh, pca=pca_enabled
                )
            except Exception:
                positions = am.get_atom_positions(s_processed, separation=sep, pca=pca_enabled)

        result.n_atoms = len(positions)

        if result.n_atoms < 5:
            result.error = "Too few atoms"
            result.time_seconds = time.time() - start
            return result

        # Quick refinement for quality assessment
        image_data = _signal_to_2d_float_array(s)
        sublattice = am.Sublattice(positions, image=image_data)
        sublattice.find_nearest_neighbors()
        sublattice.refine_atom_positions_using_center_of_mass()

        try:
            sublattice.refine_atom_positions_using_2d_gaussian()
        except Exception:
            pass

        # Compute confidence scores (simplified from full pipeline)
        confidences = []
        for atom in sublattice.atom_list:
            factors = []

            sigma_x = getattr(atom, "sigma_x", np.nan)
            sigma_y = getattr(atom, "sigma_y", np.nan)
            if not np.isnan(sigma_x) and not np.isnan(sigma_y):
                sigma_avg = (sigma_x + sigma_y) / 2
                if 0.5 <= sigma_avg <= 5.0:
                    factors.append(1.0)
                elif sigma_avg < 0.5:
                    factors.append(max(0.0, sigma_avg / 0.5))
                else:
                    factors.append(max(0.0, 1.0 - (sigma_avg - 5.0) / 5.0))
            else:
                factors.append(0.0)

            amp = getattr(atom, "amplitude_max_intensity", np.nan)
            if not np.isnan(amp):
                factors.append(min(1.0, max(0.0, (amp - 0.1) / 0.4)))
            else:
                factors.append(0.5)

            nn_list = getattr(atom, "nearest_neighbor_list", None)
            if nn_list:
                n_nn = len(nn_list)
                if 4 <= n_nn <= 6:
                    factors.append(1.0)
                elif n_nn < 4:
                    factors.append(n_nn / 4.0)
                else:
                    factors.append(max(0.0, 1.0 - (n_nn - 6) / 4.0))
            else:
                factors.append(0.0)

            conf = float(np.prod(factors) ** (1.0 / len(factors))) if factors else 0.0
            confidences.append(conf)

        confidences = np.array(confidences)
        result.n_high_conf = int(np.sum(confidences >= confidence_threshold))
        result.mean_confidence = float(np.mean(confidences))
        result.unk_fraction = 1.0 - (result.n_high_conf / max(result.n_atoms, 1))

        # Scoring: high-confidence atoms minus UNK penalty
        result.score = result.n_high_conf * (1.0 - result.unk_fraction)

        # 4. Visual Plotting (Obligatory)
        if plot_dir:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(image_data, cmap="gray", interpolation="nearest")
            
            # positions is [[x,y], ...] from atomap
            pos_arr = np.array(sublattice.atom_positions)
            x_coords = pos_arr[:, 0]  # column (x in image space)
            y_coords = pos_arr[:, 1]  # row (y in image space)
            
            # All detected atoms as solid cyan points
            ax.scatter(x_coords, y_coords, s=5, c="cyan", alpha=0.9, label="Detected")

            # High-confidence atoms as solid yellow points
            high_conf_mask = confidences >= confidence_threshold
            if high_conf_mask.any():
                ax.scatter(x_coords[high_conf_mask], y_coords[high_conf_mask], 
                           s=5, c="yellow", label="High-Conf")

            ax.set_title(
                f"sep={sep} | thresh={thresh:.2f} | n={len(positions)} | "
                f"hi-conf={result.n_high_conf} | score={result.score:.1f}",
                fontsize=9, pad=4
            )
            ax.axis("off")
            if flip_h: ax.invert_xaxis()
            if flip_v: ax.invert_yaxis()
            ax.legend(loc="lower right", fontsize=6, framealpha=0.5)

            plot_path = os.path.join(plot_dir, f"trial_sep{sep}_th{thresh:.2f}.png")
            fig.savefig(plot_path, dpi=110, bbox_inches="tight")
            plt.close(fig)

    except Exception as exc:
        result.error = str(exc)
        logger.warning("[%s] Trial sep=%d thresh=%.2f failed: %s", core_info, sep, thresh, exc)
        
        # Plot anyway if we have any atoms
        if plot_dir and s is not None and positions is not None and len(positions) > 0:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(_signal_to_2d_float_array(s), cmap="gray", interpolation="nearest")
                # Plot all detected atoms as solid red points
                pos_arr = np.array(positions)
                ax.scatter(pos_arr[:, 0], pos_arr[:, 1], s=5, c="red", alpha=0.9, label="Failed/Detected")
                ax.set_title(f"FAILED: sep={sep} th={thresh:.2f}\n{exc[:50]}...", fontsize=8, color="red")
                ax.axis("off")
                if flip_h: ax.invert_xaxis()
                if flip_v: ax.invert_yaxis()
                plot_path = os.path.join(plot_dir, f"trial_sep{sep}_th{thresh:.2f}.png")
                fig.savefig(plot_path, dpi=110, bbox_inches="tight")
                plt.close(fig)
            except:
                pass

    result.time_seconds = time.time() - start
    
    # Embed the specific core info block in the error field for the log to read
    if result.error is None:
        result.error = f"Ran on {core_info}"
        
    return result


class GridSearchEngine:
    """
    Automated parameter optimization for peak detection.

    Usage::

        engine = GridSearchEngine(max_workers=8)
        best_params, results = engine.search(
            image_path="path/to/image.png",
            config=cfg.grid_search,
            tracker=mlflow_tracker,  # optional
        )
    """

    def __init__(self, max_workers: int = 8):
        self.max_workers = min(max_workers, os.cpu_count() or 4)

    def search(
        self,
        image_path: str,
        config: dict,
        tracker=None,  # Optional ExperimentTracker
        confidence_threshold: float = 0.3,
        plot_dir: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        hardware: str = "cpu",
        flip_h: bool = False,
        flip_v: bool = False,
    ) -> Tuple[dict, List[GridSearchResult]]:
        """
        Run grid search over parameter space.

        Args:
            image_path: Path to a single STEM image
            config: Grid search config with separations, thresholds, pca_options
            tracker: Optional MLflow ExperimentTracker

        Returns:
            (best_params, all_results)
        """
        separations = config.get("separations", [4, 6, 8, 10])
        thresholds = config.get("thresholds", [0.3, 0.4, 0.5, 0.6])
        pca_options = config.get("pca_options", [True])

        combos = list(itertools.product(separations, thresholds, pca_options))
        total = len(combos)
        logger.info(
            "Grid search: %d combinations (%d sep × %d thresh × %d pca), %d workers",
            total,
            len(separations),
            len(thresholds),
            len(pca_options),
            self.max_workers,
        )

        # --- Optimization: Preprocess ONCE before grid search ---
        from core.preprocessing import preprocess_image
        
        logger.info("Reducing image size before grid search...")
        s_raw = _load_signal2d(image_path)
        pre_config = config.get("preprocessing", {"rescale_factor": 1.0, "binning": 1})
        
        # Apply preprocessing (rescale, bin, blur)
        processed_data = preprocess_image(_signal_to_2d_float_array(s_raw), pre_config)
        
        # Save to a temporary file for workers to share
        temp_dir = tempfile.gettempdir()
        temp_img_path = os.path.join(temp_dir, f"gs_temp_{int(time.time())}.tif")
        s_reduced = hs.signals.Signal2D(processed_data)
        s_reduced.save(temp_img_path, overwrite=True, file_format="tif")
        
        logger.info("Image reduced: %s -> %s pixels. Temp file: %s", 
                    _signal_to_2d_float_array(s_raw).shape, processed_data.shape, temp_img_path)

        # Prepare worker args
        # For workers, we bypass preprocessing since it's already done
        ref_cfg = config.get("refinement", {"percent": 0.25, "radius": None})
        worker_pre_config = {"rescale_factor": 1.0, "binning": 1, "crop_edge_pixels": 0}
        worker_args = [
            (temp_img_path, sep, thresh, pca, confidence_threshold, worker_pre_config, plot_dir, ref_cfg, hardware, flip_h, flip_v)
            for sep, thresh, pca in combos
        ]

        results: List[GridSearchResult] = []
        start_time = time.time()

        try:
            import json
            prog_file = os.path.join(plot_dir, "grid_search_live.json") if plot_dir else None
            
            def export_progress(i, res=None):
                if progress_callback:
                    progress_callback(i, total)
                if prog_file:
                    with open(prog_file, "w") as f:
                        json.dump({
                            "current": i, 
                            "total": total, 
                            "latest_result": res.to_dict() if res else None,
                            "all_results": [r.to_dict() for r in results]
                        }, f, indent=2)

            if hardware == "gpu":
                # Detailed GPU check
                gpu_name = "Unknown"
                try:
                    import cupy
                    if cupy.cuda.is_available():
                        gpu_name = cupy.cuda.runtime.getDeviceProperties(0)['name'].decode()
                    else:
                        gpu_name = "CUDA not found"
                except ImportError:
                    gpu_name = "CuPy missing"
                    
                logger.info(f"Starting GPU Sequential Grid Search... (Detected GPU: {gpu_name})")
                for i, args in enumerate(worker_args):
                    result = _evaluate_params(args)
                    # clean up the fake error
                    core_inf = result.error
                    result.error = None if str(result.error).startswith("Ran on") else result.error
                    results.append(result)
                    
                    export_progress(i + 1, result)
                    logger.info("[%d/%d] GPU ✓ sep=%d thresh=%.2f → atoms=%d (%s)", 
                                 i+1, total, result.separation, result.threshold, result.n_atoms, core_inf)
            else:
                real_cores = os.cpu_count() or 1
                import multiprocessing
                ctx = multiprocessing.get_context("spawn")
                logger.info("Starting CPU Parallel Grid Search with %d workers (System total available CPUs: %d)...", self.max_workers, real_cores)
                with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctx) as executor:
                    futures = {
                        executor.submit(_evaluate_params, args): args
                        for args in worker_args
                    }

                    for i, future in enumerate(as_completed(futures)):
                        result = future.result()
                        core_inf = result.error
                        result.error = None if str(result.error).startswith("Ran on") else result.error
                        results.append(result)

                        export_progress(i + 1, result)

                        # Log to MLflow if tracker provided
                        if tracker:
                            try:
                                with tracker.start_child_run(
                                    f"sep{result.separation}_th{result.threshold}_pca{result.pca}"
                                ):
                                    tracker.log_params({
                                        "sep": result.separation,
                                        "thresh": result.threshold,
                                        "pca": result.pca,
                                    })
                                    tracker.log_metrics({
                                        "n_atoms": result.n_atoms,
                                        "n_high_conf": result.n_high_conf,
                                        "unk_fraction": result.unk_fraction,
                                        "mean_confidence": result.mean_confidence,
                                        "score": result.score,
                                        "time_seconds": result.time_seconds,
                                    })
                            except Exception:
                                pass

                        status = "✓" if result.error is None else "✗"
                        logger.info(
                            "[%d/%d] %s sep=%d thresh=%.2f pca=%s → %d atoms, score=%.1f (%.1fs) [%s]",
                            i + 1,
                            total,
                            status,
                            result.separation,
                            result.threshold,
                            result.pca,
                            result.n_atoms,
                            result.score,
                            result.time_seconds,
                            core_inf
                        )
        finally:
            # Cleanup temp file
            if os.path.exists(temp_img_path):
                try:
                    os.remove(temp_img_path)
                except Exception:
                    pass

        # Build one final contact sheet that includes EVERYTHING (sorted by best score)
        if plot_dir and results:
            self._build_contact_sheet(results, plot_dir)

        # Find best (lenient logic)
        valid_results = [r for r in results if r.n_atoms > 0]
        if not valid_results:
            logger.error("Grid search failed: absolutely no atoms found in any trial!")
            return {"separation": 6, "threshold": 0.5, "pca": True}, results
        
        # Prefer results without errors, fall back to any result with atoms
        perfect_results = [r for r in valid_results if r.error is None]
        best_source = perfect_results if perfect_results else valid_results
        best = max(best_source, key=lambda r: r.score if r.score > 0 else r.n_atoms / 1000.0)

        best_params = {
            "separation": best.separation,
            "threshold": best.threshold,
            "pca": best.pca,
        }

        elapsed = time.time() - start_time
        logger.info(
            "Grid search done in %.1fs. Best: sep=%d, thresh=%.2f, pca=%s → "
            "%d atoms, score=%.1f",
            elapsed,
            best.separation,
            best.threshold,
            best.pca,
            best.n_atoms,
            best.score,
        )

        # Log summary to MLflow parent run
        if tracker:
            tracker.log_grid_search_results(
                results=[r.to_dict() for r in results],
                best_params=best_params,
                best_score=best.score,
            )

        return best_params, results

    @staticmethod
    def _build_contact_sheet(results: List["GridSearchResult"], plot_dir: str):
        """Assemble all per-trial PNGs into a single sorted grid overview image."""
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from pathlib import Path

        # Sort: Perfect results first (by score), then Failed results (by atom count)
        results_sorted = sorted(results, key=lambda r: (r.error is None, r.score if r.score > 0 else r.n_atoms / 1e6), reverse=True)

        imgs = []
        labels = []
        imgs = []
        labels = []
        for r in results_sorted:
            p = os.path.join(plot_dir, f"trial_sep{r.separation}_th{r.threshold:.2f}.png")
            if os.path.exists(p):
                imgs.append(mpimg.imread(p))
                label = f"sep={r.separation} th={r.threshold:.2f}\nn={r.n_atoms} score={r.score:.1f}"
                if r.error:
                    label += "\n[FAILED]"
                labels.append(label)

        if not imgs:
            return

        n = len(imgs)
        ncols = min(8, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.8),
                                  facecolor="#1a1a2e")
        axes = np.array(axes).flatten()

        for i, (img, label) in enumerate(zip(imgs, labels)):
            ax = axes[i]
            ax.imshow(img)
            ax.set_title(label, fontsize=6, color="white", pad=2)
            ax.axis("off")
            # Gold border on the best result
            if i == 0:
                for spine in ax.spines.values():
                    spine.set_edgecolor("gold")
                    spine.set_linewidth(2.5)
                    ax.set_visible(True)

        # Hide unused axes
        for j in range(len(imgs), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Grid Search Contact Sheet — sorted by score (best = top-left)",
                     fontsize=10, color="white", y=1.01)
        fig.tight_layout(pad=0.3)
        sheet_path = os.path.join(plot_dir, "summary_contact_sheet.png")
        fig.savefig(sheet_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Contact sheet saved: %s", os.path.abspath(sheet_path))
