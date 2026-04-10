"""
CLI Entry Point
================
Production CLI for STEM Atom Finder.
Supports: single image, batch, grid search, and sublattice analysis.

Usage::

    # Grid search on a single image
    python -m core.cli grid-search --image path/to/image.png

    # Batch processing with best params
    python -m core.cli batch --folder path/to/images --out results/

    # Ga/As Sublattice separation
    python -m core.cli sublattice --csv results/GaAs_01/GaAs_01_atoms.csv --image data/GaAs_01.tif
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from omegaconf import OmegaConf

# --- SPEED OPTIMIZATION: Bypassing heavy library scans & fixing Windows DLL errors ---
os.environ["HYPERSPY_DISABLE_PLUGIN_SCAN"] = "1"
os.environ["HYPERSPY_GUI_BACKEND"] = "none" # Disable trying to load GUIs
os.environ["NUMBA_CACHE_DIR"] = str(Path.cwd() / ".numba_cache")
os.environ["OMP_NUM_THREADS"] = "1" # Fix for threadpoolctl delay-load DLL errors
os.environ["MKL_NUM_THREADS"] = "1"

# Ensure the root directory is on the path so 'import core' works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stem-atom-finder")

def _run_grid_search(args, cfg, tracker):
    from core.grid_search import GridSearchEngine
    engine = GridSearchEngine(max_workers=cfg.grid_search.get("parallel_workers", 8))

    with tracker.start_run(run_name=f"grid_search", tags={"command": "grid-search"}):
        tracker.log_config(OmegaConf.to_container(cfg, resolve=True))
        import json, pandas as pd
        out_dir = Path("results/optimisation")
        plot_dir = out_dir / "trials"
        plot_dir.mkdir(parents=True, exist_ok=True)

        gs_config = OmegaConf.to_container(cfg.grid_search)
        gs_config["preprocessing"] = OmegaConf.to_container(cfg.preprocessing)
        gs_config["refinement"] = OmegaConf.to_container(cfg.peak_detection.refinement)

        best_params, results = engine.search(
            image_path=args.image,
            config=gs_config,
            tracker=tracker,
            confidence_threshold=cfg.confidence.threshold,
            plot_dir=str(plot_dir),
        )

        logger.info("=" * 60)
        logger.info("BEST PARAMETERS:")
        for k, v in best_params.items():
            logger.info("  %s = %s", k, v)
        logger.info("=" * 60)

        with open(out_dir / "best_parameters.json", "w") as f:
            json.dump(best_params, f, indent=4)

        results_df = pd.DataFrame([r.to_dict() for r in results])
        results_df.sort_values("score", ascending=False, inplace=True)
        results_df.to_csv(out_dir / "grid_search_results.csv", index=False)

        sheet = str(plot_dir / "summary_contact_sheet.png")
        if os.path.exists(sheet):
            tracker.log_artifact(sheet, "grid_search")

def _run_batch(args, cfg, tracker):
    from core.batch_processor import BatchProcessor
    processor = BatchProcessor(max_workers=cfg.batch.max_workers)

    with tracker.start_run(run_name="batch_processing", tags={"command": "batch"}):
        tracker.log_config(OmegaConf.to_container(cfg, resolve=True))
        batch_config = OmegaConf.to_container(cfg, resolve=True)
        summary = processor.run(
            input_folder=args.folder,
            output_folder=args.out,
            config=batch_config,
        )
        tracker.log_metrics(summary.to_dict())
        logger.info("=" * 60)
        logger.info("BATCH SUMMARY:")
        for k, v in summary.to_dict().items():
            logger.info("  %s = %s", k, v)
        logger.info("=" * 60)

def _run_analyze(args, cfg, tracker):
    logger.info("Single image analysis: %s (mode=%s)", args.image, args.mode)
    with tracker.start_run(run_name=f"analyze_{os.path.basename(args.image)}", tags={"command": "analyze"}):
        tracker.log_config(OmegaConf.to_container(cfg, resolve=True))
        logger.info("Full analysis pipeline logic goes here.")

def _run_sublattice(args, cfg, tracker):
    from core.sublattice_analysis import separate_sublattices
    logger.info("Ga/As Sublattice Separation: %s", args.csv)
    with tracker.start_run(run_name=f"sublattice_{os.path.basename(args.csv)}", tags={"command": "sublattice"}):
        tracker.log_config(OmegaConf.to_container(cfg, resolve=True))
        ga_csv, as_csv = separate_sublattices(
            csv_path=args.csv,
            image_path=args.image,
            out_dir=args.out if args.out else None,
            max_dist=getattr(args, "max_dist", None),
            min_dist=getattr(args, "min_dist", 0.0),
        )
        tracker.log_artifact(ga_csv, "sublattice")
        tracker.log_artifact(as_csv, "sublattice")
        png = os.path.join(os.path.dirname(ga_csv), os.path.basename(ga_csv).replace("_atoms.csv", "_sublattice.png"))
        if os.path.exists(png):
            tracker.log_artifact(png, "sublattice")

def _run_cluster(args, cfg, tracker):
    from ml.feature_engineering import run_clustering
    logger.info("Atom Clustering: %s", args.csv)
    with tracker.start_run(run_name=f"cluster_{os.path.basename(args.csv)}", tags={"command": "cluster"}):
        tracker.log_config(OmegaConf.to_container(cfg, resolve=True))
        out_csv = run_clustering(
            csv_path=args.csv,
            image_path=getattr(args, "image", None),
            out_dir=args.out if args.out else None,
            min_cluster_size=getattr(args, "min_cluster_size", 15),
            gmm_components=getattr(args, "gmm_components", 4),
        )
        tracker.log_artifact(out_csv, "cluster")

def _run_defects(args, cfg, tracker):
    from ml.defect_detection import detect_defects
    logger.info("Defect Detection: %s", args.csv)
    with tracker.start_run(run_name=f"defects_{os.path.basename(args.csv)}", tags={"command": "defects"}):
        tracker.log_config(OmegaConf.to_container(cfg, resolve=True))
        out_csv = detect_defects(
            csv_path=args.csv,
            image_path=getattr(args, "image", None),
            out_dir=args.out if args.out else None,
        )
        tracker.log_artifact(out_csv, "defects")

def _run_strain(args, cfg, tracker):
    from core.strain_analysis import generate_strain_maps
    logger.info("Lattice Strain Analysis: %s", args.csv)
    with tracker.start_run(run_name=f"strain_{os.path.basename(args.csv)}", tags={"command": "strain"}):
        tracker.log_config(OmegaConf.to_container(cfg, resolve=True))
        strain_csv = generate_strain_maps(
            csv_path=args.csv,
            image_path=args.image,
            out_dir=args.out if args.out else None
        )
        tracker.log_artifact(strain_csv, "strain")
        # Log available PNGs
        out_dir = os.path.dirname(strain_csv)
        for f in os.listdir(out_dir):
            if "_strain_e" in f and f.endswith(".png"):
                tracker.log_artifact(os.path.join(out_dir, f), "strain")

def _run_resize(args, cfg, tracker):
    import tifffile
    from core.preprocessing import preprocess_image
    from skimage import io, exposure
    data = tifffile.imread(args.image)
    h, w = data.shape
    pre_cfg = OmegaConf.to_container(cfg.preprocessing)
    if args.scale:
        pre_cfg["rescale_factor"] = args.scale
    if args.width:
        pre_cfg["rescale_factor"] = args.width / w
    processed_data = preprocess_image(data, pre_cfg)
    out_path = args.out
    if os.path.isdir(out_path):
        fname = os.path.basename(args.image).rsplit(".", 1)[0] + f"_r{processed_data.shape[1]}.png"
        out_path = os.path.join(out_path, fname)
    img_8bit = exposure.rescale_intensity(processed_data, out_range=(0, 255)).astype("uint8")
    io.imsave(out_path, img_8bit)
    logger.info("Saved resized image to: %s", out_path)

def _run_patch(args, cfg, tracker):
    import tifffile
    import numpy as np
    from skimage import exposure, io
    data = tifffile.imread(args.image)
    h, w = data.shape
    size = args.size
    stride = size - args.overlap
    os.makedirs(args.out, exist_ok=True)
    patches = []
    if args.mode == "grid":
        y_coords = range(0, max(1, h - size + 1), stride)
        x_coords = range(0, max(1, w - size + 1), stride)
        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                patches.append((y, x, f"grid_y{i}_x{j}"))
    for y, x, name in patches:
        patch = data[y : y + size, x : x + size]
        out_path = os.path.join(args.out, f"{os.path.basename(args.image).rsplit('.', 1)[0]}_{name}.png")
        img_8bit = exposure.rescale_intensity(patch, out_range=(0, 255)).astype("uint8")
        io.imsave(out_path, img_8bit)
    logger.info("Exported %d patches to %s", len(patches), args.out)

def main():
    parser = argparse.ArgumentParser(prog="stem-find", description="STEM Atom Finder CLI")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Command: grid-search
    p = subparsers.add_parser("grid-search", help="Automated parameter optimization")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("overrides", nargs="*")

    # Command: batch
    p = subparsers.add_parser("batch", help="Batch process folder")
    p.add_argument("--folder", type=str, required=True)
    p.add_argument("--out", type=str, default="results")
    p.add_argument("overrides", nargs="*")

    # Command: sublattice
    p = subparsers.add_parser("sublattice", help="Ga/As separation")
    p.add_argument("--csv", type=str, required=True, help="Path to *_atoms.csv from batch")
    p.add_argument("--image", type=str, required=True, help="Path to original raw image")
    p.add_argument("--out", type=str, default=None, help="Output directory override")
    p.add_argument(
        "--max-dist",
        type=float,
        default=None,
        dest="max_dist",
        help=(
            "Max pixel distance to consider two atoms a dumbbell pair (default: auto). "
            "Increase to catch wider dumbbells; decrease to avoid false pairs."
        ),
    )
    p.add_argument(
        "--min-dist",
        type=float,
        default=0.0,
        dest="min_dist",
        help="Min pixel distance to consider a pair valid (default: 0.0)",
    )
    p.add_argument("overrides", nargs="*")
    
    # Command: strain
    p = subparsers.add_parser("strain", help="Geometric strain mapping")
    p.add_argument("--csv", type=str, required=True, help="Path to a single sublattice CSV (Ga or As)")
    p.add_argument("--image", type=str, required=True, help="Path to original raw image")
    p.add_argument("--out", type=str, default=None, help="Output folder")
    p.add_argument("overrides", nargs="*")

    # Command: cluster
    p = subparsers.add_parser("cluster", help="HDBSCAN/GMM atom environment clustering")
    p.add_argument("--csv", type=str, required=True, help="Path to _strain.csv")
    p.add_argument("--image", type=str, default=None, help="Path to original image (for overlay)")
    p.add_argument("--out", type=str, default=None, help="Output folder")
    p.add_argument("--min-cluster-size", type=int, default=15, dest="min_cluster_size",
                   help="HDBSCAN min_cluster_size (default: 15)")
    p.add_argument("--gmm-components", type=int, default=4, dest="gmm_components",
                   help="Number of GMM components (default: 4)")
    p.add_argument("overrides", nargs="*")

    # Command: defects
    p = subparsers.add_parser("defects", help="Automatic defect detection from clustered data")
    p.add_argument("--csv", type=str, required=True, help="Path to _clustered.csv")
    p.add_argument("--image", type=str, default=None, help="Path to original image (for overlay)")
    p.add_argument("--out", type=str, default=None, help="Output folder")
    p.add_argument("overrides", nargs="*")

    # Command: analyze
    p = subparsers.add_parser("analyze", help="Single image analysis")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--mode", choices=["full", "tile", "crop"], default="full")
    p.add_argument("overrides", nargs="*")

    # Command: resize
    p = subparsers.add_parser("resize", help="Rescale image")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--scale", type=float)
    p.add_argument("--width", type=int)
    p.add_argument("overrides", nargs="*")

    # Command: patch
    p = subparsers.add_parser("patch", help="Subdivide into tiles")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--mode", choices=["grid", "center", "random"], default="grid")
    p.add_argument("overrides", nargs="*")

    args = parser.parse_args()
    
    from core.config import load_config
    from ml.tracking import ExperimentTracker
    cfg = load_config(args.config, getattr(args, "overrides", None))
    tracker = ExperimentTracker.from_config(OmegaConf.to_container(cfg.mlflow))

    handlers = {
        "grid-search": _run_grid_search,
        "batch": _run_batch,
        "analyze": _run_analyze,
        "sublattice": _run_sublattice,
        "strain": _run_strain,
        "cluster": _run_cluster,
        "defects": _run_defects,
        "resize": _run_resize,
        "patch": _run_patch,
    }
    handlers[args.command](args, cfg, tracker)

if __name__ == "__main__":
    main()
