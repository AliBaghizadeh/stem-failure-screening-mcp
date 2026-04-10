"""
MLflow Experiment Tracking
===========================
Production-grade experiment tracking for STEM atom finding.

Tracks:
- All configuration parameters (peak detection, Y-filtering, coherence, etc.)
- Per-image metrics (atom count, confidence, UNK fraction, quality status)
- Grid search runs with nested child runs
- ML pipeline parameters (clustering, strain analysis)
- Artifacts (plots, CSVs, JSON reports)

Best practices:
- Flat param logging (MLflow doesn't support nested dicts natively)
- Batch metric logging (reduces I/O overhead)
- Run context manager for clean start/end
- Parent/child runs for grid search and batch processing
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    MLflow-based experiment tracker for STEM atom finding.

    Usage::

        tracker = ExperimentTracker.from_config(cfg.mlflow)

        # Single image analysis
        with tracker.start_run(run_name="image_001") as run:
            tracker.log_config(cfg)
            # ... run analysis ...
            tracker.log_image_metrics(results)
            tracker.log_artifact("results/image_001_plot.png")

        # Grid search with nested runs
        with tracker.start_run(run_name="grid_search_001") as parent:
            for sep, thresh in param_grid:
                with tracker.start_child_run(f"sep{sep}_th{thresh}"):
                    tracker.log_params({"sep": sep, "thresh": thresh})
                    # ... run ...
                    tracker.log_metrics({"n_atoms": n, "unk_frac": uf})
    """

    def __init__(
        self,
        tracking_uri: str = "mlruns",
        experiment_name: str = "stem-atom-finder",
        log_artifacts: bool = True,
        auto_log_params: bool = True,
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.log_artifacts_flag = log_artifacts
        self.auto_log_params = auto_log_params
        self._active_run = None
        self._parent_run_id = None

        # Configure MLflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient(tracking_uri)

        logger.info(
            "MLflow tracker initialised: experiment='%s', uri='%s'",
            experiment_name,
            tracking_uri,
        )

    @classmethod
    def from_config(cls, mlflow_cfg: dict) -> "ExperimentTracker":
        return cls(
            tracking_uri=mlflow_cfg.get("tracking_uri", "mlruns"),
            experiment_name=mlflow_cfg.get("experiment_name", "stem-atom-finder"),
            log_artifacts=mlflow_cfg.get("log_artifacts", True),
            auto_log_params=mlflow_cfg.get("auto_log_params", True),
        )

    # -- Run management ---------------------------------------------------------

    @contextmanager
    def start_run(self, run_name: str, tags: Optional[dict] = None):
        """Start a top-level MLflow run."""
        with mlflow.start_run(run_name=run_name, tags=tags) as run:
            self._active_run = run
            self._parent_run_id = run.info.run_id
            start_time = time.time()
            logger.info("MLflow run started: %s (%s)", run_name, run.info.run_id)

            try:
                yield run
            finally:
                elapsed = time.time() - start_time
                mlflow.log_metric("total_time_seconds", elapsed)
                self._active_run = None
                self._parent_run_id = None
                logger.info("MLflow run finished: %.1fs", elapsed)

    @contextmanager
    def start_child_run(self, run_name: str, tags: Optional[dict] = None):
        """Start a nested child run (e.g., one grid search combo or one image in batch)."""
        with mlflow.start_run(
            run_name=run_name,
            nested=True,
            tags=tags,
        ) as child_run:
            start_time = time.time()
            try:
                yield child_run
            finally:
                elapsed = time.time() - start_time
                mlflow.log_metric("run_time_seconds", elapsed)

    # -- Parameter logging ------------------------------------------------------

    def log_config(self, cfg: dict, prefix: str = ""):
        """
        Recursively flatten and log all config parameters.

        Converts nested dict::

            {"peak_detection": {"separation": 6}}

        into flat MLflow params::

            peak_detection.separation = 6
        """
        flat = self._flatten_dict(cfg, prefix)
        # MLflow has a 100-param batch limit
        items = list(flat.items())
        for i in range(0, len(items), 100):
            batch = dict(items[i : i + 100])
            mlflow.log_params(batch)
        logger.info("Logged %d config parameters.", len(flat))

    def log_params(self, params: Dict[str, Any]):
        """Log a flat dict of parameters."""
        mlflow.log_params(params)

    # -- Metric logging ----------------------------------------------------------

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log a dict of metrics (all at once for efficiency)."""
        mlflow.log_metrics(metrics, step=step)

    def log_image_metrics(self, result: dict, image_name: str = ""):
        """
        Log standardised per-image result metrics.

        Expected keys in result dict:
        - n_atoms_total, n_atoms_high_conf, unk_fraction
        - tile_confidence, explained_variance, unique_fraction
        - median_residual, status (OK/UNRELIABLE/REJECTED)
        - state_counts: {A: n, B: n, UNK: n}
        """
        prefix = f"{image_name}." if image_name else ""
        metrics = {}

        for key in [
            "n_atoms_total",
            "n_atoms_high_conf",
            "unk_fraction",
            "tile_confidence",
            "explained_variance",
            "unique_fraction",
            "median_residual",
        ]:
            if key in result:
                metrics[f"{prefix}{key}"] = float(result[key])

        if "state_counts" in result:
            for state, count in result["state_counts"].items():
                metrics[f"{prefix}state_{state}"] = count

        if "status" in result:
            status_code = {"OK": 1.0, "UNRELIABLE": 0.5, "REJECTED": 0.0}
            metrics[f"{prefix}quality_score"] = status_code.get(
                result["status"], 0.0
            )

        if metrics:
            mlflow.log_metrics(metrics)

    # -- Grid search logging ----------------------------------------------------

    def log_grid_search_results(
        self,
        results: List[dict],
        best_params: dict,
        best_score: float,
    ):
        """Log grid search summary to parent run."""
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_score", best_score)
        mlflow.log_metric("n_combinations_tested", len(results))

        # Save full results table as artifact
        import tempfile
        tmp_dir = tempfile.gettempdir()
        results_path = os.path.join(tmp_dir, "grid_search_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        mlflow.log_artifact(results_path, "grid_search")

    # -- Artifact logging -------------------------------------------------------

    def log_artifact(self, local_path: str, artifact_subdir: str = ""):
        """Log a file as an MLflow artifact."""
        if self.log_artifacts_flag and os.path.exists(local_path):
            mlflow.log_artifact(local_path, artifact_subdir)

    def log_figure(self, fig, filename: str, artifact_subdir: str = "plots"):
        """Log a matplotlib figure as an artifact."""
        if not self.log_artifacts_flag:
            return
        import tempfile
        tmp_dir = tempfile.gettempdir()
        tmp_path = os.path.join(tmp_dir, filename)
        fig.savefig(tmp_path, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(tmp_path, artifact_subdir)

    def log_dataframe(self, df, filename: str, artifact_subdir: str = "data"):
        """Log a pandas DataFrame as CSV artifact."""
        if not self.log_artifacts_flag:
            return
        import tempfile
        tmp_dir = tempfile.gettempdir()
        tmp_path = os.path.join(tmp_dir, filename)
        df.to_csv(tmp_path, index=False)
        mlflow.log_artifact(tmp_path, artifact_subdir)

    # -- Helpers ----------------------------------------------------------------

    @staticmethod
    def _flatten_dict(d: dict, prefix: str = "", sep: str = ".") -> dict:
        """Flatten nested dict for MLflow param logging."""
        items = {}
        for k, v in d.items():
            key = f"{prefix}{sep}{k}" if prefix else k
            if isinstance(v, dict):
                items.update(ExperimentTracker._flatten_dict(v, key, sep))
            elif isinstance(v, (list, tuple)):
                items[key] = str(v)
            else:
                items[key] = v
        return items

    def get_best_run(self, metric: str = "best_score", order: str = "DESC") -> dict:
        """Query the best run from the current experiment."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return {}
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )
        if runs:
            return {
                "run_id": runs[0].info.run_id,
                "params": runs[0].data.params,
                "metrics": runs[0].data.metrics,
            }
        return {}
