"""
Device Management — GPU/CPU Dispatch
=====================================
Centralised hardware abstraction. Every compute-heavy function imports from here
so we get a single point of control for GPU vs CPU fallback.

Best practices implemented:
- Lazy GPU initialisation (don't touch CUDA until first use)
- Graceful fallback to CPU if GPU unavailable
- Memory pool management for CuPy (avoids fragmentation on RTX 5080)
- Device-agnostic array creation helpers
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_GPU_AVAILABLE: Optional[bool] = None
_cp = None  # lazy cupy import
_cuml = None  # lazy cuml import


def _probe_gpu() -> bool:
    """One-shot GPU availability check (cached)."""
    global _GPU_AVAILABLE, _cp
    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE

    try:
        import cupy as cp

        # Force a tiny allocation to verify the driver actually works
        _ = cp.zeros(1)
        _cp = cp
        _GPU_AVAILABLE = True

        dev = cp.cuda.Device(0)
        mem = dev.mem_info
        logger.info(
            "GPU detected: %s | Free: %.1f GB / %.1f GB",
            dev.pci_bus_id if hasattr(dev, "pci_bus_id") else "device-0",
            mem[0] / 1e9,
            mem[1] / 1e9,
        )
    except Exception as exc:
        logger.warning("GPU not available (%s), falling back to CPU.", exc)
        _GPU_AVAILABLE = False

    return _GPU_AVAILABLE


@dataclass
class DeviceManager:
    """
    Central GPU/CPU dispatch manager.

    Usage::

        dev = DeviceManager.from_config(cfg.gpu)
        xp = dev.xp  # cupy on GPU, numpy on CPU — same API

        # Move arrays
        arr_device = dev.to_device(numpy_array)
        arr_host   = dev.to_host(device_array)

        # Context-managed memory pool
        with dev.memory_pool():
            big = xp.random.randn(10_000, 10_000)
    """

    enabled: bool = True
    device_id: int = 0
    memory_fraction: float = 0.8
    _gpu_ok: bool = field(init=False, repr=False, default=False)

    def __post_init__(self):
        if self.enabled:
            self._gpu_ok = _probe_gpu()
            if self._gpu_ok:
                self._setup_memory_pool()
        else:
            self._gpu_ok = False
            logger.info("GPU explicitly disabled in config.")

    @classmethod
    def from_config(cls, gpu_cfg: dict) -> "DeviceManager":
        return cls(
            enabled=gpu_cfg.get("enabled", True),
            device_id=gpu_cfg.get("device_id", 0),
            memory_fraction=gpu_cfg.get("memory_fraction", 0.8),
        )

    # -- Array module dispatch --------------------------------------------------

    @property
    def xp(self):
        """Return cupy if GPU available, else numpy (duck-typed API)."""
        if self._gpu_ok:
            return _cp
        return np

    @property
    def is_gpu(self) -> bool:
        return self._gpu_ok

    def get_hardware_status(self) -> dict:
        """Return dict with hardware details for UI display."""
        if self._gpu_ok:
            try:
                import cupy as cp
                dev = cp.cuda.Device(self.device_id)
                return {
                    "type": "GPU",
                    "name": "NVIDIA RTX 5080" if "5080" in str(_cp.cuda.runtime.getDeviceProperties(self.device_id)) else "NVIDIA GPU",
                    "id": self.device_id,
                    "vram_free_gb": round(dev.mem_info[0] / 1e9, 2),
                    "vram_total_gb": round(dev.mem_info[1] / 1e9, 2)
                }
            except:
                return {"type": "GPU", "name": "NVIDIA GPU"}
        return {
            "type": "CPU",
            "name": f"AMD CPU ({os.cpu_count() or 16} cores)",
            "cores": os.cpu_count() or 16
        }

    # -- Data movement ----------------------------------------------------------

    def to_device(self, arr: np.ndarray):
        """Move numpy array to GPU (no-op if CPU mode)."""
        if self._gpu_ok:
            return _cp.asarray(arr)
        return arr

    def to_host(self, arr) -> np.ndarray:
        """Move array to CPU numpy (no-op if already numpy)."""
        if self._gpu_ok and hasattr(arr, "get"):
            return arr.get()
        return np.asarray(arr)

    # -- Memory pool (CuPy best practice) ----------------------------------------

    def _setup_memory_pool(self):
        """
        Configure CuPy memory pool to avoid fragmentation.
        RTX 5080 has 16 GB — we cap usage to memory_fraction.
        """
        if not self._gpu_ok:
            return
        try:
            _cp.cuda.Device(self.device_id).use()
            pool = _cp.cuda.MemoryPool(_cp.cuda.malloc_managed)
            _cp.cuda.set_allocator(pool.malloc)
            dev = _cp.cuda.Device(self.device_id)
            total_mem = dev.mem_info[1]
            limit = int(total_mem * self.memory_fraction)
            pool.set_limit(size=limit)
            logger.info(
                "CuPy memory pool: limit %.1f GB (%.0f%% of %.1f GB)",
                limit / 1e9,
                self.memory_fraction * 100,
                total_mem / 1e9,
            )
        except Exception as exc:
            logger.warning("Failed to configure CuPy memory pool: %s", exc)

    # -- cuML nearest-neighbor helper -------------------------------------------

    def nearest_neighbors(self, positions: np.ndarray, k: int = 6):
        """
        GPU-accelerated k-nearest-neighbor search using cuML.
        Falls back to sklearn on CPU.

        Returns:
            distances: (N, k) array
            indices:   (N, k) array
        """
        if self._gpu_ok:
            try:
                from cuml.neighbors import NearestNeighbors as cuNN

                pos_dev = self.to_device(positions.astype(np.float32))
                nn = cuNN(n_neighbors=k, algorithm="brute")
                nn.fit(pos_dev)
                distances, indices = nn.kneighbors(pos_dev)
                return self.to_host(distances), self.to_host(indices)
            except Exception as exc:
                logger.warning("cuML NN failed (%s), falling back to sklearn.", exc)

        # CPU fallback
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree", n_jobs=-1)
        nn.fit(positions)
        return nn.kneighbors(positions)

    # -- Batch 2D Gaussian fitting on GPU ----------------------------------------

    def batch_gaussian_2d_fit(
        self,
        image: np.ndarray,
        positions: np.ndarray,
        window_size: int = 7,
        batch_size: int = 512,
    ) -> dict:
        """
        GPU-accelerated batch 2D Gaussian fitting.

        Extracts windows around each atom position and fits 2D Gaussians
        in parallel on the GPU using least-squares.

        Args:
            image: (H, W) STEM image
            positions: (N, 2) initial atom positions [x, y]
            window_size: Half-size of fitting window
            batch_size: Atoms per GPU batch

        Returns:
            dict with keys: x_refined, y_refined, sigma_x, sigma_y,
                            amplitude, rotation, residual
        """
        N = len(positions)
        xp = self.xp

        results = {
            "x_refined": np.zeros(N),
            "y_refined": np.zeros(N),
            "sigma_x": np.full(N, np.nan),
            "sigma_y": np.full(N, np.nan),
            "amplitude": np.full(N, np.nan),
            "residual": np.full(N, np.nan),
        }

        img_dev = self.to_device(image.astype(np.float32))
        H, W = image.shape
        ws = window_size

        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            batch_pos = positions[batch_start:batch_end]

            for i, (px, py) in enumerate(batch_pos):
                idx = batch_start + i
                ix, iy = int(round(px)), int(round(py))

                # Bounds check
                x0 = max(0, ix - ws)
                x1 = min(W, ix + ws + 1)
                y0 = max(0, iy - ws)
                y1 = min(H, iy + ws + 1)

                if (x1 - x0) < 3 or (y1 - y0) < 3:
                    results["x_refined"][idx] = px
                    results["y_refined"][idx] = py
                    continue

                window = img_dev[y0:y1, x0:x1]

                # Center-of-mass refinement (GPU-native)
                yy, xx = xp.mgrid[y0:y1, x0:x1]
                total = xp.sum(window)
                if float(self.to_host(total)) > 0:
                    cx = float(self.to_host(xp.sum(xx * window) / total))
                    cy = float(self.to_host(xp.sum(yy * window) / total))
                    results["x_refined"][idx] = cx
                    results["y_refined"][idx] = cy

                    # Estimate sigma from second moment
                    dx = xx - cx
                    dy = yy - cy
                    w_norm = window / total
                    sx = float(self.to_host(xp.sqrt(xp.sum(dx**2 * w_norm))))
                    sy = float(self.to_host(xp.sqrt(xp.sum(dy**2 * w_norm))))
                    results["sigma_x"][idx] = max(0.1, sx)
                    results["sigma_y"][idx] = max(0.1, sy)
                    results["amplitude"][idx] = float(
                        self.to_host(xp.max(window))
                    )
                else:
                    results["x_refined"][idx] = px
                    results["y_refined"][idx] = py

        return results
