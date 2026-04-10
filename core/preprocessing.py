"""
Preprocessing Module
====================
Image manipulation before peak detection:
- Rescaling (interpolation)
- Binning (area averaging)
- Denoising (Gaussian blur, Median filter)
- Cropping

All operations are tracked via MLflow to ensure pixel -> nanometer traceablity.
"""

from __future__ import annotations
import logging
import numpy as np
from skimage.transform import rescale, downscale_local_mean
from skimage.filters import gaussian

logger = logging.getLogger(__name__)

def preprocess_image(image: np.ndarray, config: dict) -> np.ndarray:
    """
    Apply preprocessing pipeline based on config.
    
    Args:
        image: Original raw image
        config: preprocessing section of default.yaml
        
    Returns:
        Processed image (potentially different shape)
    """
    processed = image.copy()
    
    # 1. Binning (Spatial reduction, conservative)
    bin_factor = config.get("binning", 1)
    if bin_factor > 1:
        prev_shape = processed.shape
        # Use local mean for high-quality binning (area average)
        processed = downscale_local_mean(processed, (bin_factor, bin_factor))
        logger.info("Binned image: %s -> %s (factor %d)", prev_shape, processed.shape, bin_factor)
        
    # 2. Rescaling (Sub-pixel reduction, interpolation)
    rescale_f = config.get("rescale_factor", 1.0)
    if rescale_f != 1.0:
        prev_shape = processed.shape
        # Use anti-aliasing to avoid Moire patterns in lattice
        processed = rescale(processed, rescale_f, anti_aliasing=True, preserve_range=True)
        logger.info("Rescaled image: %s -> %s (factor %.2f)", prev_shape, processed.shape, rescale_f)
        
    # 3. Denoising
    if config.get("low_pass_filter", False):
        sigma = config.get("sigma_blur", 1.0)
        processed = gaussian(processed, sigma=sigma)
        logger.info("Applied Gaussian blur (sigma=%.1f)", sigma)
        
    # 4. Edge Cropping
    crop = config.get("crop_edge_pixels", 0)
    if crop > 0:
        h, w = processed.shape
        processed = processed[crop:h-crop, crop:w-crop]
        logger.info("Cropped edges: removed %d pixels", crop)
        
    return processed.astype(np.float32)

def transform_coordinates_back(coords: np.ndarray, config: dict) -> np.ndarray:
    """
    Map coordinates from processed image space back to original raw image space.
    Crucial for physical atom positions.
    """
    back_coords = coords.copy()
    
    # Reverse Edge Cropping
    crop = config.get("crop_edge_pixels", 0)
    back_coords += crop
    
    # Reverse Rescaling
    rescale_f = config.get("rescale_factor", 1.0)
    if rescale_f != 1.0:
        back_coords /= rescale_f
        
    # Reverse Binning
    bin_factor = config.get("binning", 1)
    if bin_factor > 1:
        back_coords *= bin_factor
        
    return back_coords
