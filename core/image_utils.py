from __future__ import annotations

import numpy as np


def normalize_hyperspy_load(loaded, image_path: str):
    if isinstance(loaded, list):
        if not loaded:
            raise ValueError(f"No signals were loaded from {image_path}")
        loaded = loaded[0]
    return loaded


def signal_to_2d_float_array(signal) -> np.ndarray:
    data = np.asarray(signal.data)

    if data.dtype.fields:
        field_names = list(data.dtype.fields.keys())
        if not field_names:
            raise ValueError("Structured image has no fields to convert.")
        channels = [data[name].astype(np.float32, copy=False) for name in field_names[:4]]
        data = np.mean(np.stack(channels, axis=-1), axis=-1)
    elif data.ndim > 2:
        data = np.mean(data.astype(np.float32, copy=False), axis=-1)

    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D image after normalization, got shape {data.shape}.")
    return data

