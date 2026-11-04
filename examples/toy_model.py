"""Minimal custom model for 21cmEVAL.

    eval21 run --data demo/data --model toy=python:examples/toy_model.py \
        --out results/toy

The only requirement is ``predict(sample) -> {name: array}``. ``describe()``
and ``close()`` are optional.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def predict(sample):
    density = np.asarray(sample.inputs["density"], dtype=np.float64)
    delta = np.log(np.maximum(density, 1e-6))
    smoothed = gaussian_filter(delta, sigma=(1.5, 1.5, 0))
    z = sample.meta.z_grid
    progress = np.clip((z - z.min()) / (z.max() - z.min()), 0.0, 1.0)
    assumed_xbar = (0.5 * (1.0 - np.cos(np.pi * progress))) ** 1.4

    xhi = np.empty(smoothed.shape, dtype=np.float64)
    for k in range(smoothed.shape[2]):
        threshold = np.quantile(smoothed[:, :, k], assumed_xbar[k])
        xhi[:, :, k] = 1.0 / (1.0 + np.exp((smoothed[:, :, k] - threshold) / 0.1))
    return {"xhi": xhi}


def describe():
    return {"framework": "example", "parameters": 0}
