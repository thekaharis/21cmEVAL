"""A deliberately simple density -> x_HI "model", for demos and smoke tests.

It smooths the log density and thresholds it. The output is smooth, bounded,
and correlated with the truth, which is exactly what is needed to see every
metric produce a sensible non-degenerate number. It says nothing about any
real emulator; do not quote its scores.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from ..spec import Sample
from .base import BaseAdapter


class SmoothThresholdModel(BaseAdapter):
    framework = "synthetic"

    def __init__(self, sigma: float = 1.5, width: float = 0.1,
                 threshold: float | None = None):
        self.sigma = float(sigma)
        self.width = float(width)
        self.threshold = threshold

    def predict(self, sample: Sample) -> dict:
        density = np.asarray(sample.inputs["density"], dtype=np.float64)   # overdensity delta
        delta = np.log(np.maximum(1.0 + density, 1e-6))
        smoothed = gaussian_filter(delta, sigma=(self.sigma, self.sigma, 0))
        z = sample.meta.z_grid
        shape = smoothed.shape
        if z is None:
            assumed = np.full(shape[2], 0.5)
        else:
            progress = np.clip((np.asarray(z) - z.min())
                               / (z.max() - z.min()), 0.0, 1.0)
            assumed = (0.5 * (1.0 - np.cos(np.pi * progress))) ** 1.4
        xhi = np.empty(shape, dtype=np.float64)
        for k in range(shape[2]):
            threshold = np.quantile(smoothed[:, :, k], assumed[k])
            xhi[:, :, k] = 1.0 / (
                1.0 + np.exp((smoothed[:, :, k] - threshold) / self.width))
        return {"xhi": xhi}

    def describe(self) -> dict:
        info = super().describe()
        info.update({"sigma": self.sigma, "width": self.width})
        return info


class ToyGenerativeModel(SmoothThresholdModel):
    """A parameters-only "generative model": x_HI from a fresh random field per draw.

    It reads no input field: each draw thresholds a Gaussian random field (seeded by the sample
    id and the draw index) with the same z-dependent ionized fraction as
    :class:`SmoothThresholdModel`. Its maps do not align with the truth, its statistics roughly
    do -- the case ``--mode generative --draws N`` is built for.
    """

    framework = "synthetic-generative"

    def predict(self, sample: Sample, draw: int = 0) -> dict:
        import zlib

        shape = sample.shape
        seed = (zlib.crc32(str(sample.id).encode()) * 1009 + int(draw)) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        field = rng.standard_normal(shape)
        fake = sample.__class__(id=sample.id, inputs={"density": np.exp(0.5 * field) - 1.0},
                                targets={}, meta=sample.meta)
        return super().predict(fake)
