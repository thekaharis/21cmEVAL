"""Transverse isotropy: is the prediction invariant under x <-> y?

The transverse plane has no preferred axis, so a model whose output changes
when the input is transposed is fitting grid artefacts rather than physics.
The runner re-runs the adapter on the transposed sample and this metric
reports the RMS difference, both absolute and in units of the prediction's
standard deviation. It needs no target, only a re-runnable adapter.
"""
from __future__ import annotations

import numpy as np

from ..registry import register
from .base import Metric, nanmedian


@register("symmetry")
class SymmetryMetrics(Metric):
    pointwise = True
    needs = ()
    requires_rerun = True

    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        self.per_sample: list[float] = []
        self.std: list[float] = []

    def update(self, sample, prediction, aux=None, runtime=None):
        if not aux or "xy" not in aux:
            return
        pred = self.prediction(prediction)
        swapped = np.asarray(aux["xy"][self.settings.target], dtype=np.float64)
        if swapped.shape != pred.shape:
            return
        self.per_sample.append(float(np.sqrt(np.mean((pred - swapped) ** 2))))
        self.std.append(float(pred.std()))

    def result(self):
        if not self.per_sample:
            return {"note": "adapter was not re-run under x<->y"}
        rms = np.asarray(self.per_sample)
        std = np.asarray(self.std)
        with np.errstate(invalid="ignore", divide="ignore"):
            relative = np.where(std > 0, rms / std, np.nan)
        return {
            "xy_rms": float(np.median(rms)),
            "xy_rms_p84": float(np.percentile(rms, 84)),
            "xy_relative": float(nanmedian(relative))
            if np.isfinite(relative).any() else None,
            "n_samples": len(self.per_sample),
        }

    def arrays(self):
        return {"symmetry_xy_rms": np.asarray(self.per_sample)} if self.per_sample else {}
