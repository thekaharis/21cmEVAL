"""Calibration: which part of the x_HI range does the model get wrong?

A binned parity curve. Voxels are binned by their **true** value; within each
bin we report the mean and spread of the prediction, its 16/50/84 percentiles
and the RMSE. Systematic departure of the conditional mean from the identity
is conditional bias ``E[pred | true] - true`` -- the hedging-toward-the-mean
signature of an MSE-trained emulator in the transition range. The bin
occupancy is reported with it, because x_HI is strongly bimodal and the sparse
mid-range bins would otherwise look as trustworthy as the saturated ones.

This follows ``fno-21cm/viz/parity_diagnostic.py``: 50 true bins on [0, 1]
with values clipped in (so x_HI = 1.0 is counted), and percentiles from a
50 x 400 count histogram rather than from stored samples.
"""
from __future__ import annotations

import numpy as np

from ..registry import register
from .base import Metric


@register("parity")
class ParityMetrics(Metric):
    pointwise = True
    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        self.n_true = 50
        self.n_pred = 400
        self.count = np.zeros(self.n_true, dtype=np.int64)
        self.sum_pred = np.zeros(self.n_true)
        self.sum_pred2 = np.zeros(self.n_true)
        self.sum_truth = np.zeros(self.n_true)
        self.sum_err2 = np.zeros(self.n_true)
        self.h2d = np.zeros(self.n_true * self.n_pred, dtype=np.int64)
        self.lo, self.hi = settings.partial_range
        self._partial_ss = 0.0
        self._partial_count = 0

    def update(self, sample, prediction, aux=None, runtime=None):
        truth = np.clip(self.target(sample).ravel(), 0.0, 1.0)
        pred = np.clip(self.prediction(prediction).ravel(), 0.0, 1.0)
        ti = np.minimum((truth * self.n_true).astype(np.int64),
                        self.n_true - 1)
        pi = np.minimum((pred * self.n_pred).astype(np.int64),
                        self.n_pred - 1)
        error2 = (pred - truth) ** 2
        self.count += np.bincount(ti, minlength=self.n_true)
        self.sum_pred += np.bincount(ti, weights=pred, minlength=self.n_true)
        self.sum_pred2 += np.bincount(ti, weights=pred ** 2,
                                      minlength=self.n_true)
        self.sum_truth += np.bincount(ti, weights=truth, minlength=self.n_true)
        self.sum_err2 += np.bincount(ti, weights=error2, minlength=self.n_true)
        self.h2d += np.bincount(ti * self.n_pred + pi,
                                minlength=self.n_true * self.n_pred)
        partial = (truth > self.lo) & (truth < self.hi)
        self._partial_ss += float(error2[partial].sum())
        self._partial_count += int(partial.sum())

    def _percentiles(self):
        hist = self.h2d.reshape(self.n_true, self.n_pred)
        centers = (np.arange(self.n_pred) + 0.5) / self.n_pred
        out = np.full((self.n_true, 3), np.nan)
        for i in range(self.n_true):
            total = hist[i].sum()
            if total == 0:
                continue
            cumulative = np.cumsum(hist[i])
            for j, q in enumerate((16.0, 50.0, 84.0)):
                index = int(np.searchsorted(cumulative, q / 100.0 * total))
                out[i, j] = centers[min(index, self.n_pred - 1)]
        return out

    def result(self):
        if self.count.sum() == 0:
            return {}
        with np.errstate(invalid="ignore", divide="ignore"):
            n = np.maximum(self.count, 1).astype(np.float64)
            mean = self.sum_pred / n
            variance = np.clip(self.sum_pred2 / n - mean ** 2, 0.0, None)
            truth_mean = self.sum_truth / n
            rmse = np.sqrt(self.sum_err2 / n)
            bias = mean - truth_mean
        percentiles = self._percentiles()
        empty = self.count == 0
        mean[empty] = np.nan
        truth_mean[empty] = np.nan
        rmse[empty] = np.nan
        bias[empty] = np.nan
        minimum = max(5, int(0.001 * self.count.sum()))
        occupied = self.count >= minimum
        bias_max = (_f(np.nanmax(np.abs(bias[occupied])))
                    if occupied.any() else None)
        bias_max_count = (int(self.count[occupied][
            int(np.nanargmax(np.abs(bias[occupied])))]) if occupied.any() else None)
        bins = [
            {
                "center": float((i + 0.5) / self.n_true),
                "count": int(self.count[i]),
                "truth_mean": _f(truth_mean[i]),
                "pred_mean": _f(mean[i]),
                "pred_std": _f(np.sqrt(variance[i])),
                "pred_p16": _f(percentiles[i, 0]),
                "pred_p50": _f(percentiles[i, 1]),
                "pred_p84": _f(percentiles[i, 2]),
                "bias": _f(bias[i]),
                "rmse": _f(rmse[i]),
            }
            for i in range(self.n_true) if self.count[i] > 0
        ]
        return {
            "bins": bins,
            "headline": {
                "conditional_bias_max": bias_max,
                "conditional_bias_max_count": bias_max_count,
                "conditional_bias_min_count": minimum,
                "partial_rmse": (_f(np.sqrt(self._partial_ss / self._partial_count))
                                 if self._partial_count else None),
                "partial_voxels": int(self._partial_count),
            },
            "n_voxels": int(self.count.sum()),
        }

def _f(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None
