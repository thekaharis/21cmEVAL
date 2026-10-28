"""Calibration: which part of the x_HI range does the model get wrong?

A binned parity curve. Voxels are binned by their **true** value; within each
bin we report the mean and spread of the prediction and the RMSE. Systematic
departure of the conditional mean from the identity is conditional bias
``E[pred | true] - true`` -- the hedging-toward-the-mean signature of an
MSE-trained emulator in the transition range. The bin occupancy is reported
with it, because x_HI is strongly bimodal and the sparse mid-range bins would
otherwise look as trustworthy as the saturated ones.

Everything is streaming per bin, so the cost is one pass and a few kB.
"""
from __future__ import annotations

import numpy as np

from ..registry import register
from .base import Metric


@register("parity")
class ParityMetrics(Metric):
    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        lo, hi = settings.xhi_range
        self.edges = np.linspace(max(0.0, lo), min(1.0, hi), 41)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        n = len(self.centers)
        self.count = np.zeros(n)
        self.sum_pred = np.zeros(n)
        self.sum_pred2 = np.zeros(n)
        self.sum_err2 = np.zeros(n)
        self.sum_truth = np.zeros(n)
        self.partial_ss = 0.0
        self.partial_n = 0
        self.lo, self.hi = settings.partial_range

    def update(self, sample, prediction, aux=None, runtime=None):
        truth = self.target(sample)
        pred = self.prediction(prediction)
        flat_t = truth.ravel()
        flat_p = pred.ravel()
        index = np.digitize(flat_t, self.edges) - 1
        valid = (index >= 0) & (index < len(self.centers))
        if not valid.any():
            return
        index = index[valid]
        t = flat_t[valid]
        p = flat_p[valid]
        error2 = (p - t) ** 2
        n = len(self.centers)
        self.count += np.bincount(index, minlength=n)
        self.sum_pred += np.bincount(index, weights=p, minlength=n)
        self.sum_pred2 += np.bincount(index, weights=p * p, minlength=n)
        self.sum_err2 += np.bincount(index, weights=error2, minlength=n)
        self.sum_truth += np.bincount(index, weights=t, minlength=n)

        partial = (t > self.lo) & (t < self.hi)
        self.partial_ss += float(error2[partial].sum())
        self.partial_n += int(partial.sum())

    def result(self):
        if self.count.sum() == 0:
            return {}
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(self.count > 0, self.sum_pred / self.count, np.nan)
            mean2 = np.where(self.count > 0, self.sum_pred2 / self.count, np.nan)
            rmse = np.where(self.count > 0, np.sqrt(self.sum_err2 / self.count), np.nan)
            truth_mean = np.where(self.count > 0, self.sum_truth / self.count, np.nan)
            std = np.sqrt(np.maximum(mean2 - mean ** 2, 0.0))
            bias = mean - truth_mean
        occupied = self.count >= max(1, 0.001 * self.count.sum())
        bias_max = float(np.nanmax(np.abs(bias[occupied]))) if occupied.any() else np.nan
        bins = [
            {
                "center": float(self.centers[i]),
                "count": int(self.count[i]),
                "truth_mean": _f(truth_mean[i]),
                "pred_mean": _f(mean[i]),
                "bias": _f(bias[i]),
                "pred_std": _f(std[i]),
                "rmse": _f(rmse[i]),
            }
            for i in range(len(self.centers))
            if self.count[i] > 0
        ]
        return {
            "bins": bins,
            "headline": {
                "conditional_bias_max": _f(bias_max),
                "partial_rmse": _f(
                    np.sqrt(self.partial_ss / self.partial_n)
                    if self.partial_n else np.nan),
                "partial_voxels": int(self.partial_n),
            },
            "n_voxels": int(self.count.sum()),
        }

    def arrays(self):
        if self.count.sum() == 0:
            return {}
        return {
            "parity_centers": self.centers,
            "parity_count": self.count,
            "parity_rmse": np.sqrt(np.divide(
                self.sum_err2, self.count, out=np.full_like(self.count, np.nan),
                where=self.count > 0)),
            "parity_bias": np.divide(
                self.sum_pred - self.sum_truth, self.count,
                out=np.full_like(self.count, np.nan), where=self.count > 0),
        }


def _f(value):
    value = float(value)
    return value if np.isfinite(value) else None
