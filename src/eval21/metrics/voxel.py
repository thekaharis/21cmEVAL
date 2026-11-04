"""Voxel-level accuracy: RMSE, R^2, MAE, bias, Pearson r.

Definitions, stated because R^2 on a bimodal field is easy to misread::

    RMSE = sqrt(sum (p - t)^2 / N)
    R^2  = 1 - sum (p - t)^2 / sum (t - mean(t))^2

``mean(t)`` is taken over all evaluated voxels of the split, so R^2 uses the
constant-mean predictor as reference. x_HI is strongly bimodal (mass at 0 and
1), which makes that predictor poor and R^2 flattering: it is a relative
score, not an accuracy. Accumulation is streaming, so memory does not grow
with the number of samples.
"""
from __future__ import annotations

import numpy as np

from ..registry import register
from .base import Metric, median_band


@register("voxel")
class VoxelMetrics(Metric):
    pointwise = True
    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        self.n = 0
        self.ss_res = 0.0
        self.sum_abs = 0.0
        self.sum_err = 0.0
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.sum_p = 0.0
        self.sum_p2 = 0.0
        self.sum_py = 0.0
        self.per_sample_rmse: list[float] = []
        self.per_sample_r2: list[float] = []
        self.sum_history_abs = 0.0
        self.history_samples = 0
        self.grad_ss = 0.0
        self.grad_n = 0
        self.highk_pred = 0.0
        self.highk_truth = 0.0
        self.highk_cross = 0.0
        self._highk_mask = None

    def update(self, sample, prediction, aux=None, runtime=None):
        truth = self.target(sample)
        pred = self.prediction(prediction)
        if truth.shape != pred.shape:
            raise ValueError(f"shape mismatch for sample {sample.id!r}")
        error_cube = pred - truth
        t = truth.ravel()
        p = pred.ravel()
        error = error_cube.ravel()
        self.n += t.size
        self.ss_res += float(error @ error)
        self.sum_abs += float(np.abs(error).sum())
        self.sum_err += float(error.sum())
        self.sum_y += float(t.sum())
        self.sum_y2 += float(t @ t)
        self.sum_p += float(p.sum())
        self.sum_p2 += float(p @ p)
        self.sum_py += float(p @ t)

        rmse = float(np.sqrt(np.mean(error ** 2)))
        self.per_sample_rmse.append(rmse)

        # Reference fno_xhi2d.final_report: sum |mean_z error| per cube,
        # divided by the number of cubes (not by n_z).
        history_error = (pred.mean(axis=(0, 1))
                         - truth.mean(axis=(0, 1)))
        self.sum_history_abs += float(np.abs(history_error).sum())
        self.history_samples += 1

        # Reference gradient error: forward periodic differences in cell
        # units, divided by 2 * numel (dx and dy over every voxel).
        dx = np.roll(error_cube, -1, axis=0) - error_cube
        dy = np.roll(error_cube, -1, axis=1) - error_cube
        self.grad_ss += float(np.sum(dx * dx + dy * dy))
        self.grad_n += 2 * error_cube.size

        mask = self._highk_region(truth.shape[:2])
        ft = np.fft.rfft2(truth, axes=(0, 1), norm="ortho")
        fp = np.fft.rfft2(pred, axes=(0, 1), norm="ortho")
        self.highk_truth += float(np.sum(np.abs(ft[mask]) ** 2))
        self.highk_pred += float(np.sum(np.abs(fp[mask]) ** 2))
        self.highk_cross += float(np.sum(np.real(fp[mask] * np.conj(ft[mask]))))
        denominator = float(((t - t.mean()) ** 2).sum())
        if denominator > 0:
            self.per_sample_r2.append(1.0 - float(error @ error) / denominator)

    def _highk_region(self, shape):
        key = (shape[0], shape[1])
        if self._highk_mask is not None and self._highk_mask[0] == key:
            return self._highk_mask[1]
        fx = np.fft.fftfreq(shape[0])[:, None]
        fy = np.fft.rfftfreq(shape[1])[None, :]
        mask = np.sqrt(fx * fx + fy * fy) >= 0.25
        self._highk_mask = (key, mask)
        return mask

    def result(self):
        if self.n == 0:
            return {}
        mean_y = self.sum_y / self.n
        ss_tot = self.sum_y2 - self.n * mean_y ** 2
        covariance = self.sum_py - self.sum_p * self.sum_y / self.n
        variance_p = self.sum_p2 - self.sum_p ** 2 / self.n
        denominator = float(np.sqrt(max(variance_p, 0.0) * max(ss_tot, 0.0)))
        rmse = float(np.sqrt(self.ss_res / self.n))
        r2 = 1.0 - self.ss_res / ss_tot if ss_tot > 0 else float("nan")
        r2_mean = float(np.mean(self.per_sample_r2)) if self.per_sample_r2 else float("nan")
        _, r2_lo, r2_hi = median_band(self.per_sample_r2)
        return {
            "rmse": rmse,
            "mse": self.ss_res / self.n,
            "mae": self.sum_abs / self.n,
            "bias": self.sum_err / self.n,
            "r2": r2,
            "pearson_r": float(np.clip(covariance / denominator, -1, 1))
            if denominator > 1e-12 else float("nan"),
            "truth_mean": mean_y,
            "truth_std": float(np.sqrt(max(ss_tot, 0.0) / self.n)),
            "r2_per_sample_mean": float(r2_mean) if np.isfinite(r2_mean) else None,
            "gradient_rmse": (float(np.sqrt(self.grad_ss / self.grad_n))
                              if self.grad_n else None),
            "mean_xhi_mae": (self.sum_history_abs / self.history_samples
                             if self.history_samples else None),
            "high_k_power_ratio": (self.highk_pred / self.highk_truth
                                   if self.highk_truth > 0 else None),
            "high_k_cross_correlation": (
                self.highk_cross / np.sqrt(self.highk_pred * self.highk_truth)
                if self.highk_pred > 0 and self.highk_truth > 0 else None),
            "r2_per_sample_p16": float(r2_lo) if np.isfinite(r2_lo) else None,
            "r2_per_sample_p84": float(r2_hi) if np.isfinite(r2_hi) else None,
            "n_voxels": int(self.n),
            "n_samples": len(self.per_sample_rmse),
        }

    def arrays(self):
        return {
            "rmse_per_sample": np.asarray(self.per_sample_rmse),
            "r2_per_sample": np.asarray(self.per_sample_r2),
        }
