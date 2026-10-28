"""Error versus distance to the ionization front.

Whole-volume L2 is dominated by trivially correct saturated regions; the
error that matters sits in a thin shell around the ionization fronts, and it
is mostly a *gradient* error (diffuse walls). This metric bins every voxel by
its signed distance to the nearest **true** front (positive into the neutral
side), then reports

* RMSE and gradient RMS versus distance -- confirms whether error is
  front-dominated;
* band RMSE within ``front_bands_mpc`` of the front;
* the 10-90 transition width of the mean x_HI profile, truth and prediction:
  a blurrier front gives a larger width;
* saturated-region RMSE, where the truth is already 0 or 1.

Distances use physical transverse sampling (Mpc), so anisotropic cells are
handled correctly. Slices with no front (all neutral or all ionized) are
skipped; they would only add trivial zeros. This is the real-space complement
to P(k)/r(k): a diffuse front is suppressed high-k power and low coherence at
high k, and the two diagnostics should agree.
"""
from __future__ import annotations

import numpy as np

from ..physics import front_width, signed_distance_2d
from ..registry import register
from .base import Metric, nanmedian


@register("front")
class FrontMetrics(Metric):
    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        n = int(round(2 * settings.front_d_max_mpc / settings.front_bin_mpc))
        self.edges = np.linspace(-settings.front_d_max_mpc,
                                 settings.front_d_max_mpc, n + 1)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.n = np.zeros(n)
        self.sum_sq = np.zeros(n)
        self.sum_sq2 = np.zeros(n)
        self.sum_grad = np.zeros(n)
        self.sum_truth = np.zeros(n)
        self.sum_pred = np.zeros(n)
        self.band_ss = {band: 0.0 for band in settings.front_bands_mpc}
        self.band_n = {band: 0 for band in settings.front_bands_mpc}
        self.saturated_ss = 0.0
        self.saturated_n = 0
        self.widths: list[dict] = []

    def update(self, sample, prediction, aux=None, runtime=None):
        truth = self.target(sample)
        pred = self.prediction(prediction)
        box = self.box_mpc(sample)
        dx = box / truth.shape[0]
        dy = box / truth.shape[1]
        sampling = (dx, dy)
        error = pred - truth
        edges = self.edges

        for k in range(truth.shape[2]):
            t_slice = truth[:, :, k]
            mask = t_slice >= self.settings.threshold
            distance = signed_distance_2d(mask, sampling)
            if not np.isfinite(distance).any():
                continue
            index = np.digitize(distance.ravel(), edges) - 1
            valid = (index >= 0) & (index < len(self.centers))
            if not valid.any():
                continue
            index = index[valid]
            squared = (error[:, :, k].ravel()[valid]) ** 2
            grad = self._grad_sq(error[:, :, k], dx, dy).ravel()[valid]
            self.n += np.bincount(index, minlength=len(self.centers))
            self.sum_sq += np.bincount(index, weights=squared,
                                       minlength=len(self.centers))
            self.sum_sq2 += np.bincount(index, weights=squared ** 2,
                                        minlength=len(self.centers))
            self.sum_grad += np.bincount(index, weights=grad,
                                         minlength=len(self.centers))
            self.sum_truth += np.bincount(
                index, weights=t_slice.ravel()[valid], minlength=len(self.centers))
            self.sum_pred += np.bincount(
                index, weights=pred[:, :, k].ravel()[valid], minlength=len(self.centers))

            absolute = np.abs(distance.ravel()[valid])
            for band in self.band_ss:
                # A coarse grid cannot resolve a band thinner than one cell;
                # fall back to the first cell shell instead of reporting NaN.
                select = absolute <= max(band, dx, dy)
                self.band_ss[band] += float(squared[select].sum())
                self.band_n[band] += int(select.sum())

            saturated = (t_slice <= 1e-3) | (t_slice >= 1 - 1e-3)
            if saturated.any():
                self.saturated_ss += float(np.sum(error[:, :, k][saturated] ** 2))
                self.saturated_n += int(saturated.sum())

        profile = self._profile()
        if profile is not None:
            self.widths.append({
                "truth": front_width(profile["d"], profile["truth"]),
                "pred": front_width(profile["d"], profile["pred"]),
            })

    @staticmethod
    def _grad_sq(field, dx, dy):
        gx, gy = np.gradient(field, dx, dy)
        return gx * gx + gy * gy

    def _profile(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(self.n > 0, self.sum_truth / self.n, np.nan)
            pred_mean = np.where(self.n > 0, self.sum_pred / self.n, np.nan)
        return {"d": self.centers, "truth": mean, "pred": pred_mean}

    def result(self):
        profile = self._profile()
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_sq = np.where(self.n > 0, self.sum_sq / self.n, np.nan)
            grad_rms = np.where(self.n > 0, np.sqrt(self.sum_grad / self.n), np.nan)
        width_truth = nanmedian([w["truth"] for w in self.widths]) if self.widths else np.nan
        width_pred = nanmedian([w["pred"] for w in self.widths]) if self.widths else np.nan
        bands = {}
        for band in self.band_ss:
            rmse = (np.sqrt(self.band_ss[band] / self.band_n[band])
                    if self.band_n[band] else np.nan)
            bands[f"rmse_within_{band:g}mpc"] = _f(rmse)
        return {
            "d_centers_mpc": self.centers,
            "mean_truth": profile["truth"],
            "mean_pred": profile["pred"],
            "rmse_profile": np.sqrt(mean_sq),
            "gradient_rms_profile": grad_rms,
            "front_width_truth_mpc": _f(width_truth),
            "front_width_pred_mpc": _f(width_pred),
            "front_width_excess_mpc": _f(width_pred - width_truth),
            "saturated_rmse": _f(
                np.sqrt(self.saturated_ss / self.saturated_n)
                if self.saturated_n else np.nan),
            "bands": bands,
            "n_samples": len(self.widths),
        }

    def arrays(self):
        profile = self._profile()
        with np.errstate(invalid="ignore", divide="ignore"):
            rmse = np.where(self.n > 0, np.sqrt(self.sum_sq / self.n), np.nan)
            grad = np.where(self.n > 0, np.sqrt(self.sum_grad / self.n), np.nan)
        return {
            "front_d_mpc": self.centers,
            "front_mean_truth": profile["truth"],
            "front_mean_pred": profile["pred"],
            "front_rmse": rmse,
            "front_gradient_rms": grad,
            "front_counts": self.n,
        }


def _f(value):
    value = float(value)
    return value if np.isfinite(value) else None
