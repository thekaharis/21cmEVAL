"""Error versus distance to the ionization front.

Whole-volume L2 is dominated by trivially correct saturated regions; the
error that matters sits in a thin shell around the ionization fronts, and it
is mostly a *gradient* error (diffuse walls). This metric bins every voxel by
its signed distance to the nearest **true** front (positive into the neutral
side), then reports

* RMSE and gradient RMS versus distance -- confirms whether error is
  front-dominated;
* per-cone band L2, H1 and error fraction within ``front_bands_mpc`` of the
  front, averaged over samples;
* the 10-90 transition width of the pooled mean x_HI profile, truth and
  prediction: a blurrier front gives a larger width;
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
from .base import Metric


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
        self.band_grad = {band: 0.0 for band in settings.front_bands_mpc}
        self.band_n = {band: 0 for band in settings.front_bands_mpc}
        self.total_sq_err = 0.0
        self.saturated_ss = 0.0
        self.saturated_n = 0
        self.neutral_pred = 0.0
        self.neutral_n = 0
        self.ionized_pred = 0.0
        self.ionized_n = 0
        self.per_sample: list[dict] = []
        self.widths: list[dict] = []
        self.skipped = 0

    def update(self, sample, prediction, aux=None, runtime=None):
        truth = self.target(sample)
        pred = self.prediction(prediction)
        if truth.shape[0] < 2 or truth.shape[1] < 2:
            self.skipped += 1
            return
        box = self.box_mpc(sample)
        dx = box / truth.shape[0]
        dy = box / truth.shape[1]
        sampling = (dx, dy)
        error = pred - truth
        edges = self.edges
        sample_band = {}
        sample_total_sq = 0.0
        sample_total_all = 0.0          # over every voxel of a front slice (fno-21cm errfrac)
        sample_n_front = 0

        for k in range(truth.shape[2]):
            t_slice = truth[:, :, k]
            mask = t_slice >= self.settings.threshold
            distance = signed_distance_2d(mask, sampling)
            if not np.isfinite(distance).any():
                continue
            sample_total_all += float(np.sum(error[:, :, k][np.isfinite(distance)] ** 2))
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
                index, weights=pred[:, :, k].ravel()[valid],
                minlength=len(self.centers))

            absolute = np.abs(distance.ravel()[valid])
            sample_total_sq += float(squared.sum())
            sample_n_front += int(valid.sum())
            for band in self.band_ss:
                # A coarse grid cannot resolve a band thinner than one cell;
                # fall back to the first cell shell instead of reporting NaN.
                select = absolute <= max(band, dx, dy)
                self.band_ss[band] += float(squared[select].sum())
                self.band_grad[band] += float(grad[select].sum())
                self.band_n[band] += int(select.sum())
                sample_band.setdefault(band, [0.0, 0.0, 0])
                sample_band[band][0] += float(squared[select].sum())
                sample_band[band][1] += float(grad[select].sum())
                sample_band[band][2] += int(select.sum())

            neutral = t_slice >= 1 - 1e-3
            ionized = t_slice <= 1e-3
            if neutral.any():
                self.neutral_pred += float(np.sum(pred[:, :, k][neutral]))
                self.neutral_n += int(neutral.sum())
            if ionized.any():
                self.ionized_pred += float(np.sum(pred[:, :, k][ionized]))
                self.ionized_n += int(ionized.sum())
            saturated = neutral | ionized
            if saturated.any():
                self.saturated_ss += float(np.sum(error[:, :, k][saturated] ** 2))
                self.saturated_n += int(saturated.sum())

        self.total_sq_err += sample_total_sq
        record = {"n_front_vox": sample_n_front, "sample": str(sample.id)}
        for band, (ss, gs, n) in sample_band.items():
            record[f"L2_band{band:g}"] = np.sqrt(ss / n) if n else np.nan
            record[f"H1_band{band:g}"] = np.sqrt(gs / n) if n else np.nan
            record[f"errfrac_band{band:g}"] = (
                ss / sample_total_sq if sample_total_sq > 0 else np.nan)
            record[f"errfrac_all_band{band:g}"] = (
                ss / sample_total_all if sample_total_all > 0 else np.nan)
        self.per_sample.append(record)

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
        width_truth = front_width(profile["d"], profile["truth"])
        width_pred = front_width(profile["d"], profile["pred"])
        per_sample_truth = _nanmedian([w["truth"] for w in self.widths])
        per_sample_pred = _nanmedian([w["pred"] for w in self.widths])
        bands = {}
        for band in self.band_ss:
            n = self.band_n[band]
            l2 = np.sqrt(self.band_ss[band] / n) if n else np.nan
            h1 = np.sqrt(self.band_grad[band] / n) if n else np.nan
            errfrac = (self.band_ss[band] / self.total_sq_err
                       if self.total_sq_err > 0 else np.nan)
            per_l2 = [r.get(f"L2_band{band:g}", np.nan)
                      for r in self.per_sample]
            per_h1 = [r.get(f"H1_band{band:g}", np.nan)
                      for r in self.per_sample]
            per_err = [r.get(f"errfrac_band{band:g}", np.nan)
                       for r in self.per_sample]
            bands[f"rmse_within_{band:g}mpc"] = _f(l2)
            bands[f"h1_within_{band:g}mpc"] = _f(h1)
            bands[f"errfrac_within_{band:g}mpc"] = _f(errfrac)
            bands[f"rmse_within_{band:g}mpc_mean"] = _f(_nanmean(per_l2))
            bands[f"h1_within_{band:g}mpc_mean"] = _f(_nanmean(per_h1))
            bands[f"errfrac_within_{band:g}mpc_mean"] = _f(_nanmean(per_err))
        status = ("ok" if np.isfinite(width_truth) and np.isfinite(width_pred)
                  else "unresolved: the mean profile does not span 0.1-0.9 "
                       "inside the distance window")
        return {
            "d_centers_mpc": self.centers,
            "mean_truth": profile["truth"],
            "mean_pred": profile["pred"],
            "rmse_profile": np.sqrt(mean_sq),
            "gradient_rms_profile": grad_rms,
            "front_width_truth_mpc": _f(width_truth),
            "front_width_pred_mpc": _f(width_pred),
            "front_width_excess_mpc": (_f(width_pred - width_truth)
                                       if np.isfinite(width_pred)
                                       and np.isfinite(width_truth) else None),
            "front_width_per_sample_truth_mpc": _f(per_sample_truth),
            "front_width_per_sample_pred_mpc": _f(per_sample_pred),
            "front_width_status": status,
            "saturated_rmse": _f(
                np.sqrt(self.saturated_ss / self.saturated_n)
                if self.saturated_n else np.nan),
            "pred_where_truth_neutral_mean": _f(
                self.neutral_pred / self.neutral_n if self.neutral_n else np.nan),
            "pred_where_truth_ionized_mean": _f(
                self.ionized_pred / self.ionized_n if self.ionized_n else np.nan),
            "bands": bands,
            "n_samples": len(self.widths),
            "skipped_samples": self.skipped,
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
            "front_width_truth": np.asarray([w["truth"] for w in self.widths]),
            "front_width_pred": np.asarray([w["pred"] for w in self.widths]),
            **self._band_arrays("front"),
        }

    def _band_arrays(self, prefix):
        """Per-sample band metrics for ``eval21 plot edge`` (csv means and the paired
        bootstrap of fno-21cm's boundary_band_diagnostic). ``errfrac`` there divides by the
        squared error of every voxel of the front slices, not only those within d_max."""
        out = {f"{prefix}_sample_ids": np.array([r.get("sample", "") for r in self.per_sample]),
               f"{prefix}_n_front_vox": np.array([r["n_front_vox"] for r in self.per_sample], dtype=np.int64)}
        for band in self.settings.front_bands_mpc:
            for name, key in (("L2", "L2_band"), ("H1", "H1_band"), ("errfrac", "errfrac_all_band")):
                out[f"{prefix}_band_{name}_{band:g}"] = np.array(
                    [r.get(f"{key}{band:g}", np.nan) for r in self.per_sample], dtype=np.float64)
        return out


def _nanmedian(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmedian(values))


def _nanmean(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmean(values))


def _f(value):
    value = float(value)
    return value if np.isfinite(value) else None
