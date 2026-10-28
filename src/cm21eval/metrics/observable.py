"""The 21-cm observable, not just x_HI.

Voxel error does not say whether the difference matters to a radio
observation. This metric re-expresses the prediction in brightness
temperature and compares one-point statistics:

* **PDFs** of x_HI and dT_b per stage, with the Wasserstein-1 distance between
  truth and prediction. The partially ionized fraction
  (``0.1 < x_HI < 0.9``) is reported separately: an MSE-blurred field inflates
  it, a sharp field does not.
* **variance and skewness** of dT_b versus the global neutral fraction. These
  are the moments a 21-cm power-spectrum experiment can actually constrain.
* **Delta^2_21(k)** in mK^2 at fixed target wavenumbers, and its ratio.
* **power-bias significance**: ``(P_pred - P_truth) / (P_truth sqrt(2/N_modes))``,
  i.e. the emulator bias in units of the Gaussian sample variance of a single
  slice. ``1/significance^2`` is the number of independent slices an observer
  must average before the bias equals their statistical error.

dT_b uses the saturated-spin-temperature limit without redshift-space
distortions and the *same* density for truth and prediction, so differences
are purely the emulator's x_HI error propagated to the observable. It is not
the 21cmFAST ``brightness_temp`` field, which also carries T_S and velocities.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..physics import (
    KBinner, brightness_temperature_mk, histogram_w1, slice_moments,
    stage_labels, stage_masks,
)
from ..registry import register
from .base import Metric, nanmedian


@register("observable")
class ObservableMetrics(Metric):
    needs = ("density",)

    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        self.rows: list[dict] = []
        self.skipped = 0
        self._binner = None
        self.xhi_edges = np.linspace(*settings.xhi_range, settings.xhi_bins + 1)
        self.dtb_edges = np.linspace(*settings.dtb_range_mk, settings.dtb_bins + 1)
        self.xbar_edges = np.linspace(0.0, 1.0, 20 + 1)

    def _bins(self, nx, ny, box):
        key = (nx, ny, round(box, 9), self.settings.n_k_bins)
        if self._binner is None or self._binner[0] != key:
            binner = KBinner(nx, ny, box, self.settings.n_k_bins)
            self._binner = (key, binner)
        return self._binner[1]

    def update(self, sample, prediction, aux=None, runtime=None):
        z = self.z_grid(sample)
        if z is None:
            self.skipped += 1
            return
        truth = self.target(sample)
        pred = self.prediction(prediction)
        density = self.require(sample, self.settings.density)
        if density.shape != truth.shape:
            raise ValueError(f"sample {sample.id!r}: density shape mismatch")
        box = self.box_mpc(sample)
        binner = self._bins(*truth.shape[:2], box)
        cosmology = dict(omega_m=self.omega_m(sample),
                         omega_b=self.settings.omega_b, h=self.settings.hubble_h)
        dtb_truth = brightness_temperature_mk(truth, density, z, **cosmology)
        dtb_pred = brightness_temperature_mk(pred, density, z, **cosmology)

        masks = stage_masks(
            truth.mean(axis=(0, 1)),
            self.settings.stage_edges,
            self.settings.active_range,
        )
        power_t = binner.slice_sums(truth - truth.mean(axis=(0, 1), keepdims=True))
        power_p = binner.slice_sums(pred - pred.mean(axis=(0, 1), keepdims=True))
        power_dt = binner.slice_sums(
            dtb_truth - dtb_truth.mean(axis=(0, 1), keepdims=True))
        power_dp = binner.slice_sums(
            dtb_pred - dtb_pred.mean(axis=(0, 1), keepdims=True))
        counts = binner.counts
        lo, hi = self.settings.partial_range
        k = binner.centers
        scale = k ** 2 / (2 * np.pi)
        moments_truth = slice_moments(dtb_truth)
        moments_pred = slice_moments(dtb_pred)

        stages = []
        for mask in masks:
            n = int(mask.sum())
            if n == 0:
                stages.append(None)
                continue
            xhi_hist_t, _ = np.histogram(
                np.clip(truth[:, :, mask], self.xhi_edges[0], self.xhi_edges[-1]),
                bins=self.xhi_edges)
            xhi_hist_p, _ = np.histogram(
                np.clip(pred[:, :, mask], self.xhi_edges[0], self.xhi_edges[-1]),
                bins=self.xhi_edges)
            dtb_hist_t, _ = np.histogram(
                np.clip(dtb_truth[:, :, mask], self.dtb_edges[0], self.dtb_edges[-1]),
                bins=self.dtb_edges)
            dtb_hist_p, _ = np.histogram(
                np.clip(dtb_pred[:, :, mask], self.dtb_edges[0], self.dtb_edges[-1]),
                bins=self.dtb_edges)

            with np.errstate(invalid="ignore", divide="ignore"):
                sigma_t = power_dt[mask] * np.sqrt(2.0 / counts)[None, :]
                significance = (power_dp[mask] - power_dt[mask]) / sigma_t
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    sig_median = np.nanmedian(significance, axis=0)

            denominator = n * counts
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_pt = power_t[mask].sum(axis=0) / denominator
                mean_pp = power_p[mask].sum(axis=0) / denominator
                ratio = np.where(mean_pt > 0, mean_pp / mean_pt, np.nan)
            stages.append({
                "n_slices": n,
                "xhi_hist_truth": xhi_hist_t.astype(np.float64),
                "xhi_hist_pred": xhi_hist_p.astype(np.float64),
                "dtb_hist_truth": dtb_hist_t.astype(np.float64),
                "dtb_hist_pred": dtb_hist_p.astype(np.float64),
                "w1_xhi": histogram_w1(xhi_hist_t, xhi_hist_p, self.xhi_edges),
                "w1_dtb": histogram_w1(dtb_hist_t, dtb_hist_p, self.dtb_edges),
                "partial_truth": float(np.mean(
                    (truth[:, :, mask] > lo) & (truth[:, :, mask] < hi))),
                "partial_pred": float(np.mean(
                    (pred[:, :, mask] > lo) & (pred[:, :, mask] < hi))),
                "variance_truth": float(np.mean(moments_truth["variance"][mask])),
                "variance_pred": float(np.mean(moments_pred["variance"][mask])),
                "skewness_truth": float(np.nanmean(moments_truth["skewness"][mask])),
                "skewness_pred": float(np.nanmean(moments_pred["skewness"][mask])),
                "d2_ratio": ratio,
                "significance": sig_median,
                "d2_truth": scale * mean_pt,
                "d2_pred": scale * mean_pp,
            })
        self.rows.append({"sample": sample.id, "stages": stages})

    def result(self):
        if not self.rows:
            return {"note": "no z_grid or density available"}
        n_rows = len(self.rows[0]["stages"])
        labels = stage_labels(self.settings.stage_edges, self.settings.active_range)

        def gather(key):
            out = np.full((len(self.rows), n_rows), np.nan)
            for i, row in enumerate(self.rows):
                for j, stage in enumerate(row["stages"]):
                    if stage is not None:
                        out[i, j] = stage[key]
            return out

        w1_xhi = gather("w1_xhi")
        w1_dtb = gather("w1_dtb")
        partial_t = gather("partial_truth")
        partial_p = gather("partial_pred")
        variance_ratio = gather("variance_pred") / gather("variance_truth")
        skew_delta = gather("skewness_pred") - gather("skewness_truth")
        significance = np.full((len(self.rows), n_rows), np.nan)
        d2_ratio = np.full((len(self.rows), n_rows, len(self.settings.k_targets)), np.nan)
        k = self._binner[1].centers
        target_index = [int(np.argmin(np.abs(np.log(k) - np.log(t))))
                        for t in self.settings.k_targets]
        for i, row in enumerate(self.rows):
            for j, stage in enumerate(row["stages"]):
                if stage is None:
                    continue
                significance[i, j] = float(nanmedian(np.abs(stage["significance"])))
                d2_ratio[i, j] = stage["d2_ratio"][target_index]

        stages = []
        for j, label in enumerate(labels):
            stages.append({
                "label": label,
                "n_slices": int(np.nansum(
                    [r["stages"][j]["n_slices"] for r in self.rows
                     if r["stages"][j] is not None])),
                "w1_xhi": _f(nanmedian(w1_xhi[:, j])),
                "w1_dtb_mk": _f(nanmedian(w1_dtb[:, j])),
                "partial_fraction_truth": _f(nanmedian(partial_t[:, j])),
                "partial_fraction_pred": _f(nanmedian(partial_p[:, j])),
                "partial_fraction_excess": _f(
                    nanmedian(partial_p[:, j] - partial_t[:, j])),
                "dtb_variance_ratio": _f(nanmedian(variance_ratio[:, j])),
                "dtb_skewness_delta": _f(nanmedian(skew_delta[:, j])),
                "power_bias_significance": _f(nanmedian(significance[:, j])),
                "delta2_ratio": [_f(nanmedian(d2_ratio[:, j, t]))
                                 for t in range(len(target_index))],
                "delta2_k_targets": [float(k[t]) for t in target_index],
            })
        active = stages[-1]
        return {
            "stage_labels": labels,
            "stages": stages,
            "active": {
                "w1_xhi": active["w1_xhi"],
                "w1_dtb_mk": active["w1_dtb_mk"],
                "partial_fraction_excess": active["partial_fraction_excess"],
                "dtb_variance_ratio": active["dtb_variance_ratio"],
                "power_bias_significance": active["power_bias_significance"],
            },
            "skipped_samples": self.skipped,
            "n_samples": len(self.rows),
        }

    def arrays(self):
        if not self.rows:
            return {}
        out = {"observable_xhi_edges": self.xhi_edges,
               "observable_dtb_edges": self.dtb_edges}
        labels = stage_labels(self.settings.stage_edges, self.settings.active_range)
        for j, label in enumerate(labels):
            token = label.replace(" ", "_").replace("-", "_").replace(".", "p")
            for key in ("xhi_hist_truth", "xhi_hist_pred",
                        "dtb_hist_truth", "dtb_hist_pred"):
                rows = [r["stages"][j][key] for r in self.rows
                        if r["stages"][j] is not None]
                if rows:
                    total = np.sum(rows, axis=0)
                    mass = total.sum()
                    out[f"observable_{key}_{token}"] = (
                        total / mass if mass > 0 else total)
        return out


def _f(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None
