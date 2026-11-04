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
* **Delta^2_21(k)** in mK^2 from the dT_b field, and its ratio.
* **power-bias significance**: ``(P_pred - P_truth) / (P_truth sqrt(2/N_modes))``,
  i.e. the emulator bias in units of the Gaussian sample variance of a single
  slice. ``1/significance^2`` is the number of independent slices an observer
  must average before the bias equals their statistical error.
* **global-history bias** ``<x_HI,pred> - <x_HI,truth>`` per redshift.

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
        self.xbar_edges = np.linspace(0.0, 1.0, 21)

    def _bins(self, nx, ny, box):
        key = (nx, ny, round(box, 9), self.settings.n_k_bins)
        if self._binner is None or self._binner[0] != key:
            self._binner = (key, KBinner(nx, ny, box, self.settings.n_k_bins))
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

        xbar_truth = truth.mean(axis=(0, 1))
        xbar_pred = pred.mean(axis=(0, 1))
        masks = stage_masks(
            xbar_truth, self.settings.stage_edges, self.settings.active_range)
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
        target_index = [int(np.argmin(np.abs(np.log(k) - np.log(t))))
                        for t in self.settings.k_targets]

        stages = []
        # fno-21cm physical_evaluation semantics, for ``eval21 plot physical``:
        # histograms over every slice of a stage (thin stages included, as the
        # pooled PDFs there), significance averaged (not median) over slices.
        suite = _suite_stage_arrays(masks, truth, pred, dtb_truth, dtb_pred, self.xhi_edges,
                                    self.dtb_edges, power_t, power_p, power_dt, power_dp,
                                    counts, self.settings.min_stage_slices,
                                    self.settings.partial_range)
        for mask in masks:
            n = int(mask.sum())
            if n < self.settings.min_stage_slices:
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
                sig_xhi = _significance(power_t[mask], power_p[mask], counts)
                sig_dtb = _significance(power_dt[mask], power_dp[mask], counts)
                denominator = n * counts
                d2t = scale * power_dt[mask].sum(axis=0) / denominator
                d2p = scale * power_dp[mask].sum(axis=0) / denominator
                d2_ratio = np.where(d2t > 0, d2p / d2t, np.nan)
                xhi_ratio = np.where(
                    power_t[mask].sum(axis=0) > 0,
                    power_p[mask].sum(axis=0) / power_t[mask].sum(axis=0),
                    np.nan)
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
                "d2_truth": d2t,
                "d2_pred": d2p,
                "d2_ratio": d2_ratio,
                "xhi_ratio": xhi_ratio,
                "sig_xhi": sig_xhi,
                "sig_dtb": sig_dtb,
            })
        self.rows.append({
            "sample": sample.id,
            "suite": suite,
            "stages": stages,
            "target_index": target_index,
            "xbar_truth": xbar_truth,
            "xbar_pred": xbar_pred,
            "xbar_var_truth": _xbar_mean(moments_truth["variance"], xbar_truth,
                                         self.xbar_edges),
            "xbar_var_pred": _xbar_mean(moments_pred["variance"], xbar_truth,
                                        self.xbar_edges),
            "xbar_skew_truth": _xbar_mean(moments_truth["skewness"], xbar_truth,
                                          self.xbar_edges),
            "xbar_skew_pred": _xbar_mean(moments_pred["skewness"], xbar_truth,
                                         self.xbar_edges),
            "xbar_d2_truth": _xbar_mean_multi(
                _per_slice_d2(power_dt, counts, scale, target_index),
                xbar_truth, self.xbar_edges),
            "xbar_d2_pred": _xbar_mean_multi(
                _per_slice_d2(power_dp, counts, scale, target_index),
                xbar_truth, self.xbar_edges),
        })

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
        significance = _gather_reduced(self.rows, "sig_dtb", n_rows)
        xhi_significance = _gather_reduced(self.rows, "sig_xhi", n_rows)
        k = self._binner[1].centers
        targets = self.rows[0]["target_index"]
        d2_ratio = np.full((len(self.rows), n_rows, len(targets)), np.nan)
        xhi_ratio = np.full_like(d2_ratio, np.nan)
        for i, row in enumerate(self.rows):
            for j, stage in enumerate(row["stages"]):
                if stage is None:
                    continue
                d2_ratio[i, j] = stage["d2_ratio"][targets]
                xhi_ratio[i, j] = stage["xhi_ratio"][targets]

        stages = []
        for j, label in enumerate(labels):
            stages.append({
                "label": label,
                "n_slices": int(np.nansum([
                    r["stages"][j]["n_slices"] for r in self.rows
                    if r["stages"][j] is not None])),
                "w1_xhi": _f(nanmedian(w1_xhi[:, j])),
                "w1_dtb_mk": _f(nanmedian(w1_dtb[:, j])),
                "partial_fraction_truth": _f(nanmedian(partial_t[:, j])),
                "partial_fraction_pred": _f(nanmedian(partial_p[:, j])),
                "partial_fraction_excess": _f(
                    nanmedian(partial_p[:, j] - partial_t[:, j])),
                "dtb_variance_ratio": _f(nanmedian(variance_ratio[:, j])),
                "dtb_skewness_delta": _f(nanmedian(skew_delta[:, j])),
                "power_bias_significance_dtb": _f(
                    nanmedian(np.abs(significance[:, j]))),
                "power_bias_significance_xhi": _f(
                    nanmedian(np.abs(xhi_significance[:, j]))),
                "delta2_dtb_ratio": [_f(nanmedian(d2_ratio[:, j, t]))
                                     for t in range(len(targets))],
                "delta2_xhi_ratio": [_f(nanmedian(xhi_ratio[:, j, t]))
                                     for t in range(len(targets))],
                "k_targets": [float(k[t]) for t in targets],
            })
        active = stages[-1]
        dxbar = np.array([row["xbar_pred"] - row["xbar_truth"]
                          for row in self.rows])
        dxbar_med, dxbar_lo, dxbar_hi = _band_curve(dxbar)
        return {
            "stage_labels": labels,
            "stages": stages,
            "dxbar": {
                "med": _list(dxbar_med),
                "p16": _list(dxbar_lo),
                "p84": _list(dxbar_hi),
                "abs_max": _f(np.nanmax(np.abs(dxbar))) if dxbar.size else None,
                "n_z": int(dxbar.shape[1]),
            },
            "active": {
                "w1_xhi": active["w1_xhi"],
                "w1_dtb_mk": active["w1_dtb_mk"],
                "partial_fraction_excess": active["partial_fraction_excess"],
                "dtb_variance_ratio": active["dtb_variance_ratio"],
                "power_bias_significance_dtb":
                    active["power_bias_significance_dtb"],
                "power_bias_significance_xhi":
                    active["power_bias_significance_xhi"],
            },
            "skipped_samples": self.skipped,
            "n_samples": len(self.rows),
        }

    def arrays(self):
        if not self.rows:
            return {}
        out = {"observable_xhi_edges": self.xhi_edges,
               "observable_dtb_edges": self.dtb_edges,
               "observable_xbar_edges": self.xbar_edges}
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
            for key in ("d2_truth", "d2_pred", "d2_ratio", "xhi_ratio",
                        "sig_xhi", "sig_dtb"):
                rows = [r["stages"][j][key] for r in self.rows
                        if r["stages"][j] is not None]
                if rows:
                    out[f"observable_{key}_{token}"] = np.stack(rows)
        for key in ("xbar_var", "xbar_skew", "xbar_d2"):
            for which in ("truth", "pred"):
                rows = [r[f"{key}_{which}"] for r in self.rows
                        if r[f"{key}_{which}"] is not None]
                if rows:
                    out[f"observable_{key}_{which}"] = np.stack(rows)
        dxbar = np.array([row["xbar_pred"] - row["xbar_truth"]
                          for row in self.rows])
        dxbar_med, dxbar_lo, dxbar_hi = _band_curve(dxbar)
        out["observable_dxbar_med"] = dxbar_med
        out["observable_dxbar_p16"] = dxbar_lo
        out["observable_dxbar_p84"] = dxbar_hi
        out["observable_xbar_truth"] = np.stack(
            [r["xbar_truth"] for r in self.rows])
        out["observable_xbar_pred"] = np.stack(
            [r["xbar_pred"] for r in self.rows])
        out.update(_suite_arrays(self.rows, labels, self._binner[1].centers, self.settings))
        return out


def _suite_stage_arrays(masks, truth, pred, dtb_truth, dtb_pred, xhi_edges, dtb_edges,
                        power_t, power_p, power_dt, power_dp, counts, min_slices,
                        partial_range=(0.1, 0.9)):
    """Per-stage arrays of one sample with fno-21cm ``physical_evaluation`` semantics."""
    n_rows, nk = len(masks), len(counts)
    out = {"hist": {}, "w1_xhi": np.full(n_rows, np.nan), "w1_dtb": np.full(n_rows, np.nan),
           "partial_truth": np.full(n_rows, np.nan), "partial_pred": np.full(n_rows, np.nan),
           "sigmean_xhi": np.full((n_rows, nk), np.nan), "sigmean_dtb": np.full((n_rows, nk), np.nan),
           "d2_truth": np.full((n_rows, nk), np.nan), "d2_pred": np.full((n_rows, nk), np.nan),
           "n_slices": np.array([int(m.sum()) for m in masks])}
    fields = {"xhi_truth": (truth, xhi_edges), "xhi_pred": (pred, xhi_edges),
              "dtb_truth": (dtb_truth, dtb_edges), "dtb_pred": (dtb_pred, dtb_edges)}
    for key, (field, edges) in fields.items():
        out["hist"][key] = np.stack([
            np.histogram(np.clip(field[:, :, m].ravel(), edges[0], edges[-1]), bins=edges)[0]
            for m in masks]).astype(np.float64)
    lo, hi = partial_range
    for j, mask in enumerate(masks):
        n = int(mask.sum())
        if n < min_slices:
            continue
        h = out["hist"]
        out["w1_xhi"][j] = histogram_w1(h["xhi_truth"][j], h["xhi_pred"][j], xhi_edges)
        out["w1_dtb"][j] = histogram_w1(h["dtb_truth"][j], h["dtb_pred"][j], dtb_edges)
        out["partial_truth"][j] = np.mean((truth[:, :, mask] > lo) & (truth[:, :, mask] < hi))
        out["partial_pred"][j] = np.mean((pred[:, :, mask] > lo) & (pred[:, :, mask] < hi))
        out["sigmean_xhi"][j] = _significance_mean(power_t[mask], power_p[mask], counts)
        out["sigmean_dtb"][j] = _significance_mean(power_dt[mask], power_dp[mask], counts)
    return out


def _significance_mean(sum_truth, sum_pred, counts):
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = sum_truth * np.sqrt(2.0 / counts)[None, :]
        values = (sum_pred - sum_truth) / sigma
    values[~(sum_truth > 0) | ~np.isfinite(values)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=0)


def _suite_arrays(rows, labels, k, settings):
    """Stacked per-sample arrays for ``eval21 plot physical`` (fno-21cm layout)."""
    suite = [r["suite"] for r in rows if "suite" in r]
    if not suite:
        return {}
    out = {"observable_k": np.asarray(k),
           "observable_n_slices": np.stack([s["n_slices"] for s in suite])}
    for key in ("xhi_truth", "xhi_pred", "dtb_truth", "dtb_pred"):
        out[f"observable_counts_{key}"] = np.sum([s["hist"][key] for s in suite], axis=0)
    for key in ("w1_xhi", "w1_dtb", "partial_truth", "partial_pred", "sigmean_xhi", "sigmean_dtb"):
        out[f"observable_{key}"] = np.stack([s[key] for s in suite])
    d2t = np.full((len(rows), len(labels), len(k)), np.nan)
    d2p = np.full_like(d2t, np.nan)
    for i, row in enumerate(rows):
        for j, stage in enumerate(row["stages"]):
            if stage is not None:
                d2t[i, j], d2p[i, j] = stage["d2_truth"], stage["d2_pred"]
    out["observable_d2_dtb_truth"] = d2t
    out["observable_d2_dtb_pred"] = d2p
    out["observable_sample_ids"] = np.array([str(r["sample"]) for r in rows])
    return out


def _gather_reduced(rows, key, n_rows):
    """Per-stage scalar: median over wavenumbers of each stage vector."""
    out = np.full((len(rows), n_rows), np.nan)
    for i, row in enumerate(rows):
        for j, stage in enumerate(row["stages"]):
            if stage is not None:
                out[i, j] = nanmedian(stage[key])
    return out


def _significance(sum_truth, sum_pred, counts):
    """Power bias in units of single-slice Gaussian sample variance."""
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = sum_truth * np.sqrt(2.0 / counts)[None, :]
        values = (sum_pred - sum_truth) / sigma
    values[~(sum_truth > 0) | ~np.isfinite(values)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(values, axis=0)


def _per_slice_d2(power, counts, scale, targets):
    """(n_z, n_targets) Delta^2 per slice at the requested k bins."""
    target_counts = np.asarray(counts)[targets][None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        values = (power[:, targets]
                  * np.asarray(scale)[None, targets] / target_counts)
    return np.where(target_counts > 0, values, np.nan)


def _xbar_mean_multi(per_slice, xbar, edges):
    """Bin a (n_z, n_targets) per-slice array by xbar -> (n_bins, n_targets)."""
    index = np.clip(np.digitize(xbar, edges) - 1, 0, len(edges) - 2)
    out = np.full((len(edges) - 1, per_slice.shape[1]), np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for b in range(len(edges) - 1):
            select = index == b
            if select.any():
                out[b] = np.nanmean(per_slice[select], axis=0)
    return out


def _xbar_mean(per_slice, xbar, edges):
    index = np.clip(np.digitize(xbar, edges) - 1, 0, len(edges) - 2)
    out = np.full(len(edges) - 1, np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for b in range(len(edges) - 1):
            select = index == b
            if select.any():
                out[b] = np.nanmean(per_slice[select])
    return out


def _band_curve(values):
    """Median and 16-84 percentiles over samples, keeping the z axis."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return (np.nanmedian(values, axis=0), np.nanpercentile(values, 16, axis=0),
                np.nanpercentile(values, 84, axis=0))


def _list(values):
    return [None if not np.isfinite(value) else float(value) for value in values]


def _f(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None
