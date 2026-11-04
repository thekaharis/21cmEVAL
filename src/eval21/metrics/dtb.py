"""The model's own 21-cm brightness temperature, scored like fno-21cm's ``--dtb-from-npz``.

``observable`` recomputes dT_b from x_HI in the saturated limit, so it isolates the x_HI error.
This metric instead takes the T_b field the model predicts (a ``tb`` target, in mK) and compares
it with the simulated T_b, exactly the statistics ``viz/physical_evaluation.py`` draws with
``--dtb-from-npz`` (``physical_dtb``):

* stage-resolved PDFs of dT_b (pooled histogram counts, default -250..100 mK in 1 mK bins, so the
  absorption trough is covered) and their Wasserstein-1 distance per sample;
* per-slice dT_b variance and skewness, binned by the truth's slice-mean x_HI;
* the transverse 21-cm power ``Delta^2_21(k) = k^2 P(k) / 2 pi`` [mK^2] per stage and at fixed k
  versus x_HI;
* the power bias in units of single-slice Gaussian sample variance, averaged over the slices of a
  stage.

Stages are the **true x_HI** slice means, so the sample must carry the true x_HI (an ``xhi`` input
for a T_b pass, as LoReLi's ``manifest_tb.json``); without it the metric is skipped. Use
``--set stage_edges=[0.02,0.2,0.4,0.6,0.8,0.98,1.001]`` for the extra pre-reionization stage of
the original suite. ``eval21 plot physical`` combines this with the x_HI run of the same data.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..physics import KBinner, histogram_w1, slice_moments, stage_labels, stage_masks
from ..registry import register
from .base import Metric, nanmedian
from .cylindrical import true_xhi


@register("dtb")
class DtbMetrics(Metric):
    """Score the model's own dT_b field: PDFs, 21-cm power, power bias."""
    needs = ()

    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        lo, hi = self.options.get("dtb_range_mk", (-250.0, 100.0))
        bins = int(self.options.get("dtb_bins", int(round(hi - lo))))
        self.edges = np.linspace(float(lo), float(hi), bins + 1)
        self.xbar_edges = np.linspace(0.0, 1.0, 21)
        self.rows: list[dict] = []
        self._binner = None
        self.skipped = 0

    def _bins(self, nx, ny, box):
        key = (nx, ny, round(box, 9), self.settings.n_k_bins)
        if self._binner is None or self._binner[0] != key:
            self._binner = (key, KBinner(nx, ny, box, self.settings.n_k_bins))
        return self._binner[1]

    def update(self, sample, prediction, aux=None, runtime=None):
        xhi = true_xhi(sample, self.settings.target, self.settings.stage_field)
        if xhi is None or self.settings.target == "xhi":     # needs a T_b target staged by x_HI
            self.skipped += 1
            return
        truth = self.target(sample)
        pred = self.prediction(prediction)
        binner = self._bins(*truth.shape[:2], self.box_mpc(sample))
        xbar = xhi.mean(axis=(0, 1))
        masks = stage_masks(xbar, self.settings.stage_edges, self.settings.active_range)
        k = binner.centers
        counts = binner.counts
        scale = k ** 2 / (2 * np.pi)
        power_t = binner.slice_sums(truth - truth.mean(axis=(0, 1), keepdims=True))
        power_p = binner.slice_sums(pred - pred.mean(axis=(0, 1), keepdims=True))
        n_rows, nk = len(masks), k.size
        rec = {"sample": str(sample.id),
               "n_slices": np.array([int(m.sum()) for m in masks]),
               "w1": np.full(n_rows, np.nan),
               "sigmean": np.full((n_rows, nk), np.nan),
               "d2_truth": np.full((n_rows, nk), np.nan),
               "d2_pred": np.full((n_rows, nk), np.nan)}
        rec["hist_truth"] = np.stack([np.histogram(np.clip(truth[:, :, m].ravel(), self.edges[0], self.edges[-1]),
                                                   bins=self.edges)[0] for m in masks]).astype(np.float64)
        rec["hist_pred"] = np.stack([np.histogram(np.clip(pred[:, :, m].ravel(), self.edges[0], self.edges[-1]),
                                                  bins=self.edges)[0] for m in masks]).astype(np.float64)
        for j, mask in enumerate(masks):
            n = int(mask.sum())
            if n < self.settings.min_stage_slices:
                continue
            rec["w1"][j] = histogram_w1(rec["hist_truth"][j], rec["hist_pred"][j], self.edges)
            with np.errstate(invalid="ignore", divide="ignore"):
                sigma = power_t[mask] * np.sqrt(2.0 / counts)[None, :]
                values = (power_p[mask] - power_t[mask]) / sigma
                values[~(power_t[mask] > 0) | ~np.isfinite(values)] = np.nan
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    rec["sigmean"][j] = np.nanmean(values, axis=0)
                rec["d2_truth"][j] = scale * np.where(counts > 0, power_t[mask].sum(0) / counts, np.nan) / n
                rec["d2_pred"][j] = scale * np.where(counts > 0, power_p[mask].sum(0) / counts, np.nan) / n
        targets = [int(np.argmin(np.abs(np.log(k) - np.log(t)))) for t in self.settings.k_targets]
        mom_t, mom_p = slice_moments(truth), slice_moments(pred)
        with np.errstate(invalid="ignore", divide="ignore"):
            d2_t = scale[targets] * np.where(counts[targets] > 0, power_t[:, targets] / counts[targets], np.nan)
            d2_p = scale[targets] * np.where(counts[targets] > 0, power_p[:, targets] / counts[targets], np.nan)
        for which, mom, d2 in (("truth", mom_t, d2_t), ("pred", mom_p, d2_p)):
            rec[f"xbar_var_{which}"] = _xbar_mean(mom["variance"][:, None], xbar, self.xbar_edges)[:, 0]
            rec[f"xbar_skew_{which}"] = _xbar_mean(mom["skewness"][:, None], xbar, self.xbar_edges)[:, 0]
            rec[f"xbar_d2_{which}"] = _xbar_mean(d2, xbar, self.xbar_edges)
        rec["targets"] = targets
        self.rows.append(rec)

    def result(self):
        if not self.rows:
            return {"note": "needs a T_b target and the true x_HI (input 'xhi') to stage it"}
        labels = stage_labels(self.settings.stage_edges, self.settings.active_range)
        k = self._binner[1].centers
        targets = self.rows[0]["targets"]
        w1 = np.stack([r["w1"] for r in self.rows])
        d2t = np.stack([r["d2_truth"] for r in self.rows])
        d2p = np.stack([r["d2_pred"] for r in self.rows])
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(d2t > 0, d2p / d2t, np.nan)
        sig = np.stack([r["sigmean"] for r in self.rows])
        stages = []
        for j, label in enumerate(labels):
            stages.append({
                "label": label,
                "n_slices": int(sum(r["n_slices"][j] for r in self.rows)),
                "w1_dtb_mk": _f(nanmedian(w1[:, j])),
                "delta2_dtb_ratio": [_f(nanmedian(ratio[:, j, t])) for t in targets],
                "power_bias_significance_dtb": [_f(nanmedian(sig[:, j, t])) for t in targets],
                "k_targets": [float(k[t]) for t in targets],
            })
        return {"stage_labels": labels, "stages": stages, "active": stages[-1],
                "dtb_range_mk": [float(self.edges[0]), float(self.edges[-1])],
                "skipped_samples": self.skipped, "n_samples": len(self.rows)}

    def arrays(self):
        if not self.rows:
            return {}
        out = {"dtb_edges": self.edges, "dtb_xbar_edges": self.xbar_edges,
               "dtb_k": self._binner[1].centers,
               "dtb_k_targets": np.asarray(self.rows[0]["targets"]),
               "dtb_counts_truth": np.sum([r["hist_truth"] for r in self.rows], axis=0),
               "dtb_counts_pred": np.sum([r["hist_pred"] for r in self.rows], axis=0),
               "dtb_sample_ids": np.array([r["sample"] for r in self.rows])}
        for key in ("n_slices", "w1", "sigmean", "d2_truth", "d2_pred", "xbar_var_truth", "xbar_var_pred",
                    "xbar_skew_truth", "xbar_skew_pred", "xbar_d2_truth", "xbar_d2_pred"):
            out[f"dtb_{key}"] = np.stack([r[key] for r in self.rows])
        return out


def _xbar_mean(per_slice, xbar, edges):
    index = np.clip(np.digitize(xbar, edges) - 1, 0, len(edges) - 2)
    out = np.full((len(edges) - 1, per_slice.shape[1]), np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for b in range(len(edges) - 1):
            select = index == b
            if select.any():
                out[b] = np.nanmean(per_slice[select], axis=0)
    return out


def _f(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None
