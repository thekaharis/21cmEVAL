"""Error versus 3-D distance to the ionization front (fno-21cm ``boundary_band_diagnostic --mode 3d``).

The approximate cross-check of the per-slice ``front`` metric: the signed distance to the true
x_HI = ``threshold`` surface is a 3-D Euclidean distance transform with sampling
``(dx, dy, los_cell_mpc)``, and the error gradient includes the LOS derivative. A single mean LOS
cell size is used, so the distances are approximate along the lightcone axis (the original's
caveat); ``los_cell_mpc`` defaults to the original's 13 Mpc and is set in a TOML config
(``[metric_options.front3d] los_cell_mpc = ...``).

Exports the same pooled profile and per-sample band arrays as ``front`` with the prefix
``front3d_``, so ``eval21 plot edge`` draws ``edge_3d/`` like ``edge_slice/``.
"""
from __future__ import annotations

import numpy as np

from ..physics import front_width
from ..registry import register
from .base import Metric


@register("front3d")
class Front3dMetrics(Metric):
    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        self.los_cell_mpc = float(self.options.get("los_cell_mpc", 13.0))
        n = int(round(2 * settings.front_d_max_mpc / settings.front_bin_mpc))
        self.edges = np.linspace(-settings.front_d_max_mpc, settings.front_d_max_mpc, n + 1)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.n = np.zeros(n)
        self.sum_sq = np.zeros(n)
        self.sum_grad = np.zeros(n)
        self.sum_truth = np.zeros(n)
        self.sum_pred = np.zeros(n)
        self.per_sample: list[dict] = []

    def update(self, sample, prediction, aux=None, runtime=None):
        from scipy.ndimage import distance_transform_edt

        truth = self.target(sample)
        pred = self.prediction(prediction)
        box = self.box_mpc(sample)
        sampling = (box / truth.shape[0], box / truth.shape[1], self.los_cell_mpc)
        mask = truth > self.settings.threshold
        record = {"sample": str(sample.id), "n_front_vox": 0}
        if mask.all() or (~mask).all():
            self.per_sample.append(record)
            return
        d = (distance_transform_edt(mask, sampling=sampling)
             - distance_transform_edt(~mask, sampling=sampling)).ravel()
        error = pred - truth
        gx, gy, gz = np.gradient(error, *sampling)
        grad = (gx * gx + gy * gy + gz * gz).ravel()
        del gx, gy, gz
        sq = (error ** 2).ravel()
        index = np.digitize(d, self.edges) - 1
        inb = (index >= 0) & (index < len(self.centers))
        i = index[inb]
        nb = len(self.centers)
        self.n += np.bincount(i, minlength=nb)
        self.sum_sq += np.bincount(i, weights=sq[inb], minlength=nb)
        self.sum_grad += np.bincount(i, weights=grad[inb], minlength=nb)
        self.sum_truth += np.bincount(i, weights=truth.ravel()[inb], minlength=nb)
        self.sum_pred += np.bincount(i, weights=pred.ravel()[inb], minlength=nb)
        record["n_front_vox"] = int(d.size)
        total = float(sq.sum())
        for band in self.settings.front_bands_mpc:
            m = np.abs(d) <= band
            record[f"L2_band{band:g}"] = float(np.sqrt(sq[m].mean())) if m.any() else np.nan
            record[f"H1_band{band:g}"] = float(np.sqrt(grad[m].mean())) if m.any() else np.nan
            record[f"errfrac_band{band:g}"] = float(sq[m].sum() / total) if total > 0 else np.nan
        self.per_sample.append(record)

    def _profile(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            return {"truth": np.where(self.n > 0, self.sum_truth / self.n, np.nan),
                    "pred": np.where(self.n > 0, self.sum_pred / self.n, np.nan),
                    "rmse": np.where(self.n > 0, np.sqrt(self.sum_sq / self.n), np.nan),
                    "grad": np.where(self.n > 0, np.sqrt(self.sum_grad / self.n), np.nan)}

    def result(self):
        if not self.per_sample:
            return {}
        p = self._profile()
        bands = {}
        for band in self.settings.front_bands_mpc:
            for name in ("L2", "H1", "errfrac"):
                values = np.array([r.get(f"{name}_band{band:g}", np.nan) for r in self.per_sample])
                bands[f"{name}_band{band:g}_mean"] = _f(np.nanmean(values)) if np.isfinite(values).any() else None
        return {
            "los_cell_mpc": self.los_cell_mpc,
            "front_width_truth_mpc": _f(front_width(self.centers, p["truth"])),
            "front_width_pred_mpc": _f(front_width(self.centers, p["pred"])),
            "bands": bands,
            "n_samples_with_fronts": int(sum(r["n_front_vox"] > 0 for r in self.per_sample)),
            "n_samples": len(self.per_sample),
        }

    def arrays(self):
        if not self.per_sample:
            return {}
        p = self._profile()
        out = {"front3d_d_mpc": self.centers, "front3d_mean_truth": p["truth"], "front3d_mean_pred": p["pred"],
               "front3d_rmse": p["rmse"], "front3d_gradient_rms": p["grad"], "front3d_counts": self.n,
               "front3d_sample_ids": np.array([r["sample"] for r in self.per_sample]),
               "front3d_n_front_vox": np.array([r["n_front_vox"] for r in self.per_sample], dtype=np.int64)}
        for band in self.settings.front_bands_mpc:
            for name in ("L2", "H1", "errfrac"):
                out[f"front3d_band_{name}_{band:g}"] = np.array(
                    [r.get(f"{name}_band{band:g}", np.nan) for r in self.per_sample], dtype=np.float64)
        return out


def _f(value):
    value = float(value)
    return value if np.isfinite(value) else None
