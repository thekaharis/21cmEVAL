"""Ionization-front survey of the truth (fno-21cm ``legacy/xhi2d/survey_truth_sharpness.py``).

Not a model score: how much of the field is partially ionized and how sharp the fronts are,
accumulated against the slice's own mean x_HI (50 bins), over every LOS plane of every sample:

* partially ionized fraction at three thresholds (1e-3, 0.01, 0.1 from either phase);
* transverse transition width in pixels, band pixels (0.1 < x_HI < 0.9) per unit of transverse
  total variation;
* transverse and line-of-sight total variation per voxel (periodic forward differences
  transversally, interior differences along the LOS) and the 99.9th percentile of |grad_xy|;
* a 200-bin histogram of voxel values.

The same statistics of the prediction are kept next to them (``survey_pred_*``), binned by the
prediction's own slice mean. ``eval21 plot survey`` draws the truth like ``viz/report_truth_survey.py``.
The original reads the raw 2340-step lightcones; here the statistics are those of the evaluated
grid (one LOS slab per sample, so only the last plane has no LOS difference).
"""
from __future__ import annotations

import numpy as np

from ..registry import register
from .base import Metric

N_XHI_BINS = 50
N_VALUE_BINS = 200
PARTIAL_BANDS = ((1e-3, 1 - 1e-3), (0.01, 0.99), (0.1, 0.9))


def _accumulators():
    return {"n_planes": np.zeros(N_XHI_BINS), "sum_partial": np.zeros((len(PARTIAL_BANDS), N_XHI_BINS)),
            "sum_tv_xy": np.zeros(N_XHI_BINS), "sum_tv_z": np.zeros(N_XHI_BINS),
            "sum_band": np.zeros(N_XHI_BINS), "sum_width": np.zeros(N_XHI_BINS),
            "sum_peak": np.zeros(N_XHI_BINS), "value_hist": np.zeros(N_VALUE_BINS),
            "n_cones": np.zeros(1), "n_planes_total": np.zeros(1)}


def accumulate(block: np.ndarray, acc: dict) -> None:
    """Per-plane statistics of one (nx, ny, nz) block, as the original's ``_accumulate``."""
    block = np.asarray(block, dtype=np.float32)
    npl = block.shape[2]
    area = block.shape[0] * block.shape[1]
    mean = block.mean(axis=(0, 1))
    idx = np.clip((mean * N_XHI_BINS).astype(np.int64), 0, N_XHI_BINS - 1)
    gx = np.roll(block, -1, axis=0) - block
    gy = np.roll(block, -1, axis=1) - block
    gmag = np.sqrt(gx * gx + gy * gy)
    tv_xy = gmag.sum(axis=(0, 1))
    peak = np.percentile(gmag.reshape(-1, npl), 99.9, axis=0)
    gz = np.abs(np.diff(block, axis=2))
    tv_z = np.zeros(npl)
    tv_z[:-1] = gz.sum(axis=(0, 1))
    band = ((block > 0.1) & (block < 0.9)).sum(axis=(0, 1)).astype(np.float64)
    width = band / np.maximum(tv_xy, 1e-9)
    np.add.at(acc["n_planes"], idx, 1)
    np.add.at(acc["sum_tv_xy"], idx, tv_xy / area)
    np.add.at(acc["sum_tv_z"], idx, tv_z / area)
    np.add.at(acc["sum_band"], idx, band / area)
    np.add.at(acc["sum_width"], idx, width)
    np.add.at(acc["sum_peak"], idx, peak)
    for b, (lo, hi) in enumerate(PARTIAL_BANDS):
        np.add.at(acc["sum_partial"][b], idx, ((block > lo) & (block < hi)).sum(axis=(0, 1)) / area)
    acc["value_hist"] += np.histogram(block, bins=N_VALUE_BINS, range=(0.0, 1.0))[0]
    acc["n_planes_total"] += npl
    acc["n_cones"] += 1


@register("survey")
class SurveyMetrics(Metric):
    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        self.acc = {"truth": _accumulators(), "pred": _accumulators()}

    def update(self, sample, prediction, aux=None, runtime=None):
        accumulate(self.target(sample), self.acc["truth"])
        accumulate(self.prediction(prediction), self.acc["pred"])

    def result(self):
        out = {}
        for which, a in self.acc.items():
            total = a["value_hist"].sum()
            if total == 0:
                return {}
            out[which] = {"frac_ionized": float(a["value_hist"][0] / total),
                          "frac_neutral": float(a["value_hist"][-1] / total),
                          "frac_partial": float(a["value_hist"][1:-1].sum() / total),
                          "n_planes": int(a["n_planes_total"][0])}
        out["n_samples"] = int(self.acc["truth"]["n_cones"][0])
        return out

    def arrays(self):
        if self.acc["truth"]["n_cones"][0] == 0:
            return {}
        out = {"survey_partial_bands": np.array(PARTIAL_BANDS)}
        for which, a in self.acc.items():
            for key, value in a.items():
                out[f"survey_{which}_{key}"] = value
        return out
