"""Topology of the ionized regions.

A three-dimensional connected-component census of ``x_HI < threshold`` --
ionized bubbles embedded in the neutral IGM, the polarity used by
``fno-21cm``. It answers questions the spectra cannot: does the model merge
bubbles too early, and is the largest bubble the right size?

Radii are effective radii of equal volume, in cells; multiply by the cell
size to quote Mpc. IoU is reported both at the fixed threshold and with the
predicted mask restricted to the true ionized cell count, which separates
"wrong shape" from "wrong ionized fraction".
"""
from __future__ import annotations

import numpy as np

from ..physics import ionized_region_summary
from ..registry import register
from .base import Metric, nanmedian


FRACTIONS = (0.1, 0.25, 0.5, 0.75, 0.9)


@register("morphology")
class MorphologyMetrics(Metric):
    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        self.truth: list[dict] = []
        self.pred: list[dict] = []
        self.iou: dict[str, list[float]] = {
            "threshold": [], "matched": [],
            **{f"f{fraction:g}": [] for fraction in FRACTIONS}}

    def update(self, sample, prediction, aux=None, runtime=None):
        truth = self.target(sample)
        pred = self.prediction(prediction)
        threshold = self.settings.threshold
        self.truth.append(ionized_region_summary(truth, threshold))
        self.pred.append(ionized_region_summary(pred, threshold))
        self.iou["threshold"].append(_iou_at_threshold(truth, pred, threshold))
        k = int(np.count_nonzero(truth < threshold))
        self.iou["matched"].append(_iou_top_k(truth, pred, k))
        for fraction in FRACTIONS:
            n = int(round(fraction * truth.size))
            self.iou[f"f{fraction:g}"].append(_iou_top_k(truth, pred, n))

    def result(self):
        if not self.truth:
            return {}

        def summary(rows):
            keys = ("count", "mean_effective_radius_cells",
                    "volume_weighted_radius_cells", "largest_fraction")
            with np.errstate(all="ignore"):
                return {key: float(nanmedian([row[key] for row in rows]))
                        for key in keys}

        truth, pred = summary(self.truth), summary(self.pred)
        return {
            "truth": truth,
            "pred": pred,
            "count_ratio": _ratio(pred["count"], truth["count"]),
            "mean_radius_ratio": _ratio(
                pred["mean_effective_radius_cells"],
                truth["mean_effective_radius_cells"]),
            "largest_fraction_ratio": _ratio(
                pred["largest_fraction"], truth["largest_fraction"]),
            "iou": {
                key: _f(nanmedian(values))
                for key, values in self.iou.items() if values
            },
        }

    def arrays(self):
        if not self.truth:
            return {}
        keys = ("count", "mean_effective_radius_cells",
                "volume_weighted_radius_cells", "largest_fraction")
        return {
            **{f"morphology_iou_{key}": np.asarray(values)
               for key, values in self.iou.items() if values},
            **{f"morphology_truth_{k}": np.array([r[k] for r in self.truth])
               for k in keys},
            **{f"morphology_pred_{k}": np.array([r[k] for r in self.pred])
               for k in keys},
        }


def _iou_at_threshold(truth, pred, threshold: float) -> float:
    """IoU of the two ionized masks at the same fixed threshold."""
    return _iou(truth < threshold, pred < threshold)


def _iou_top_k(truth, pred, k: int) -> float:
    """IoU with both masks restricted to their k most ionized cells.

    Ties (x_HI exactly 0 or 1) are broken by position, as in the reference
    ``tools_excursion_set_diagnostic.py``.
    """
    if k <= 0:
        return 1.0
    k = min(int(k), truth.size)
    return _iou(_top_fraction(truth, k), _top_fraction(pred, k))


def _top_fraction(field, k: int) -> np.ndarray:
    """Mask of the ``k`` cells with the lowest x_HI."""
    flat = np.asarray(field, dtype=np.float64).ravel()
    mask = np.zeros(flat.size, dtype=bool)
    mask[np.argpartition(flat, k - 1)[:k]] = True
    return mask.reshape(np.shape(field))


def _iou(a, b) -> float:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.0


def _f(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _ratio(a, b):
    return float(a / b) if b and np.isfinite(a) and np.isfinite(b) else None
