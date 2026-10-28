"""Topology of the neutral regions.

A three-dimensional connected-component census of ``x_HI >= threshold``
(equivalently ``x_HI < threshold`` for ionized bubbles, which are the
complement). It answers questions the spectra cannot: does the model merge
bubbles too early, and is the largest bubble the right size?

Radii are effective radii of equal volume, in cells; multiply by the cell
size to quote Mpc.
"""
from __future__ import annotations

import numpy as np

from ..physics import neutral_region_summary
from ..registry import register
from .base import Metric, nanmedian


FRACTIONS = (0.1, 0.25, 0.5, 0.75, 0.9)


@register("morphology")
class MorphologyMetrics(Metric):
    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        self.truth: list[dict] = []
        self.pred: list[dict] = []
        self.iou: dict[float, list[float]] = {f: [] for f in FRACTIONS}

    def update(self, sample, prediction, aux=None, runtime=None):
        truth = self.target(sample)
        pred = self.prediction(prediction)
        self.truth.append(neutral_region_summary(truth, self.settings.threshold))
        self.pred.append(neutral_region_summary(pred, self.settings.threshold))
        for fraction in FRACTIONS:
            self.iou[fraction].append(_iou_at_fraction(truth, pred, fraction))

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
            "iou_at_matched_fraction": {
                str(fraction): _f(nanmedian(values))
                for fraction, values in self.iou.items() if values
            },
        }

    def arrays(self):
        if not self.truth:
            return {}
        keys = ("count", "mean_effective_radius_cells",
                "volume_weighted_radius_cells", "largest_fraction")
        return {
            **{f"morphology_iou_f{fraction:g}": np.asarray(values)
               for fraction, values in self.iou.items() if values},
            **{f"morphology_truth_{k}": np.array([r[k] for r in self.truth])
               for k in keys},
            **{f"morphology_pred_{k}": np.array([r[k] for r in self.pred])
               for k in keys},
        }


def _iou_at_fraction(truth, pred, fraction: float) -> float:
    """IoU of the ionized masks, each thresholded to the same ionized fraction."""
    t = truth < np.quantile(truth, fraction)
    p = pred < np.quantile(pred, fraction)
    union = np.count_nonzero(t | p)
    return float(np.count_nonzero(t & p) / union) if union else float("nan")


def _f(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _ratio(a, b):
    return float(a / b) if b and np.isfinite(a) and np.isfinite(b) else None
