"""Add a metric without touching the package.

    python examples/custom_metric.py --data demo/data --out results/custom

The wrapper imports the new metric (registration side effect) and then calls
the normal CLI. A metric only has to accumulate and return JSON-able values.
"""
from __future__ import annotations

import numpy as np

from cm21eval.metrics.base import Metric, nanmedian
from cm21eval.registry import register


@register("ionized_fraction")
class IonizedFraction(Metric):
    """Mean ionized fraction per stage, compared as an absolute difference."""

    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        self.rows = []

    def update(self, sample, prediction, aux=None, runtime=None):
        truth = self.target(sample)
        pred = self.prediction(prediction)
        masks = self.stages(truth)[0]
        row = []
        for mask in masks:
            if not mask.any():
                row.append((np.nan, np.nan))
                continue
            row.append((float((1 - truth[:, :, mask]).mean()),
                        float((1 - pred[:, :, mask]).mean())))
        self.rows.append(row)

    def result(self):
        if not self.rows:
            return {}
        truth = np.array([[r[0] for r in row] for row in self.rows])
        pred = np.array([[r[1] for r in row] for row in self.rows])
        labels = ["xbar_HI %.2f-%.2f" % pair for pair in
                  zip(self.settings.stage_edges[:-1], self.settings.stage_edges[1:])]
        labels.append("active")
        return {
            "stages": [
                {"label": labels[j],
                 "truth": nanmedian(truth[:, j]),
                 "pred": nanmedian(pred[:, j]),
                 "abs_error": nanmedian(np.abs(pred[:, j] - truth[:, j]))}
                for j in range(truth.shape[1])
            ]
        }


if __name__ == "__main__":
    from cm21eval.cli import main

    raise SystemExit(main())
