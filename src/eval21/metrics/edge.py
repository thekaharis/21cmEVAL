"""Where in the lightcone the error lives.

Whole-volume RMSE is dominated by trivially correct saturated regions: fully
neutral at high redshift, fully ionized at low redshift. Three cuts separate
the informative part:

* transverse **edge** vs **interior** RMSE -- periodic FFT models can leak
  error in from the box boundary;
* **active-window** RMSE -- only LOS slices whose transverse standard
  deviation exceeds ``active_variance_fraction`` of its maximum, i.e. slices
  that contain an ionization front;
* **global-history** RMSE -- error of the mean x_HI(z) curve, which is what a
  reionization observer measures first.
"""
from __future__ import annotations

import numpy as np

from ..registry import register
from .base import Metric


@register("edge")
class EdgeMetrics(Metric):
    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        self.ss_edge = 0.0
        self.n_edge = 0
        self.ss_interior = 0.0
        self.n_interior = 0
        self.ss_active = 0.0
        self.n_active = 0
        self.ss_history = 0.0
        self.n_history = 0
        self.active_range_rows: list[tuple[float, float]] = []

    def update(self, sample, prediction, aux=None, runtime=None):
        truth = self.target(sample)
        pred = self.prediction(prediction)
        error = pred - truth
        nx, ny = truth.shape[:2]

        width = max(1, int(min(nx, ny) * self.settings.edge_width_fraction))
        edge = np.zeros((nx, ny), dtype=bool)
        edge[:width], edge[-width:] = True, True
        edge[:, :width], edge[:, -width:] = True, True

        self.ss_edge += float(np.sum(error[edge, :] ** 2))
        self.n_edge += int(edge.sum()) * truth.shape[2]
        self.ss_interior += float(np.sum(error[~edge, :] ** 2))
        self.n_interior += int((~edge).sum()) * truth.shape[2]

        transverse_std = truth.std(axis=(0, 1))
        maximum = float(transverse_std.max())
        if maximum > 1e-8:
            active = transverse_std > self.settings.active_variance_fraction * maximum
        else:
            active = np.ones_like(transverse_std, dtype=bool)
        if active.any():
            self.ss_active += float(np.sum(error[..., active] ** 2))
            self.n_active += int(error[..., active].size)
            z = self.z_grid(sample)
            if z is not None and len(z) == truth.shape[2]:
                self.active_range_rows.append(
                    (float(z[active].min()), float(z[active].max()))
                )

        history_error = pred.mean(axis=(0, 1)) - truth.mean(axis=(0, 1))
        self.ss_history += float(history_error @ history_error)
        self.n_history += history_error.size

    def result(self):
        def ratio(sums, counts):
            return float(np.sqrt(sums / counts)) if counts else None

        if self.active_range_rows:
            rows = np.asarray(self.active_range_rows)
            active_z = (float(np.median(rows[:, 0])), float(np.median(rows[:, 1])))
            active_z_band = (float(np.percentile(rows[:, 0], 16)),
                             float(np.percentile(rows[:, 1], 84)))
        else:
            active_z = (None, None)
            active_z_band = (None, None)
        return {
            "transverse_edge_rmse": ratio(self.ss_edge, self.n_edge),
            "transverse_interior_rmse": ratio(self.ss_interior, self.n_interior),
            "active_window_rmse": ratio(self.ss_active, self.n_active),
            "active_window_fraction": (self.n_active / (self.n_edge + self.n_interior))
            if (self.n_edge + self.n_interior) else None,
            "active_z_min": active_z[0],
            "active_z_max": active_z[1],
            "active_z_min_p16": active_z_band[0],
            "active_z_max_p84": active_z_band[1],
            "global_history_rmse": ratio(self.ss_history, self.n_history),
        }
