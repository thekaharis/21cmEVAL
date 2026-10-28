"""Reionization history in the units observers use.

Global x_HI(z) is reduced to four numbers with a physical reference:

* ``tau_e`` -- CMB Thomson optical depth, compared to Planck 2018
  ``tau = 0.0544 +- 0.0073``;
* ``z_xhi25 / 50 / 75`` -- the redshifts where the mean neutral fraction
  crosses 0.75, 0.5 and 0.25, compared to the Planck instantaneous-reionization
  midpoint uncertainty ``sigma_z = 0.73``;
* ``delta_z = z(0.75) - z(0.25)`` -- the duration of reionization.

Errors are quoted in units of the observational uncertainty, so a value of
1.0 means "as wrong as the Planck error bar". A model can win on voxel RMSE
and still shift the history; that is the point of this metric.
"""
from __future__ import annotations

import numpy as np

from ..physics import (PLANCK_TAU, PLANCK_TAU_SIGMA, PLANCK_ZRE_SIGMA,
                       history_features, thomson_optical_depth)
from ..registry import register
from .base import Metric, nanmedian


@register("history")
class HistoryMetrics(Metric):
    needs = ()

    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        self.rows: list[dict] = []

    def update(self, sample, prediction, aux=None, runtime=None):
        truth = self.target(sample)
        pred = self.prediction(prediction)
        z = self.z_grid(sample)
        if z is None:
            return
        if len(z) != truth.shape[2]:
            raise ValueError(
                f"sample {sample.id!r}: z_grid length {len(z)} does not match "
                f"the LOS dimension {truth.shape[2]}"
            )
        omega = self.omega_m(sample)
        cosmology = dict(omega_m=omega, omega_b=self.settings.omega_b,
                         h=self.settings.hubble_h)
        xbar_t = truth.mean(axis=(0, 1))
        xbar_p = pred.mean(axis=(0, 1))
        self.rows.append({
            "z": z,
            "tau_truth": thomson_optical_depth(z, xbar_t, **cosmology),
            "tau_pred": thomson_optical_depth(z, xbar_p, **cosmology),
            "history_truth": history_features(z, xbar_t),
            "history_pred": history_features(z, xbar_p),
            "history_rmse": float(np.sqrt(np.mean((xbar_p - xbar_t) ** 2))),
        })

    def result(self):
        if not self.rows:
            return {}
        tau_t = np.array([r["tau_truth"] for r in self.rows])
        tau_p = np.array([r["tau_pred"] for r in self.rows])
        tau = {
            "truth_med": _f(nanmedian(tau_t)),
            "pred_med": _f(nanmedian(tau_p)),
            "error_med": _f(nanmedian(tau_p - tau_t)),
            "error_sigma_med": _f(nanmedian(np.abs(tau_p - tau_t) / PLANCK_TAU_SIGMA)),
            "tau_truth_pull_from_planck": _f(
                nanmedian((tau_t - PLANCK_TAU) / PLANCK_TAU_SIGMA)),
        }
        history = {}
        for key in ("z_xhi25", "z_xhi50", "z_xhi75", "delta_z"):
            truth = np.array([r["history_truth"][key] for r in self.rows])
            pred = np.array([r["history_pred"][key] for r in self.rows])
            entry = {
                "truth_med": _f(nanmedian(truth)),
                "pred_med": _f(nanmedian(pred)),
                "error_med": _f(nanmedian(pred - truth)),
            }
            if key != "delta_z":
                entry["error_sigma_med"] = _f(
                    nanmedian(np.abs(pred - truth) / PLANCK_ZRE_SIGMA))
            history[key] = entry
        return {
            "tau": tau,
            "history": history,
            "history_rmse_med": _f(nanmedian(
                [r["history_rmse"] for r in self.rows])),
            "n_samples": len(self.rows),
        }

    def arrays(self):
        if not self.rows:
            return {}
        return {
            "history_z": self.rows[0]["z"],
            "history_tau_truth": np.array([r["tau_truth"] for r in self.rows]),
            "history_tau_pred": np.array([r["tau_pred"] for r in self.rows]),
            **{f"history_{key}_truth": np.array(
                [r["history_truth"][key] for r in self.rows])
               for key in ("z_xhi25", "z_xhi50", "z_xhi75", "delta_z")},
            **{f"history_{key}_pred": np.array(
                [r["history_pred"][key] for r in self.rows])
               for key in ("z_xhi25", "z_xhi50", "z_xhi75", "delta_z")},
        }


def _f(value):
    value = float(value)
    return value if np.isfinite(value) else None
