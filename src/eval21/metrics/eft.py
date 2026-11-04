"""Effective-field-theory bias diagnostics.

With ``u = x_HI - <x_HI>`` per slice and the input density ``delta``::

    T(k)     = P_u,delta(k) / P_delta(k)                     transfer
    P_eps(k) = P_uu(k) - P_u,delta(k)^2 / P_delta(k)         stochasticity
    r_u,d    = P_u,delta / sqrt(P_uu P_delta)                correlation

Per stage, ``T`` is fitted as ``b_1 + b_{k2} k^2`` and ``P_eps`` as
``P_eps,0 + P_eps,2 k^2`` for ``k <= eft_k_max``. Truth and prediction
coefficients are compared cone by cone and the difference is scaled by the
truth fit's error.

The normalisation follows ``fno-21cm`` exactly: ``n_modes = N_modes *
n_slices``, ``n_independent = n_modes / 2``, ``P_eps`` is a per-mode power,
non-positive ``P_eps`` bins are dropped, and the fit variances are
``P_eps / (n_independent P_dd)`` for the transfer function and
``2 P_eps^2 / n_independent`` for the stochastic term.

Interpretation: a conditional-mean (MSE-trained) emulator usually reproduces
T(k) but under-predicts P_eps -- it produces the mean field given the density
and drops the part of x_HI that the density does not determine. That is a
physical failure mode invisible to RMSE.

Caveats carried from fno-21cm: these are 2-D transverse spectra of thin
slices, so k here corresponds to 3-D wavenumbers of roughly 1-2 k_perp, and
the fit errors assume Gaussian mode counting with slices treated as
independent (they are not). Read coefficient *differences*, not the absolute
errors.
"""
from __future__ import annotations

import numpy as np

from ..physics import (KBinner, fit_even_polynomial, stage_labels,
                       stage_masks)
from ..registry import register
from .base import Metric, nanmedian


@register("eft")
class EftMetrics(Metric):
    pointwise = False           # compares field-density relations, not truth voxels
    needs = ("density",)

    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        self.rows: list[dict] = []
        self._binner = None

    def _bins(self, nx, ny, box):
        key = (nx, ny, round(box, 9), self.settings.n_k_bins)
        if self._binner is None or self._binner[0] != key:
            self._binner = (key, KBinner(nx, ny, box, self.settings.n_k_bins))
        return self._binner[1]

    def update(self, sample, prediction, aux=None, runtime=None):
        truth = self.target(sample)
        pred = self.prediction(prediction)
        density = self.require(sample, self.settings.density)
        if density.shape != truth.shape:
            raise ValueError(f"sample {sample.id!r}: density shape mismatch")
        binner = self._bins(*truth.shape[:2], self.box_mpc(sample))
        u_t = truth - truth.mean(axis=(0, 1), keepdims=True)
        u_p = pred - pred.mean(axis=(0, 1), keepdims=True)
        delta = density - density.mean(axis=(0, 1), keepdims=True)

        sums = {
            "dd": binner.slice_sums(delta),
            "td": binner.cross_sums(u_t, delta),
            "pd": binner.cross_sums(u_p, delta),
            "tt": binner.slice_sums(u_t),
            "pp": binner.slice_sums(u_p),
        }
        masks = stage_masks(
            self.stage_xbar(sample, truth),
            self.settings.stage_edges,
            self.settings.active_range,
        )
        stages = []
        for mask in masks:
            n = int(mask.sum())
            if n < self.settings.min_stage_slices:
                stages.append(None)
                continue
            stage_sums = {name: array[mask].sum(axis=0)
                          for name, array in sums.items()}
            stages.append(self._fit_stage(stage_sums, binner.counts, n))
        self.rows.append({"sample": sample.id, "stages": stages})

    def _fit_stage(self, sums, counts, n_slices: int) -> dict:
        k = self._binner[1].centers
        n_modes = counts * n_slices
        n_independent = n_modes / 2.0
        valid = (n_modes > 0) & (sums["dd"] > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            p_dd = np.where(valid, sums["dd"] / n_modes, np.nan)
            t_truth = np.where(valid, sums["td"] / sums["dd"], np.nan)
            t_pred = np.where(valid, sums["pd"] / sums["dd"], np.nan)
            eps_truth = np.where(
                valid, (sums["tt"] - sums["td"] ** 2 / sums["dd"]) / n_modes,
                np.nan)
            eps_pred = np.where(
                valid, (sums["pp"] - sums["pd"] ** 2 / sums["dd"]) / n_modes,
                np.nan)
            # The reference drops non-positive stochastic power before fitting.
            eps_truth = np.where(eps_truth > 0, eps_truth, np.nan)
            eps_pred = np.where(eps_pred > 0, eps_pred, np.nan)
            r_truth = np.where(
                valid & (sums["tt"] > 0),
                sums["td"] / np.sqrt(sums["tt"] * sums["dd"]), np.nan)
            r_pred = np.where(
                valid & (sums["pp"] > 0),
                sums["pd"] / np.sqrt(sums["pp"] * sums["dd"]), np.nan)
            var_t_truth = eps_truth / (n_independent * p_dd)
            var_t_pred = eps_pred / (n_independent * p_dd)
            var_e_truth = 2.0 * eps_truth ** 2 / n_independent
            var_e_pred = 2.0 * eps_pred ** 2 / n_independent

        fit_t = fit_even_polynomial(k, t_truth, var_t_truth,
                                    self.settings.eft_k_max, 2)
        fit_p = fit_even_polynomial(k, t_pred, var_t_pred,
                                    self.settings.eft_k_max, 2)
        fit_et = fit_even_polynomial(k, eps_truth, var_e_truth,
                                     self.settings.eft_k_max, 2)
        fit_ep = fit_even_polynomial(k, eps_pred, var_e_pred,
                                     self.settings.eft_k_max, 2)
        targets = self._target_index()
        return {
            "k": k,
            "transfer_truth": t_truth,
            "transfer_pred": t_pred,
            "eps_truth": eps_truth,
            "eps_pred": eps_pred,
            "r_truth": r_truth,
            "r_pred": r_pred,
            "transfer_fit_truth": fit_t,
            "transfer_fit_pred": fit_p,
            "eps_fit_truth": fit_et,
            "eps_fit_pred": fit_ep,
            "transfer_at_k": [t_truth[i] for i in targets],
            "transfer_pred_at_k": [t_pred[i] for i in targets],
            "r_at_k": [r_truth[i] for i in targets],
            "r_pred_at_k": [r_pred[i] for i in targets],
        }

    def _target_index(self):
        k = self._binner[1].centers
        return [int(np.argmin(np.abs(np.log(k) - np.log(target))))
                for target in self.settings.k_targets]

    def _stage_values(self, j, key):
        values = []
        for row in self.rows:
            stage = row["stages"][j]
            if stage is not None:
                values.append(stage[key])
        return values

    def _coef(self, j, key, term):
        values = [v["coef"][term] for v in self._stage_values(j, key)
                  if np.isfinite(v["coef"][term])]
        return float(np.median(values)) if values else float("nan")

    def _sigma(self, j, key, term):
        values = [v["sigma"][term] for v in self._stage_values(j, key)
                  if np.isfinite(v["sigma"][term])]
        return float(np.median(values)) if values else float("nan")

    def _ratio(self, j, target_index):
        t = self._stage_values(j, "transfer_at_k")
        p = self._stage_values(j, "transfer_pred_at_k")
        if not t:
            return float("nan")
        with np.errstate(invalid="ignore", divide="ignore"):
            values = [float(p[i][target_index]) / float(t[i][target_index])
                      for i in range(len(t))]
        return nanmedian(values)

    def _median_at_k(self, j, key, target_index):
        values = []
        for stage in self._stage_values(j, key):
            value = float(stage[target_index])
            if np.isfinite(value):
                values.append(value)
        return float(np.median(values)) if values else float("nan")

    def result(self):
        if not self.rows:
            return {}
        labels = stage_labels(self.settings.stage_edges,
                              self.settings.active_range)
        targets = self._target_index()
        k = self._binner[1].centers

        stages = []
        for j, label in enumerate(labels):
            b1_t = self._coef(j, "transfer_fit_truth", 0)
            b1_p = self._coef(j, "transfer_fit_pred", 0)
            b2_t = self._coef(j, "transfer_fit_truth", 1)
            b2_p = self._coef(j, "transfer_fit_pred", 1)
            e0_t = self._coef(j, "eps_fit_truth", 0)
            e0_p = self._coef(j, "eps_fit_pred", 0)
            sigma = self._sigma(j, "transfer_fit_truth", 0)
            delta_b1 = (b1_p - b1_t) / sigma if sigma else float("nan")
            stages.append({
                "label": label,
                "b1_truth": _f(b1_t),
                "b1_pred": _f(b1_p),
                "b1_delta_sigma": _f(delta_b1),
                "bk2_truth": _f(b2_t),
                "bk2_pred": _f(b2_p),
                "eps0_truth": _f(e0_t),
                "eps0_pred": _f(e0_p),
                "eps0_ratio": _f(e0_p / e0_t) if e0_t else None,
                "transfer_ratio": [_f(self._ratio(j, t))
                                   for t in range(len(targets))],
                "r_ud_truth": _f(self._median_at_k(j, "r_at_k",
                                                   len(targets) // 2)),
                "r_ud_pred": _f(self._median_at_k(j, "r_pred_at_k",
                                                  len(targets) // 2)),
                "k_targets": [float(k[i]) for i in targets],
            })
        active = stages[-1]
        return {
            "stage_labels": labels,
            "stages": stages,
            "active": {
                "b1_truth": active["b1_truth"],
                "b1_pred": active["b1_pred"],
                "b1_delta_sigma": active["b1_delta_sigma"],
                "eps0_ratio": active["eps0_ratio"],
                "transfer_ratio": active["transfer_ratio"],
                "r_ud_truth": active["r_ud_truth"],
                "r_ud_pred": active["r_ud_pred"],
            },
            "n_samples": len(self.rows),
        }

    def arrays(self):
        if not self.rows:
            return {}
        k = self._binner[1].centers
        out = {"eft_k": k}
        labels = stage_labels(self.settings.stage_edges,
                              self.settings.active_range)
        for j, label in enumerate(labels):
            token = label.replace(" ", "_").replace("-", "_").replace(".", "p")
            for key in ("transfer_truth", "transfer_pred", "eps_truth",
                        "eps_pred", "r_truth", "r_pred"):
                rows = [row["stages"][j][key] for row in self.rows
                        if row["stages"][j] is not None]
                if rows:
                    out[f"eft_{key}_{token}"] = np.stack(rows)
        # Per-sample fit coefficients (n_samples, n_rows, 2), NaN for thin stages: the
        # truth-vs-prediction scatter of ``eval21 plot physical`` (fno-21cm layout).
        for name, key, part in (("bias_truth", "transfer_fit_truth", "coef"),
                                ("bias_pred", "transfer_fit_pred", "coef"),
                                ("bias_sigma_truth", "transfer_fit_truth", "sigma"),
                                ("stoch_truth", "eps_fit_truth", "coef"),
                                ("stoch_pred", "eps_fit_pred", "coef"),
                                ("stoch_sigma_truth", "eps_fit_truth", "sigma")):
            values = np.full((len(self.rows), len(labels), 2), np.nan)
            for i, row in enumerate(self.rows):
                for j, stage in enumerate(row["stages"]):
                    if stage is not None:
                        values[i, j] = np.asarray(stage[key][part], dtype=np.float64)[:2]
            out[f"eft_fit_{name}"] = values
        return out


def _f(value):
    value = float(value)
    return value if np.isfinite(value) else None
