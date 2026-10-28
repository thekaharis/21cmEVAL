"""Scale-resolved power: transverse P(k), power ratio, coherence, Delta^2.

For every LOS slice the 2-D transverse power spectrum is binned radially in
physical 1/Mpc, then aggregated by reionization stage (the **truth** slice
mean x_HI, so truth and prediction are paired). Reported per stage:

    ratio(k)      = P_pred(k) / P_truth(k)
    r(k)          = P_cross(k) / sqrt(P_truth P_pred)     (coherence)
    Delta^2(k)    = k^2 P(k) / 2 pi                       (dimensionless)
    k_r90         = smallest k where r(k) drops below ``coherence_level``

The LOS axis is deliberately not Fourier transformed: it mixes geometry with
cosmic evolution, so a 2-D transverse spectrum is the well-defined statistic.
"""
from __future__ import annotations

import numpy as np

from ..physics import KBinner, delta2, k_crossing, safe_ratio, stage_labels, stage_masks
from ..registry import register
from .base import Metric, median_band


@register("spectrum")
class SpectrumMetrics(Metric):
    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        self.records: list[dict] = []
        self._binner = None
        self._k = None

    def _bins(self, nx, ny, box):
        key = (nx, ny, round(box, 9), self.settings.n_k_bins)
        if self._binner is None or self._binner[0] != key:
            binner = KBinner(nx, ny, box, self.settings.n_k_bins)
            self._binner = (key, binner)
            self._k = binner.centers
        return self._binner[1]

    def update(self, sample, prediction, aux=None, runtime=None):
        truth = self.target(sample)
        pred = self.prediction(prediction)
        nx, ny, nz = truth.shape
        binner = self._bins(nx, ny, self.box_mpc(sample))

        t = truth - truth.mean(axis=(0, 1), keepdims=True)
        p = pred - pred.mean(axis=(0, 1), keepdims=True)
        pt = binner.slice_sums(t)
        pp = binner.slice_sums(p)
        cross = binner.cross_sums(t, p)

        masks = stage_masks(
            truth.mean(axis=(0, 1)),
            self.settings.stage_edges,
            self.settings.active_range,
        )
        nb = binner.centers.size
        n_rows = len(masks)
        stage_pt = np.zeros((n_rows, nb))
        stage_pp = np.zeros((n_rows, nb))
        stage_cross = np.zeros((n_rows, nb))
        stage_n = np.zeros(n_rows, dtype=np.int64)
        for row, mask in enumerate(masks):
            stage_pt[row] = pt[mask].sum(axis=0)
            stage_pp[row] = pp[mask].sum(axis=0)
            stage_cross[row] = cross[mask].sum(axis=0)
            stage_n[row] = int(mask.sum())

        denominator = np.maximum(stage_n, 1)[:, None] * binner.counts[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_pt = stage_pt / denominator
            mean_pp = stage_pp / denominator
        ratio, r = safe_ratio(stage_pt, stage_pp, stage_cross)

        thin = stage_n < self.settings.min_stage_slices
        ratio[thin] = np.nan
        r[thin] = np.nan
        scale = binner.centers ** 2 / (2 * np.pi)
        self.records.append({
            "sample": sample.id,
            "n_stage": stage_n,
            "ratio": ratio,
            "r": r,
            "d2_truth": scale * mean_pt,
            "d2_pred": scale * mean_pp,
        })

    def _maps(self):
        """(n_samples, n_rows, nb) arrays for ratio, r, Delta^2."""
        return tuple(
            np.stack([record[key] for record in self.records])
            for key in ("ratio", "r", "d2_truth", "d2_pred")
        )

    def result(self):
        if not self.records:
            return {}
        ratio, r, d2t, d2p = self._maps()
        ratio_med, ratio_lo, ratio_hi = median_band(ratio, axis=0)
        r_med, r_lo, r_hi = median_band(r, axis=0)
        d2t_med, _, _ = median_band(d2t, axis=0)
        d2p_med, _, _ = median_band(d2p, axis=0)
        k = np.asarray(self._k)
        labels = stage_labels(self.settings.stage_edges, self.settings.active_range)

        total_slices = np.sum([record["n_stage"] for record in self.records], axis=0)
        stages = []
        for row, label in enumerate(labels):
            n = int(total_slices[row])
            entry = {
                "label": label,
                "n_slices": n,
                "power_ratio_med": ratio_med[row],
                "power_ratio_p16": ratio_lo[row],
                "power_ratio_p84": ratio_hi[row],
                "coherence_med": r_med[row],
                "coherence_p16": r_lo[row],
                "coherence_p84": r_hi[row],
                "delta2_truth_med": d2t_med[row],
                "delta2_pred_med": d2p_med[row],
                "k_r90": k_crossing(k, r_med[row], self.settings.coherence_level),
            }
            stages.append(entry)

        active = stages[-1]
        headline = {}
        for target in self.settings.k_targets:
            index = int(np.argmin(np.abs(np.log(k) - np.log(target))))
            name = f"{k[index]:.3g}"
            headline[f"ratio_k{name}"] = _value(active["power_ratio_med"], index)
            headline[f"coherence_k{name}"] = _value(active["coherence_med"], index)
        headline["k_r90"] = active["k_r90"]
        reference = int(np.argmin(np.abs(np.log(k) - np.log(0.2))))
        headline["k_ref"] = float(k[reference])
        headline["ratio_ref"] = _value(active["power_ratio_med"], reference)
        headline["coherence_ref"] = _value(active["coherence_med"], reference)
        return {
            "k_centers": k,
            "stage_labels": labels,
            "stages": stages,
            "active": headline,
            "n_samples": len(self.records),
        }

    def arrays(self):
        if not self.records:
            return {}
        ratio, r, d2t, d2p = self._maps()
        out = {"spectrum_k": np.asarray(self._k)}
        labels = stage_labels(self.settings.stage_edges, self.settings.active_range)
        for row, label in enumerate(labels):
            token = label.replace(" ", "_").replace("-", "_").replace(".", "p")
            out[f"spectrum_ratio_{token}"] = ratio[:, row]
            out[f"spectrum_coherence_{token}"] = r[:, row]
            out[f"spectrum_delta2_truth_{token}"] = d2t[:, row]
            out[f"spectrum_delta2_pred_{token}"] = d2p[:, row]
        return out


def _value(array, index):
    value = float(array[index])
    return value if np.isfinite(value) else None
