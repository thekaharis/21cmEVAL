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
        self._highk = None

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
        highk = self._highk_scores(t, p)
        highk_stage = np.array([
            (float(np.mean(highk[mask])) if mask.any() else np.nan)
            for mask in masks])
        xbar = truth.mean(axis=(0, 1))
        active = (xbar > 0.001) & (xbar < 0.999)       # fno-21cm viz/ps_coherence_highk
        lin = self._linear_spectra(t, p)
        self.records.append({
            "sample": sample.id,
            "lin": lin,
            "highk_slices": np.array([float(highk[active].sum()), float(active.sum()),
                                      float(highk.sum()), float(highk.size)]),
            "n_stage": stage_n,
            "ratio": ratio,
            "r": r,
            "d2_truth": scale * mean_pt,
            "d2_pred": scale * mean_pp,
            "highk_logmse": highk_stage,
        })

    def _linear_spectra(self, truth, pred, n_bins=24):
        """Radially binned auto/cross power in cycles per pixel, summed over slices.

        The 2-D spectra of ``viz/ps_coherence_highk.py``: linear bins up to the per-axis
        Nyquist (0.5 cycles/px), DC and corner modes beyond it dropped, ``norm="ortho"``.
        Returns (3, n_bins): [pred, truth, cross] per-mode means summed over slices.
        """
        nx, ny = truth.shape[:2]
        key = ("lin", nx, ny, n_bins)
        if getattr(self, "_lin", None) is None or self._lin[0] != key:
            radius = np.sqrt(np.fft.fftfreq(nx)[:, None] ** 2 + np.fft.rfftfreq(ny)[None, :] ** 2).ravel()
            edges = np.linspace(0.0, 0.5, n_bins + 1)
            index = np.digitize(radius, edges) - 1
            index[(radius >= 0.5) | (radius == 0) | (index < 0)] = -1
            keep = index >= 0
            counts = np.bincount(index[keep], minlength=n_bins).astype(float)
            self._lin = (key, index, keep, counts, 0.5 * (edges[:-1] + edges[1:]))
        _, index, keep, counts, _ = self._lin
        fp = np.fft.rfft2(pred, axes=(0, 1), norm="ortho")
        ft = np.fft.rfft2(truth, axes=(0, 1), norm="ortho")
        out = []
        for field in ((fp * fp.conj()).real, (ft * ft.conj()).real, (fp * ft.conj()).real):
            flat = field.sum(axis=2).ravel()
            acc = np.bincount(index[keep], weights=flat[keep], minlength=n_bins)
            out.append(acc / np.maximum(counts, 1.0))
        return np.stack(out)

    def _highk_matrix(self, nx, ny):
        key = (nx, ny, self.settings.highk_k_min, self.settings.highk_bins)
        if self._highk is None or self._highk[0] != key:
            fx = np.fft.fftfreq(nx)
            fy = np.fft.rfftfreq(ny)
            radius = np.sqrt(fx[:, None] ** 2 + fy[None, :] ** 2).ravel()
            edges = np.linspace(self.settings.highk_k_min,
                                float(radius.max()),
                                self.settings.highk_bins + 1)
            index = np.digitize(radius, edges) - 1
            valid = (radius >= self.settings.highk_k_min) & (
                index < self.settings.highk_bins) & (radius > 0)
            matrix = np.zeros((int(valid.sum()), self.settings.highk_bins))
            matrix[np.arange(matrix.shape[0]), index[valid]] = 1.0
            counts = np.maximum(matrix.sum(axis=0), 1.0)
            self._highk = (key, valid, matrix, counts)
        return self._highk[1:]

    def _highk_scores(self, truth, pred):
        """Per-slice mean (log P_pred/P_true)^2 above ``highk_k_min``."""
        valid, matrix, counts = self._highk_matrix(*truth.shape[:2])
        ft = np.fft.rfft2(truth, axes=(0, 1), norm="ortho")
        fp = np.fft.rfft2(pred, axes=(0, 1), norm="ortho")
        pt = (np.abs(ft) ** 2).reshape(-1, truth.shape[2])[valid]
        pp = (np.abs(fp) ** 2).reshape(-1, pred.shape[2])[valid]
        binned_t = (matrix.T @ pt).T / counts[None, :]
        binned_p = (matrix.T @ pp).T / counts[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.log((binned_p + 1e-12) / (binned_t + 1e-12))
        return np.mean(ratio ** 2, axis=1)

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
        thin = np.stack([record["n_stage"] < self.settings.min_stage_slices
                         for record in self.records])
        d2t[thin] = np.nan
        d2p[thin] = np.nan
        highk = np.stack([record["highk_logmse"] for record in self.records])
        highk_med, _, _ = median_band(highk, axis=0)
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
                "highk_logmse": _value(highk_med, row),
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
        headline["highk_logmse"] = active["highk_logmse"]
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
        out = {"spectrum_k": np.asarray(self._k),
               "spectrum_highk_logmse": np.stack([r["highk_logmse"] for r in self.records])}
        labels = stage_labels(self.settings.stage_edges, self.settings.active_range)
        for row, label in enumerate(labels):
            token = label.replace(" ", "_").replace("-", "_").replace(".", "p")
            out[f"spectrum_ratio_{token}"] = ratio[:, row]
            out[f"spectrum_coherence_{token}"] = r[:, row]
            out[f"spectrum_delta2_truth_{token}"] = d2t[:, row]
            out[f"spectrum_delta2_pred_{token}"] = d2p[:, row]
        # ps_coherence_highk layout: linear cycles/px spectra and per-slice highk sums.
        if all("lin" in r for r in self.records):
            out["spectrum_lin_k"] = self._lin[4]
            out["spectrum_lin_power"] = np.stack([r["lin"] for r in self.records])
            out["spectrum_highk_slices"] = np.stack([r["highk_slices"] for r in self.records])
            out["spectrum_highk_k_min"] = np.asarray(self.settings.highk_k_min)
            out["spectrum_sample_ids"] = np.array([str(r["sample"]) for r in self.records])
            out["spectrum_n_stage"] = np.stack([r["n_stage"] for r in self.records])
        return out


def _value(array, index):
    value = float(array[index])
    return value if np.isfinite(value) else None
