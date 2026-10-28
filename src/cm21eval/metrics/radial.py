"""Isotropic 3-D power ratio and coherence -- a cross-check, not the headline.

The lightcone axis is **not** periodic (it spans z = 5..25) and the 256-cell
grid under-samples LOS structure, so a 3-D transform mixes geometry with
evolution. This metric therefore exists only as a cross-check of the
transverse result: a Hanning window along z suppresses the edge discontinuity,
and the radial binning uses physical cell sizes with a mean LOS spacing.

Reported per k shell: ``P_pred/P_truth`` and ``r = P_cross/sqrt(P_t P_p)``.
Because normalization cancels in both, no absolute power is quoted.
"""
from __future__ import annotations

import numpy as np

from ..physics import k_crossing, los_cell_size_mpc
from ..registry import register
from .base import Metric, median_band


@register("radial")
class RadialMetrics(Metric):
    needs = ()

    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        self.records: list[dict] = []
        self._grid = None

    def _bins(self, shape, box, dz):
        nx, ny, nz = shape
        dx, dy = box / nx, box / ny
        kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
        kz = 2 * np.pi * np.fft.rfftfreq(nz, d=dz)
        k = np.sqrt(
            kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
        )
        k_fund = 2 * np.pi / max(box, box, dz * nz)
        k_ny = np.pi / max(dx, dy, dz)
        if not k_fund < k_ny:
            return None
        edges = np.geomspace(k_fund, k_ny, self.settings.n_k_bins + 1)
        edges[0] *= 1 - 1e-9
        edges[-1] *= 1 + 1e-9
        index = np.digitize(k.ravel(), edges) - 1
        valid = (index >= 0) & (index < self.settings.n_k_bins) & (k.ravel() > 0)
        return edges, index[valid], valid, np.sqrt(edges[:-1] * edges[1:])

    def update(self, sample, prediction, aux=None, runtime=None):
        truth = self.target(sample)
        pred = self.prediction(prediction)
        nx, ny, nz = truth.shape
        box = self.box_mpc(sample)
        z = self.z_grid(sample)
        if z is not None and len(z) > 1:
            dz = los_cell_size_mpc(float(np.mean(z)), float(z[1] - z[0]),
                                   omega_m=self.omega_m(sample),
                                   h=self.settings.hubble_h)
        else:
            dz = box / nz
        bins = self._bins((nx, ny, nz), box, dz)
        if bins is None:
            self.skipped = getattr(self, "skipped", 0) + 1
            return
        edges, index, valid, centers = bins
        # A 2-sample Hann window is identically zero; use a rectangular
        # window when the LOS axis is too short for a taper.
        window = np.hanning(nz)[None, None, :] if nz > 2 else np.ones((1, 1, nz))

        def shell(field):
            centered = (field - field.mean()) * window
            fft = np.fft.rfftn(centered)
            power = (np.abs(fft) ** 2).ravel()[valid]
            return np.bincount(index, weights=power,
                               minlength=len(centers))

        count = np.bincount(index, minlength=len(centers))
        t = shell(truth)
        p = shell(pred)
        # Cross power needs both fields: recompute jointly.
        centered_t = (truth - truth.mean()) * window
        centered_p = (pred - pred.mean()) * window
        ft = np.fft.rfftn(centered_t).ravel()[valid]
        fp = np.fft.rfftn(centered_p).ravel()[valid]
        cross = np.bincount(index, weights=np.real(ft * np.conj(fp)),
                            minlength=len(centers))
        self.records.append({"count": count, "t": t, "p": p, "cross": cross})
        self._grid = centers

    def result(self):
        if not self.records:
            return {}
        with np.errstate(invalid="ignore", divide="ignore"):
            ratios, coherences = [], []
            for record in self.records:
                count = np.maximum(record["count"], 1)
                pt = record["t"] / count
                pp = record["p"] / count
                cross = record["cross"] / count
                ratio = np.where(pt > 0, pp / pt, np.nan)
                denominator = np.sqrt(pt * pp)
                r = np.where(denominator > 0, cross / denominator, np.nan)
                ratios.append(ratio)
                coherences.append(r)
        ratio = np.stack(ratios)
        coherence = np.stack(coherences)
        ratio_med, ratio_lo, ratio_hi = median_band(ratio, axis=0)
        r_med, r_lo, r_hi = median_band(coherence, axis=0)
        k = self._grid
        reference = int(np.argmin(np.abs(np.log(k) - np.log(0.2))))
        return {
            "k_centers": k,
            "power_ratio_med": ratio_med,
            "power_ratio_p16": ratio_lo,
            "power_ratio_p84": ratio_hi,
            "coherence_med": r_med,
            "coherence_p16": r_lo,
            "coherence_p84": r_hi,
            "headline": {
                "k_ref": float(k[reference]),
                "ratio_ref": _f(ratio_med[reference]),
                "coherence_ref": _f(r_med[reference]),
                "k_r90": _f(k_crossing(k, r_med, self.settings.coherence_level)),
            },
            "note": "3-D transform over a non-periodic LOS axis; cross-check only",
            "n_samples": len(self.records),
        }

    def arrays(self):
        if not self.records:
            return {}
        return {"radial_k": self._grid}


def _f(value):
    value = float(value)
    return value if np.isfinite(value) else None
