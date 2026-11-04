"""Cylindrical (k_perp, k_par) power ratio and coherence.

An emulator can have the right isotropic spectrum and still get the
line-of-sight anisotropy wrong: the LOS axis mixes cosmic evolution with
geometry, and a network that smooths along z loses small-scale structure that
a transverse-only diagnostic cannot see.

For LOS chunks centred on ``chunk_z_centers`` the chunk is Hann-windowed
along z, transformed, and binned in ``(k_perp, k_par)``. Reported per chunk:
the ratio ``P_pred/P_truth`` and coherence ``r = P_cross/sqrt(P_t P_p)`` on
the 2-D grid (exported to ``curves.npz``), plus a headline split into
low- and high-``k_par`` halves.

The k_par = 0 plane is dropped, so both axes are strictly positive
wavenumbers. Ratios need no physical normalization: it cancels cell by cell.
"""
from __future__ import annotations

import numpy as np

from ..physics import KBinner, los_cell_size_mpc, safe_ratio
from ..registry import register
from .base import Metric, nanmedian


@register("anisotropy")
class AnisotropyMetrics(Metric):
    needs = ()

    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        self.chunk_z = tuple(options.get("chunk_z_centers", (7.0, 9.0, 11.0)))
        self.chunk_cells = int(options.get("chunk_cells", 32))
        self.min_truth_std = float(options.get("min_truth_std", 1e-3))
        self.rows: list[dict] = []
        self._binner = None
        self._kperp = None
        self._kpar = None
        self.skipped = 0

    def _bins(self, nx, ny, box):
        key = (nx, ny, round(box, 9), self.settings.n_k_bins)
        if self._binner is None or self._binner[0] != key:
            binner = KBinner(nx, ny, box, self.settings.n_k_bins)
            self._binner = (key, binner)
            self._kperp = binner.centers
        return self._binner[1]

    def update(self, sample, prediction, aux=None, runtime=None):
        z = self.z_grid(sample)
        if z is None or len(z) < 2:
            self.skipped += 1
            return
        truth = self.target(sample)
        pred = self.prediction(prediction)
        nx, ny, nz = truth.shape
        chunk = min(self.chunk_cells, nz)
        if chunk < 8:
            self.skipped += 1
            return
        box = self.box_mpc(sample)
        binner = self._bins(nx, ny, box)
        slice_std = truth.std(axis=(0, 1))
        maps = []
        for target_z in self.chunk_z:
            index = int(np.argmin(np.abs(z - target_z)))
            start = min(max(index - chunk // 2, 0), nz - chunk)
            stop = start + chunk
            # Coarse z grids cannot centre a window on z=7; clamp it and report
            # the actual centre instead of silently skipping the diagnostic.
            cell = los_cell_size_mpc(
                float(z[start:stop].mean()), float(z[1] - z[0]),
                omega_m=self.omega_m(sample), h=self.settings.hubble_h)
            freq = 2 * np.pi * np.fft.rfftfreq(chunk, d=cell)
            kpar = freq[1:]
            window = np.hanning(chunk)
            t = (truth[:, :, start:stop]
                 - truth[:, :, start:stop].mean(axis=(0, 1), keepdims=True))
            p = (pred[:, :, start:stop]
                 - pred[:, :, start:stop].mean(axis=(0, 1), keepdims=True))
            ft = np.fft.fftn(np.fft.rfft(t * window, axis=2), axes=(0, 1))
            fp = np.fft.fftn(np.fft.rfft(p * window, axis=2), axes=(0, 1))
            pt = self._bin(ft, binner)
            pp = self._bin(fp, binner)
            cross = self._bin_cross(ft, fp, binner)
            maps.append({"z": float(z[start:stop].mean()), "kpar": kpar,
                         "pt": pt, "pp": pp, "cross": cross,
                         "ok": bool(slice_std[start:stop].mean()
                                    > self.min_truth_std)})
        self.rows.append({"sample": sample.id, "maps": maps})

    @staticmethod
    def _bin(fft, binner):
        flat = fft.reshape(-1, fft.shape[2])
        return (binner.matrix.T @ (np.abs(flat) ** 2)[binner.valid]).T[1:]

    @staticmethod
    def _bin_cross(ft, fp, binner):
        flat = np.real(ft * np.conj(fp)).reshape(-1, ft.shape[2])
        return (binner.matrix.T @ flat[binner.valid]).T[1:]

    def result(self):
        if not self.rows:
            return {"note": "no z_grid or chunk shorter than chunk_cells"}
        n_chunks = max(len(row["maps"]) for row in self.rows)
        if n_chunks == 0:
            return {"note": "no LOS chunk fits the redshift range"}
        chunks = []
        kperp = self._kperp
        for c in range(n_chunks):
            ratios, coherences = [], []
            z_label = None
            kpar = None
            for row in self.rows:
                if c >= len(row["maps"]):
                    continue
                record = row["maps"][c]
                if not record["ok"]:
                    continue
                z_label = record["z"]
                kpar = record["kpar"]
                ratio, coherence = safe_ratio(
                    record["pt"], record["pp"], record["cross"])
                ratios.append(ratio)
                coherences.append(coherence)
            if not ratios:
                continue
            with np.errstate(all="ignore"):
                ratio_med = _nanmedian(np.stack(ratios), axis=0)
                coherence_med = _nanmedian(np.stack(coherences), axis=0)
            if not (np.isfinite(ratio_med).any()
                    or np.isfinite(coherence_med).any()):
                continue
            high = kpar > np.nanmedian(kpar)
            low = ~high
            ratio_low = nanmedian(ratio_med[low, :])
            ratio_high = nanmedian(ratio_med[high, :])
            chunks.append({
                "z": _f(z_label),
                "kperp_centers": [float(v) for v in kperp],
                "kpar_centers": [float(v) for v in kpar],
                "ratio_low_kpar": _f(ratio_low),
                "ratio_high_kpar": _f(ratio_high),
                "coherence_low_kpar": _f(nanmedian(coherence_med[low, :])),
                "coherence_high_kpar": _f(nanmedian(coherence_med[high, :])),
                "anisotropy": _ratio(ratio_high, ratio_low),
            })
        return {
            "chunks": chunks,
            "headline": {
                "anisotropy_median": _f(nanmedian(
                    [c["anisotropy"] for c in chunks if c["anisotropy"]])),
                "coherence_high_kpar": _f(nanmedian(
                    [c["coherence_high_kpar"] for c in chunks
                     if c["coherence_high_kpar"] is not None])),
                "coherence_low_kpar": _f(nanmedian(
                    [c["coherence_low_kpar"] for c in chunks
                     if c["coherence_low_kpar"] is not None])),
            },
            "skipped_samples": self.skipped,
            "n_samples": len(self.rows),
        }

    def arrays(self):
        if not self.rows:
            return {}
        out = {}
        kperp = self._kperp
        if kperp is not None:
            out["anisotropy_kperp"] = kperp
        n_chunks = max(len(row["maps"]) for row in self.rows)
        for c in range(n_chunks):
            ratios = []
            z_label = None
            kpar = None
            for row in self.rows:
                if c >= len(row["maps"]):
                    continue
                record = row["maps"][c]
                if not record["ok"]:
                    continue
                z_label = record["z"]
                kpar = record["kpar"]
                ratio, _ = safe_ratio(record["pt"], record["pp"],
                                      record["cross"])
                ratios.append(ratio)
            if ratios:
                with np.errstate(all="ignore"):
                    median = _nanmedian(np.stack(ratios), axis=0)
                if np.isfinite(median).any():
                    out[f"anisotropy_ratio_z{z_label:g}"] = median
                out[f"anisotropy_kpar_z{z_label:g}"] = kpar
        out.update(self._suite_arrays())
        return out

    def _suite_arrays(self):
        """Per-sample cylindrical maps (n_samples, n_chunks, n_kpar, n_kperp) for
        ``eval21 plot power`` (ps_cyl_ratio / ps_cyl_r), pooled over groups there; chunks
        with ~uniform truth are NaN, as in fno-21cm's power_spectrum_evaluation."""
        n_chunks = max(len(row["maps"]) for row in self.rows)
        if n_chunks == 0 or any(len(row["maps"]) != n_chunks for row in self.rows):
            return {}
        shape = self.rows[0]["maps"][0]["pt"].shape
        ratio = np.full((len(self.rows), n_chunks) + shape, np.nan)
        coherence = np.full_like(ratio, np.nan)
        z = np.full((len(self.rows), n_chunks), np.nan)
        kpar = np.full((len(self.rows), n_chunks, shape[0]), np.nan)
        for i, row in enumerate(self.rows):
            for c, record in enumerate(row["maps"]):
                z[i, c] = record["z"]
                kpar[i, c] = record["kpar"]
                if record["ok"] and record["pt"].shape == shape:
                    ratio[i, c], coherence[i, c] = safe_ratio(record["pt"], record["pp"], record["cross"])
        return {"anisotropy_cyl_ratio": ratio, "anisotropy_cyl_r": coherence, "anisotropy_cyl_z": z,
                "anisotropy_cyl_kpar": kpar, "anisotropy_cyl_z_target": np.asarray(self.chunk_z, float),
                "anisotropy_kperp_edges": self._binner[1].edges}


def _nanmedian(values, axis=0):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def _f(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _ratio(a, b):
    if a is None or b is None or b == 0:
        return None
    return _f(a / b)
