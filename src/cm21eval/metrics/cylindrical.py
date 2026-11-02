"""Cylindrical power P(k_perp, k_par) on lightcone chunks: ratio, coherence, in-band share.

The 2-D (k_perp, k_par) power spectrum of the 21-cm literature, measured on chunks of the
lightcone. It is the LOS counterpart of ``spectrum`` (transverse only): structure along the line
of sight, slice-to-slice jitter of slice-wise models and LOS smoothing all show up at k_par.

Per chunk of ``cyl_chunk_cells`` LOS cells of one transverse cell each (140 = 200 Mpc on the
21cmFAST grid): the per-slice mean is removed, a Hann window applied along the LOS (a chunk is
not periodic), then a 3-D FFT; |F_t|^2, |F_p|^2 and Re(F_t F_p*) are summed into bins of k_perp
(the suite's ``n_k_bins`` log bins, ``KBinner``) and k_par (one row per LOS frequency, DC
dropped). Per sample and chunk:

    ratio(k_perp, k_par) = P_pred / P_truth
    r(k_perp, k_par)     = P_cross / sqrt(P_truth P_pred)

both masked when the truth chunk is ~uniform (mean transverse std <= 1e-3). Reported per chunk
column: the median over samples, and the share of resolved (k_perp, k_par) cells whose median
|ratio - 1| is within ``cyl_dead_band`` (``cm21eval plot cyl`` draws those maps). This is the
computation of rhizome-neural-operator's ``ebm21cm.native_cyl`` (fno-21cm's
``_cylindrical_sums``), which it reproduces on identical chunks.

Chunk placement. By default per sample at ionization stages (``cyl_stages``, true slice-mean
x_HI), so that simulations or samples that reionize at different redshifts are compared like
for like: the chunk is centred on the slice whose true x_HI is nearest the stage; a sample whose
history comes no closer than 0.05 is skipped for that stage, and a chunk that would leave the
cone is shifted inside it and dropped if its centre then sits more than 0.1 from the stage.
With ``cyl_z_centers`` the chunks sit at fixed redshifts instead (dropped when they leave the
cone). The true x_HI comes from the ``xhi`` target or, in a T_b pass, an ``xhi`` input.

LOS geometry. Positions follow from the sample's ``z_grid`` (comoving distance, flat LCDM with the
suite's cosmology). A grid already uniform at the transverse cell size (within 0.2 %) is sliced
directly; any other grid (e.g. uniform in z) is linearly resampled onto the transverse cell
inside the chunk, and k_par above pi / (coarsest grid spacing in the chunk) is marked as not
resolved. ``cyl_native_cell_mpc`` and ``cyl_native_los_dz`` / ``cyl_native_los_mpc`` declare
that the truth itself was interpolated from a coarser native grid (LoReLi II: 2.3 Mpc cells,
planes uniform in z with dz = 10/1535) and add those limits. Unresolved cells are excluded from
the in-band share and hatched in the plots.

``anisotropy`` is the older, lighter variant (fixed-z 32-cell chunks, median maps only, a low/high
k_par headline); ``cylindrical`` keeps per-sample maps so that runs can be pooled and plotted.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..physics import KBinner, comoving_distance_mpc, los_cell_size_mpc
from ..registry import register
from .base import Metric, finite_or_none

STAGE_TOL, STAGE_SHIFT_TOL, MIN_TRUTH_STD, UNIFORM_TOL = 0.05, 0.1, 1e-3, 2e-3


def true_xhi(sample, target: str):
    """True x_HI cube of a sample (target or input ``xhi``), or None."""
    for source in (sample.targets, sample.inputs):
        value = source.get("xhi")
        if value is not None and np.ndim(value) == 3:
            return np.asarray(value, dtype=np.float64)
    return None


def column_label(mode: str, value: float) -> str:
    text = f"{value:g}".replace(".", "p").replace("-", "m")
    return f"xbar_{text}" if mode == "stage" else f"z_{text}"


@register("cylindrical")
class CylindricalMetrics(Metric):
    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        s = self.settings
        self.mode = "z" if s.cyl_z_centers else "stage"
        self.columns = tuple(float(v) for v in (s.cyl_z_centers if self.mode == "z" else s.cyl_stages))
        self.n = int(s.cyl_chunk_cells)
        self.window = np.hanning(self.n)
        self._binner = None
        self.kpar_edges = None
        self.cell_los = None
        self.ratio, self.coherence, self.chunk_z, self.chunk_xbar, self.kpar_max = [], [], [], [], []
        self.have: list[np.ndarray] = []
        self.kperp_max = np.inf
        self.ids: list[str] = []

    # -- geometry ----------------------------------------------------------- #
    def _setup(self, nx, ny, box, cell_los):
        if self._binner is None:
            self._binner = KBinner(nx, ny, box, self.settings.n_k_bins)
            kpar = 2 * np.pi * np.fft.rfftfreq(self.n, d=cell_los)
            self.kpar_edges = np.concatenate([[kpar[1] / 2], kpar[1:] + kpar[1] / 2])
            self.cell_los = cell_los
            native = self.settings.cyl_native_cell_mpc
            self.kperp_max = np.pi / native if native else np.inf
        elif abs(cell_los / self.cell_los - 1) > 1e-6 or self._binner.edges.size != self.settings.n_k_bins + 1:
            raise ValueError("cylindrical: LOS cell size changed between samples; one geometry per run")
        return self._binner

    def _native_kpar(self, z_center: float) -> float:
        s = self.settings
        if s.cyl_native_los_mpc:
            return np.pi / s.cyl_native_los_mpc
        if s.cyl_native_los_dz:
            return np.pi / los_cell_size_mpc(z_center, s.cyl_native_los_dz, self.settings.omega_m,
                                             self.settings.hubble_h)
        return np.inf

    def _centers(self, z, xbar):
        """Centre index on the sample grid per column (None: no chunk)."""
        out = []
        for value in self.columns:
            if self.mode == "z":
                j = int(np.argmin(np.abs(z - value)))
                out.append(j if z.min() <= value <= z.max() else None)
            else:
                j = int(np.argmin(np.abs(xbar - value)))
                out.append(j if abs(xbar[j] - value) <= STAGE_TOL else None)
        return out

    def _chunk(self, fields, dist, j, xbar):
        """(chunk fields, centre distance) for a chunk centred at index j, or None."""
        n, cell = self.n, self.cell_los
        uniform = self._uniform
        if uniform:
            half = n // 2
            start = j - half
            if self.mode == "stage":
                start = min(max(start, 0), len(dist) - n)
            if start < 0 or start + n > len(dist):
                return None
            if self.mode == "stage" and abs(xbar[start + half] - self.columns[self._col]) > STAGE_SHIFT_TOL:
                return None
            return [f[..., start:start + n] for f in fields], float(dist[start + half]), (start, start + n)
        length = n * cell
        lo = dist[j] - (n // 2) * cell
        if self.mode == "stage":
            lo = min(max(lo, dist[0]), dist[-1] - (n - 1) * cell)
        positions = lo + np.arange(n) * cell
        if positions[0] < dist[0] - 1e-9 or positions[-1] > dist[-1] + 1e-9 or length > dist[-1] - dist[0] + cell:
            return None
        center = positions[n // 2]
        if self.mode == "stage" and abs(np.interp(center, dist, xbar) - self.columns[self._col]) > STAGE_SHIFT_TOL:
            return None
        idx = np.clip(np.searchsorted(dist, positions) - 1, 0, len(dist) - 2)
        w = ((positions - dist[idx]) / (dist[idx + 1] - dist[idx]))
        out = [f[..., idx] * (1 - w) + f[..., idx + 1] * w for f in fields]
        return out, float(center), (idx[0], idx[-1] + 1)

    # -- accumulation ------------------------------------------------------- #
    def _sums(self, binner, truth, pred):
        def spectrum(x):
            x = (x - x.mean(axis=(0, 1), keepdims=True)) * self.window
            return np.fft.fftn(np.fft.rfft(x, axis=2), axes=(0, 1))

        def binned(x):
            flat = x.reshape(-1, x.shape[2])[binner.valid][:, 1:]        # (modes, n_kpar)  DC dropped
            return binner.matrix.T @ flat                                 # (n_kperp, n_kpar)

        tf, pf = spectrum(truth), spectrum(pred)
        return (binned(np.abs(tf) ** 2).T, binned(np.abs(pf) ** 2).T,
                binned(np.real(tf * np.conj(pf))).T)                     # (n_kpar, n_kperp)

    def update(self, sample, prediction, aux=None, runtime=None):
        truth = self.target(sample)
        pred = self.prediction(prediction)
        nx, ny, nz = truth.shape
        box = self.box_mpc(sample)
        cell = box / nx
        z = self.z_grid(sample)
        if z is None:
            z_sorted, dist = None, np.arange(nz) * cell
        else:
            dist = comoving_distance_mpc(z, self.settings.omega_m, self.settings.hubble_h)
            if np.any(np.diff(dist) <= 0):
                raise ValueError(f"cylindrical: z_grid of sample {sample.id!r} must increase strictly")
            z_sorted = z
        spacing = np.diff(dist)
        self._uniform = bool(np.all(np.abs(spacing / cell - 1) <= UNIFORM_TOL))
        cell_los = float(spacing.mean()) if self._uniform else cell
        binner = self._setup(nx, ny, box, cell_los)

        xhi = true_xhi(sample, self.settings.target)
        if self.mode == "stage" and xhi is None:
            raise ValueError("cylindrical: stage chunks need the true x_HI (an 'xhi' target or input); "
                             "set cyl_z_centers for fixed-redshift chunks")
        xbar = None if xhi is None else xhi.mean(axis=(0, 1))
        if self.mode == "z" and z_sorted is None:
            raise ValueError("cylindrical: fixed-redshift chunks need the sample's z_grid")

        n_kpar, n_kperp = self.kpar_edges.size - 1, binner.centers.size
        ratio = np.full((len(self.columns), n_kpar, n_kperp), np.nan)
        coherence = np.full_like(ratio, np.nan)
        zc = np.full(len(self.columns), np.nan)
        xc = np.full(len(self.columns), np.nan)
        kpar_max = np.full(len(self.columns), np.nan)
        have = np.zeros(len(self.columns), bool)
        for col, j in enumerate(self._centers(z_sorted, xbar)):
            if j is None:
                continue
            self._col = col
            got = self._chunk([truth, pred], dist, j, xbar)
            if got is None:
                continue
            (t, p), center_d, where = got
            have[col] = True
            pt, pp, cross = self._sums(binner, t, p)
            if t.std(axis=(0, 1)).mean() > MIN_TRUTH_STD:
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio[col] = np.where(pt > 0, pp / pt, np.nan)
                    coherence[col] = np.where(pt * pp > 0, cross / np.sqrt(pt * pp), np.nan)
            z_center = float(np.interp(center_d, dist, z_sorted)) if z_sorted is not None else np.nan
            zc[col] = z_center
            xc[col] = float(np.interp(center_d, dist, xbar)) if xbar is not None else np.nan
            limit = self._native_kpar(z_center) if np.isfinite(z_center) else np.inf
            if not self._uniform:
                lo, hi = where
                coarsest = float(np.max(spacing[max(lo - 1, 0):min(hi + 1, len(spacing))]))
                limit = min(limit, np.pi / coarsest)
            kpar_max[col] = limit
        self.ratio.append(ratio.astype(np.float32))
        self.coherence.append(coherence.astype(np.float32))
        self.chunk_z.append(zc)
        self.chunk_xbar.append(xc)
        self.kpar_max.append(kpar_max)
        self.have.append(have)
        self.ids.append(str(sample.id))

    # -- results ------------------------------------------------------------ #
    def _medians(self):
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            ratio = np.nanmedian(np.stack(self.ratio), axis=0)
            coherence = np.nanmedian(np.stack(self.coherence), axis=0)
            kpar_max = np.nanmedian(np.stack(self.kpar_max), axis=0)
            z_med = np.nanmedian(np.stack(self.chunk_z), axis=0)
        return ratio, coherence, kpar_max, z_med

    def resolved(self, kpar_max_col: float) -> np.ndarray:
        kperp_c = self._binner.centers
        kpar_c = np.sqrt(self.kpar_edges[1:] * self.kpar_edges[:-1])
        return (kperp_c[None, :] <= self.kperp_max) & (kpar_c[:, None] <= kpar_max_col)

    def result(self):
        if not self.ratio:
            return {}
        ratio, coherence, kpar_max, z_med = self._medians()
        n_chunks = np.stack(self.have).sum(axis=0)
        dead = self.settings.cyl_dead_band
        out = {"mode": self.mode, "chunk_cells": self.n, "chunk_mpc": self.n * self.cell_los,
               "dead_band": dead, "stages": []}
        shares = []
        for col, value in enumerate(self.columns):
            err = ratio[col] - 1.0
            mask = self.resolved(kpar_max[col] if np.isfinite(kpar_max[col]) else np.inf) & np.isfinite(err)
            entry = {"label": column_label(self.mode, value), "value": value, "n_samples": int(n_chunks[col]),
                     "z_median": float(z_med[col]) if np.isfinite(z_med[col]) else None}
            if mask.any():
                share = float(np.mean(np.abs(err[mask]) <= dead))
                shares.append(share)
                entry.update(within_dead_band=share,
                             median_abs_error=float(np.median(np.abs(err[mask]))),
                             median_coherence=float(np.nanmedian(coherence[col][mask])),
                             resolved_cells=int(mask.sum()))
            out["stages"].append(entry)
        if shares:
            out["within_dead_band_mean"] = float(np.mean(shares))
        out["kperp_max"] = finite_or_none([self.kperp_max])[0]
        return out

    def arrays(self):
        if not self.ratio:
            return {}
        ratio, coherence, kpar_max, z_med = self._medians()
        return {
            "cyl_mode": np.array(self.mode),
            "cyl_columns": np.array(self.columns, float),
            "cyl_kperp_edges": self._binner.edges,
            "cyl_kpar_edges": self.kpar_edges,
            "cyl_kperp_max": np.array(self.kperp_max, float),
            "cyl_ids": np.array(self.ids),
            "cyl_ratio": np.stack(self.ratio),               # (samples, columns, k_par, k_perp)
            "cyl_coherence": np.stack(self.coherence),
            "cyl_chunk_z": np.stack(self.chunk_z),           # (samples, columns)
            "cyl_chunk_xbar": np.stack(self.chunk_xbar),
            "cyl_kpar_max": np.stack(self.kpar_max),
            "cyl_have": np.stack(self.have),
            "cyl_ratio_median": ratio,                       # (columns, k_par, k_perp)
            "cyl_coherence_median": coherence,
        }
