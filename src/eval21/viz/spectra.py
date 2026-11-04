"""Power-spectrum figures: transverse P(k) per stage, cylindrical coherence maps,
coherence vs small-scale power planes.

Ports of fno-21cm's ``viz/power_spectrum_evaluation.py``, ``viz/ps_coherence_highk.py`` and
``viz/cyl_anisotropy_decomposition.py`` and of rhizome-neural-operator's ``ebm21cm/ps_plane3d.py``
(same layouts, axes, colour maps, colour limits, reference lines and labels). One file per target
(``<figure>_xhi``, ``<figure>_tb``); every model of the catalog is drawn (catalog labels, colours
and line styles; the first model is the reference). Per-sample curves are pooled over the groups
of a target and summarized by the median and the 16-84 % band, as the originals did over cones.

============================  ===============================================  ==============================================  ======
figure                        ports                                            shows                                           needs
============================  ===============================================  ==============================================  ======
ps_overlay                    power_spectrum_evaluation.plot_overlay           active-stage Delta^2(k_perp) truth vs models,    curves
                              (``ps_overlay.png``)                             P_pred/P_true (log y, grey +-5 % band) and
                                                                               coherence r(k_perp) (label: k where r < 0.9)
ps_stage_curves               power_spectrum_evaluation.plot_stage_curves      per model (rows): ratio(k) and r(k) per          curves
                              (``ps_stage_curves.png``)                        reionization stage (viridis)
ps_cyl_r                      power_spectrum_evaluation.plot_cylindrical_maps  coherence r(k_perp, k_par), viridis,             curves
                              (``ps_cyl_r.png``)                               vmin = min(0.9, 2nd pct), vmax = 1
cyl_anisotropy_decomposition  cyl_anisotropy_decomposition.main                amplitude |log10 ratio| and decoherence 1 - r:   curves
                                                                               marginals over k_perp / k_par per model and the
                                                                               focus model's advantage map over the worst
                                                                               other model, with the two-way variance split
                                                                               (k_par / k_perp / residual); + csv
ps_coherence_highk            ps_coherence_highk.main                          r(k) and P_pred/P_true vs k in cycles/pixel      curves
                                                                               (dotted: highk cutoff) and the plane mean
                                                                               r(k >= cutoff) vs highk log-MSE; + csv
ps_plane3d                    ebm21cm.ps_plane3d.main                          small-scale coherence r(k > 1/Mpc) vs            curves
                                                                               |P_pred/P_true - 1| (k > 1/Mpc), active stage,
                                                                               one point per model; + csv
============================  ===============================================  ==============================================  ======

Data. Transverse curves come from the ``spectrum`` metric (``curves.npz``: per-sample ratio,
coherence and Delta^2 per stage, ``spectrum_highk_logmse``). Cylindrical maps come from the
``cylindrical`` metric (per-sample ``cyl_ratio`` / ``cyl_coherence``; columns are its chunks,
at ionization stages or fixed redshifts -- ``cyl_z_centers=7,9,11`` with
``cyl_chunk_cells=32`` reproduces the fno-21cm chunks); ``--opt cyl_source=anisotropy`` uses
the ``anisotropy`` metric's fixed-z median ratio maps instead (ratio only: that metric does not
export coherence maps). The ``cyl`` figure (``viz/cylindrical.py``, P_pred/P_true - 1 with a
white dead band) is a different look and stays separate.

Generative runs (``--mode generative``) drop the coherence from ``results.json``; the coherence
curves, maps and points of such models are left out.

Options (``--opt key=value``): ``cyl_source`` (cylindrical | anisotropy), ``model`` (focus model
of cyl_anisotropy_decomposition, default: the first), ``zbin`` (its chunk column, default: the
middle one), ``highk_k_min`` (cycles/pixel, default: the run's ``highk_k_min``), ``plane_kmin``
(1/Mpc, default 1), ``note`` and ``legend_loc`` (footnote / legend place of ps_plane3d).

Not ported: fno-21cm ``viz/los_grid_evaluation.py`` (a round-trip test of LOS sampling grids for
the cube cache on native 2340-slice lightcones; it involves no model and writes only tables).
"""
from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np

from .common import (field_name, med_band, model_handles, pane_values, pyplot, save, shared_legend, style_key,
                     wrap_label)
from .registry import SkipFigure, figure

K_PERP = r"$k_\perp$  [Mpc$^{-1}$]"
K_PAR = r"$k_\parallel$  [Mpc$^{-1}$]"
RATIO = r"$P_{\rm pred}/P_{\rm true}$"
DPI = 150

# ps_coherence_highk / ps_plane3d look
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e3de"


# --------------------------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------------------------- #
def _nanmedian(a, axis=0):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.asarray(a, float), axis=axis)

def _finite(x, y):
    """Drop empty bins (NaN) so curves stay connected, as in the originals."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]



def _token(label: str) -> str:
    return label.replace(" ", "_").replace("-", "_").replace(".", "p")


def pretty_stage(label: str) -> str:
    """``xbar_HI 0.02-0.20`` -> ``x̄_HI 0.02–0.20`` (the original's stage labels)."""
    return label.replace("xbar_HI", "x̄_HI").replace("-", "–")


def _coherence_dropped(metric: dict) -> bool:
    """True when a generative run stripped the coherence from this metric's results."""
    return any("coherence" in key for key in (metric.get("_generative_dropped") or []))


def _has_coherence(metric: dict) -> bool:
    if _coherence_dropped(metric):
        return False
    stages = metric.get("stages") or [{}]
    return "coherence_med" in stages[-1]


class Spectrum:
    """Stage-resolved transverse spectra of one (target, model), pooled over the groups."""

    def __init__(self, cat, target, model):
        runs = cat.runs_for(target, model)
        metrics = [run.metric(model, "spectrum") for run in runs]
        if not all(metrics) or not all("stage_labels" in m for m in metrics):
            raise SkipFigure(f"no 'spectrum' results for target {target!r}, model {model!r}")
        self.labels = list(metrics[0]["stage_labels"])
        self.settings = runs[0].settings
        self.coherence = all(_has_coherence(m) for m in metrics)
        curves = [run.curves(model) for run in runs]
        ks = [c.get("spectrum_k") for c in curves]
        tokens = [_token(lab) for lab in self.labels]
        have = all(k is not None for k in ks) and all(
            f"spectrum_ratio_{t}" in c for c in curves for t in tokens)
        if have:
            for k in ks[1:]:
                if np.shape(k) != np.shape(ks[0]) or not np.allclose(k, ks[0], rtol=1e-6):
                    raise SkipFigure(f"spectrum k bins differ between the groups of target {target!r}")
            self.k = np.asarray(ks[0], float)

            def stack(name):
                return np.stack([cat.pooled(model, f"spectrum_{name}_{t}", target) for t in tokens], 1)

            ratio = stack("ratio")                                   # (n, S, K)
            thin = np.all(~np.isfinite(ratio), axis=2, keepdims=True)
            r = stack("coherence")
            d2t, d2p = stack("delta2_truth"), stack("delta2_pred")
            d2t = np.where(thin, np.nan, d2t)
            d2p = np.where(thin, np.nan, d2p)
            self.n = ratio.shape[0]
            self.ratio = med_band(ratio, 0)
            self.r = med_band(r, 0)
            self.d2_truth = _nanmedian(d2t, 0)
            self.d2_pred = _nanmedian(d2p, 0)
            highk = cat.pooled(model, "spectrum_highk_logmse", target)
            self.highk = _nanmedian(highk, 0) if highk.size else np.full(len(self.labels), np.nan)
        else:
            if len(metrics) > 1:
                raise SkipFigure(f"per-sample spectrum curves missing for target {target!r}; "
                                 "cannot pool several groups")
            m = metrics[0]
            self.k = np.asarray(m["k_centers"], float)
            stages = m["stages"]

            def arr(key):
                return np.array([[np.nan if v is None else v for v in s.get(key) or [None] * len(self.k)]
                                 for s in stages], float)

            self.n = int(m.get("n_samples", 0))
            self.ratio = (arr("power_ratio_med"), arr("power_ratio_p16"), arr("power_ratio_p84"))
            self.r = (arr("coherence_med"), arr("coherence_p16"), arr("coherence_p84"))
            self.d2_truth = arr("delta2_truth_med")
            self.d2_pred = arr("delta2_pred_med")
            self.highk = np.array([np.nan if s.get("highk_logmse") is None else s["highk_logmse"]
                                   for s in stages], float)
        if not self.coherence:
            nan = np.full_like(self.r[0], np.nan)
            self.r = (nan, nan, nan)

    @property
    def k_nyquist(self) -> float:
        """Upper edge of the log k bins (the transverse Nyquist pi N / L)."""
        return float(self.k[-1] * np.sqrt(self.k[1] / self.k[0]))


def _spectra(cat, target):
    if not any(run.metric(m, "spectrum") for m in cat.models for run in cat.runs_for(target, m)):
        return None
    return {m: Spectrum(cat, target, m) for m in cat.models}


def k_crossing(k, curve, level):
    """Smallest k where ``curve`` first drops below ``level`` (log-interp), as the original."""
    m = np.isfinite(curve)
    k, c = np.asarray(k)[m], np.asarray(curve)[m]
    if len(k) == 0:
        return np.nan
    below = np.flatnonzero(c < level)
    if below.size == 0:
        return np.nan
    i = int(below[0])
    if i == 0:
        return float(k[0])
    f = (level - c[i - 1]) / (c[i] - c[i - 1])
    return float(np.exp(np.log(k[i - 1]) + f * (np.log(k[i]) - np.log(k[i - 1]))))


def _d2_label(target):
    sub = {"xhi": r"x_{\rm HI}", "tb": r"T_b"}.get(target, target)
    unit = "  [mK$^2$]" if target == "tb" else ""
    return rf"$\Delta^2_{{{sub}}}(k_\perp)$" + unit


# --------------------------------------------------------------------------------------------- #
# ps_overlay, ps_stage_curves
# --------------------------------------------------------------------------------------------- #
@figure("ps_overlay", group="spectra",
        help="active-stage Delta^2(k), P_pred/P_true and r(k) per model (fno-21cm ps_overlay)")
def ps_overlay(ctx):
    cat = ctx.catalog
    plt = pyplot()
    written = []
    for target in cat.targets:
        spec = _spectra(cat, target)
        if spec is None:
            continue
        first = spec[cat.models[0]]
        k = first.k
        level = float(first.settings.get("coherence_level", 0.9))
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

        ax = axes[0]
        ax.loglog(k, first.d2_truth[-1], "k-", lw=2.2, label="truth")
        for m in cat.models:
            ax.loglog(k, spec[m].d2_pred[-1], color=cat.color(m), ls=cat.ls(m), label=cat.label(m))
        ax.set_xlabel(K_PERP)
        ax.set_ylabel(_d2_label(target))
        ax.set_title("Transverse power (active slices, cone median)")

        ax = axes[1]
        ax.axhspan(0.95, 1.05, color="grey", alpha=0.18, lw=0)
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        for m in cat.models:
            med, lo, hi = (a[-1] for a in spec[m].ratio)
            ax.plot(*_finite(k, med), color=cat.color(m), ls=cat.ls(m), label=cat.label(m))
            ax.fill_between(k, lo, hi, color=cat.color(m), alpha=0.18, lw=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(K_PERP)
        ax.set_ylabel(RATIO)
        ax.set_title("Power ratio (grey band: ±5%)")

        ax = axes[2]
        ax.axhline(level, color="grey", lw=0.8, ls="--")
        crossings = []
        for m in cat.models:
            if not spec[m].coherence:
                continue
            med, lo, hi = (a[-1] for a in spec[m].r)
            crossings.append((cat.color(m), cat.ls(m), k_crossing(k, med, level)))
            ax.plot(*_finite(k, med), color=cat.color(m), ls=cat.ls(m), label=cat.label(m))
            ax.fill_between(k, lo, hi, color=cat.color(m), alpha=0.18, lw=0)
        pane_values(ax, crossings, label=f"k(r < {level:g}) [Mpc⁻¹]", fmt="{:.2f}")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(K_PERP)
        ax.set_ylabel(r"$r(k_\perp)$")
        ax.set_title("Cross-correlation (phase fidelity)")
        if not ax.get_legend_handles_labels()[0]:
            ax.set_xlim(axes[1].get_xlim())
            ax.text(0.5, 0.5, "no coherence (generative runs)", ha="center", va="center",
                    transform=ax.transAxes, color="0.4")
        fig.tight_layout()
        shared_legend(fig, handles=model_handles(cat),
                      extra=style_key([("16–84 % over cones", {"kind": "band", "color": "0.6", "alpha": 0.35})]))
        written.append(save(fig, ctx.out, f"ps_overlay_{target}", dpi=DPI, quiet=ctx.quiet))
    if not written:
        raise SkipFigure("no 'spectrum' results")
    return written


@figure("ps_stage_curves", group="spectra",
        help="per model: P_pred/P_true and r(k) per reionization stage (fno-21cm ps_stage_curves)")
def ps_stage_curves(ctx):
    cat = ctx.catalog
    plt = pyplot()
    written = []
    for target in cat.targets:
        spec = _spectra(cat, target)
        if spec is None:
            continue
        first = spec[cat.models[0]]
        k = first.k
        level = float(first.settings.get("coherence_level", 0.9))
        labels = [pretty_stage(s) for s in first.labels[:-1]]          # exclude "active"
        cmap = plt.get_cmap("viridis")
        n = len(cat.models)
        fig, axes = plt.subplots(n, 2, figsize=(12, 3.6 * n), squeeze=False)
        for row, m in enumerate(cat.models):
            for col, (data, ylabel, ref) in enumerate([
                    (spec[m].ratio[0], RATIO, 1.0),
                    (spec[m].r[0], r"$r(k_\perp)$", level)]):
                ax = axes[row][col]
                ax.axhline(ref, color="grey", lw=0.8, ls="--")
                for s, label in enumerate(labels):
                    ax.plot(*_finite(k, data[s]), color=cmap(s / max(len(labels) - 1, 1)), label=label)
                ax.set_xscale("log")
                if col == 0:
                    if np.any(np.isfinite(data[:-1]) & (data[:-1] > 0)):
                        ax.set_yscale("log")
                else:
                    ax.set_ylim(0, 1.05)
                    if not spec[m].coherence:
                        ax.text(0.5, 0.5, "no coherence (generative run)", ha="center", va="center",
                                transform=ax.transAxes, color="0.4")
                ax.set_xlabel(K_PERP)
                ax.set_ylabel(ylabel)
                ax.set_title(wrap_label(f"{cat.label(m)} by reionization stage", 60))
        fig.tight_layout()
        shared_legend(fig, axes=axes[0])                 # the stage colours, once
        written.append(save(fig, ctx.out, f"ps_stage_curves_{target}", dpi=DPI, quiet=ctx.quiet))
    if not written:
        raise SkipFigure("no 'spectrum' results")
    return written


# --------------------------------------------------------------------------------------------- #
# cylindrical maps
# --------------------------------------------------------------------------------------------- #
def _close(a, b, rtol):
    return np.shape(a) == np.shape(b) and np.allclose(a, b, rtol=rtol, atol=0, equal_nan=True)


def _geometric_edges(centers):
    c = np.asarray(centers, float)
    q = np.sqrt(c[1] / c[0])
    return np.concatenate([c / q, [c[-1] * q]])


def _linear_edges(centers):
    c = np.asarray(centers, float)
    d = float(c[1] - c[0]) if c.size > 1 else float(c[0])
    return np.concatenate([[c[0] - d / 2], c + d / 2])


def cyl_maps(cat, target, source=None):
    """Median cylindrical maps per model, pooled over groups.

    Returns ``{"kperp_edges", "kperp", "columns": [{"title", "kpar_edges", "kpar", "n",
    "ratio": {m: map}, "r": {m: map or None}}], "source"}`` or None (no data). Maps are
    (n_kpar, n_kperp); empty columns are dropped.
    """
    models = cat.models
    curves = {m: [run.curves(m) for run in cat.runs_for(target, m)] for m in models}
    has_cyl = all("cyl_ratio" in c for m in models for c in curves[m])
    has_aniso = all(any(k.startswith("anisotropy_ratio_z") for k in c) for m in models for c in curves[m])
    source = source or ("cylindrical" if has_cyl else "anisotropy" if has_aniso else None)
    if source == "cylindrical" and has_cyl:
        return _cyl_from_cylindrical(cat, target, curves)
    if source == "anisotropy" and has_aniso:
        return _cyl_from_anisotropy(cat, target, curves)
    return None


def _cyl_from_cylindrical(cat, target, curves):
    models = cat.models
    all_c = [c for m in models for c in curves[m]]
    ref = all_c[0]
    for key, rtol in (("cyl_kperp_edges", 1e-9), ("cyl_kpar_edges", 2e-3), ("cyl_columns", 1e-9)):
        if not all(_close(c[key], ref[key], rtol) for c in all_c[1:]):
            raise SkipFigure(f"cylindrical {key} differs between runs/models of target {target!r}")
    modes = {str(c["cyl_mode"]) for c in all_c}
    if len(modes) != 1:
        raise SkipFigure(f"runs of target {target!r} mix cylindrical chunk modes {sorted(modes)}")
    mode = modes.pop()
    kperp_edges = np.asarray(ref["cyl_kperp_edges"], float)
    kpar_edges = np.asarray(ref["cyl_kpar_edges"], float)
    m0 = models[0]

    def pooled(m, key):
        return np.concatenate([np.asarray(c[key], float) for c in curves[m]], 0)

    ratio = {m: _nanmedian(pooled(m, "cyl_ratio"), 0) for m in models}
    coherent = {m: not any(_coherence_dropped(run.metric(m, "cylindrical")) for run in cat.runs_for(target, m))
                for m in models}
    coh = {m: _nanmedian(pooled(m, "cyl_coherence"), 0) if coherent[m] else None for m in models}
    zc = _nanmedian(pooled(m0, "cyl_chunk_z"), 0)
    have = pooled(m0, "cyl_have").sum(0)
    columns = []
    for col, value in enumerate(np.asarray(ref["cyl_columns"], float)):
        if not any(np.isfinite(ratio[m][col]).any() for m in models):
            continue
        if mode == "z":
            title = f"z ≈ {round(float(value), 1):g}"
        else:
            title = r"$\bar{x}_{\rm HI}$" + f" ≈ {value:g} (z ≈ {zc[col]:.1f})"
        columns.append({"title": title, "kpar_edges": kpar_edges,
                        "kpar": 0.5 * (kpar_edges[:-1] + kpar_edges[1:]), "n": int(have[col]),
                        "ratio": {m: ratio[m][col] for m in models},
                        "r": {m: None if coh[m] is None else coh[m][col] for m in models}})
    return {"kperp_edges": kperp_edges, "kperp": np.sqrt(kperp_edges[:-1] * kperp_edges[1:]),
            "columns": columns, "source": "cylindrical"}


def _cyl_from_anisotropy(cat, target, curves):
    models = cat.models
    ref = curves[models[0]][0]
    kperp = np.asarray(ref["anisotropy_kperp"], float)
    names = sorted((k for k in ref if k.startswith("anisotropy_ratio_z")),
                   key=lambda k: float(k[len("anisotropy_ratio_z"):]))
    columns = []
    for i, _ in enumerate(names):
        maps, kpar = {}, None
        for m in models:
            per_group = []
            for c in curves[m]:
                keys = sorted((k for k in c if k.startswith("anisotropy_ratio_z")),
                              key=lambda k: float(k[len("anisotropy_ratio_z"):]))
                if len(keys) != len(names):
                    raise SkipFigure(f"anisotropy chunks differ between the groups of target {target!r}")
                per_group.append(np.asarray(c[keys[i]], float))
                kp = np.asarray(c["anisotropy_kpar_z" + keys[i][len("anisotropy_ratio_z"):]], float)
                if kpar is None:
                    kpar = kp
                elif not _close(kp, kpar, 2e-3):
                    raise SkipFigure(f"anisotropy k_par differs between runs of target {target!r}")
            maps[m] = _nanmedian(np.stack(per_group), 0)
        zs = [float(k[len("anisotropy_ratio_z"):]) for c in curves[models[0]]
              for k in [sorted((k for k in c if k.startswith("anisotropy_ratio_z")),
                               key=lambda k: float(k[len("anisotropy_ratio_z"):]))[i]]]
        n = sum(int(run.metric(models[0], "anisotropy").get("n_samples", 0))
                for run in cat.runs_for(target, models[0]))
        title = f"z ≈ {round(float(np.mean(zs)), 1):g}"
        columns.append({"title": title, "kpar_edges": _linear_edges(kpar), "kpar": kpar, "n": n,
                        "ratio": maps, "r": {m: None for m in models}})
    return {"kperp_edges": _geometric_edges(kperp), "kperp": kperp, "columns": columns,
            "source": "anisotropy"}


def _cyl_or_skip(ctx, target):
    source = ctx.options.get("cyl_source")
    if source not in (None, "cylindrical", "anisotropy"):
        raise SystemExit(f"cyl_source must be 'cylindrical' or 'anisotropy', got {source!r}")
    data = cyl_maps(ctx.catalog, target, source)
    if data is None or not data["columns"]:
        return None
    return data


@figure("ps_cyl_r", group="spectra",
        help="coherence r(k_perp, k_par), rows = models (fno-21cm ps_cyl_r)")
def ps_cyl_r(ctx):
    """Coherence r(k_perp, k_par): rows = models, columns = chunks (power errors: the `cyl` figure)."""
    cat = ctx.catalog
    plt = pyplot()
    written, reasons = [], []
    for target in cat.targets:
        data = _cyl_or_skip(ctx, target)
        if data is None:
            reasons.append(f"{target}: no cylindrical/anisotropy maps")
            continue
        cols = data["columns"]
        models = [m for m in cat.models if all(c["r"][m] is not None for c in cols)]
        if not models:
            reasons.append(f"{target}: no coherence maps ({data['source']} source or generative runs)")
            continue
        vals = np.concatenate([c["r"][m][np.isfinite(c["r"][m])] for c in cols for m in models])
        if not vals.size:
            reasons.append(f"{target}: coherence maps are empty")
            continue
        cmap = plt.get_cmap("viridis").with_extremes(bad="0.85")
        vmin = min(0.9, float(np.percentile(vals, 2)))
        n_col = len(cols)
        fig, axes = plt.subplots(len(models), n_col, figsize=(4.0 * n_col + 0.6, 3.5 * len(models) + 0.4),
                                 squeeze=False)
        for row, m in enumerate(models):
            for col, c in enumerate(cols):
                ax = axes[row][col]
                pcm = ax.pcolormesh(data["kperp_edges"], c["kpar_edges"], np.ma.masked_invalid(c["r"][m]),
                                    cmap=cmap, vmin=vmin, vmax=1.0)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_box_aspect(1)                      # square (k_perp, k_par) panels
                ax.set_xlabel(K_PERP)
                # the model names the row (first column), the chunk the column (first row)
                ax.set_ylabel(f"{wrap_label(cat.label(m), 24)}\n{K_PAR}" if col == 0 else K_PAR)
                if row == 0:
                    ax.set_title(c["title"])
                fig.colorbar(pcm, ax=ax, pad=0.02, label=r"$r(k_\perp, k_\parallel)$", shrink=0.85)
        fig.tight_layout()
        written.append(save(fig, ctx.out, f"ps_cyl_r_{target}", dpi=DPI, quiet=ctx.quiet))
    if not written:
        raise SkipFigure("; ".join(reasons) or "no cylindrical results")
    return written


# --------------------------------------------------------------------------------------------- #
# cyl_anisotropy_decomposition
# --------------------------------------------------------------------------------------------- #
def error_channels(ratio, coherence):
    """(n_kpar, n_kperp) error maps, lower is better (as the original)."""
    ratio = np.asarray(ratio, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {"amplitude": np.abs(np.log10(np.where(ratio > 0, ratio, np.nan)))}
    if coherence is not None:
        out["decoherence"] = 1.0 - np.asarray(coherence, float)
    return out


def two_way_decomposition(adv):
    """Variance share of an advantage map attributable to k_par vs k_perp (original's ANOVA split)."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(adv)
        a = np.nanmean(adv, axis=1) - mu
        b = np.nanmean(adv, axis=0) - mu
        resid = adv - (mu + a[:, None] + b[None, :])
        ss_par = adv.shape[1] * np.nansum(a ** 2)
        ss_perp = adv.shape[0] * np.nansum(b ** 2)
        ss_res = np.nansum(resid ** 2)
    total = ss_par + ss_perp + ss_res
    if not total > 0:
        return {"par_share": np.nan, "perp_share": np.nan, "resid_share": np.nan}
    return {"par_share": ss_par / total, "perp_share": ss_perp / total, "resid_share": ss_res / total}


def _nanmean(a, axis=None):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


@figure("cyl_anisotropy_decomposition", group="spectra",
        help="cylindrical error split into k_par / k_perp marginals + advantage map (fno-21cm)")
def cyl_anisotropy_decomposition(ctx):
    cat = ctx.catalog
    plt = pyplot()
    if len(cat.models) < 2:
        raise SkipFigure("needs at least two models")
    focus = str(ctx.options.get("model", cat.models[0]))
    if focus not in cat.models:
        raise SystemExit(f"cyl_anisotropy_decomposition: model {focus!r} not in {cat.models}")
    others = [m for m in cat.models if m != focus]
    written, reasons = [], []
    for target in cat.targets:
        data = _cyl_or_skip(ctx, target)
        if data is None:
            reasons.append(f"{target}: no cylindrical/anisotropy maps")
            continue
        cols = data["columns"]
        kperp = data["kperp"]
        with_r = all(c["r"][m] is not None for c in cols for m in cat.models)
        errs = [{m: error_channels(c["ratio"][m], c["r"][m] if with_r else None) for m in cat.models}
                for c in cols]
        channels = ["amplitude"] + (["decoherence"] if with_r else [])

        rows = []
        for channel in channels:
            for other in others:
                for zi, c in enumerate(cols):
                    adv = errs[zi][other][channel] - errs[zi][focus][channel]
                    dec = two_way_decomposition(adv)
                    rows.append({"channel": channel, "model": focus, "reference": other,
                                 "column": c["title"].replace(r"$\bar{x}_{\rm HI}$", "xbar_HI"),
                                 "mean_advantage": float(_nanmean(adv)),
                                 "adv_lowest_kpar": float(_nanmean(adv[0])),
                                 "adv_highest_kpar": float(_nanmean(adv[-1])),
                                 "adv_lowest_kperp": float(_nanmean(adv[:, 0])),
                                 "adv_highest_kperp": float(_nanmean(adv[:, -1])),
                                 **{k: float(v) for k, v in dec.items()}})
        ctx.out.mkdir(parents=True, exist_ok=True)
        with open(ctx.out / f"cyl_anisotropy_decomposition_{target}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            for row in rows:
                w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})

        zi = int(ctx.options.get("zbin", len(cols) // 2))
        zi = min(max(zi, 0), len(cols) - 1)
        c = cols[zi]
        kpar = c["kpar"]
        fig, axes = plt.subplots(len(channels), 3, figsize=(15.5, 4.2 * len(channels)), squeeze=False)
        for ri, channel in enumerate(channels):
            me = errs[zi][focus][channel]
            ax_par, ax_perp, ax_map = axes[ri]
            for name in [focus] + others:
                e = errs[zi][name][channel]
                style = dict(lw=2.6, color=cat.color(name), zorder=5) if name == focus else \
                    dict(lw=1.3, color=cat.color(name), ls=cat.ls(name), alpha=0.85)
                ax_par.plot(kpar, _nanmean(e, axis=1), label=cat.label(name), **style)
                ax_perp.plot(kperp, _nanmean(e, axis=0), label=cat.label(name), **style)
            for ax, xlab in ((ax_par, K_PAR), (ax_perp, K_PERP)):
                ax.set_xscale("log")
                ax.set_xlabel(xlab)
                ax.grid(alpha=0.25)
            ylab = (r"$|\log_{10} P_{\rm pred}/P_{\rm truth}|$" if channel == "amplitude" else r"$1-r(k)$")
            ax_par.set_ylabel(f"{channel}\n{ylab}")
            ax_par.set_title(r"marginal over $k_\perp$", fontsize=10)
            ax_perp.set_title(r"marginal over $k_\parallel$", fontsize=10)

            worst = max(others, key=lambda n: np.nan_to_num(_nanmean(errs[zi][n][channel]), nan=-np.inf))
            adv = errs[zi][worst][channel] - me
            vmax = float(np.nanmax(np.abs(adv))) if np.isfinite(adv).any() else 1.0
            vmax = vmax or 1.0
            cmap = plt.get_cmap("RdBu_r").with_extremes(bad="0.85")
            im = ax_map.pcolormesh(kperp, kpar, np.ma.masked_invalid(adv), cmap=cmap,
                                   vmin=-vmax, vmax=vmax, shading="nearest")
            ax_map.set_xscale("log")
            ax_map.set_yscale("log")
            ax_map.set_box_aspect(1)                      # square (k_perp, k_par) map
            ax_map.set_xlabel(r"$k_\perp$")
            ax_map.set_ylabel(r"$k_\parallel$")
            dec = two_way_decomposition(adv)
            ax_map.set_title(
                f"advantage over {wrap_label(cat.label(worst), 34)}\n"
                f"$k_\\parallel$ {100 * dec['par_share']:.0f}%  "
                f"$k_\\perp$ {100 * dec['perp_share']:.0f}%  "
                f"resid {100 * dec['resid_share']:.0f}%", fontsize=9)
            fig.colorbar(im, ax=ax_map, label="error reduction")
        fig.suptitle(f"Cylindrical error split, {field_name(target)}  —  {c['title']}, {c['n']} cones",
                     fontsize=13)
        fig.tight_layout()
        shared_legend(fig, axes=[axes[0, 0]], fontsize=8.5)
        written.append(save(fig, ctx.out, f"cyl_anisotropy_decomposition_{target}", dpi=DPI,
                            quiet=ctx.quiet))
    if not written:
        raise SkipFigure("; ".join(reasons) or "no cylindrical results")
    return written


# --------------------------------------------------------------------------------------------- #
# ps_coherence_highk, ps_plane3d
# --------------------------------------------------------------------------------------------- #
def _surface(fig, axes):
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)


def coherence_scale(centers, coherence, threshold=0.9):
    """First k where r drops below the threshold (linear interp); NaN if it never does."""
    m = np.isfinite(coherence)
    centers, coherence = np.asarray(centers)[m], np.asarray(coherence)[m]
    below = np.where(coherence < threshold)[0]
    if below.size == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return float(centers[0])
    x0, x1 = centers[i - 1], centers[i]
    y0, y1 = coherence[i - 1], coherence[i]
    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x1)


@figure("ps_coherence_highk", group="spectra",
        help="coherence r(k) vs small-scale power fidelity, with the placement/amount plane (fno-21cm)")
def ps_coherence_highk(ctx):
    cat = ctx.catalog
    plt = pyplot()
    written = []
    for target in cat.targets:
        spec = _spectra(cat, target)
        if spec is None:
            continue
        first = spec[cat.models[0]]
        k_min = float(ctx.options.get("highk_k_min", first.settings.get("highk_k_min", 0.2)))
        level = float(first.settings.get("coherence_level", 0.9))
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
        _surface(fig, axes)
        rows = []
        for i, m in enumerate(cat.models):
            s = spec[m]
            centers = s.k / (2.0 * s.k_nyquist)            # 1/Mpc -> cycles / pixel
            coh, ratio = s.r[0][-1], s.ratio[0][-1]
            band = centers >= k_min
            coh_hk = float(_nanmean(coh[band])) if band.any() else float("nan")
            hk = float(s.highk[-1])
            colour = cat.color(m)
            if s.coherence:
                axes[0].plot(centers, coh, ls=cat.ls(m), color=colour, lw=1.8, label=cat.label(m))
            axes[1].plot(centers, ratio, ls=cat.ls(m), color=colour, lw=1.8, label=cat.label(m))
            if np.isfinite(coh_hk) and np.isfinite(hk):
                axes[2].plot(coh_hk, hk, "o", ms=11, color=colour, mec=SURFACE, mew=2)
                if ctx.opt("point_labels", 0, int):
                    axes[2].annotate(wrap_label(cat.label(m), 24), (coh_hk, hk), textcoords="offset points",
                                     xytext=(0, 13 if i % 2 == 0 else -19), ha="center", fontsize=8.5, color=INK_2)
            rows.append({"model": m, "coherence_mean_k_ge_kmin": coh_hk,
                         f"coherence_k_r{level:g}": coherence_scale(centers, coh, level),
                         "highk_active": hk, "n_samples": s.n})
        axes[0].axhline(level, color=INK_MUTED, lw=1.0, ls="--")
        axes[0].set_ylabel("coherence  $r(k)$", fontsize=10, color=INK_2)
        axes[0].set_title("Is the structure in the right place?", fontsize=11, color=INK, loc="left")
        axes[0].set_ylim(0, 1.02)
        axes[1].axhline(1.0, color=INK_MUTED, lw=1.0, ls="--")
        axes[1].axvline(k_min, color=INK_MUTED, lw=1.0, ls=":")
        axes[1].set_ylabel("$P_\\mathrm{pred}/P_\\mathrm{true}$", fontsize=10, color=INK_2)
        axes[1].set_title(f"Is there the right amount?  (dotted: highk cutoff {k_min:g})",
                          fontsize=11, color=INK, loc="left")
        axes[1].set_yscale("log")
        for ax in axes[:2]:
            ax.set_xlabel("$k$  (cycles / pixel)", fontsize=10, color=INK_2)
        axes[2].set_xlabel(f"mean coherence $r(k)$ for $k \\geq {k_min:g}$  (higher better)",
                           fontsize=10, color=INK_2)
        axes[2].set_ylabel("highk on active slices  (lower better)", fontsize=10, color=INK_2)
        axes[2].set_title("Placement vs amount of small-scale structure", fontsize=11, color=INK, loc="left")
        for ax in axes:
            ax.grid(True, color=GRID, lw=0.8, zorder=0)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(GRID)
            ax.tick_params(colors=INK_2, labelsize=9)
        fig.set_facecolor(SURFACE)
        fig.tight_layout()
        shared_legend(fig, handles=model_handles(cat, truth=False, markers=["o"]), fontsize=8.5)
        written.append(save(fig, ctx.out, f"ps_coherence_highk_{target}", dpi=160, quiet=ctx.quiet))
        with open(ctx.out / f"ps_coherence_highk_{target}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if not written:
        raise SkipFigure("no 'spectrum' results")
    return written


def place_labels(ax, labels, fixed=None):
    """Label points, choosing per label the first offset whose text box (estimated in display
    pixels) overlaps no earlier label and no point (rhizome ps_plane3d)."""
    ax.figure.canvas.draw()
    to_px = ax.transData.transform
    dpi = ax.figure.dpi / 72
    points = [to_px((x, y)) for _, x, y, _ in labels]
    fixed = fixed or {}
    boxes = []
    candidates = [(7, 4, "left"), (7, -12, "left"), (-7, 4, "right"), (-7, -12, "right"),
                  (7, 14, "left"), (-7, 14, "right"), (7, -22, "left"), (-7, -22, "right")]
    for text, x, y, hollow in labels:
        size = 7.5 if hollow else 8.5
        w, h = 0.6 * size * len(text) * dpi, size * 1.2 * dpi
        px, py = to_px((x, y))
        own = [(fixed[text][0], fixed[text][1], "right" if fixed[text][0] < 0 else "left")] if text in fixed else []
        for dx, dy, ha in own or candidates:
            x0 = px + dx * dpi - (w if ha == "right" else 0)
            y0 = py + dy * dpi
            box = (x0, y0, x0 + w, y0 + h)
            clash = any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in boxes)
            clash |= any(box[0] - 4 < qx < box[2] + 4 and box[1] - 4 < qy < box[3] + 4 for qx, qy in points)
            if not clash or own:
                break
        boxes.append(box)
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=size,
                    color=INK_MUTED if hollow else INK_2)


@figure("ps_plane3d", group="spectra",
        help="small-scale coherence r(k>1) vs |P_pred/P_true - 1| (k>1), active stage (rhizome ps_plane3d)")
def ps_plane3d(ctx):
    cat = ctx.catalog
    plt = pyplot()
    kmin = float(ctx.options.get("plane_kmin", 1.0))
    note = str(ctx.options.get("note", "") or "")
    written, reasons = [], []
    for target in cat.targets:
        spec = _spectra(cat, target)
        if spec is None:
            continue
        points = {}
        for m in cat.models:
            s = spec[m]
            band = s.k >= kmin
            if not band.any():
                break
            r = float(_nanmedian(s.r[0][-1][band])) if s.coherence else float("nan")
            err = float(_nanmedian(np.abs(s.ratio[0][-1][band] - 1.0)))
            points[m] = (r, err, s.n)
        if not points:
            reasons.append(f"{target}: no k bin above {kmin:g} / Mpc")
            continue
        fig, ax = plt.subplots(figsize=(9.5, 6.4), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        lines = ["group,model,coherence_r_small,power_err_small,n_cones"]
        groups = "+".join(cat.group_labels(target))
        labels = []
        for i, (m, (r, err, n)) in enumerate(points.items()):
            lines.append(f"{groups},{m},{r:.4f},{err:.4f},{n}")
            if not (np.isfinite(r) and np.isfinite(err)):
                continue
            color = cat.color(m)
            hollow = i == 0
            ax.scatter([r], [err], s=70, label=cat.label(m), zorder=3, linewidths=1.6,
                       facecolors="none" if hollow else color, edgecolors=color)
            labels.append((cat.label(m), r, err, hollow))
        if labels:
            ax.margins(0.12)
            if ctx.opt("point_labels", 0, int):
                place_labels(ax, [(wrap_label(t, 24), *rest) for t, *rest in labels])
        else:
            ax.text(0.5, 0.5, "no coherence (generative runs)", ha="center", va="center",
                    transform=ax.transAxes, color=INK_MUTED)
        ax.set_xlabel(f"small-scale coherence  r(k > {kmin:g} Mpc$^{{-1}}$)  (higher better)", color=INK)
        ax.set_ylabel(f"small-scale power error  |P$_{{pred}}$/P$_{{true}}$ − 1|, k > {kmin:g}  (lower better)",
                      color=INK)
        ax.set_title(f"Placement vs amount of small-scale structure ({field_name(target)}, active stage)",
                     loc="left", color=INK)
        ax.grid(color=GRID, lw=0.8)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=INK_2)
        if note:
            fig.text(0.01, 0.01, note, fontsize=7.5, color=INK_MUTED, ha="left", va="bottom", wrap=True)
        fig.tight_layout(rect=(0, 0.06 if note else 0, 1, 1))
        if labels:
            shared_legend(fig, fontsize=9)
        fig.set_facecolor(SURFACE)
        written.append(save(fig, ctx.out, f"ps_plane3d_{target}", dpi=150, quiet=ctx.quiet))
        (ctx.out / f"ps_plane3d_{target}.csv").write_text("\n".join(lines) + "\n")
    if not written:
        raise SkipFigure("; ".join(reasons) or "no 'spectrum' results")
    return written
