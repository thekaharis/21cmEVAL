"""Cylindrical power-error maps: P_pred/P_true - 1 over (k_perp, k_par), white dead band.

Port of rhizome-neural-operator's ``ebm21cm.plot_cyl_error`` (same colour map, axes, merging of
empty bins, titles and hatching), fed by the ``cylindrical`` metric: per column the median over
all samples of all groups of the per-sample ratio maps. Rows: models; columns: chunk stages (or
fixed redshifts). Colour: |error| <= dead band is flat white; beyond it the diverging map starts
at a visibly tinted level and saturates at +-vmax (red: too much power); grey: no truth power.
Wavenumbers beyond the truth's resolution are hatched behind dashed lines and not counted.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from .registry import figure
from .common import wrap_label
from .common import field_name, pyplot, save


def dead_band_cmap(vmax, dead, base="RdBu_r", start=0.12, n=1024):
    """Diverging colormap on [-vmax, vmax]: white for |v| <= dead, then base from 0.5 +- start outward."""
    plt = pyplot()
    from matplotlib.colors import ListedColormap

    cmap = plt.get_cmap(base)
    v = np.linspace(-vmax, vmax, n)
    frac = np.clip((np.abs(v) - dead) / (vmax - dead), 0, 1)
    pos = 0.5 + np.sign(v) * (start + (0.5 - start) * frac)
    colors = cmap(pos)
    colors[np.abs(v) <= dead] = (1.0, 1.0, 1.0, 1.0)
    out = ListedColormap(colors)
    out.set_bad("0.85")
    return out


def merge_empty_bins(edges, maps, axis):
    """Absorb bins that are empty in every map (no Fourier modes) into the next larger bin."""
    edges = np.asarray(edges, float)
    maps = [np.asarray(m, float) for m in maps]
    empty = np.all([np.all(np.isnan(m), axis=1 - axis) for m in maps], axis=0)
    keep_bins = ~empty
    if not keep_bins.any() or not empty.any():
        return edges, maps
    keep_edges = np.ones(len(edges), bool)
    for i in np.flatnonzero(empty):
        keep_edges[i + 1 if i + 1 < len(edges) - 1 else i] = False
    return edges[keep_edges], [np.compress(keep_bins, m, axis=axis) for m in maps]


def _pooled(cat, target, model, key):
    parts = [run.curves(model)[key] for run in cat.runs_for(target, model)]
    return np.concatenate(parts, 0)


def _same(cat, target, key, rtol=1e-9):
    values = [run.curves(m)[key] for m in cat.models for run in cat.runs_for(target, m)]
    for v in values[1:]:
        if np.shape(v) != np.shape(values[0]) or not np.allclose(v, values[0], rtol=rtol, atol=0, equal_nan=True):
            raise SystemExit(f"cylindrical: {key} differs between runs/models of target {target!r}; "
                             "the maps can only be pooled on one chunk geometry")
    return values[0]


def plot_cylindrical(cat, out: Path, fields=None, vmax=0.4, dead=None, prefix="cyl_error", title=None,
                     quiet=False):
    """One figure per target field. Returns {field: {model: [share per column]}}."""
    plt = pyplot()
    summary = {}
    for target in fields or cat.targets:
        if not cat.has(target, "cyl_ratio"):
            if not quiet:
                print(f"cylindrical: no 'cylindrical' results for target {target!r}; skipped")
            continue
        kperp = _same(cat, target, "cyl_kperp_edges")
        kpar_all = _same(cat, target, "cyl_kpar_edges", rtol=2e-3)   # same LOS frequencies, cells within 0.2 %
        columns = _same(cat, target, "cyl_columns")
        mode = str(_same_mode(cat, target))
        run0 = cat.runs_for(target, cat.models[0])
        band = dead if dead is not None else float(run0[0].settings.get("cyl_dead_band", 0.1))
        kperp_max = float(run0[0].curves(cat.models[0])["cyl_kperp_max"])
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            m0 = cat.models[0]
            zc = np.nanmedian(_pooled(cat, target, m0, "cyl_chunk_z"), 0)
            kpar_max = np.nanmedian(_pooled(cat, target, m0, "cyl_kpar_max"), 0)
            n_col = _pooled(cat, target, m0, "cyl_have").sum(0)
            medians = {m: np.nanmedian(_pooled(cat, target, m, "cyl_ratio"), 0) for m in cat.models}
        kpar_max = np.where(np.isfinite(kpar_max), kpar_max, np.inf)
        limited = np.isfinite(kperp_max) or np.isfinite(kpar_max).any()
        n_samples = len(_pooled(cat, target, m0, "cyl_have"))
        cmap = dead_band_cmap(vmax, band)
        # square (k_perp, k_par) panels: the figure follows the grid, not the other way round
        fig, axes = plt.subplots(len(cat.models), len(columns),
                                 figsize=(3.3 * len(columns) + 1.8, 3.25 * len(cat.models) + 1.1),
                                 squeeze=False, constrained_layout=True)
        pcm = None
        shares = {}
        for col, value in enumerate(columns):
            errs = [medians[m][col] - 1.0 for m in cat.models]
            kpar, errs = merge_empty_bins(kpar_all, errs, axis=0)
            kp, errs = merge_empty_bins(kperp, errs, axis=1)
            for row, (m, err) in enumerate(zip(cat.models, errs)):
                ax = axes[row][col]
                err = np.ma.masked_invalid(err)
                pcm = ax.pcolormesh(kp, kpar, err, cmap=cmap, vmin=-vmax, vmax=vmax)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_box_aspect(1)
                kp_c, kpar_c = np.sqrt(kp[1:] * kp[:-1]), np.sqrt(kpar[1:] * kpar[:-1])
                resolved = (kp_c[None, :] <= kperp_max) & (kpar_c[:, None] <= kpar_max[col])
                for lim, span, line, top in ((kperp_max, ax.axvspan, ax.axvline, kp[-1]),
                                             (kpar_max[col], ax.axhspan, ax.axhline, kpar[-1])):
                    if np.isfinite(lim) and lim < top:
                        span(lim, top, facecolor="none", edgecolor="0.35", hatch="////", linewidth=0)
                        line(lim, color="0.25", linestyle="--", linewidth=0.8)
                vals = np.ma.masked_where(~resolved, err).compressed()
                inside = float(np.mean(np.abs(vals) <= band)) if vals.size else float("nan")
                shares.setdefault(m, [None] * len(columns))[col] = inside
                cells = "resolved cells" if limited else "cells"
                if mode == "z":
                    ax.set_title(f"z ≈ {round(float(value), 1):g}: {100 * inside:.0f}% of {cells} within ±{band:g}",
                                 fontsize=10)
                else:
                    where = r"$\bar{x}_{\rm HI}$" + f" ≈ {value:g} (z ≈ {zc[col]:.1f}, {int(n_col[col])} cones)"
                    ax.set_title(f"{where}\n{100 * inside:.0f}% of {cells} within ±{band:g}", fontsize=10)
                if row == len(cat.models) - 1:
                    ax.set_xlabel(r"$k_\perp$  [Mpc$^{-1}$]")
                if col == 0:
                    ax.set_ylabel(f"{wrap_label(cat.label(m), 24)}\n" + r"$k_\parallel$  [Mpc$^{-1}$]", fontsize=10)
        bar = fig.colorbar(pcm, ax=axes, shrink=0.6, aspect=30, extend="both")
        bar.set_label(r"$P_{\rm pred}/P_{\rm true} - 1$  (white: within $\pm$" + f"{band:g})")
        ticks = sorted({-vmax, -band, 0.0, band, vmax} | set(np.round(np.arange(-vmax, vmax + 1e-9, 0.2), 2)))
        bar.set_ticks(ticks)
        text = title or f"{n_samples} test cones"
        head = f"Cylindrical {field_name(target)} power error, cone median over {text}"
        if mode == "stage":
            head += "; chunks centred per cone at the ionization stage"
        if limited:
            head += "\nhatched: beyond the truth's native resolution (not counted)"
        fig.suptitle(head)
        save(fig, Path(out), f"{prefix}_{target}", dpi=150, quiet=quiet)
        summary[target] = shares
        if not quiet:
            name = (lambda c: f"z≈{round(float(columns[c]), 1):g}") if mode == "z" else \
                (lambda c: f"x̄≈{columns[c]:g}")
            for m, fr in shares.items():
                print(f"{cat.label(m):28s} share within ±{band:g}: "
                      + " ".join(f"{name(c)} {100 * f:.0f}%" for c, f in enumerate(fr)))
    return summary


def _same_mode(cat, target):
    modes = {str(run.curves(m)["cyl_mode"]) for m in cat.models for run in cat.runs_for(target, m)}
    if len(modes) != 1:
        raise SystemExit(f"cylindrical: runs of target {target!r} mix chunk modes {sorted(modes)}")
    return modes.pop()


@figure("cyl", group="spectra", help="cylindrical P_pred/P_true - 1 maps (white dead band), rows = models")
def _fig_cyl(ctx):
    plot_cylindrical(ctx.catalog, ctx.out, vmax=float(ctx.options.get("vmax", 0.4)),
                     dead=None if "dead" not in ctx.options else float(ctx.options["dead"]),
                     title=ctx.title or None, quiet=ctx.quiet)
