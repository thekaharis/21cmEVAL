"""Calibration and front diagnostics.

``parity_overlay`` draws the binned parity relation E[pred x_HI | true x_HI] with its 16-84 %
band and bin occupancy from the ``parity`` metric. ``boundary_band_overlay`` draws RMSE, mean
x_HI (10-90 front width) and gradient error versus signed distance to the true front from the
``front`` metric. Both are x_HI figures; a T_b run skips them.
"""
from __future__ import annotations

import re

import numpy as np

from ..physics import front_width
from .common import model_handles, pane_values, pyplot, save, shared_legend, style_key
from .registry import SkipFigure, figure

TARGET = "xhi"
N_TRUE = 50           # parity metric: true bins on [0, 1]

def _finite(x, y):
    """Drop empty bins (NaN) so curves stay connected, as in the originals."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]



def _title(text, ctx):
    return f"{text} — {ctx.title}" if ctx.title else text


def _safe(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "group"


# ----------------------------------------------------------------------------------------------- #
# parity_diagnostic.py
# ----------------------------------------------------------------------------------------------- #
def _parity_arrays(metric: dict) -> dict:
    """Rebuild the full per-bin arrays of ``ParityAccumulator.stats`` from the metric's bin list."""
    keys = ("truth_mean", "pred_mean", "pred_std", "pred_p16", "pred_p50", "pred_p84", "bias", "rmse")
    out = {k: np.full(N_TRUE, np.nan) for k in keys}
    out["count"] = np.zeros(N_TRUE, dtype=np.int64)
    out["true_edges"] = np.linspace(0.0, 1.0, N_TRUE + 1)
    for b in metric.get("bins", []):
        i = int(np.clip(np.floor(float(b["center"]) * N_TRUE), 0, N_TRUE - 1))
        out["count"][i] = int(b["count"])
        for k in keys:
            v = b.get(k)
            out[k][i] = np.nan if v is None else float(v)
    return out


@figure("parity_overlay", group="diagnostics", needs=("curves",),
        help="binned parity plot E[pred x_HI | true x_HI] with 16-84% band and bin occupancy (fno-21cm parity_diagnostic)")
def parity_overlay(ctx):
    plt = pyplot()
    cat = ctx.catalog
    if TARGET not in cat.targets:
        raise SkipFigure("needs an x_HI run")
    groups = cat.group_labels(TARGET)
    written = []
    for gi, group in enumerate(groups):
        stats = {m: _parity_arrays(cat.runs_for(TARGET, m)[gi].metric(m, "parity")) for m in cat.models}
        if not any(s["count"].sum() for s in stats.values()):
            continue
        fig, (ax, axn) = plt.subplots(2, 1, figsize=(7.5, 8.5), sharex=True,
                                      gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08})
        ax.plot([0, 1], [0, 1], color="0.4", lw=1.0, ls="--", label="identity (y = x)")
        for m, st in stats.items():
            if not st["count"].sum():
                continue
            c, name = cat.color(m), cat.label(m)
            x = st["truth_mean"]
            ax.fill_between(x, st["pred_p16"], st["pred_p84"], color=c, alpha=0.22, linewidth=0)
            ax.plot(x, st["pred_p50"], color=c, lw=1.8)
            ax.plot(x, st["pred_mean"], color=c, lw=1.0, ls=":")
        ax.set_ylabel("predicted x_HI")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        # colour = model (solid line samples); line style / band = statistic, once for all models
        shown = [m for m, st in stats.items() if st["count"].sum()]
        shared_legend(fig, handles=model_handles(cat, models=shown, truth=False, kind="patch"),
                      extra=style_key([("median", "-"), ("mean", ":"),
                                       ("16–84 %", {"kind": "band", "color": "0.5", "alpha": 0.3}),
                                       ("identity", {"ls": "--", "color": "0.4", "lw": 1.0})]), fontsize=8.5)
        title = "Binned parity plot: E[predicted x_HI | true x_HI]"
        if len(groups) > 1:
            title += f" ({group})"
        ax.set_title(_title(title, ctx))
        for m, st in stats.items():          # counts are truth-only: identical for every model
            if not st["count"].sum():
                continue
            edges = st["true_edges"]
            axn.step(edges[:-1], np.maximum(st["count"], 0.5), where="post", color=cat.color(m), lw=1.2,
                     label=cat.label(m))
            break
        axn.set_yscale("log")
        axn.set_xlabel("true x_HI")
        axn.set_ylabel("voxels / bin")
        axn.grid(alpha=0.25)
        name = "parity_overlay" if len(groups) == 1 else f"parity_overlay_{_safe(group)}"
        written.append(save(fig, ctx.out, name, dpi=150, quiet=ctx.quiet))
    if not written:
        raise SkipFigure("no `parity` metric in the x_HI runs (dropped in generative mode)")
    return written


# ----------------------------------------------------------------------------------------------- #
# boundary_band_diagnostic.py / edge_metrics_xhi2d.py
# ----------------------------------------------------------------------------------------------- #
def _front_profile(cat, m):
    """Count-weighted pool of the front profiles over all x_HI groups of model ``m``."""
    d, n = None, 0.0
    sums = {"front_mean_truth": 0.0, "front_mean_pred": 0.0, "front_rmse": 0.0, "front_gradient_rms": 0.0}
    present = set()
    for run in cat.runs_for(TARGET, m):
        c = run.curves(m)
        if "front_d_mpc" not in c or "front_counts" not in c:
            continue
        if d is None:
            d = np.asarray(c["front_d_mpc"], float)
        elif len(c["front_d_mpc"]) != len(d) or not np.allclose(c["front_d_mpc"], d):
            raise SkipFigure("front distance bins differ between groups")
        counts = np.asarray(c["front_counts"], float)
        n = n + counts
        for key in sums:
            if key in c:
                v = np.nan_to_num(np.asarray(c[key], float))
                # RMS profiles pool as mean squares
                sums[key] = sums[key] + counts * (v ** 2 if key in ("front_rmse", "front_gradient_rms") else v)
                present.add(key)
    if d is None:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {"d_mpc": d, "count": n}
        for key, s in sums.items():
            if key not in present:
                out[key] = np.full(len(d), np.nan)
                continue
            v = np.where(n > 0, s / np.where(n > 0, n, 1), np.nan)
            out[key] = np.sqrt(v) if key in ("front_rmse", "front_gradient_rms") else v
    return out


@figure("boundary_band_overlay", group="diagnostics", needs=("curves",),
        help="RMSE, mean x_HI (front width) and gradient error vs distance to the true front (fno-21cm boundary_band_diagnostic)")
def boundary_band_overlay(ctx):
    plt = pyplot()
    cat = ctx.catalog
    if TARGET not in cat.targets:
        raise SkipFigure("needs an x_HI run (the front metric is x_HI only)")
    profiles = {m: _front_profile(cat, m) for m in cat.models}
    profiles = {m: p for m, p in profiles.items() if p is not None}
    if not profiles:
        raise SkipFigure("no `front` metric in the x_HI runs")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # (1) RMSE vs signed distance
    ax = axes[0]
    for m, p in profiles.items():
        ax.plot(*_finite(p["d_mpc"], p["front_rmse"]), label=cat.label(m), color=cat.color(m), ls=cat.ls(m))
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("signed distance to true front  [Mpc]\n(- ionized   +  neutral)")
    ax.set_ylabel("RMSE per voxel")
    ax.set_title("Error localised at the front")

    # (2) mean x_HI vs signed distance: front sharpness
    ax = axes[1]
    any_truth = next(iter(profiles.values()))
    ax.plot(*_finite(any_truth["d_mpc"], any_truth["front_mean_truth"]), "k-", lw=2.2, label="truth")
    widths = []
    for m, p in profiles.items():
        widths.append((cat.color(m), cat.ls(m), front_width(p["d_mpc"], p["front_mean_pred"])))
        ax.plot(*_finite(p["d_mpc"], p["front_mean_pred"]), color=cat.color(m), ls=cat.ls(m),
                label=cat.label(m))
    wt = front_width(any_truth["d_mpc"], any_truth["front_mean_truth"])
    ax.axhline(0.1, color="grey", lw=0.6, ls="--")
    ax.axhline(0.9, color="grey", lw=0.6, ls="--")
    ax.set_xlabel("signed distance to true front  [Mpc]")
    ax.set_ylabel(r"$\langle x_{\rm HI}\rangle$")
    ax.set_title(f"Front sharpness (truth 10-90 = {wt:.1f} Mpc)")
    pane_values(ax, widths, label="10–90 width [Mpc]", fmt="{:.1f}")

    # (3) gradient (H1 seminorm) RMS vs signed distance
    ax = axes[2]
    for m, p in profiles.items():
        ax.plot(*_finite(p["d_mpc"], p["front_gradient_rms"]), color=cat.color(m), ls=cat.ls(m), label=cat.label(m))
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("signed distance to true front  [Mpc]")
    ax.set_ylabel(r"RMS $|\nabla(\hat{x}-x)|$")
    ax.set_title("Gradient error (where the H1 win lives)")

    if ctx.title:
        fig.suptitle(ctx.title)
    fig.tight_layout()
    shared_legend(fig, handles=model_handles(cat, models=list(profiles)))
    return [save(fig, ctx.out, "boundary_band_overlay", dpi=130, quiet=ctx.quiet)]
