"""Diagnostic figures: calibration (parity) and error near the ionization fronts.

Ports of fno-21cm's x_HI diagnostics.

=====================  ===============================================  ========================================================  =======
figure                 original                                         shows                                                     needs
=====================  ===============================================  ========================================================  =======
parity_overlay         fno-21cm ``viz/parity_diagnostic.py``            binned parity plot: per true-x_HI bin (50 on [0, 1]) the  curves
                       ``plot_parity``                                  median (solid), mean (dotted) and 16-84 % band of the
                                                                        prediction against the identity line; voxel occupancy
                                                                        per bin (log) underneath
parity_comparison      fno-21cm ``legacy/xhi2d/compare_xhi2d_to_3d.py``  one hexbin panel per model of predicted vs true x_HI     store
                       ``plot_parity``                                  (300 000 random voxels of the stored samples, log count,
                                                                        magma, white dashed identity)
boundary_band_overlay  fno-21cm ``viz/boundary_band_diagnostic.py``     three panels vs signed distance to the true front         curves
                       ``plot_overlay`` (also drawn by                  (2-D transverse, - ionized / + neutral): RMSE per voxel,
                       ``viz/edge_metrics_xhi2d.py`` as                 mean x_HI with 10-90 % front widths under the panel and
                       ``edge_band_overlay``)                           title, RMS in-plane gradient error
=====================  ===============================================  ========================================================  =======

Data: ``parity_overlay`` reads the ``parity`` metric (same 50 x 400 histogram construction as the
original); ``boundary_band_overlay`` the ``front`` metric's distance profiles (``front_*`` in
``curves.npz``; 1.5 Mpc bins to +-30 Mpc by default, as the original). Both are x_HI figures, as
the originals (the ``parity`` metric clips T_b to [0, 1], so T_b runs are not drawn).

Differences to the originals: models, labels and colours come from the catalog (the original used
the matplotlib colour cycle); model lines of ``boundary_band_overlay`` take the catalog line style.
``parity_overlay`` percentiles cannot be pooled across groups (data sets), so with several groups
one file per group is written (``parity_overlay_<group>``); ``boundary_band_overlay`` pools the
profiles of all groups weighted by voxel counts and computes the 10-90 % widths from the pooled
profiles. ``parity_comparison`` reads the stored prediction (the scored draw for generative
models) and accepts ``--ids`` and ``--opt voxels=300000``. The CSV / bootstrap side outputs of the
originals are not figures and are covered by ``results.json`` (``front.bands``,
``parity.bins``).
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
# legacy/xhi2d/compare_xhi2d_to_3d.py
# ----------------------------------------------------------------------------------------------- #
def _stored_xhi(store, model, sid):
    pred = store.prediction(model, sid)
    if TARGET in pred:
        return pred[TARGET]
    draws = pred.get(f"{TARGET}__draws")
    return None if draws is None else draws[0]


@figure("parity_comparison", group="diagnostics", needs=("store",),
        help="hexbin of predicted vs true x_HI per model, log counts (fno-21cm compare_xhi2d_to_3d plot_parity)")
def parity_comparison(ctx):
    plt = pyplot()
    cat, store = ctx.catalog, ctx.store
    models = [m for m in cat.models if m in store.models]
    if not models:
        raise SkipFigure("no catalog model has stored predictions")
    ids = [str(i) for i in (ctx.ids or store.samples) if str(i) in store.samples]
    truths, preds = [], {m: [] for m in models}
    for sid in ids:
        truth = store.truth(sid)["targets"].get(TARGET)
        if truth is None:
            continue
        cubes = {m: _stored_xhi(store, m, sid) for m in models}
        if any(c is None or c.shape != truth.shape for c in cubes.values()):
            continue
        truths.append(truth.ravel())
        for m in models:
            preds[m].append(cubes[m].ravel())
    if not truths:
        raise SkipFigure("no stored sample has x_HI truth and every model's x_HI prediction")
    flat_truth = np.concatenate(truths)
    rng = np.random.default_rng(42)
    count = min(int(ctx.opt("voxels", 300_000)), len(flat_truth))
    indices = rng.choice(len(flat_truth), size=count, replace=False)
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(16 / 3 * max(n, 1) + (0.6 if n == 1 else 0), 5),
                             constrained_layout=True, squeeze=False)
    for axis, m in zip(axes[0], models):
        flat_pred = np.concatenate(preds[m])
        rendered = axis.hexbin(flat_truth[indices], flat_pred[indices], gridsize=75, bins="log", mincnt=1,
                               cmap="magma")
        axis.plot([0, 1], [0, 1], "w--", linewidth=1)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_aspect("equal")
        axis.set_xlabel("truth $x_{HI}$")
        axis.set_ylabel("prediction $x_{HI}$")
        axis.set_title(cat.label(m))
        fig.colorbar(rendered, ax=axis, label="log count")
    if ctx.title:
        fig.suptitle(ctx.title)
    return [save(fig, ctx.out, "parity_comparison", dpi=170, quiet=ctx.quiet)]


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
