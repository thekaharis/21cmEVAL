"""Figures of stochastic / generative models (``eval21 run --draws N``).

=====================  ==========================================================  =========
figure                 shows                                                       needs
=====================  ==========================================================  =========
ensemble_calibration   per target: voxel rank histogram (flat = calibrated), PIT    curves
                       histograms of the summary statistics (x_bar(z), history,
                       power per stage), coverage vs its nominal value, and the
                       spread/skill ratio per reionization stage
ensemble_draws         per stored sample: truth, ensemble mean, ensemble std and    store
                       up to four draws at the gallery stages (x_HI viridis 0-1,
                       T_b magma, std in greys) -- what a generative model's
                       variety looks like next to the truth
=====================  ==========================================================  =========

Both follow the gallery's conventions (``viz/slices.py``): stages from the true slice-mean
x_HI, one row per stage, a colour bar per column.
"""
from __future__ import annotations

import numpy as np

from .common import field_name, pyplot, save
from .registry import SkipFigure, figure
from .slices import cone_name

FAMILY_LABELS = {"xbar": r"$\bar x(z)$", "history": "history", "power": r"$\Delta^2(k)$ per stage"}


def _ensemble(cat, target, model):
    """The ensemble result of one model, pooled over groups (first group's stage labels)."""
    entries = [run.metric(model, "ensemble") for run in cat.runs_for(target, model)]
    entries = [e for e in entries if e and "n_draws" in e]
    return entries


@figure("ensemble_calibration", group="ensemble",
        help="rank histogram, PIT histograms, coverage vs nominal, spread/skill per stage")
def ensemble_calibration(ctx):
    cat = ctx.catalog
    plt = pyplot()
    written = []
    for target in cat.targets:
        models = [m for m in cat.models if _ensemble(cat, target, m)]
        if not models:
            continue
        families = []
        for m in models:
            for e in _ensemble(cat, target, m):
                families += [f for f in e.get("statistics", {}) if f not in families]
        has_voxel = any("voxel" in e for m in models for e in _ensemble(cat, target, m))
        ncols = len(families) + (2 if has_voxel else 0) + 1
        fig, axes = plt.subplots(1, ncols, figsize=(3.6 * ncols, 3.3), squeeze=False)
        axes = list(axes[0])
        col = 0
        if has_voxel:
            ax = axes[col]; col += 1
            for m in models:
                ranks = np.sum([np.asarray(e["voxel"]["rank_histogram"], float)
                                for e in _ensemble(cat, target, m) if "voxel" in e], axis=0)
                if np.ndim(ranks) == 0:
                    continue
                freq = ranks / ranks.sum() * ranks.size
                ax.step(np.arange(ranks.size + 1) - 0.5, np.r_[freq, freq[-1]], where="post",
                        color=cat.color(m), ls=cat.ls(m), label=cat.label(m))
            ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
            ax.set(xlabel="rank of truth among draws", ylabel="relative frequency",
                   title="voxel rank histogram")
            ax = axes[col]; col += 1
            for m in models:
                rows = [e["voxel_stages"] for e in _ensemble(cat, target, m) if e.get("voxel_stages")]
                if not rows:
                    continue
                labels = [s["label"] for s in rows[0]]
                values = np.nanmean([[np.nan if s.get("spread_skill") is None else s["spread_skill"]
                                      for s in r] for r in rows], axis=0)
                ax.plot(range(len(labels)), values, marker="o", color=cat.color(m), ls=cat.ls(m), label=cat.label(m))
                ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
            ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
            ax.set(ylabel="spread / skill", title=r"spread/skill per stage ($\bar x_{\rm HI}$)")
        for family in families:
            ax = axes[col]; col += 1
            for m in models:
                hists = [np.asarray(e["statistics"][family]["pit_hist"], float)
                         for e in _ensemble(cat, target, m) if family in e.get("statistics", {})]
                if not hists:
                    continue
                h = np.sum(hists, axis=0)
                density = h / h.sum() * h.size
                edges = np.linspace(0, 1, h.size + 1)
                ax.step(edges, np.r_[density, density[-1]], where="post", color=cat.color(m),
                        ls=cat.ls(m), label=cat.label(m))
            ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
            ax.set(xlim=(0, 1), xlabel="PIT", ylabel="density", title=f"PIT: {FAMILY_LABELS.get(family, family)}")
        ax = axes[col]
        names, observed, nominal, colors = [], [], [], []
        for m in models:
            for e in _ensemble(cat, target, m)[:1]:
                if "voxel" in e:
                    for level in (50, 90):
                        names.append(f"{cat.label(m)}\nvoxel {level}")
                        observed.append(e["voxel"][f"coverage{level}"])
                        nominal.append(e["voxel"][f"coverage{level}_nominal"])
                        colors.append(cat.color(m))
                for family, s in e.get("statistics", {}).items():
                    names.append(f"{cat.label(m)}\n{family} 68")
                    observed.append(s["coverage68"])
                    nominal.append(s["coverage68_nominal"])
                    colors.append(cat.color(m))
        y = np.arange(len(names))
        ax.barh(y, observed, color=colors, alpha=0.7)
        ax.scatter(nominal, y, marker="|", s=200, color="k", label="nominal", zorder=3)
        ax.set_yticks(y, names, fontsize=6)
        ax.set(xlim=(0, 1), xlabel="coverage", title="coverage vs nominal")
        ax.invert_yaxis()
        axes[0].legend(fontsize=7, frameon=False)
        n_draws = sorted({e["n_draws"] for m in models for e in _ensemble(cat, target, m)})
        fig.suptitle(f"{field_name(target)} ensemble calibration ({'/'.join(map(str, n_draws))} draws"
                     + (f", {ctx.title}" if ctx.title else "") + ")", fontsize=10)
        fig.tight_layout()
        written.append(save(fig, ctx.out, f"ensemble_calibration_{target}", quiet=ctx.quiet))
    if not written:
        raise SkipFigure("no model was run with --draws N (no 'ensemble' results)")
    return written


@figure("ensemble_draws", group="ensemble", needs=("store",),
        help="truth, ensemble mean/std and draws at the gallery stages, per stored sample")
def ensemble_draws(ctx):
    store = ctx.store
    models = [m for m in ctx.catalog.models if m in store.models and store.draws(m) > 1]
    if not models:
        raise SkipFigure("no stored draws (run with --draws N --save-predictions DIR)")
    plt = pyplot()
    stages = tuple(float(s) for s in str(ctx.options.get("stages", "0.8,0.6,0.4,0.2")).split(","))
    n_show = int(ctx.options.get("draws_shown", 4))
    ids = ctx.ids or store.samples
    written = []
    for sid in ids:
        truth = store.truth(sid)
        xhi_true = truth["targets"].get("xhi")
        for m in models:
            pred = store.prediction(m, sid)
            for field in [k[: -len("__draws")] for k in pred if k.endswith("__draws")]:
                draws = pred[f"{field}__draws"]
                true = truth["targets"].get(field)
                if true is None:
                    continue
                stage_source = xhi_true if xhi_true is not None else true
                xbar = stage_source.mean(axis=(0, 1))
                planes = [int(np.argmin(np.abs(xbar - s))) for s in stages]
                shown = draws[:n_show]
                mean, std = draws.mean(0), draws.std(0)
                cols = [("truth", true), ("ensemble mean", mean), ("ensemble std", std)] + \
                       [(f"draw {i}", d) for i, d in enumerate(shown)]
                if field == "xhi":
                    cmap, vmin, vmax = "viridis", 0.0, 1.0
                else:
                    cmap = "magma"
                    vmin, vmax = np.percentile(true, [1, 99])
                fig, grid = plt.subplots(len(planes) + 1, len(cols),
                                         figsize=(1.9 * len(cols), 1.9 * len(planes) + 0.5), squeeze=False,
                                         gridspec_kw={"height_ratios": [1] * len(planes) + [0.07]})
                axes, cbar_axes = grid[:-1], grid[-1]
                smax = max(float(np.percentile(std, 99)), 1e-6)
                images = {}
                for r, p in enumerate(planes):
                    for c, (name, cube) in enumerate(cols):
                        ax = axes[r, c]
                        is_std = name == "ensemble std"
                        im = ax.imshow(cube[:, :, p].T, origin="lower", cmap="Greys" if is_std else cmap,
                                       vmin=0 if is_std else vmin, vmax=smax if is_std else vmax)
                        ax.set_xticks([]); ax.set_yticks([])
                        if r == 0:
                            ax.set_title(name, fontsize=8)
                        if c == 0:
                            z = truth["z_grid"]
                            ax.set_ylabel(rf"$\bar x$={xbar[p]:.2f}" + ("" if z is None else f"\nz={z[p]:.2f}"),
                                          fontsize=7)
                        images[c] = im
                for c, cax in enumerate(cbar_axes):
                    if c in (0, 2):
                        fig.colorbar(images[c], cax=cax, orientation="horizontal")
                        cax.tick_params(labelsize=7)
                    else:
                        cax.axis("off")
                fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.05, wspace=0.05, hspace=0.08)
                fig.suptitle(f"{cone_name(sid)}: {ctx.catalog.label(m)}, {field_name(field)}, "
                             f"{store.draws(m)} draws ({len(shown)} shown)", fontsize=9)
                written.append(save(fig, ctx.out / "ensemble_draws",
                                    f"{cone_name(sid)}_{m}_{field}", quiet=ctx.quiet))
    return written
