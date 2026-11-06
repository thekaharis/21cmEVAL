"""Bubble-size figures: transverse mean-free-path (MFP) ionized-bubble size distributions.

Ports of fno-21cm's bubble-size evaluation figures, fed by the ``bubble`` metric (per-stage
histogram of MFP distances on the metric's log radius bins, ``curves.npz``
``bubble_mass_{truth,pred}_<stage>`` per sample; per-stage medians of W1, bias, underflow and
censored fractions in ``results.json``). Only x_HI runs carry the metric.

=========================  ================================================  ======================================================  =======
figure                     original                                          shows                                                   needs
=========================  ================================================  ======================================================  =======
bubble_size_distribution   fno-21cm ``viz/bubble_size_evaluation.py``        3-column grid, one panel per stage (5 x_HI stages +     curves
                           ``plot_distributions``                            "active"): mean probability per log bin vs MFP
                                                                             distance (log x), 16-84 % band over samples, truth
                                                                             black; v / ^ markers at the first / last edge for
                                                                             "no ionized pixels" / "censored at box length"
bubble_size_summary        ``bubble_size_evaluation.py`` ``plot_summary``    grouped bars per stage: median restricted Wasserstein   curves
                                                                             distance [cMpc] and restricted mean-MFP bias [%]
bsd_clean                  fno-21cm ``viz/replot_bubble_size.py`` ``main``   the distribution grid redrawn without bands (mean or    curves
                                                                             median curves only, "21cmFAST truth", math stage
                                                                             titles, y from 0); readable with many models
bsd_clean_summary          ``replot_bubble_size.py`` ``plot_summary``        the Wasserstein / bias bars with math stage labels      curves
=========================  ================================================  ======================================================  =======

Options (``--opt``): ``stat=mean|med`` (central curve of ``bsd_clean``, default ``mean``),
``stages=0,2,5`` (panels of ``bsd_clean`` / ``bsd_clean_summary``), ``markers=0`` (drop the
underflow / censored markers), ``ncols=3``.

Differences to the originals, all forced by what ``eval21 run`` records: models come from the
catalog (labels, colours; model lines use the catalog line style, the reference model dashed);
the underflow / censored markers are the per-stage **medians** over samples (the metric stores no
per-sample fractions, so the mean curve of ``bubble_size_distribution`` is paired with median
markers); the bar heights are the metric's per-stage medians. With several groups (data sets) of
one target the mass curves are pooled over all samples of all groups, and the marker / bar values
are the sample-count-weighted mean of the per-group medians. Groups with different radius bins
(different box or ``max_distance_mpc``) are drawn into separate files ``<name>_g<i>``.
"""
from __future__ import annotations

import warnings

import numpy as np

from .common import model_handles, pyplot, save, shared_legend, style_key
from .registry import SkipFigure, figure

TARGET = "xhi"


# ----------------------------------------------------------------------------------------------- #
# data
# ----------------------------------------------------------------------------------------------- #
def _curve_key(label: str) -> str:
    """results.json stage label -> curves.npz suffix ('xbar_HI 0.02-0.20' -> 'xbar0.02_0.20')."""
    label = str(label)
    if label.startswith("active"):
        return "active"
    lo, hi = label.split()[-1].split("-")
    return f"xbar{lo}_{hi}"


def _percentile(a, q):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(a, q, axis=0)


def _nanmean(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _weighted(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    ok = np.isfinite(values) & (weights > 0)
    return float(np.sum(values[ok] * weights[ok]) / np.sum(weights[ok])) if ok.any() else np.nan


def _collect(cat):
    """Per-model bubble statistics, pooled over compatible groups.

    Returns a list of (suffix, data) with data = {"edges", "centers", "labels",
    "models": {m: {"truth_mass", "pred_mass": [stage] -> (N, bins) arrays,
                   "truth_underflow", "pred_underflow", "truth_censored", "pred_censored",
                   "w1", "bias": [stage] floats}}}.
    """
    if TARGET not in cat.targets:
        raise SkipFigure("needs an x_HI run (the bubble metric is x_HI only)")
    first = cat.models[0]
    runs0 = cat.runs_for(TARGET, first)
    metrics0 = [run.metric(first, "bubble") for run in runs0]
    if not any(metrics0):
        raise SkipFigure("no `bubble` metric in the x_HI runs")
    # partition the groups by radius binning
    parts = []
    for gi, met in enumerate(metrics0):
        if not met:
            continue
        edges = np.asarray(met["radius_edges_mpc"], float)
        for part in parts:
            if part["edges"].shape == edges.shape and np.allclose(part["edges"], edges):
                part["groups"].append(gi)
                break
        else:
            parts.append({"edges": edges, "groups": [gi], "labels": list(met["stage_labels"]),
                          "centers": np.asarray(met["radius_centers_mpc"], float)})
    out = []
    for pi, part in enumerate(parts):
        labels = part["labels"]
        models = {}
        for m in cat.models:
            truth_mass = [[] for _ in labels]
            pred_mass = [[] for _ in labels]
            scal = {k: [[] for _ in labels] for k in ("truth_underflow", "pred_underflow", "truth_censored",
                                                      "pred_censored", "w1", "bias")}
            weights = [[] for _ in labels]
            for gi in part["groups"]:
                run = cat.runs_for(TARGET, m)[gi]
                met = run.metric(m, "bubble")
                curves = run.curves(m)
                if not met:
                    continue
                n = float(met.get("n_samples") or 1)
                for si, label in enumerate(labels):
                    key = _curve_key(label)
                    for name, dest in (("truth", truth_mass), ("pred", pred_mass)):
                        arr = curves.get(f"bubble_mass_{name}_{key}")
                        if arr is not None and np.size(arr):
                            dest[si].append(np.atleast_2d(arr).astype(float))
                    st = met["stages"][si] if si < len(met.get("stages", [])) else {}
                    for k, src in (("truth_underflow", "underflow_truth"), ("pred_underflow", "underflow_pred"),
                                   ("truth_censored", "censored_truth"), ("pred_censored", "censored_pred"),
                                   ("w1", "wasserstein_med_mpc"), ("bias", "relative_bias_med")):
                        v = st.get(src)
                        scal[k][si].append(np.nan if v is None else float(v))
                    weights[si].append(n)
            nb = len(part["centers"])
            entry = {
                "truth_mass": [np.concatenate(a, 0) if a else np.full((1, nb), np.nan) for a in truth_mass],
                "pred_mass": [np.concatenate(a, 0) if a else np.full((1, nb), np.nan) for a in pred_mass],
            }
            for k, vals in scal.items():
                entry[k] = np.array([_weighted(v, w) for v, w in zip(vals, weights)])
            models[m] = entry
        suffix = "" if len(parts) == 1 else f"_g{pi}"
        out.append((suffix, {"edges": part["edges"], "centers": part["centers"], "labels": labels,
                             "models": models}))
    return out


def _title(text, ctx):
    return f"{text} — {ctx.title}" if ctx.title else text


# ----------------------------------------------------------------------------------------------- #
# bubble_size_evaluation.py
# ----------------------------------------------------------------------------------------------- #
@figure("bubble_size_distribution", group="bubbles", needs=("curves",),
        help="MFP bubble-size PDF per stage, truth vs models with 16-84% bands (fno-21cm bubble_size_evaluation)")
def bubble_size_distribution(ctx):
    plt = pyplot()
    cat = ctx.catalog
    written = []
    for suffix, data in _collect(cat):
        labels, centers, edges = data["labels"], data["centers"], data["edges"]
        n_cols = 3
        n_rows = int(np.ceil(len(labels) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.2 * n_rows), squeeze=False)
        axes = axes.ravel()
        truth = data["models"][cat.models[0]]
        for si, label in enumerate(labels):
            ax = axes[si]
            tm = truth["truth_mass"][si]
            ax.plot(centers, _nanmean(tm), color="black", linewidth=2, label="truth")
            ax.fill_between(centers, _percentile(tm, 16), _percentile(tm, 84), color="black", alpha=0.12)
            ax.scatter(edges[0], truth["truth_underflow"][si], color="black", marker="v", s=26)
            ax.scatter(edges[-1], truth["truth_censored"][si], color="black", marker="^", s=26)
            for m in cat.models:
                d = data["models"][m]
                pm = d["pred_mass"][si]
                color = cat.color(m)
                ax.plot(centers, _nanmean(pm), color=color, ls=cat.ls(m), linewidth=1.8, label=cat.label(m))
                ax.fill_between(centers, _percentile(pm, 16), _percentile(pm, 84), color=color, alpha=0.10)
                ax.scatter(edges[0], d["pred_underflow"][si], color=color, marker="v", s=24)
                ax.scatter(edges[-1], d["pred_censored"][si], color=color, marker="^", s=24)
            ax.set_xscale("log")
            ax.set_xlim(edges[0], edges[-1])
            ax.set(title=label, xlabel="MFP distance R [cMpc]", ylabel="probability per log bin")
            ax.grid(alpha=0.2)
        for ax in axes[len(labels):]:
            ax.set_visible(False)
        fig.suptitle(_title("Transverse mean-free-path ionized-bubble distributions", ctx))
        fig.tight_layout()
        shared_legend(fig, handles=model_handles(cat), fontsize=9,
                      extra=style_key([("16–84 % over cones", {"kind": "band", "color": "0.5", "alpha": 0.25}),
                                       ("no ionized pixels", {"marker": "v", "ms": 6}),
                                       ("censored at box length", {"marker": "^", "ms": 6})]))
        written.append(save(fig, ctx.out, f"bubble_size_distribution{suffix}", dpi=180, quiet=ctx.quiet))
    return written


def _bars(ctx, data, stages, labels_fn, figsize, rotation, title0, legend_kw, name, dpi=180):
    plt = pyplot()
    cat = ctx.catalog
    labels = data["labels"]
    x = np.arange(len(stages))
    width = 0.8 / max(len(cat.models), 1)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for index, m in enumerate(cat.models):
        d = data["models"][m]
        offset = (index - (len(cat.models) - 1) / 2) * width
        axes[0].bar(x + offset, d["w1"][stages], width, label=cat.label(m), color=cat.color(m))
        axes[1].bar(x + offset, 100 * d["bias"][stages], width, label=cat.label(m), color=cat.color(m))
    axes[0].set(ylabel="restricted Wasserstein distance [cMpc]", title=title0)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(ylabel="restricted mean MFP bias [%]", title="Bubble-size bias")
    for ax in axes:
        ax.set_xticks(x, [labels_fn(labels[i]) for i in stages], rotation=rotation, ha="right")
        ax.grid(axis="y", alpha=0.2)
    if ctx.title:
        fig.suptitle(ctx.title)
    fig.tight_layout()
    shared_legend(fig, handles=model_handles(cat, truth=False, kind="patch"),
                  fontsize=legend_kw.get("fontsize", 9))
    return save(fig, ctx.out, name, dpi=dpi, quiet=ctx.quiet)


@figure("bubble_size_summary", group="bubbles", needs=("curves",),
        help="per-stage restricted Wasserstein distance and mean-MFP bias bars (fno-21cm bubble_size_evaluation)")
def bubble_size_summary(ctx):
    written = []
    for suffix, data in _collect(ctx.catalog):
        stages = list(range(len(data["labels"])))
        written.append(_bars(ctx, data, stages, str, (15, 5), 30,
                             "BSD distribution error (0 / box-length caps)", {},
                             f"bubble_size_summary{suffix}"))
    return written
