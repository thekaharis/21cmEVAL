"""Slice figures from the ``gallery`` metric: per-sample galleries and prediction - truth pages.

``plot_gallery`` is rhizome-neural-operator's ``ebm21cm.compare_cones3d`` page: a row per
ionization stage; x_HI truth, each model, each model - truth, then the same for T_b when the runs
provide it. x_HI in viridis on [0, 1], T_b in magma on the truth's 1-99.5 % range per row,
residuals in RdBu_r (x_HI +-1, T_b symmetric 99 % of the largest model residual per row), slice
RMSE in each model panel. ``plot_deltas`` is ``ebm21cm.delta_slices``: per sample a row per stage
(truth, then each model's prediction - truth, red: too neutral), and an overview page with the
x_HI ~ 0.4 slice of every sample.

Fields are merged across runs by sample id (an x_HI pass and a T_b pass with the true x_HI as
input give both blocks); truth must agree between models.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .registry import figure
from .common import pyplot, save

LABEL = {"xhi": "x$_{HI}$", "tb": "T$_b$"}
OVERVIEW_ROWS = 10


def cone_name(sid: str) -> str:
    """'cone12316' / 'cone_12316' / '12316' -> '12316'; other ids unchanged."""
    m = re.fullmatch(r"(?:cone_?)?(\d+)", str(sid))
    return m.group(1) if m else str(sid)
UNIT = {"xhi": "", "tb": " [mK]"}


def collect(cat):
    """{sample id: {"z", "xbar", "stages", "truth": {field: (S,H,W)}, "pred": {model: {field: ...}}}}."""
    samples = {}
    for run in cat.runs:
        for m in run.models:
            if m not in cat.models:
                continue
            c = run.curves(m)
            if "gallery_ids" not in c:
                continue
            fields = sorted({k[len("gallery_"):-len("_truth")] for k in c if k.startswith("gallery_")
                             and k.endswith("_truth")})
            for i, sid in enumerate(c["gallery_ids"]):
                sid = str(sid)
                entry = samples.setdefault(sid, {"z": c["gallery_z"][i], "xbar": c["gallery_xbar"][i],
                                                 "stages": c["gallery_stages"], "truth": {}, "pred": {}})
                for f in fields:
                    truth, pred = c[f"gallery_{f}_truth"][i], c[f"gallery_{f}_pred"][i]
                    if not np.isfinite(truth).any():
                        continue
                    if f in entry["truth"]:
                        if not np.allclose(entry["truth"][f], truth, atol=1e-4, rtol=1e-4, equal_nan=True):
                            raise SystemExit(f"gallery: truth of {f!r} for sample {sid} differs between runs")
                    else:
                        entry["truth"][f] = truth
                    entry["pred"].setdefault(m, {})[f] = pred
    return {sid: e for sid, e in samples.items() if all(m in e["pred"] for m in cat.models)}


def plot_gallery(cat, out: Path, ids=None, quiet=False):
    plt = pyplot()
    samples = collect(cat)
    if not samples:
        if not quiet:
            print("gallery: no stored slices (run with --metrics ...,gallery)")
        return []
    names, n_m, written = cat.models, len(cat.models), []
    label = lambda m: cat.label(m)
    for sid, e in samples.items():
        if ids and sid not in ids:
            continue
        fields = [f for f in ("xhi", "tb") if f in e["truth"] and all(f in e["pred"][m] for m in names)]
        fields += [f for f in e["truth"] if f not in fields and all(f in e["pred"][m] for m in names)]
        n_st = len(e["stages"])
        per = 1 + 2 * n_m
        fig, ax = plt.subplots(n_st, per * len(fields), figsize=(2.1 * per * len(fields) + 0.6, 2.75 * n_st),
                               constrained_layout=True, squeeze=False)
        for i in range(n_st):
            for b, f in enumerate(fields):
                tr = e["truth"][f][i]
                ps = [e["pred"][m][f][i] for m in names]
                if f == "xhi":
                    cm, v0, v1, dl = "viridis", 0.0, 1.0, 1.0
                else:
                    cm = "magma"
                    v0, v1 = np.percentile(tr, [1, 99.5])
                    dl = max(float(np.percentile(np.abs(p - tr), 99)) for p in ps) or 1.0
                lab_f, unit = LABEL.get(f, f), UNIT.get(f, "")
                panels = [(f"{lab_f} truth{unit}", tr, cm, v0, v1, None)]
                panels += [(f"{lab_f} {label(m)}", p, cm, v0, v1, p - tr) for m, p in zip(names, ps)]
                panels += [(f"{lab_f} {label(m)}\n− truth{unit}", p - tr, "RdBu_r", -dl, dl, None)
                           for m, p in zip(names, ps)]
                for j, (title, img, c, lo, hi, err) in enumerate(panels):
                    a = ax[i, b * per + j]
                    im = a.imshow(img, cmap=c, vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
                    a.set_xticks([])
                    a.set_yticks([])
                    if i == 0:
                        a.set_title(title, fontsize=9)
                    if err is not None:
                        rmse = float(np.sqrt(np.mean(err ** 2)))
                        a.text(0.02, 0.97, f"{rmse:.3f}" if f == "xhi" else f"{rmse:.1f} mK", transform=a.transAxes,
                               va="top", fontsize=8, color="w", bbox=dict(facecolor="k", alpha=0.55, lw=0))
                    if j in (n_m, per - 1):
                        fig.colorbar(im, ax=a, shrink=0.8)
            ax[i, 0].set_ylabel(f"cone {cone_name(sid)}\nz = {e['z'][i]:.2f}, <x$_{{HI}}$> = {e['xbar'][i]:.2f}",
                                fontsize=9)
        h, w = e["truth"][fields[0]].shape[1:]
        fig.suptitle(f"Test cone {cone_name(sid)}: transverse slices at four ionization stages; numbers are slice "
                     f"RMSE. {h}×{w} cells.", fontsize=10)
        written.append(save(fig, Path(out), f"cone_{cone_name(sid)}", dpi=170, quiet=quiet))
    return written


def plot_deltas(cat, out: Path, field="xhi", ids=None, quiet=False):
    plt = pyplot()
    samples = {sid: e for sid, e in collect(cat).items() if field in e["truth"] and (not ids or sid in ids)}
    if not samples:
        if not quiet:
            print(f"deltas: no stored {field} slices")
        return []
    names = cat.models
    vlim = 1.0 if field == "xhi" else None
    cbar = "signed error (red: too neutral)" if field == "xhi" else f"signed error{UNIT.get(field, '')}"

    def page(rows, path, title):
        fig, axes = plt.subplots(len(rows), 1 + len(names), figsize=(3.0 * (1 + len(names)) + 0.6, 3.0 * len(rows) + 0.6),
                                 squeeze=False)
        im = None
        for i, (truth, deltas, z, xbar, label_row) in enumerate(rows):
            axes[i, 0].imshow(truth, origin="lower", cmap="viridis" if field == "xhi" else "magma",
                              vmin=0 if field == "xhi" else None, vmax=1 if field == "xhi" else None,
                              interpolation="nearest")
            axes[i, 0].set_ylabel(f"{label_row}\nz={z:.2f}  <x_HI>={xbar:.2f}", fontsize=8)
            lim = vlim or max(float(np.percentile(np.abs(d), 99)) for d in deltas.values()) or 1.0
            for a, (m, d) in zip(axes[i, 1:], deltas.items()):
                im = a.imshow(d, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                rmse = float(np.sqrt(np.mean(d ** 2)))
                a.text(0.02, 0.97, f"RMSE {rmse:.3f}", transform=a.transAxes, va="top", fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
            for a in axes[i]:
                a.set_xticks([])
                a.set_yticks([])
        for j, name in enumerate([f"truth {field if field != 'xhi' else 'x_HI'}"]
                                 + [f"{cat.label(m)}\nprediction − truth" for m in names]):
            axes[0, j].set_title(name, fontsize=9)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout(rect=(0, 0, 0.86, 0.98))
        cax = fig.add_axes((0.88, 0.25, 0.015, 0.5))
        fig.colorbar(im, cax=cax).set_label(cbar, fontsize=8)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        if not quiet:
            print("wrote", path)
        return path

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written, overview = [], []
    for sid, e in samples.items():
        order = np.argsort(e["stages"])                      # x-bar 0.2 ... 0.8 top to bottom
        truth = e["truth"][field]
        rows = [(truth[k], {m: e["pred"][m][field][k] - truth[k] for m in names}, e["z"][k], e["xbar"][k],
                 f"x̄≈{e['stages'][k]:.1f}") for k in order]
        written.append(page(rows, out / f"cone_{cone_name(sid)}_deltas.png",
                            f"Test cone {cone_name(sid)}: prediction − truth per slice"))
        mid = order[len(order) // 2 - (1 if len(order) % 2 == 0 else 0)]
        overview.append((truth[mid], {m: e["pred"][m][field][mid] - truth[mid] for m in names}, e["z"][mid],
                         e["xbar"][mid], f"cone {cone_name(sid)}"))
    if overview:
        stage = next(iter(samples.values()))["stages"][mid]
        pages = [overview[i:i + OVERVIEW_ROWS] for i in range(0, len(overview), OVERVIEW_ROWS)]
        for n, rows in enumerate(pages, 1):
            name = "overview_deltas.png" if len(pages) == 1 else f"overview_deltas_{n}.png"
            suffix = "" if len(pages) == 1 else f" ({n}/{len(pages)})"
            written.append(page(rows, out / name, f"Prediction − truth at x̄_HI ≈ {stage:.1f}{suffix}"))
    return written


@figure("gallery", group="maps", help="per-sample gallery: stages x (x_HI | T_b) truth / models / residuals")
def _fig_gallery(ctx):
    return plot_gallery(ctx.catalog, ctx.out / "gallery", ids=ctx.ids, quiet=ctx.quiet)


@figure("deltas", group="maps", help="prediction - truth per stage, per sample, plus overview pages")
def _fig_deltas(ctx):
    return plot_deltas(ctx.catalog, ctx.out / "deltas", ids=ctx.ids, quiet=ctx.quiet)
