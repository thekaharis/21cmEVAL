"""Map, slice and lightcone figures from stored cubes (ports of fno-21cm / rhizome viz scripts).

Every figure reads the prediction store (``eval21 run --save-predictions DIR``), draws each target
field that the truth holds and every shown model predicted (x_HI and/or T_b), and works for any
number of models: where an original drew one prediction, each catalog model gets its own
prediction (and error) panel, in catalog order, labelled with the catalog label. x_HI is drawn in
viridis on [0, 1] everywhere; T_b keeps the original script's colour map (inferno in the
fno-21cm multi-field scripts, magma in ``multifield_detailed``, RdBu_r in rhizome's ``examples``).
Errors are prediction - truth in RdBu_r (red: prediction too high / too neutral).

=========================  ==================================================  ==========================================  =====
figure                     ports                                               shows                                       needs
=========================  ==================================================  ==========================================  =====
comparison_3d              fno-21cm viz/visualize_3d.py ``plot_z_slices``      per sample/field: rows of z-slices;         store
                           (``comparison_3d_<split>_cone<id>``); with          density | truth | per model prediction,
                           ``--opt detailed=1`` visualize_3d_detailed.py       pred - true (MSE); active-z picker and
                                                                               low-z crop in detailed mode
lightcone_3d               visualize_3d.py ``plot_lightcone_strip``            per sample/field: edge-on x-z strip at      store
                           (``lightcone_3d_<split>_cone<id>``)                 y = Ny/2 over the LOS: density, truth,
                                                                               per model prediction and pred - true
scatter_3d                 visualize_3d.py ``plot_scatter``                    per sample/field: voxel hexbin truth vs     store
                           (``scatter_3d_<split>_cone<id>``)                   prediction, one panel per model, R^2/RMSE
lightcone_grid_3d          visualize_3d.py ``plot_lightcone_summary_grid``     per field: one row per sample (sorted by    store
                           (``lightcone_grid_3d_<split>``)                     <x_HI> at z = 7): truth | prediction |
                                                                               pred - true strips
multifield_slices          viz/multifield_slices.py ``figure``                 per sample/field: transverse slices at the  store
                           (``cone<id>_<field>_slices``)                       active redshifts; truth / predictions /
                                                                               pred - truth rows, RMSE per column
multifield_los             multifield_slices.py ``los_figure``                 per sample/field: (x, z) planes at several  store
                           (``cone<id>_<field>_los``)                          y, cropped to the active band
multifield_strips          viz/multifield_lightcone_strips.py ``build``        per field: stacked full-LOS strips, groups  store
                           (``<field>_strips[_active][_partN]``)               of truth / predictions / errors per sample
                                                                               on a common colour scale
multifield_detailed        viz/multifield_detailed.py ``page``                 per sample/field/model: x-z slice triptych, store
                           (``cone<id>_<field>``)                              transverse maps, error vs z, parity,
                                                                               error histogram, P_perp ratio, r(k_perp)
lightcone_animation        viz/animate_lightcone.py ``animate``                GIF (or MP4) of transverse slices stepping  store
                           (``cone<id>_lightcone[_compare].gif``)              through z: truth and each model, per field
slice_comparison           viz/visualize_xhi2d.py ``plot_comparison``          per field: 4 slices spread over all stored  store
                           (``comparison_<split>``)                            (sample, z) slices; density | truth |
                                                                               prediction | pred - true per model
slice_scatter              visualize_xhi2d.py ``plot_scatter``                 per field: hexbin over 32 stored slices,    store
                           (``scatter_<split>``)                               one panel per model, R^2/RMSE
representative_z_slices    viz/visualize_xhi2d_representative.py               per field: slices at z quantiles (local-    store
                           ``render_slices`` (``representative_z_slices``)     median x_HI); truth | predictions | errors
                                                                               with RMSE/MAE
representative_z_compare   viz/compare_xhi2d_models.py ``plot_predictions``    per field: truth and every model at z       store
                           (``representative_z_predictions`` / ``_errors``)    quantiles, plus the signed-error page
ebm_examples               rhizome ebm21cm/viz.py ``examples``                 per model: rows of samples (slice at        store
                                                                               x_HI ~ 0.5): delta | per field truth,
                                                                               draws, ensemble mean, ensemble std
regression_examples        rhizome ebm21cm/viz.py ``regression_examples``      per model/field: rows of samples: log       store
                                                                               density | truth | prediction | error
=========================  ==================================================  ==========================================  =====

Options (``--opt key=value``): ``slices`` (comparison_3d / multifield rows), ``detailed=1``
(comparison/lightcone/scatter/grid: active-z slices and shared low-z crop),
``zmin=auto|<z>`` (crop), ``stratify_z`` (grid ranking, 7.0), ``cones`` (grid rows),
``crop=1`` / ``y_frac`` / ``per_figure`` / ``error=0`` (strips), ``full_z=1`` (los),
``frames`` / ``fps`` / ``dpi`` / ``format=gif|mp4`` / ``animate_n`` (animation, default 1 sample,
48 frames), ``quantiles`` (representative), ``stage`` (examples, 0.5), ``draws_shown`` (2).

Not ported: z_re maps (``visualize_zre_detailed.py``: the store holds 3-D fields only, no z_re
map); ``physical_3d`` diagnostics and ``metrics_comparison`` bars (statistics, other modules);
rhizome ``spectra`` / ``bubble_sizes`` (statistics) and ``energy_gaps`` (model internal).
"""
from __future__ import annotations

import re

import numpy as np

from .common import field_name, pyplot, save, wrap_label
from .registry import SkipFigure, figure
from .slices import cone_name

PLAIN = {"xhi": "x_HI", "tb": "T_b"}
UNITS = {"xhi": "dimensionless", "tb": "mK"}
CMAP_MF = {"xhi": "viridis", "tb": "inferno"}           # multi-field scripts (x_HI: viridis, see above)
BOUNDS = {"xhi": (0.0, 1.0)}


# ------------------------------------------------------------------------------- helpers
def _plain(field):
    return PLAIN.get(field, field)


def _units(field):
    return UNITS.get(field, "")


def _cmap(field):
    return CMAP_MF.get(field, "viridis")


def _flag(ctx, key, default=False):
    value = ctx.options.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _floats(value, default):
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(v) for v in re.split(r"[,;+\s]+", str(value)) if v]


def edges(centres):
    """Cell edges from monotone cell centres (fno-21cm ``viz._zaxis.edges``)."""
    c = np.asarray(centres, dtype=float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[1:] + c[:-1])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def find_low_z_cutoff_index(history, target_z, min_change=0.01, baseline_points=8, smoothing_points=5,
                            min_consecutive=3, buffer_points=1) -> int:
    """fno-21cm ``util.metrics_21cm.find_low_z_cutoff_index``: first index where the (smoothed)
    history departs from its settled low-z value for ``min_consecutive`` slices."""
    history = np.asarray(history, np.float64)
    target_z = np.asarray(target_z, np.float64)
    n = len(history)
    if n == 0 or len(target_z) != n or np.any(np.diff(target_z) <= 0):
        return 0
    baseline_points = max(1, min(int(baseline_points), n))
    smoothing_points = max(1, min(int(smoothing_points), n))
    min_consecutive = max(1, min(int(min_consecutive), n))
    if smoothing_points > 1:
        left = smoothing_points // 2
        padded = np.pad(history, (left, smoothing_points - 1 - left), mode="edge")
        smoothed = np.convolve(padded, np.full(smoothing_points, 1.0 / smoothing_points), mode="valid")
    else:
        smoothed = history
    baseline = float(np.median(smoothed[:baseline_points]))
    changed = np.abs(smoothed - baseline) >= float(min_change)
    sustained = np.convolve(changed.astype(np.int8), np.ones(min_consecutive, np.int8), mode="valid")
    starts = np.flatnonzero(sustained == min_consecutive)
    return 0 if starts.size == 0 else max(0, int(starts[0]) - max(0, int(buffer_points)))


class _Cone:
    """One stored sample: truth fields, inputs, and each model's predicted fields."""

    def __init__(self, store, sid, models):
        self.id = str(sid)
        self.name = cone_name(sid)
        t = store.truth(sid)
        self.truth = t["targets"]
        self.inputs = t["inputs"]
        self.preds = {m: store.prediction(m, sid) for m in models}
        nz = next(iter(self.truth.values())).shape[-1]
        z = t.get("z_grid")
        self.has_z = z is not None and len(z) == nz
        self.z = np.asarray(z, float) if self.has_z else np.arange(nz, dtype=float)
        self.zlabel = "redshift z" if self.has_z else "LOS cell"
        box = t.get("box_mpc")
        self.box = float(box) if box is not None and np.isfinite(box) else None
        self.density = self.inputs.get("density")
        if self.density is not None and np.ndim(self.density) != 3:
            self.density = None

    def fields(self, models):
        order = [f for f in ("xhi", "tb") if f in self.truth] + \
                [f for f in self.truth if f not in ("xhi", "tb") and "__" not in f]
        return [f for f in order if all(f in self.preds[m] for m in models)]

    def xbar(self, field=None):
        src = self.truth.get("xhi")
        if src is None:
            src = self.truth[field or next(iter(self.truth))]
        return src.mean(axis=(0, 1))

    def stage_index(self, stage, field=None):
        if "xhi" in self.truth:
            return int(np.argmin(np.abs(self.xbar() - stage)))
        return self.z.size // 2


def _setup(ctx):
    store = ctx.store
    models = [m for m in ctx.catalog.models if m in store.models]
    if not models:
        raise SkipFigure("none of the shown models is in the prediction store")
    ids = [i for i in (ctx.ids or store.samples) if i in store.samples and all(store.has(m, i) for m in models)]
    if not ids:
        raise SkipFigure("no stored sample with predictions of every shown model")
    return models, ids


def _cones(ctx, models, ids):
    cones = [_Cone(ctx.store, sid, models) for sid in ids]
    fields = []
    for c in cones:
        for f in c.fields(models):
            if f not in fields:
                fields.append(f)
    if not fields:
        raise SkipFigure("no target field predicted by every shown model")
    return cones, fields


def _mlabel(cat, m, models, base):
    """Panel title: the original's text, with the model label when several models are shown."""
    return base if len(models) == 1 else f"{wrap_label(cat.label(m), 28)}\n{base}"


def _limits(field, ref, pct=(0.5, 99.5)):
    if field in BOUNDS:
        return BOUNDS[field]
    ref = np.asarray(ref, float)
    lo, hi = np.percentile(ref, pct)
    if not hi > lo:
        lo, hi = float(ref.min()), float(ref.max())
        hi = hi if hi > lo else lo + 1e-6
    return float(lo), float(hi)


def _rgb_limits(field, ref):
    """Colour limits of the visualize_3d / xhi2d figures: x_HI on [0, 1], T_b percentiles."""
    return _limits(field, ref)


def _sym(errs, floor=0.01):
    """visualize_3d's error scale: max |err| (at least ``floor``), shared by the models."""
    return max([floor] + [float(np.max(np.abs(e))) for e in errs])


def _z_start(ctx, cones):
    """Low-z crop shared by every plot of a figure (visualize_3d_detailed ``choose_plot_z_start``)."""
    zmin = ctx.options.get("zmin")
    if zmin is None:
        zmin = "auto" if _flag(ctx, "detailed") else None
    if zmin is None or not cones:
        return 0
    z = cones[0].z
    if str(zmin).lower() != "auto":
        return min(len(z) - 1, int(np.searchsorted(z, float(zmin), side="left")))
    if not cones[0].has_z or any(len(c.z) != len(z) for c in cones):
        return 0
    history = np.mean([c.xbar() for c in cones], axis=0)
    return find_low_z_cutoff_index(history, z, min_change=float(ctx.opt("low_z_min_change", 0.01, float)))


def pick_active_z_slices(truth, n_slices, frac=0.05):
    """visualize_3d_detailed ``pick_active_z_slices``: n indices across the window where the
    transverse std exceeds ``frac`` of its maximum."""
    per_z_std = truth.std(axis=(0, 1))
    max_std = float(per_z_std.max())
    if max_std < 1e-6:
        return np.linspace(0, len(per_z_std) - 1, n_slices, dtype=int).tolist()
    active = np.where(per_z_std > frac * max_std)[0]
    if len(active) < n_slices:
        return sorted(np.argsort(-per_z_std)[:n_slices].tolist())
    picks = np.linspace(active[0], active[-1], n_slices, dtype=int)
    return sorted(dict.fromkeys(picks.tolist()))


def _strip_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])


# ------------------------------------------------------------- visualize_3d: z-slice panels
def _slice_rows_figure(plt, cat, models, field, rows, suptitle, density_title, err_title, cbar_label):
    """visualize_3d ``plot_z_slices`` / visualize_xhi2d ``plot_comparison`` layout: one row per
    slice; density | truth | per model prediction and pred - true | error colour bar."""
    from matplotlib.gridspec import GridSpec

    has_d = any(r["dens"] is not None for r in rows)
    ncol = int(has_d) + 1 + 2 * len(models)
    n = len(rows)
    fig = plt.figure(figsize=(3.5 * ncol, 4 * n))
    gs = GridSpec(n, ncol + 1, figure=fig, width_ratios=[1] * ncol + [0.05], hspace=0.3, wspace=0.3)
    lab = _plain(field)
    for row, r in enumerate(rows):
        t = r["truth"]
        col = 0
        if has_d:
            ax = fig.add_subplot(gs[row, col]); col += 1
            if r["dens"] is not None:
                ax.imshow(r["dens"], cmap="plasma", origin="lower")
            ax.set_title(density_title(r))
            _strip_axes(ax)
        v0, v1 = _rgb_limits(field, t)
        cmap = "viridis" if field == "xhi" else "inferno"
        ax = fig.add_subplot(gs[row, col]); col += 1
        ax.imshow(t, cmap=cmap, origin="lower", vmin=v0, vmax=v1)
        ax.set_title(f"True {lab}")
        _strip_axes(ax)
        errs = {m: r["preds"][m] - t for m in models}
        vmax_err = _sym(errs.values())
        im3 = None
        for m in models:
            p = r["preds"][m]
            ax = fig.add_subplot(gs[row, col]); col += 1
            ax.imshow(p, cmap=cmap, origin="lower", vmin=v0, vmax=v1)
            ax.set_title(_mlabel(cat, m, models, f"Predicted {lab}"))
            _strip_axes(ax)
            ax = fig.add_subplot(gs[row, col]); col += 1
            mse = float(np.mean(errs[m] ** 2))
            im3 = ax.imshow(errs[m], cmap="RdBu_r", origin="lower", vmin=-vmax_err, vmax=vmax_err)
            ax.set_title((err_title if len(models) == 1 else f"{wrap_label(cat.label(m), 28)}: {err_title}")
                         + (f"\nMSE = {mse:.4f}" if field == "xhi" else f"\nMSE = {mse:.4g}"))
            _strip_axes(ax)
        cax = fig.add_subplot(gs[row, ncol])
        plt.colorbar(im3, cax=cax, label=cbar_label)
    fig.suptitle(suptitle, fontsize=13, y=0.995)
    return fig


@figure("comparison_3d", group="maps", needs=("store",),
        help="z-slice rows per sample: density | truth | prediction | pred - true per model")
def comparison_3d(ctx):
    """visualize_3d ``plot_z_slices`` (``--opt detailed=1``: visualize_3d_detailed's active-z picker)."""
    plt = pyplot()
    cat = ctx.catalog
    models, ids = _setup(ctx)
    cones, _ = _cones(ctx, models, ids)
    detailed = _flag(ctx, "detailed")
    n_slices = int(ctx.opt("slices", 6 if detailed else 4, int))
    z0 = _z_start(ctx, cones)
    split = ctx.title or "test"
    written = []
    for c in cones:
        for field in c.fields(models):
            truth = c.truth[field]
            nz = truth.shape[-1]
            if detailed:
                ref = c.truth.get("xhi", truth)
                idxs = [i for i in pick_active_z_slices(ref, n_slices) if i >= z0]
                if not idxs:
                    idxs = np.linspace(z0, nz - 1, n_slices, dtype=int).tolist()
            else:
                idxs = np.linspace(0, nz - 1, n_slices, dtype=int).tolist()
            rows = [{"dens": None if c.density is None else c.density[:, :, k], "truth": truth[:, :, k],
                     "preds": {m: c.preds[m][field][:, :, k] for m in models}, "z": c.z[k], "cone": c.name}
                    for k in idxs]
            dens_title = (lambda r: f"Density δ\nz={r['z']:.2f}  cone={r['cone']}") if c.has_z else \
                (lambda r: f"Density δ\nLOS cell {int(r['z'])}  cone={r['cone']}")
            fig = _slice_rows_figure(plt, cat, models, field, rows,
                                     f"3-D predictions, {field_name(field)} ({split}, cone {c.name})",
                                     dens_title, "Pred - True", f"d{_plain(field)}"
                                     + ("" if field == "xhi" else f" [{_units(field)}]"))
            written.append(save(fig, ctx.out / "comparison_3d", f"comparison_3d_cone{c.name}_{field}",
                                dpi=150, quiet=ctx.quiet))
    return written


# ------------------------------------------------------------- visualize_3d: x-z strips
@figure("lightcone_3d", group="maps", needs=("store",),
        help="edge-on x-z lightcone strip at y = Ny/2: density, truth, prediction, pred - true")
def lightcone_3d(ctx):
    """visualize_3d ``plot_lightcone_strip``."""
    plt = pyplot()
    cat = ctx.catalog
    models, ids = _setup(ctx)
    cones, _ = _cones(ctx, models, ids)
    z0 = _z_start(ctx, cones)
    split = ctx.title or "test"
    written = []
    for c in cones:
        for field in c.fields(models):
            truth = c.truth[field]
            j = truth.shape[1] // 2
            ze = edges(c.z[z0:])
            xe = np.arange(truth.shape[0] + 1)
            t_strip = truth[:, j, z0:]
            p_strips = {m: c.preds[m][field][:, j, z0:] for m in models}
            e_strips = {m: p_strips[m] - t_strip for m in models}
            vmax_err = _sym(e_strips.values())
            v0, v1 = _rgb_limits(field, t_strip)
            cmap = "viridis" if field == "xhi" else "inferno"
            panels = []
            if c.density is not None:
                panels.append((f"Density δ  (y = Ny/2)   cone {c.name}, {split}", c.density[:, j, z0:],
                               dict(cmap="plasma")))
            panels.append((f"True {_plain(field)}" + ("" if panels else f"  (y = Ny/2)   cone {c.name}, {split}"),
                           t_strip, dict(cmap=cmap, vmin=v0, vmax=v1)))
            for m in models:
                panels.append((_mlabel(cat, m, models, f"Predicted {_plain(field)}").replace("\n", ": "),
                               p_strips[m], dict(cmap=cmap, vmin=v0, vmax=v1)))
                panels.append(((f"{wrap_label(cat.label(m), 28)}: " if len(models) > 1 else "") + "Pred - True",
                               e_strips[m], dict(cmap="RdBu_r", vmin=-vmax_err, vmax=vmax_err)))
            fig, axes = plt.subplots(len(panels), 1, figsize=(14, 2 * len(panels)), sharex=True, squeeze=False)
            for ax, (title, img, kw) in zip(axes[:, 0], panels):
                ax.pcolormesh(ze, xe, img, shading="flat", rasterized=True, **kw)
                ax.set_title(title)
                ax.set_ylabel("x cell")
            axes[-1, 0].set_xlabel(c.zlabel)
            axes[-1, 0].set_xlim(ze[0], ze[-1])
            fig.tight_layout()
            written.append(save(fig, ctx.out / "lightcone_3d", f"lightcone_3d_cone{c.name}_{field}",
                                dpi=150, quiet=ctx.quiet))
    return written


# ------------------------------------------------------------- visualize_3d: voxel scatter
def _hexbin_panel(ax, t_flat, p_flat, field, title):
    max_pts = 50_000
    if len(t_flat) > max_pts:
        idx = np.random.default_rng(42).choice(len(t_flat), max_pts, replace=False)
        t_flat, p_flat = t_flat[idx], p_flat[idx]
    if field == "xhi":
        lo, hi = 0.0, 1.0
    else:
        lo = float(min(t_flat.min(), p_flat.min()))
        hi = float(max(t_flat.max(), p_flat.max()))
        hi = hi if hi > lo else lo + 1e-6
    ax.hexbin(t_flat, p_flat, gridsize=100, cmap="Blues", mincnt=1, bins="log",
              extent=None if field == "xhi" else (lo, hi, lo, hi))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="perfect")
    unit = "" if field == "xhi" else f" [{_units(field)}]"
    ax.set_xlabel(f"True {_plain(field)}{unit}")
    ax.set_ylabel(f"Predicted {_plain(field)}{unit}")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend()
    with np.errstate(all="ignore"):
        r2 = float(np.corrcoef(t_flat, p_flat)[0, 1] ** 2)
    rmse = float(np.sqrt(np.mean((p_flat - t_flat) ** 2)))
    ax.text(0.05, 0.95, f"R^2 = {r2:.4f}\nRMSE = {rmse:.4f}" if field == "xhi" else
            f"R^2 = {r2:.4f}\nRMSE = {rmse:.4g}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))


@figure("scatter_3d", group="maps", needs=("store",),
        help="voxel hexbin of truth vs prediction per sample, one panel per model, R^2 / RMSE")
def scatter_3d(ctx):
    """visualize_3d ``plot_scatter``."""
    plt = pyplot()
    cat = ctx.catalog
    models, ids = _setup(ctx)
    cones, _ = _cones(ctx, models, ids)
    z0 = _z_start(ctx, cones)
    split = ctx.title or "test"
    written = []
    for c in cones:
        z_note = "" if z0 == 0 else f", z >= {float(c.z[z0]):.2f}"
        title = f"Scatter - {split} (cone {c.name}{z_note})"
        for field in c.fields(models):
            t_flat = c.truth[field][..., z0:].ravel()
            fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), squeeze=False)
            for ax, m in zip(axes[0], models):
                _hexbin_panel(ax, t_flat, c.preds[m][field][..., z0:].ravel(), field,
                              title if len(models) == 1 else wrap_label(cat.label(m), 28))
            if len(models) > 1:
                fig.suptitle(title)
            fig.tight_layout()
            written.append(save(fig, ctx.out / "scatter_3d", f"scatter_3d_cone{c.name}_{field}",
                                dpi=150, quiet=ctx.quiet))
    return written


# ------------------------------------------------------------- visualize_3d: strip grid
def _pick_by_summary(summaries, n):
    """visualize_3d ``pick_cones_by_reion_behavior``: evenly spaced percentiles (5..95)."""
    count = len(summaries)
    if n >= count:
        order = list(range(count))
    else:
        targets = np.percentile(summaries, np.linspace(5.0, 95.0, n))
        order = list(dict.fromkeys(int(np.argmin(np.abs(summaries - t))) for t in targets))
        for k in range(count):
            if len(order) >= n:
                break
            if k not in order:
                order.append(k)
    return sorted(order, key=lambda i: summaries[i])


@figure("lightcone_grid_3d", group="maps", needs=("store",),
        help="one x-z strip row per sample (sorted by <x_HI> at z = 7): truth | prediction | pred - true")
def lightcone_grid_3d(ctx):
    """visualize_3d ``plot_lightcone_summary_grid``."""
    plt = pyplot()
    cat = ctx.catalog
    models, ids = _setup(ctx)
    cones, fields = _cones(ctx, models, ids)
    stratify_z = float(ctx.opt("stratify_z", 7.0, float))
    summaries = []
    for c in cones:
        k = int(np.argmin(np.abs(c.z - stratify_z))) if c.has_z else c.z.size // 2
        src = c.truth.get("xhi", next(iter(c.truth.values())))
        summaries.append(float(src[:, :, k].mean()))
    summaries = np.asarray(summaries)
    order = _pick_by_summary(summaries, int(ctx.opt("cones", len(cones), int)))
    z0 = _z_start(ctx, cones)
    split = ctx.title or "test"
    written = []
    for field in fields:
        rows = [i for i in order if field in cones[i].fields(models)]
        if not rows:
            continue
        n = len(rows)
        ncol = 1 + 2 * len(models)
        fig, axes = plt.subplots(n, ncol, figsize=(6 * ncol, 2.0 * n + 1), squeeze=False)
        z_ref = cones[rows[0]].z
        for row, i in enumerate(rows):
            c = cones[i]
            truth = c.truth[field]
            j = truth.shape[1] // 2
            ze, xe = edges(c.z[z0:]), np.arange(truth.shape[0] + 1)
            t_strip = truth[:, j, z0:]
            p_strips = [c.preds[m][field][:, j, z0:] for m in models]
            e_strips = [p - t_strip for p in p_strips]
            vmax_err = _sym(e_strips)
            v0, v1 = _rgb_limits(field, t_strip)
            cmap = "viridis" if field == "xhi" else "inferno"
            panels = [(t_strip, dict(cmap=cmap, vmin=v0, vmax=v1), f"True {_plain(field)}")]
            panels += [(p, dict(cmap=cmap, vmin=v0, vmax=v1), _mlabel(cat, m, models, f"Predicted {_plain(field)}"))
                       for m, p in zip(models, p_strips)]
            panels += [(e, dict(cmap="RdBu_r", vmin=-vmax_err, vmax=vmax_err),
                        (f"{wrap_label(cat.label(m), 28)}: " if len(models) > 1 else "") + "Pred - True")
                       for m, e in zip(models, e_strips)]
            for col, (img, kw, title) in enumerate(panels):
                ax = axes[row, col]
                ax.pcolormesh(ze, xe, img, shading="flat", rasterized=True, **kw)
                ax.set_xlim(ze[0], ze[-1])
                if row == 0:
                    ax.set_title(title)
                if row == n - 1:
                    ax.set_xlabel(c.zlabel)
                else:
                    ax.set_xticklabels([])
            axes[row, 0].set_ylabel(f"cone {c.name}\n<x_HI>={summaries[i]:.2f}", fontsize=9)
        zlabel = f"z >= {float(z_ref[z0]):.2f}" if cones[rows[0]].has_z else f"LOS cell >= {z0}"
        fig.suptitle(f"Lightcone-strip grid across {n} reionization regimes  ({split}, {zlabel})",
                     y=1.0, fontsize=11)
        fig.tight_layout()
        written.append(save(fig, ctx.out / "lightcone_grid_3d", f"lightcone_grid_3d_{field}", dpi=150,
                            quiet=ctx.quiet))
    return written


# ------------------------------------------------------------- multifield_slices
def active_slices(truth, n):
    """multifield_slices ``active_slices``: n indices over the band with std > 15 % of the peak."""
    sd = truth.std(axis=(0, 1))
    if not np.any(sd > 0):
        return np.linspace(0, truth.shape[2] - 1, n).astype(int)
    active = np.flatnonzero(sd > 0.15 * sd.max())
    return np.unique(np.linspace(active[0], active[-1], n).astype(int))


def active_band(truth, pad=12):
    sd = truth.std(axis=(0, 1))
    if not np.any(sd > 0):
        return 0, truth.shape[2] - 1
    active = np.flatnonzero(sd > 0.05 * sd.max())
    return max(0, active[0] - pad), min(truth.shape[2] - 1, active[-1] + pad)


def _row_labels(cat, models, show_error=True):
    rows = [("truth", None, False)]
    rows += [("prediction" if len(models) == 1 else wrap_label(cat.label(m), 28), m, False) for m in models]
    if show_error:
        rows += [("pred - truth" if len(models) == 1 else f"{wrap_label(cat.label(m), 28)}\n- truth", m, True) for m in models]
    return rows


@figure("multifield_slices", group="maps", needs=("store",),
        help="transverse slices at the active redshifts: truth / predictions / pred - truth rows")
def multifield_slices(ctx):
    """multifield_slices ``figure`` (transverse orientation)."""
    from matplotlib.colors import TwoSlopeNorm

    plt = pyplot()
    cat = ctx.catalog
    models, ids = _setup(ctx)
    cones, _ = _cones(ctx, models, ids)
    n_slices = int(ctx.opt("slices", 5, int))
    written = []
    for c in cones:
        box = c.box or c.truth[next(iter(c.truth))].shape[0]
        for field in c.fields(models):
            truth = c.truth[field]
            idx = active_slices(truth, n_slices)
            n = len(idx)
            rows = _row_labels(cat, models)
            fig, axes = plt.subplots(len(rows), n, figsize=(3.05 * n + 1.6, 3.2 * len(rows)),
                                     gridspec_kw=dict(hspace=0.16, wspace=0.06), squeeze=False)
            cmap = _cmap(field)
            ext = [0, box, 0, box]
            units = _units(field)
            n_img = 1 + len(models)
            for col, j in enumerate(idx):
                t = truth[:, :, j]
                if field in BOUNDS:
                    v0, v1 = BOUNDS[field]
                else:
                    v0, v1 = np.percentile(t, [0.5, 99.5])
                    if v0 == v1:
                        v0, v1 = t.min(), max(t.max(), t.min() + 1e-6)
                errs = {m: c.preds[m][field][:, :, j] - t for m in models}
                lim = max(float(np.percentile(np.abs(e), 99)) for e in errs.values()) or 1.0
                last = im = None
                for r, (label, m, is_err) in enumerate(rows):
                    ax = axes[r, col]
                    if is_err:
                        e = errs[m]
                        im = ax.imshow(e.T, origin="lower", cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -lim, lim),
                                       extent=ext)
                        ax.set_xlabel(f"RMSE {np.sqrt((e ** 2).mean()):.3g}", fontsize=9)
                    else:
                        data = t if m is None else c.preds[m][field][:, :, j]
                        last = ax.imshow(data.T, origin="lower", cmap=cmap, vmin=v0, vmax=v1, extent=ext)
                    _strip_axes(ax)
                    if col == 0:
                        ax.set_ylabel(label, fontsize=12)
                    if r == 0:
                        ax.set_title(f"z = {c.z[j]:.2f}" if c.has_z else f"LOS cell {j}", fontsize=12)
                if field not in BOUNDS:
                    fig.colorbar(last, ax=axes[:n_img, col].tolist(), fraction=0.046, pad=0.02,
                                 label=units if col == n - 1 else "")
                elif col == n - 1:
                    fig.colorbar(last, ax=axes[:n_img, :].ravel().tolist(), fraction=0.023, pad=0.02,
                                 label=units)
                if col == n - 1:
                    fig.colorbar(im, ax=axes[n_img:, col].tolist(), fraction=0.046, pad=0.02, label=units)
            scale_note = "shared colour scale" if field in BOUNDS else "colour scale set per column"
            size = f"{box:.0f} x {box:.0f} Mpc" if c.box else f"{box} x {box} cells"
            fig.suptitle(f"{field_name(field)} — cone {c.name}   ({size} transverse slices, {scale_note})",
                         fontsize=15, y=0.965 if len(rows) <= 3 else 0.93)
            written.append(save(fig, ctx.out / "multifield_slices", f"cone{c.name}_{field}_slices", dpi=150,
                                quiet=ctx.quiet))
    return written


@figure("multifield_los", group="maps", needs=("store",),
        help="line-of-sight (x, z) planes at several y, cropped to the active band (--opt full_z=1)")
def multifield_los(ctx):
    """multifield_slices ``los_figure``."""
    from matplotlib.colors import TwoSlopeNorm

    plt = pyplot()
    cat = ctx.catalog
    models, ids = _setup(ctx)
    cones, _ = _cones(ctx, models, ids)
    n_slices = int(ctx.opt("slices", 4, int))
    full_z = _flag(ctx, "full_z")
    written = []
    for c in cones:
        for field in c.fields(models):
            truth = c.truth[field]
            box = c.box or truth.shape[0]
            lo, hi = (0, truth.shape[2] - 1) if full_z else active_band(truth)
            zc = c.z[lo:hi + 1]
            ys = np.unique((np.linspace(0.12, 0.88, n_slices) * truth.shape[1]).astype(int))
            n = len(ys)
            rows = _row_labels(cat, models)
            n_img = 1 + len(models)
            fig, axes = plt.subplots(len(rows), n, figsize=(3.0 * n + 1.8, 3.4 * len(rows)),
                                     gridspec_kw=dict(hspace=0.2, wspace=0.08), squeeze=False)
            cmap = _cmap(field)
            xe, ze = np.linspace(0.0, box, truth.shape[0] + 1), edges(zc)
            units = _units(field)
            if field in BOUNDS:
                v0, v1 = BOUNDS[field]
            else:
                v0, v1 = np.percentile(truth[:, :, lo:hi + 1], [0.5, 99.5])
            zlab = "redshift" if c.has_z else "LOS cell"
            for col, y in enumerate(ys):
                tt = truth[:, y, lo:hi + 1]
                errs = {m: c.preds[m][field][:, y, lo:hi + 1] - tt for m in models}
                lim = max(float(np.percentile(np.abs(e), 99)) for e in errs.values()) or 1.0
                last = im = None
                for r, (label, m, is_err) in enumerate(rows):
                    ax = axes[r, col]
                    if is_err:
                        e = errs[m]
                        im = ax.pcolormesh(xe, ze, e.T, cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -lim, lim),
                                           shading="flat", rasterized=True)
                    else:
                        data = tt if m is None else c.preds[m][field][:, y, lo:hi + 1]
                        last = ax.pcolormesh(xe, ze, data.T, cmap=cmap, vmin=v0, vmax=v1, shading="flat",
                                             rasterized=True)
                    if col == 0:
                        ax.set_ylabel(f"{label}\n{zlab}", fontsize=11)
                    else:
                        ax.set_yticks([])
                    if r == 0:
                        ax.set_title(f"y = {y * box / truth.shape[1]:.0f} {'Mpc' if c.box else 'cells'}",
                                     fontsize=11)
                    if r == len(rows) - 1:
                        rmse = np.sqrt((errs[m] ** 2).mean())
                        ax.set_xlabel(f"x [{'Mpc' if c.box else 'cells'}]\nRMSE {rmse:.3g}", fontsize=9)
                    elif is_err:
                        ax.set_xticks([])
                        ax.set_xlabel(f"RMSE {np.sqrt((errs[m] ** 2).mean()):.3g}", fontsize=9)
                    else:
                        ax.set_xticks([])
                if col == n - 1:
                    fig.colorbar(last, ax=axes[:n_img, :].ravel().tolist(), fraction=0.022, pad=0.02, label=units)
                    fig.colorbar(im, ax=axes[n_img:, col].tolist(), fraction=0.046, pad=0.02, label=units)
            crop = "full z range" if full_z else (f"cropped to z = {zc[0]:.1f}-{zc[-1]:.1f}" if c.has_z
                                                  else f"cropped to LOS cells {lo}-{hi}")
            scale = "shared colour scale" if field in BOUNDS else "shared percentile scale"
            fig.suptitle(f"{field_name(field)} — cone {c.name}   (line-of-sight slices, {crop}, {scale})",
                         fontsize=15, y=0.965 if len(rows) <= 3 else 0.93)
            written.append(save(fig, ctx.out / "multifield_los", f"cone{c.name}_{field}_los", dpi=150,
                                quiet=ctx.quiet))
    return written


# ------------------------------------------------------------- multifield_lightcone_strips
@figure("multifield_strips", group="maps", needs=("store",),
        help="stacked full-LOS lightcone strips per field: truth / predictions / errors per sample")
def multifield_strips(ctx):
    """multifield_lightcone_strips ``build`` (neutral_fraction_strips / brightness_temp_strips)."""
    from matplotlib.colors import TwoSlopeNorm

    plt = pyplot()
    cat = ctx.catalog
    models, ids = _setup(ctx)
    cones, fields = _cones(ctx, models, ids)
    y_frac = float(ctx.opt("y_frac", 0.5, float))
    strip_h = float(ctx.opt("strip_height", 0.85, float))
    crop = _flag(ctx, "crop")
    show_error = _flag(ctx, "error", True)
    per_figure = max(1, int(ctx.opt("per_figure", 4, int)))
    written = []
    for field in fields:
        entries = [c for c in cones if field in c.fields(models)]
        groups = [entries[i:i + per_figure] for i in range(0, len(entries), per_figure)]
        for gi, group in enumerate(groups, start=1):
            if field in BOUNDS:
                v0, v1 = BOUNDS[field]
            else:
                pool = np.concatenate([c.truth[field].ravel()[::97] for c in group])
                v0, v1 = np.percentile(pool, [0.5, 99.5])
            elim = float(np.percentile(np.abs(np.concatenate(
                [(c.preds[m][field] - c.truth[field]).ravel()[::97] for c in group for m in models])), 99)) or 1.0
            rows_per = _row_labels(cat, models, show_error)
            per_cone = len(rows_per)
            n_rows = per_cone * len(group)
            fig, axes = plt.subplots(n_rows, 1, figsize=(16.5, strip_h * n_rows + 1.7), sharex=True,
                                     gridspec_kw=dict(hspace=0.09), squeeze=False)
            axes = axes[:, 0]
            cmap = _cmap(field)
            units = _units(field)
            last_img = last_err = None
            zspan = [np.inf, -np.inf]
            if crop:
                bands = []
                for c in group:
                    lo, hi = active_band(c.truth[field], pad=8)
                    bands.append((c.z[lo], c.z[hi]))
                zlo, zhi = min(b[0] for b in bands), max(b[1] for b in bands)
            box = None
            for ci, c in enumerate(group):
                truth = c.truth[field]
                keep = (c.z >= zlo) & (c.z <= zhi) if crop else np.ones(len(c.z), bool)
                y = int(y_frac * truth.shape[1])
                tt = truth[:, y, keep]
                ze = edges(c.z[keep])
                box = c.box
                xe = np.linspace(0.0, c.box or truth.shape[0], tt.shape[0] + 1)
                zspan = [min(zspan[0], ze[0]), max(zspan[1], ze[-1])]
                for pi, (label, m, is_err) in enumerate(rows_per):
                    ax = axes[ci * per_cone + pi]
                    pp = None if m is None else c.preds[m][field][:, y, keep]
                    if is_err:
                        last_err = ax.pcolormesh(ze, xe, pp - tt, cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -elim, elim),
                                                 shading="flat", rasterized=True)
                    else:
                        last_img = ax.pcolormesh(ze, xe, tt if m is None else pp, cmap=cmap, vmin=v0, vmax=v1,
                                                 shading="flat", rasterized=True)
                    ax.set_yticks([])
                    ax.set_ylabel(label, fontsize=8.5, rotation=0, ha="right", va="center", labelpad=6)
                    if ci == len(group) - 1 and pi == len(rows_per) - 1:
                        ax.set_xlabel("redshift" if c.has_z else "LOS cell", fontsize=11)
                    else:
                        ax.tick_params(axis="x", labelbottom=False)
                    if len(models) == 1 and pi == 0:
                        rmse = float(np.sqrt(((c.preds[models[0]][field][:, y, keep] - tt) ** 2).mean()))
                        ax.text(1.004, 0.5, f"cone {c.name}\nRMSE {rmse:.3g}", transform=ax.transAxes,
                                fontsize=8.5, va="center", ha="left")
                    elif pi == 0:
                        ax.text(1.004, 0.5, f"cone {c.name}", transform=ax.transAxes, fontsize=8.5,
                                va="center", ha="left")
                    elif len(models) > 1 and m is not None and not is_err:
                        rmse = float(np.sqrt(((pp - tt) ** 2).mean()))
                        ax.text(1.004, 0.5, f"RMSE {rmse:.3g}", transform=ax.transAxes, fontsize=8.5,
                                va="center", ha="left")
            axes[-1].set_xlim(*zspan)
            fig.subplots_adjust(left=0.075, right=0.845, top=0.945, bottom=0.075)
            cax = fig.add_axes([0.935, 0.52, 0.011, 0.36])
            fig.colorbar(last_img, cax=cax, label=units)
            if show_error:
                cax2 = fig.add_axes([0.935, 0.10, 0.011, 0.32])
                fig.colorbar(last_err, cax=cax2, label=f"error [{units}]")
            z_all = group[0].z
            span = "active band" if crop else (f"z = {z_all.min():.0f} to {z_all.max():.0f}" if group[0].has_z
                                               else "full LOS")
            row = f"{y_frac * box:.0f} Mpc" if box else f"{y_frac:.2f} Ny"
            fig.suptitle(f"{field_name(field)} lightcone strips — transverse row y = {row}, {span}", fontsize=14)
            part = f"_part{gi}" if len(groups) > 1 else ""
            written.append(save(fig, ctx.out / "multifield_strips",
                                f"{field}_strips{'_active' if crop else ''}{part}", dpi=160, quiet=ctx.quiet))
    return written


# ------------------------------------------------------------- multifield_detailed
def transverse_spectrum(a, b, box, n_bins=12):
    """multifield_detailed ``transverse_spectrum``: ring-averaged P(k_perp) ratio and r(k_perp)."""
    nx, ny = a.shape[:2]
    kx = np.fft.fftfreq(nx, d=box / nx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=box / ny) * 2 * np.pi
    kk = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    bin_edges = np.logspace(np.log10(kk[kk > 0].min()), np.log10(kk.max() * 0.7), n_bins + 1)
    idx = np.digitize(kk.ravel(), bin_edges) - 1
    fa = np.fft.fft2(a - a.mean(axis=(0, 1)), axes=(0, 1))
    fb = np.fft.fft2(b - b.mean(axis=(0, 1)), axes=(0, 1))
    pa = (np.abs(fa) ** 2).mean(axis=2).ravel()
    pb = (np.abs(fb) ** 2).mean(axis=2).ravel()
    px = (fa * np.conj(fb)).mean(axis=2).real.ravel()
    centers, ratio, coh = [], [], []
    for i in range(n_bins):
        m = idx == i
        if m.sum() < 4:
            continue
        A, B, X = pa[m].mean(), pb[m].mean(), px[m].mean()
        centers.append(np.sqrt(bin_edges[i] * bin_edges[i + 1]))
        ratio.append(B / A if A > 0 else np.nan)
        coh.append(X / np.sqrt(A * B) if A > 0 and B > 0 else np.nan)
    return np.array(centers), np.array(ratio), np.array(coh)


def _detailed_page(plt, field, truth, pred, c, label):
    from matplotlib.colors import TwoSlopeNorm

    z = c.z
    box = c.box or truth.shape[0]
    mpc = "Mpc" if c.box else "cells"
    units = _units(field)
    err = pred - truth
    nx, ny, nz = truth.shape
    activity = truth.std(axis=(0, 1))
    zc = int(np.argmax(activity))
    active = np.flatnonzero(activity > 0.05 * activity.max()) if activity.max() > 0 else np.arange(nz)
    lo, hi = int(active[0]), int(active[-1])
    picks = sorted({int(round(lo + f * (hi - lo))) for f in (0.2, 0.5, 0.8)})
    cmap = "viridis" if field == "xhi" else "magma"
    fig = plt.figure(figsize=(17, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.34, wspace=0.24, height_ratios=[1.0, 1.0, 0.85, 0.85])
    zlab = "redshift" if c.has_z else "LOS cell"

    mid = nx // 2
    if field in BOUNDS:
        v_lo, v_hi = BOUNDS[field]
    else:
        v_lo, v_hi = np.percentile(truth[mid], [1, 99])
    xe, ze = np.linspace(0.0, box, ny + 1), edges(z)
    for col, (name, data) in enumerate((("truth", truth[mid]), ("prediction", pred[mid]))):
        ax = fig.add_subplot(gs[0, col])
        im = ax.pcolormesh(xe, ze, data.T, cmap=cmap, vmin=v_lo, vmax=v_hi, shading="flat", rasterized=True)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel(f"x [{mpc}]")
        ax.set_ylabel(zlab if col == 0 else "")
        fig.colorbar(im, ax=ax, label=units if col == 1 else "")
    ax = fig.add_subplot(gs[0, 2])
    lim = float(np.percentile(np.abs(err[mid]), 99)) or 1.0
    im = ax.pcolormesh(xe, ze, err[mid].T, cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -lim, lim), shading="flat",
                       rasterized=True)
    ax.set_title("prediction - truth", fontsize=11)
    ax.set_xlabel(f"x [{mpc}]")
    fig.colorbar(im, ax=ax, label=units)

    for col, j in enumerate(picks):
        ax = fig.add_subplot(gs[1, col])
        pair = np.concatenate([truth[:, :, j], pred[:, :, j]], axis=0)
        if field in BOUNDS:
            v0, v1 = BOUNDS[field]
        else:
            v0, v1 = np.percentile(truth[:, :, j], [1, 99])
        im = ax.imshow(pair.T, origin="lower", cmap=cmap, vmin=v0, vmax=v1, extent=[0, 2 * box, 0, box])
        ax.axvline(box, color="w", lw=1.2)
        ztxt = f"z = {z[j]:.2f}" if c.has_z else f"LOS cell {j}"
        ax.set_title(f"{ztxt}   truth | prediction", fontsize=10)
        ax.set_xlabel(f"x [{mpc}]")
        ax.set_ylabel(f"y [{mpc}]" if col == 0 else "")
        fig.colorbar(im, ax=ax, fraction=0.025)

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(z, np.sqrt((err ** 2).mean(axis=(0, 1))), color="#c0392b", label="RMSE")
    ax.plot(z, err.mean(axis=(0, 1)), color="#2c6fbb", label="mean bias")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(zlab)
    ax.set_ylabel(units)
    ax.set_title("error vs redshift", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[2, 1])
    sub = (slice(None, None, 3), slice(None, None, 3), slice(None, None, 2))
    t, p = truth[sub].ravel(), pred[sub].ravel()
    ax.hexbin(t, p, gridsize=70, bins="log", cmap="inferno", mincnt=1)
    span = [min(t.min(), p.min()), max(t.max(), p.max())]
    ax.plot(span, span, "w--", lw=1.0)
    ax.set_xlabel(f"truth [{units}]")
    ax.set_ylabel(f"prediction [{units}]")
    t64, p64 = t.astype(np.float64), p.astype(np.float64)
    var_t = t64.var()
    slope = float(((t64 - t64.mean()) * (p64 - p64.mean())).mean() / var_t) if var_t > 0 else float("nan")
    ax.set_title(f"parity — slope {slope:.3f}", fontsize=10)

    ax = fig.add_subplot(gs[2, 2])
    ax.hist(err[sub].ravel(), bins=120, color="#555", log=True)
    ax.axvline(0, color="r", lw=0.8)
    ax.set_xlabel(f"error [{units}]")
    ax.set_ylabel("voxels")
    rmse = float(np.sqrt((err ** 2).mean()))
    ax.set_title(f"error distribution — RMSE {rmse:.4g}", fontsize=10)

    band = slice(max(0, zc - 24), min(nz, zc + 24))
    k, ratio, coh = transverse_spectrum(truth[:, :, band], pred[:, :, band], box)
    kunit = r"Mpc$^{-1}$" if c.box else r"cell$^{-1}$"
    ax = fig.add_subplot(gs[3, 0])
    ax.semilogx(k, ratio, "o-", color="#2c6fbb")
    ax.axhline(1.0, color="k", lw=0.7, ls="--")
    ax.set_xlabel(rf"$k_\perp$ [{kunit}]")
    ax.set_ylabel(r"$P_{\rm pred}/P_{\rm truth}$")
    ax.set_title("transverse power ratio", fontsize=10)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[3, 1])
    ax.semilogx(k, coh, "o-", color="#c0392b")
    ax.axhline(0.9, color="k", lw=0.7, ls=":")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(rf"$k_\perp$ [{kunit}]")
    ax.set_ylabel(r"$r(k_\perp)$")
    ax.set_title("cross-correlation", fontsize=10)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[3, 2])
    ax.axis("off")
    finite = np.isfinite(coh)
    below = k[finite][coh[finite] < 0.9]
    with np.errstate(all="ignore"):
        r = np.corrcoef(t, p)[0, 1]
    lines = [
        f"cone {c.name}   field: {field}",
        f"model: {label}",
        f"units: {units}",
        f"RMSE            {rmse:.5g}",
        f"MAE             {np.abs(err).mean():.5g}",
        f"mean bias       {err.mean():+.5g}",
        f"Pearson r       {r:.5f}",
        f"parity slope    {slope:.4f}",
        f"truth std       {truth.std():.5g}",
        f"pred  std       {pred.std():.5g}",
        f"std ratio       {pred.std() / (truth.std() or 1):.4f}",
        f"r(k) < 0.9 from {below[0]:.3f} {'Mpc' if c.box else 'cell'}^-1" if len(below) else "r(k) >= 0.9 throughout",
    ]
    ax.text(0.0, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10.5)
    fig.suptitle(f"{field_name(field)} — cone {c.name}   {label}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return fig


@figure("multifield_detailed", group="maps", needs=("store",),
        help="per sample/field/model page: x-z slices, maps, error vs z, parity, histogram, P_perp, r(k)")
def multifield_detailed(ctx):
    """multifield_detailed ``page``."""
    plt = pyplot()
    cat = ctx.catalog
    models, ids = _setup(ctx)
    cones, _ = _cones(ctx, models, ids)
    written = []
    for c in cones:
        for field in c.fields(models):
            for m in models:
                fig = _detailed_page(plt, field, c.truth[field], c.preds[m][field], c, cat.label(m))
                written.append(save(fig, ctx.out / "multifield_detailed", f"cone{c.name}_{field}_{m}", dpi=145,
                                    quiet=ctx.quiet))
    return written


# ------------------------------------------------------------- animate_lightcone
@figure("lightcone_animation", group="maps", needs=("store",),
        help="GIF of transverse slices stepping through z, truth and each model (--opt frames=, animate_n=)")
def lightcone_animation(ctx):
    """animate_lightcone ``animate``; cheap by default (first sample, 48 frames, 72 dpi)."""
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    plt = pyplot()
    cat = ctx.catalog
    models, ids = _setup(ctx)
    if not ctx.ids:
        ids = ids[:max(1, int(ctx.opt("animate_n", 1, int)))]
    cones, _ = _cones(ctx, models, ids)
    max_frames = max(1, int(ctx.opt("frames", 48, int)))
    fps = int(ctx.opt("fps", 20, int))
    dpi = int(ctx.opt("dpi", 72, int))
    fmt = str(ctx.options.get("format", "gif")).lower()
    out_dir = ctx.out / "lightcone_animation"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for c in cones:
        fields = c.fields(models)
        if not fields:
            continue
        n = len(c.z)
        frames = list(range(0, n, max(1, int(np.ceil(n / max_frames)))))
        box = c.box or c.truth[fields[0]].shape[0]
        columns = [("truth", None)] + [("prediction" if len(models) == 1 else wrap_label(cat.label(m), 28), m) for m in models]
        fig, axes = plt.subplots(len(fields), len(columns), figsize=(3.7 * len(columns), 3.75 * len(fields)),
                                 squeeze=False)
        images = {}
        for r, field in enumerate(fields):
            truth = c.truth[field]
            if field in BOUNDS:
                v0, v1 = BOUNDS[field]
            else:
                v0, v1 = np.percentile(truth[::4, ::4, ::4], [0.5, 99.5])
                if not v1 > v0:
                    v0, v1 = float(truth.min()), float(truth.max()) + 1e-6
            for col, (name, m) in enumerate(columns):
                data = truth if m is None else c.preds[m][field]
                ax = axes[r][col]
                images[(field, col)] = ax.imshow(data[:, :, 0].T, origin="lower", cmap=_cmap(field), vmin=v0,
                                                 vmax=v1, extent=[0, box, 0, box], interpolation="nearest")
                _strip_axes(ax)
                if r == 0:
                    ax.set_title(name, fontsize=12)
                if col == 0:
                    size = f"{box:.0f} x {box:.0f} Mpc" if c.box else f"{box} x {box} cells"
                    ax.set_ylabel(f"{field_name(field)}\n{size}", fontsize=10)
            fig.colorbar(images[(field, 0)], ax=axes[r].tolist(), fraction=0.046 / max(1, len(columns) - 1),
                         pad=0.02, label=_units(field))
        title = fig.suptitle("", fontsize=13)

        def update(j, c=c, fields=fields, columns=columns, images=images, title=title, n=n):
            for field in fields:
                for col, (_, m) in enumerate(columns):
                    data = c.truth[field] if m is None else c.preds[m][field]
                    images[(field, col)].set_data(data[:, :, j].T)
            ztxt = f"z = {c.z[j]:.2f}" if c.has_z else f"LOS cell {j}"
            title.set_text(f"cone {c.name}    {ztxt}    slice {j + 1}/{n}")
            return list(images.values()) + [title]

        suffix = "_compare" if len(models) > 1 else ""
        anim = FuncAnimation(fig, update, frames=frames, blit=False)
        if fmt == "mp4":
            path = out_dir / f"cone{c.name}_lightcone{suffix}.mp4"
            anim.save(path, writer=FFMpegWriter(fps=fps), dpi=dpi)
        else:
            path = out_dir / f"cone{c.name}_lightcone{suffix}.gif"
            anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
        plt.close(fig)
        if not ctx.quiet:
            print(f"wrote {path}  ({len(frames)} frames of {n} slices)")
        written.append(path)
    return written


# ------------------------------------------------------------- 2-D slice figures (xhi2d)
def _slice_pool(cones, field):
    """All (cone, z index) slices of a field, in store order (the 2-D slice cache's rows)."""
    return [(ci, k) for ci, c in enumerate(cones) if field in c.truth for k in range(c.truth[field].shape[-1])]


@figure("slice_comparison", group="maps", needs=("store",),
        help="4 slices spread over all stored slices: density | truth | prediction | pred - true per model")
def slice_comparison(ctx):
    """visualize_xhi2d ``plot_comparison`` on slices of the stored cubes."""
    plt = pyplot()
    cat = ctx.catalog
    models, ids = _setup(ctx)
    cones, fields = _cones(ctx, models, ids)
    n_slices = int(ctx.opt("slices", 4, int))
    split = ctx.title or "test"
    written = []
    for field in fields:
        pool = [(ci, k) for ci, k in _slice_pool(cones, field) if field in cones[ci].fields(models)]
        picks = [pool[i] for i in (range(len(pool)) if len(pool) <= n_slices
                                   else np.linspace(0, len(pool) - 1, n_slices, dtype=int))]
        rows = []
        for ci, k in picks:
            c = cones[ci]
            rows.append({"dens": None if c.density is None else c.density[:, :, k], "truth": c.truth[field][:, :, k],
                         "preds": {m: c.preds[m][field][:, :, k] for m in models}, "z": c.z[k], "cone": c.name,
                         "has_z": c.has_z})
        fig = _slice_rows_figure(
            plt, cat, models, field, rows, f"Predictions — {split} split, {field_name(field)}",
            lambda r: (f"Density δ\nz={r['z']:.2f}  file={r['cone']}" if r["has_z"]
                       else f"Density δ\nLOS cell {int(r['z'])}  file={r['cone']}"),
            "Pred − True", f"Δ {_plain(field)}" + ("" if field == "xhi" else f" [{_units(field)}]"))
        written.append(save(fig, ctx.out / "slice_comparison", f"comparison_{field}", dpi=150, quiet=ctx.quiet))
    return written


@figure("slice_scatter", group="maps", needs=("store",),
        help="hexbin of truth vs prediction over 32 stored slices, one panel per model, R^2 / RMSE")
def slice_scatter(ctx):
    """visualize_xhi2d ``plot_scatter``."""
    plt = pyplot()
    cat = ctx.catalog
    models, ids = _setup(ctx)
    cones, fields = _cones(ctx, models, ids)
    split = ctx.title or "test"
    written = []
    for field in fields:
        pool = [(ci, k) for ci, k in _slice_pool(cones, field) if field in cones[ci].fields(models)]
        n_scatter = min(int(ctx.opt("n_scatter", 32, int)), len(pool))
        picks = [pool[i] for i in np.linspace(0, len(pool) - 1, n_scatter, dtype=int)]
        t_flat = np.concatenate([cones[ci].truth[field][:, :, k].ravel() for ci, k in picks])
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), squeeze=False)
        for ax, m in zip(axes[0], models):
            p_flat = np.concatenate([cones[ci].preds[m][field][:, :, k].ravel() for ci, k in picks])
            _hexbin_panel(ax, t_flat, p_flat, field,
                          f"Scatter — {split}" if len(models) == 1 else wrap_label(cat.label(m), 28))
        if len(models) > 1:
            fig.suptitle(f"Scatter — {split}")
        fig.tight_layout()
        written.append(save(fig, ctx.out / "slice_scatter", f"scatter_{field}", dpi=150, quiet=ctx.quiet))
    return written


def representative_indices(z, means, quantiles):
    """compare_xhi2d_models ``representative_indices``: per z quantile, the slice nearest in z whose
    mean is closest to the local median mean (no slice twice)."""
    z = np.asarray(z, float)
    means = np.asarray(means, float)
    targets = np.quantile(z, quantiles)
    chosen, used = [], set()
    local_count = max(32, len(z) // 100)
    for target in targets:
        nearest = np.argsort(np.abs(z - target), kind="stable")[:local_count]
        local_median = float(np.median(means[nearest]))
        ranked = nearest[np.argsort(np.abs(means[nearest] - local_median), kind="stable")]
        pick = next((int(i) for i in ranked if int(i) not in used), None)
        if pick is None:
            pick = next(int(i) for i in np.argsort(np.abs(z - target)) if int(i) not in used)
        chosen.append(pick)
        used.add(pick)
    return chosen


def _representative(cones, models, field, quantiles):
    pool = [(ci, k) for ci, k in _slice_pool(cones, field) if field in cones[ci].fields(models)]
    pool = pool[:]
    quantiles = list(quantiles)[:len(pool)]
    z = [cones[ci].z[k] for ci, k in pool]
    means = [float(cones[ci].truth[field][:, :, k].mean()) for ci, k in pool]
    picks = [pool[i] for i in representative_indices(z, means, quantiles)]
    info = [{"quantile": q, "cone_id": cones[ci].name, "z": float(cones[ci].z[k]), "has_z": cones[ci].has_z,
             "xhi_mean": float(cones[ci].truth[field][:, :, k].mean())} for q, (ci, k) in zip(quantiles, picks)]
    truth = np.stack([cones[ci].truth[field][:, :, k] for ci, k in picks])
    preds = {m: np.stack([cones[ci].preds[m][field][:, :, k] for ci, k in picks]) for m in models}
    return truth, preds, info


def _zt(info):
    return f"z={info['z']:.2f}" if info["has_z"] else f"LOS cell {int(info['z'])}"


@figure("representative_z_slices", group="maps", needs=("store",),
        help="slices at z quantiles (local-median mean): truth | predictions | errors with RMSE / MAE")
def representative_z_slices(ctx):
    """visualize_xhi2d_representative ``render_slices``."""
    plt = pyplot()
    cat = ctx.catalog
    models, ids = _setup(ctx)
    cones, fields = _cones(ctx, models, ids)
    quantiles = _floats(ctx.options.get("quantiles"), [0.05, 0.2, 0.4, 0.6, 0.8, 0.95])
    written = []
    for field in fields:
        truth, preds, info = _representative(cones, models, field, quantiles)
        errors = {m: preds[m] - truth for m in models}
        all_err = np.concatenate([np.abs(e).ravel() for e in errors.values()])
        error_limit = max(0.05, float(np.quantile(all_err, 0.99))) if field == "xhi" else \
            (float(np.quantile(all_err, 0.99)) or 1.0)
        v0, v1 = _limits(field, truth)
        cmap = "viridis" if field == "xhi" else "inferno"
        ncol = 1 + 2 * len(models)
        fig, axes = plt.subplots(len(info), ncol, figsize=(12.5 / 3 * ncol, 3.25 * len(info)),
                                 constrained_layout=True, squeeze=False)
        field_image = error_image = None
        for row, inf in enumerate(info):
            panels = [(truth[row], cmap, v0, v1, None)]
            panels += [(preds[m][row], cmap, v0, v1, m) for m in models]
            panels += [(errors[m][row], "RdBu_r", -error_limit, error_limit, None) for m in models]
            for col, (image, cm, lo, hi, m) in enumerate(panels):
                ax = axes[row, col]
                rendered = ax.imshow(image, origin="lower", cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
                if col <= len(models):
                    field_image = rendered
                else:
                    error_image = rendered
                _strip_axes(ax)
                if m is not None:
                    e = errors[m][row]
                    rmse, mae = float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e)))
                    ax.text(0.02, 0.98, f"RMSE={rmse:.3f}\nMAE={mae:.3f}" if field == "xhi" else
                            f"RMSE={rmse:.3g}\nMAE={mae:.3g}", transform=ax.transAxes, va="top", ha="left",
                            fontsize=8.5, color="white", bbox={"facecolor": "black", "alpha": 0.55, "pad": 3})
            axes[row, 0].set_ylabel(f"q={inf['quantile']:.2f}  {_zt(inf)}\ncone={inf['cone_id']}  "
                                    f"mean={inf['xhi_mean']:.2f}", fontsize=9)
        lab = field_name(field)
        axes[0, 0].set_title(f"Truth {lab}")
        for j, m in enumerate(models):
            axes[0, 1 + j].set_title(f"Prediction {lab}" if len(models) == 1 else f"{wrap_label(cat.label(m), 28)}\nprediction {lab}")
            axes[0, 1 + len(models) + j].set_title("Prediction - truth" if len(models) == 1
                                                   else f"{wrap_label(cat.label(m), 28)}\nprediction - truth")
        fig.colorbar(field_image, ax=axes[:, :1 + len(models)], shrink=0.8,
                     label=lab + ("" if field == "xhi" else f" [{_units(field)}]"))
        fig.colorbar(error_image, ax=axes[:, 1 + len(models):], shrink=0.8, label="signed error")
        run_label = cat.label(models[0]) if len(models) == 1 else (ctx.title or "models")
        fig.suptitle(f"{run_label}: representative held-out redshift slices", fontsize=14)
        written.append(save(fig, ctx.out / "representative_z_slices", f"representative_z_slices_{field}", dpi=170,
                            quiet=ctx.quiet))
    return written


@figure("representative_z_compare", group="maps", needs=("store",),
        help="truth and every model at z quantiles, plus the signed-error page")
def representative_z_compare(ctx):
    """compare_xhi2d_models ``plot_predictions`` (representative_z_predictions / _errors)."""
    plt = pyplot()
    cat = ctx.catalog
    models, ids = _setup(ctx)
    cones, fields = _cones(ctx, models, ids)
    quantiles = _floats(ctx.options.get("quantiles"), [0.25, 0.5, 0.75, 0.9])
    written = []
    for field in fields:
        truth, preds, info = _representative(cones, models, field, quantiles)
        lab = field_name(field)
        v0, v1 = _limits(field, truth)
        cmap = "viridis" if field == "xhi" else "inferno"
        columns = ["Truth"] + [wrap_label(cat.label(m), 28) for m in models]
        fig, axes = plt.subplots(len(info), len(columns), figsize=(2.8 * len(columns), 3.25 * len(info)),
                                 constrained_layout=True, squeeze=False)
        rendered = None
        for row, inf in enumerate(info):
            for col, image in enumerate([truth[row]] + [preds[m][row] for m in models]):
                ax = axes[row, col]
                rendered = ax.imshow(image, origin="lower", cmap=cmap, vmin=v0, vmax=v1)
                if row == 0:
                    ax.set_title(columns[col], fontsize=10)
                _strip_axes(ax)
            axes[row, 0].set_ylabel(f"{_zt(inf)}\nmean {lab}={inf['xhi_mean']:.2f}", fontsize=10)
        fig.colorbar(rendered, ax=axes, shrink=0.75, label=lab + ("" if field == "xhi" else f" [{_units(field)}]"))
        fig.suptitle("Held-out representative redshift slices", fontsize=14)
        written.append(save(fig, ctx.out / "representative_z_compare", f"representative_z_predictions_{field}",
                            dpi=170, quiet=ctx.quiet))

        errors = {m: preds[m] - truth for m in models}
        q99 = float(np.quantile(np.abs(np.concatenate([e.ravel() for e in errors.values()])), 0.99))
        limit = max(0.05, q99) if field == "xhi" else (q99 or 1.0)
        fig, axes = plt.subplots(len(info), len(models), figsize=(2.75 * len(models), 3.25 * len(info)),
                                 constrained_layout=True, squeeze=False)
        for row, inf in enumerate(info):
            for col, m in enumerate(models):
                ax = axes[row, col]
                rendered = ax.imshow(errors[m][row], origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
                if row == 0:
                    ax.set_title(wrap_label(cat.label(m), 28))
                _strip_axes(ax)
            axes[row, 0].set_ylabel(_zt(inf))
        fig.colorbar(rendered, ax=axes, shrink=0.75,
                     label="prediction - truth" + ("" if field == "xhi" else f" [{_units(field)}]"))
        fig.suptitle("Signed errors on representative redshift slices", fontsize=14)
        written.append(save(fig, ctx.out / "representative_z_compare", f"representative_z_errors_{field}",
                            dpi=170, quiet=ctx.quiet))
    return written


# ------------------------------------------------------------- rhizome ebm21cm/viz.py
@figure("ebm_examples", group="maps", needs=("store",),
        help="rhizome examples: per model, rows of samples: delta | truth, draws, ensemble mean / std per field")
def ebm_examples(ctx):
    """rhizome ``ebm21cm.viz.examples`` (draws from ``<field>__draws``; one draw for regressors)."""
    plt = pyplot()
    cat = ctx.catalog
    models, ids = _setup(ctx)
    cones, _ = _cones(ctx, models, ids)
    stage = float(ctx.opt("stage", 0.5, float))
    n_show = int(ctx.opt("draws_shown", 2, int))
    written = []
    for m in models:
        rows = []
        for c in cones:
            fields = [f for f in ("xhi", "tb") if f in c.fields([m])] or c.fields([m])
            if not fields:
                continue
            k = c.stage_index(stage)
            samples = {}
            for f in fields:
                d = c.preds[m].get(f"{f}__draws")
                samples[f] = d[:, :, :, k] if d is not None else c.preds[m][f][None, :, :, k]
            rows.append((c, k, fields, samples))
        if not rows:
            continue
        all_fields = []
        for _, _, fields, _ in rows:
            all_fields += [f for f in fields if f not in all_fields]
        n_draw = max(s.shape[0] for _, _, _, smp in rows for s in smp.values())
        mm = min(n_show, n_draw)
        per_field = 1 + mm + (2 if n_draw > 1 else 0)
        has_d = any(c.density is not None for c, *_ in rows)
        cols = int(has_d) + len(all_fields) * per_field
        n = len(rows)
        fig, axes = plt.subplots(n, cols, figsize=(1.6 * cols, 1.7 * n + 0.4), squeeze=False)
        for i, (c, k, fields, samples) in enumerate(rows):
            panels = []
            if has_d:
                panels.append((None if c.density is None else c.density[:, :, k], "delta", "cividis", None, None))
            for f in all_fields:
                if f not in fields:
                    panels += [(None, "", None, None, None)] * per_field
                    continue
                t, s = c.truth[f][:, :, k], samples[f]
                name = _plain(f)
                if f == "xhi":
                    cmap, vmin, vmax = "viridis", 0, 1
                elif f == "tb":
                    lim = max(float(np.percentile(np.abs(t), 99.5)), 1.0)
                    cmap, vmin, vmax = "RdBu_r", -lim, lim
                else:
                    cmap, (vmin, vmax) = "viridis", _limits(f, t)
                panels.append((t, f"{name} truth", cmap, vmin, vmax))
                panels += [(s[j] if j < s.shape[0] else None, f"{name} sample {j + 1}", cmap, vmin, vmax)
                           for j in range(mm)]
                if n_draw > 1:
                    many = s.shape[0] > 1
                    panels.append((s.mean(0) if many else None, f"{name} ens. mean ({s.shape[0]})", cmap, vmin, vmax))
                    panels.append((s.std(0) if many else None, f"{name} ens. std", "magma", 0, None))
            for j, (img, title, cmap, vmin, vmax) in enumerate(panels):
                ax = axes[i, j]
                if img is not None:
                    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
                _strip_axes(ax)
                if i == 0:
                    ax.set_title(title, fontsize=7)
            axes[i, 0].set_ylabel((f"z = {c.z[k]:.2f}" if c.has_z else f"cell {k}") + f"\n{c.name}", fontsize=8)
        fig.suptitle(cat.label(m), fontsize=9)
        fig.tight_layout()
        written.append(save(fig, ctx.out / "ebm_examples", f"examples_{m}", dpi=120, quiet=ctx.quiet))
    return written


@figure("regression_examples", group="maps", needs=("store",),
        help="rhizome regression_examples: rows of samples: log density | truth | prediction | error")
def regression_examples(ctx):
    """rhizome ``ebm21cm.viz.regression_examples``."""
    plt = pyplot()
    cat = ctx.catalog
    models, ids = _setup(ctx)
    cones, fields = _cones(ctx, models, ids)
    stage = float(ctx.opt("stage", 0.5, float))
    written = []
    for m in models:
        for field in fields:
            rows = [(c, c.stage_index(stage)) for c in cones if field in c.fields([m])]
            if not rows:
                continue
            n = len(rows)
            fig, axes = plt.subplots(n, 4, figsize=(9.2, 2.3 * n + 0.5), squeeze=False)
            name = _plain(field)
            names = ("density (log 1+δ)", f"{name} truth", f"{name} prediction", "prediction − truth")
            for i, (c, k) in enumerate(rows):
                truth, pred = c.truth[field][:, :, k], c.preds[m][field][:, :, k]
                rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
                if field == "xhi":
                    cm, v0, v1, el = "viridis", 0, 1, 1.0
                else:
                    cm = "inferno"
                    v0, v1 = _limits(field, truth)
                    el = float(np.percentile(np.abs(pred - truth), 99)) or 1.0
                dens = None if c.density is None else np.log1p(np.maximum(c.density[:, :, k], -0.99))
                panels = ((dens, "cividis", None, None), (truth, cm, v0, v1), (pred, cm, v0, v1),
                          (pred - truth, "RdBu_r", -el, el))
                for j, (image, cmap, vmin, vmax) in enumerate(panels):
                    ax = axes[i, j]
                    if image is not None:
                        im = ax.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                                       interpolation="nearest")
                        if j == 3:
                            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                    _strip_axes(ax)
                    if i == 0:
                        ax.set_title(names[j], fontsize=9)
                ztxt = f"z={c.z[k]:.2f}" if c.has_z else f"cell {k}"
                axes[i, 0].set_ylabel(f"{ztxt}\n<x>={truth.mean():.2f}|{pred.mean():.2f}\nrmse {rmse:.3g}", fontsize=8)
            fig.suptitle(f"{cat.label(m)}: {field_name(field)}", fontsize=10)
            fig.tight_layout()
            written.append(save(fig, ctx.out / "regression_examples", f"regression_examples_{m}_{field}", dpi=130,
                                quiet=ctx.quiet))
    return written
