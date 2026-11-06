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
