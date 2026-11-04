"""Physical / summary-statistics figures ported from fno-21cm.

Each figure keeps the look of its fno-21cm original (layout, panels, scales, colour maps, reference
lines, dpi); models come from the catalog (label, colour, line style; the truth is black). All
statistics are pooled over the groups of a target (per-sample arrays are concatenated).

=============================  ===============================================  ========================================================  ===========
figure (files)                 original (fno-21cm)                              shows                                                     needs
=============================  ===============================================  ========================================================  ===========
physical_history               viz/physical_evaluation.py ``plot_history``      global x_HI(z) error band, Delta tau / sigma_Planck       curves
                               (``physical_history.png``)                       histogram, z(x_HI=0.5) and duration truth-vs-prediction   (history,
                                                                                                                                          observable)
physical_pdfs                  ``plot_pdfs`` (``physical_pdfs.png``)            stage-resolved PDFs of x_HI and dT_b (log), W1 in legend  curves
                                                                                                                                          (observable)
physical_21cm                  ``plot_observable`` (``physical_21cm.png``)      Delta^2_21(k_perp) (active), Delta^2_21 vs x_HI at fixed  curves
                                                                                k, Var / skewness of dT_b vs x_HI, power-bias             (observable)
                                                                                significance for dT_b and x_HI
physical_eft                   ``plot_eft`` (``physical_eft.png``)              T(k), P_eps(k), r(k) (active) and per-cone b_1, b_k2,     curves (eft)
(``physical_eft_<target>``)                                                     P_eps,0 truth vs prediction, coloured by stage
physical_3d                    viz/visualize_3d.py                              per sample: global field vs z, 3-D P(k) (grid units),     store
(``physical_3d/physical_3d_``  ``plot_physical_diagnostics``                    Fourier cross-correlation (one curve per model)
``<target>_<id>``)             (``physical_3d_<split>_cone<id>.png``)
rmse_r2_comparison             legacy/xhi2d/compare_xhi2d_to_3d.py              RMSE and R^2 bars per reionization stage + overall        store (else
(``..._<target>``)             ``plot_metrics``                                                                                           curves:
                               (``rmse_r2_comparison.png``)                                                                               overall only)
rmse_r2_distributions          ``plot_metric_distributions``                    per-slice RMSE / R^2 box plots per stage                  store (else
(``..._<target>``)             (``rmse_r2_distributions.png``)                                                                            curves:
                                                                                                                                          per cone)
params_vs_accuracy             viz/plot_params_vs_accuracy.py                   model size vs R^2 / RMSE with Pareto front                curves
(``params_vs_accuracy_`` /     (``params_vs_accuracy.png``,                                                                               (efficiency
``params_vs_rmse_<target>``)   ``params_vs_rmse.png``)                                                                                    parameters,
                                                                                                                                          voxel)
inference_speed                viz/plot_inference_speed.py                      throughput vs RMSE (Pareto front); parameters vs          curves
(``inference_speed_vs_rmse_``  (``inference_speed_vs_rmse.png``,                throughput (needs parameter counts)                       (efficiency,
/ ``..._vs_size_<target>``)    ``inference_speed_vs_size.png``)                                                                           voxel)
truth_sharpness_survey         viz/report_truth_survey.py (+ legacy/xhi2d/      truth voxel population, partially ionized fraction,       store
                               survey_truth_sharpness.py)                       front width, transverse vs LOS total variation vs slice
                               (``truth_sharpness_survey.png``)                 mean x_HI (+ ``truth_sharpness_survey.json``)
=============================  ===============================================  ========================================================  ===========

Adaptations: model colours from the catalog instead of the matplotlib cycle / the original
LOCAL-operator palette; the per-cone EFT coefficients (fno-21cm fits them per cone in the
accumulator) are refitted here from the per-cone curves in ``curves.npz`` with the same weights
(the density power and mode counts are recovered from T, P_eps and r and the k binning); the
rmse/R^2 stages are the original five x_HI stages of the slice-mean *true* x_HI, computed from all
slices of the stored samples (the original sampled a few slices per stage); the truth survey reads
the stored truth cubes (the original read the raw native-LOS lightcones). The partially ionized
fraction per stage is in ``results.json`` (``observable.stages``), as in the original (CSV only).
"""
from __future__ import annotations

import json
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .common import field_name, med_band, pyplot, save
from .registry import SkipFigure, figure

PLANCK_TAU_SIGMA = 0.0073
PLANCK_ZRE_SIGMA = 0.73
TRUTH = "k"

# legacy/xhi2d/compare_xhi2d_to_3d.py STAGES: (name, target, lower, upper) of the slice-mean x_HI
RMSE_STAGES = (
    ("ionized", 0.03, 0.00, 0.10),
    ("mostly ionized", 0.20, 0.10, 0.35),
    ("mid reionization", 0.50, 0.35, 0.65),
    ("mostly neutral", 0.80, 0.65, 0.90),
    ("neutral", 0.97, 0.90, 1.000001),
)

# plot_params_vs_accuracy.py / plot_inference_speed.py / report_truth_survey.py palette
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e3de"
SURVEY_HUES = ["#2a78d6", "#eb6834", "#1baf7a"]

# legacy/xhi2d/survey_truth_sharpness.py
N_XHI_BINS = 50
N_VALUE_BINS = 200
PARTIAL_BANDS = ((1e-3, 1 - 1e-3), (0.01, 0.99), (0.1, 0.9))


@contextmanager
def _quiet():
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        yield


# --------------------------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------------------------- #
def _token(label: str) -> str:
    return label.replace(" ", "_").replace("-", "_").replace(".", "p")


def _stage_info(cat, target):
    """(stage labels incl. trailing active, curve tokens) from the run settings."""
    run = cat.runs_for(target, cat.models[0])[0]
    edges = run.settings.get("stage_edges", [0.02, 0.2, 0.4, 0.6, 0.8, 0.98])
    lo, hi = run.settings.get("active_range", [0.05, 0.95])
    labels = [f"xbar_HI {a:.2f}-{b:.2f}" for a, b in zip(edges[:-1], edges[1:])]
    labels.append(f"active {lo:.2f}-{hi:.2f}")
    return labels, [_token(s) for s in labels]


def _setting(cat, target, key, default=None):
    return cat.runs_for(target, cat.models[0])[0].settings.get(key, default)


def _pool(cat, m, key, target):
    """Per-sample arrays concatenated over groups; groups whose trailing shape differs are dropped."""
    parts = []
    for run in cat.runs_for(target, m):
        c = run.curves(m)
        if key in c:
            parts.append(np.atleast_1d(np.asarray(c[key], float)))
    if not parts:
        return np.array([])
    shape = parts[0].shape[1:]
    return np.concatenate([p for p in parts if p.shape[1:] == shape], 0)


def _band(ax, x, med, lo, hi, color, label, **kwargs):
    ax.plot(x, med, color=color, label=label, **kwargs)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)


def _one_to_one(ax, *arrays):
    flat = [np.ravel(a)[np.isfinite(np.ravel(a))] for a in arrays if np.size(a)]
    values = np.concatenate(flat) if flat else np.array([])
    if values.size:
        lo, hi = float(values.min()), float(values.max())
        pad = 0.05 * (hi - lo or 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k:", lw=0.8)


def _nanmedian(values) -> float:
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _k_axis(cat, target):
    """Transverse k-bin centres (1/Mpc) shared by spectrum / eft / observable."""
    for m in cat.models:
        c = cat.curves(target, m)
        for key in ("eft_k", "spectrum_k", "radial_k"):
            if key in c:
                return np.asarray(c[key], float)
    run = cat.runs_for(target, cat.models[0])[0]
    ds = run.results if hasattr(run, "results") else run.runs[0].results
    dataset = ds.get("dataset", {})
    shape = dataset.get("shape")
    if shape:
        from ..physics import KBinner
        box = dataset.get("box_mpc") or run.settings.get("box_mpc", 200.0)
        return KBinner(int(shape[0]), int(shape[1]), float(box), int(run.settings.get("n_k_bins", 15))).centers
    raise SkipFigure("no k axis (run the spectrum or eft metric)")


def _mode_counts(k_centers):
    """Number of 2-D modes per KBinner bin, reconstructed from its geometric bin centres."""
    from ..physics import KBinner
    k = np.asarray(k_centers, float)
    q = k[1] / k[0]
    k_fund = k[0] / np.sqrt(q)
    box = float(f"{2 * np.pi / k_fund:.7g}")       # round-off would move modes across bin edges
    k_ny = k[-1] * np.sqrt(q)
    n = int(round(k_ny * box / np.pi))
    binner = KBinner(n, n, box, len(k))
    if np.allclose(binner.centers, k, rtol=1e-6):
        return binner.counts
    return 2 * np.pi * k * np.diff(np.geomspace(k_fund, k_ny, len(k) + 1))       # annulus area


def _results_root(run):
    return run.results if hasattr(run, "results") else run.runs[0].results


def _model_info(cat, target, m):
    run = cat.runs_for(target, m)[0]
    for r in getattr(run, "runs", [run]):
        info = r.results["models"][m].get("info", {})
        if info:
            return info
    return {}


def _pooled_voxel(cat, m, target):
    """Pooled RMSE and R^2 over the groups (exact from the per-group sums)."""
    n = sse = s1 = s2 = 0.0
    for run in cat.runs_for(target, m):
        v = run.metric(m, "voxel")
        if not v or "mse" not in v or "n_voxels" not in v:
            continue
        nv = float(v["n_voxels"])
        mean, std = float(v.get("truth_mean", np.nan)), float(v.get("truth_std", np.nan))
        n += nv
        sse += float(v["mse"]) * nv
        s1 += mean * nv
        s2 += (std ** 2 + mean ** 2) * nv
    if not n:
        return float("nan"), float("nan")
    ss_tot = s2 - s1 ** 2 / n
    return float(np.sqrt(sse / n)), (1.0 - sse / ss_tot if ss_tot > 0 else float("nan"))


def _n_samples(cat, target):
    return int(sum(int(r.n_samples or 0) for r in cat.runs_for(target, cat.models[0])))


def _need_xhi(cat):
    if "xhi" not in cat.targets:
        raise SkipFigure("needs an x_HI run")


# --------------------------------------------------------------------------------------------- #
# physical_evaluation.py: history
# --------------------------------------------------------------------------------------------- #
@figure("physical_history", group="physical", needs=("curves",),
        help="history error band, tau / Planck histogram, midpoint and duration (fno-21cm physical_history)")
def physical_history(ctx):
    cat = ctx.catalog
    _need_xhi(cat)
    if not cat.has("xhi", "history_tau_truth"):
        raise SkipFigure("no history metric (needs a z grid)")
    plt = pyplot()
    models = cat.models
    with _quiet():
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))

        ax = axes[0]
        ax.axhline(0.0, color="k", lw=0.8, ls=":")
        n_groups = len(cat.groups["xhi"])
        zs = [cat.curves("xhi", models[0], g).get("history_z") for g in range(n_groups)]
        same_z = all(z is not None and zs[0] is not None and len(z) == len(zs[0]) and np.allclose(z, zs[0])
                     for z in zs)
        drawn = False
        for m in models:
            parts = []
            for g in range(n_groups):
                c = cat.curves("xhi", m, g)
                if "observable_xbar_pred" in c and "observable_xbar_truth" in c and zs[g] is not None:
                    parts.append((zs[g], c["observable_xbar_pred"] - c["observable_xbar_truth"]))
            if not parts:
                continue
            drawn = True
            if same_z:
                parts = [(parts[0][0], np.concatenate([p[1] for p in parts], 0))]
            for i, (z, dx) in enumerate(parts):
                md, lo, hi = med_band(dx)
                _band(ax, z, md, lo, hi, cat.color(m), cat.label(m) if i == 0 else None, ls=cat.ls(m))
        if not drawn:
            ax.text(0.5, 0.5, "no per-slice history\n(run the observable metric)", ha="center", va="center",
                    transform=ax.transAxes, color="0.4")
        ax.set_xlabel("z")
        ax.set_ylabel(r"$\bar{x}_{\rm HI}^{\rm pred} - \bar{x}_{\rm HI}^{\rm true}$")
        ax.set_title("Global history error (cone median, 16-84%)")
        if drawn:
            ax.legend(fontsize=8)

        ax = axes[1]
        ax.axvspan(-1, 1, color="grey", alpha=0.18, lw=0, label=r"$\pm1\sigma_{\rm Planck}$")
        pulls = [(_pool(cat, m, "history_tau_pred", "xhi") - _pool(cat, m, "history_tau_truth", "xhi"))
                 / PLANCK_TAU_SIGMA for m in models]
        finite = np.concatenate([p[np.isfinite(p)] for p in pulls]) if pulls else np.array([])
        limit = max(1.5, float(np.max(np.abs(finite))) if finite.size else 1.5)
        bins = np.linspace(-limit, limit, 41)
        for m, pull in zip(models, pulls):
            ax.hist(pull[np.isfinite(pull)], bins=bins, histtype="step", lw=1.6, color=cat.color(m),
                    ls=cat.ls(m), label=f"{cat.label(m)} (med |Δτ| = {np.nanmedian(np.abs(pull)):.2f}σ)")
        ax.set_xlabel(r"$\Delta\tau_e / \sigma_{\rm Planck}$  ($\sigma = 0.0073$)")
        ax.set_ylabel("cones")
        ax.set_title("Thomson optical depth error")
        ax.legend(fontsize=8)

        for ax, key, label in ((axes[2], "z_xhi50", r"$z(\bar{x}_{\rm HI} = 0.5)$"),
                               (axes[3], "delta_z", r"$\Delta z = z_{0.75} - z_{0.25}$")):
            preds = []
            for m in models:
                t, p = _pool(cat, m, f"history_{key}_truth", "xhi"), _pool(cat, m, f"history_{key}_pred", "xhi")
                preds.append(p)
                error = np.nanmedian(np.abs(p - t)) if t.size else np.nan
                ax.scatter(t, p, s=10, alpha=0.7, color=cat.color(m), label=f"{cat.label(m)} (med |err| = {error:.3f})")
            _one_to_one(ax, _pool(cat, models[0], f"history_{key}_truth", "xhi"), *preds)
            ax.set_xlabel(f"truth {label}")
            ax.set_ylabel(f"prediction {label}")
            ax.legend(fontsize=8)
        axes[2].set_title(rf"Midpoint ($\sigma_{{z_{{re}}}}^{{\rm Planck}}$ = {PLANCK_ZRE_SIGMA})")
        axes[3].set_title("Duration")
        fig.tight_layout()
    return [save(fig, ctx.out, "physical_history", dpi=130, quiet=ctx.quiet)]


# --------------------------------------------------------------------------------------------- #
# physical_evaluation.py: PDFs
# --------------------------------------------------------------------------------------------- #
def _pooled_hist(cat, m, key, token, j):
    """Stage histogram pooled over groups, weighted by each group's slice count."""
    total, mass = None, 0.0
    for run in cat.runs_for("xhi", m):
        c = run.curves(m)
        name = f"observable_{key}_{token}"
        if name not in c:
            continue
        stages = run.metric(m, "observable").get("stages", [])
        weight = float(stages[j].get("n_slices", 1)) if j < len(stages) else 1.0
        h = np.asarray(c[name], float)
        total = h * weight if total is None else total + h * weight
        mass += weight
    return None if total is None or mass <= 0 else total / mass


def _stage_value(cat, m, j, key):
    values = []
    for run in cat.runs_for("xhi", m):
        stages = run.metric(m, "observable").get("stages", [])
        if j < len(stages) and stages[j].get(key) is not None:
            values.append(float(stages[j][key]))
    return float(np.median(values)) if values else float("nan")


@figure("physical_pdfs", group="physical", needs=("curves",),
        help="stage-resolved PDFs of x_HI and dT_b with W1 (fno-21cm physical_pdfs)")
def physical_pdfs(ctx):
    cat = ctx.catalog
    _need_xhi(cat)
    c0 = cat.curves("xhi", cat.models[0])
    if "observable_xhi_edges" not in c0:
        raise SkipFigure("no observable metric (needs density and a z grid)")
    plt = pyplot()
    labels, tokens = _stage_info(cat, "xhi")
    labels, tokens = labels[:-1], tokens[:-1]
    specs = [("xhi", r"$x_{\rm HI}$", "w1_xhi")]
    if "observable_dtb_edges" in c0:
        specs.append(("dtb", r"$\delta T_b$ [mK]", "w1_dtb_mk"))
    with _quiet():
        fig, axes = plt.subplots(len(specs), len(labels), figsize=(4.2 * len(labels), 3.6 * len(specs)),
                                 squeeze=False)
        for row, (name, xlabel, w1_key) in enumerate(specs):
            edges = np.asarray(c0[f"observable_{name}_edges"], float)
            centers = 0.5 * (edges[:-1] + edges[1:])
            widths = np.diff(edges)
            for s, (stage, token) in enumerate(zip(labels, tokens)):
                ax = axes[row][s]
                truth = _pooled_hist(cat, cat.models[0], f"{name}_hist_truth", token, s)
                if truth is not None:
                    ax.step(centers, truth / widths, where="mid", color=TRUTH, lw=2.0, label="truth")
                for m in cat.models:
                    pred = _pooled_hist(cat, m, f"{name}_hist_pred", token, s)
                    if pred is None:
                        continue
                    w1 = _stage_value(cat, m, s, w1_key)
                    ax.step(centers, pred / widths, where="mid", color=cat.color(m), ls=cat.ls(m),
                            label=f"{cat.label(m)} (W1={w1:.3g})")
                if truth is not None:
                    ax.set_yscale("log")
                    ax.legend(fontsize=7)
                else:
                    ax.text(0.5, 0.5, "stage not reached", ha="center", va="center", transform=ax.transAxes,
                            color="0.4")
                ax.set_xlabel(xlabel)
                ax.set_title(stage, fontsize=9)
            axes[row][0].set_ylabel("PDF (pooled over cones)")
        fig.tight_layout()
    return [save(fig, ctx.out, "physical_pdfs", dpi=130, quiet=ctx.quiet)]


# --------------------------------------------------------------------------------------------- #
# physical_evaluation.py: 21-cm observable
# --------------------------------------------------------------------------------------------- #
@figure("physical_21cm", group="physical", needs=("curves",),
        help="21-cm power, Delta^2 vs x_HI, dT_b moments, power-bias significance (fno-21cm physical_21cm)")
def physical_21cm(ctx):
    cat = ctx.catalog
    _need_xhi(cat)
    c0 = cat.curves("xhi", cat.models[0])
    if "observable_xbar_edges" not in c0:
        raise SkipFigure("no observable metric (needs density and a z grid)")
    plt = pyplot()
    models = cat.models
    _, tokens = _stage_info(cat, "xhi")
    active = tokens[-1]
    k = _k_axis(cat, "xhi")
    xe = np.asarray(c0["observable_xbar_edges"], float)
    xbar = 0.5 * (xe[:-1] + xe[1:])
    stages = cat.runs_for("xhi", models[0])[0].metric(models[0], "observable").get("stages", [])
    k_targets = stages[-1].get("k_targets") if stages else None
    if not k_targets:
        k_targets = list(_setting(cat, "xhi", "k_targets", [0.1, 0.2, 0.5]))
    pool = lambda m, key: _pool(cat, m, key, "xhi")                                   # noqa: E731
    with _quiet():
        fig, axes = plt.subplots(2, 3, figsize=(18, 9))

        ax = axes[0][0]
        t = pool(models[0], f"observable_d2_truth_{active}")
        if t.size:
            ax.loglog(k, med_band(t)[0], "k-", lw=2.2, label="truth")
        for m in models:
            p = pool(m, f"observable_d2_pred_{active}")
            if p.size:
                ax.loglog(k, med_band(p)[0], color=cat.color(m), ls=cat.ls(m), label=cat.label(m))
        ax.set_xlabel(r"$k_\perp$ [Mpc$^{-1}$]")
        ax.set_ylabel(r"$\Delta^2_{21,\perp}(k)$ [mK$^2$]")
        ax.set_title("Transverse 21-cm power (active slices, cone median)")
        ax.legend(fontsize=8)

        ax = axes[0][1]
        styles = ["-", "--", ":", "-."]
        t = pool(models[0], "observable_xbar_d2_truth")
        preds = {m: pool(m, "observable_xbar_d2_pred") for m in models}
        for j, kt in enumerate(k_targets):
            ls = styles[j % len(styles)]
            if t.ndim == 3 and j < t.shape[2]:
                ax.plot(xbar, med_band(t)[0][:, j], color="k", ls=ls, lw=2.0, label=f"truth k={kt:.2f}")
            for m in models:
                p = preds[m]
                if p.ndim == 3 and j < p.shape[2]:
                    ax.plot(xbar, med_band(p)[0][:, j], color=cat.color(m), ls=ls,
                            label=f"{cat.label(m)} k={kt:.2f}" if j == 0 else None)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\bar{x}_{\rm HI}$ (truth)")
        ax.set_ylabel(r"$\Delta^2_{21,\perp}$ [mK$^2$]")
        ax.set_title("21-cm power evolution at fixed k")
        ax.legend(fontsize=7)

        for ax, key, title in ((axes[1][0], "xbar_var", r"Var($\delta T_b$) [mK$^2$]"),
                               (axes[1][1], "xbar_skew", r"Skewness($\delta T_b$)")):
            t = pool(models[0], f"observable_{key}_truth")
            if t.size:
                _band(ax, xbar, *med_band(t), "k", "truth", lw=2.0)
            for m in models:
                p = pool(m, f"observable_{key}_pred")
                if p.size:
                    _band(ax, xbar, *med_band(p), cat.color(m), cat.label(m), ls=cat.ls(m))
            ax.set_xlabel(r"$\bar{x}_{\rm HI}$ (truth)")
            ax.set_title(title)
            ax.legend(fontsize=8)
        axes[1][1].axhline(0.0, color="k", lw=0.6, ls=":")

        for ax, key, title in ((axes[0][2], "sig_dtb", r"$\delta T_b$"), (axes[1][2], "sig_xhi", r"$x_{\rm HI}$")):
            ax.axhspan(-1, 1, color="grey", alpha=0.18, lw=0)
            ax.axhline(0.0, color="k", lw=0.6, ls=":")
            for m in models:
                v = pool(m, f"observable_{key}_{active}")
                if v.size:
                    _band(ax, k, *med_band(v), cat.color(m), cat.label(m), ls=cat.ls(m))
            ax.set_xscale("log")
            ax.set_xlabel(r"$k_\perp$ [Mpc$^{-1}$]")
            ax.set_ylabel(r"$(P_{\rm pred}-P_{\rm true})\,/\,\sigma_{\rm SV,slice}$")
            ax.set_title(f"{title} power bias / single-slice sample variance (active)")
            ax.legend(fontsize=8)
        fig.tight_layout()
    return [save(fig, ctx.out, "physical_21cm", dpi=130, quiet=ctx.quiet)]


# --------------------------------------------------------------------------------------------- #
# physical_evaluation.py: EFT
# --------------------------------------------------------------------------------------------- #
def _eft_fits(k, counts, T_t, T_p, e_t, e_p, r_t, k_max, n_terms=2):
    """Per-cone WLS fits of T(k) = b1 + b_k2 k^2 and P_eps = P_eps0 + P_eps2 k^2, fno-21cm weights.

    The fit variances are ``P_eps / (n_ind P_dd)`` (transfer) and ``2 P_eps^2 / n_ind`` (stochastic),
    ``n_ind = counts * n_slices / 2``; the slice count is a constant factor and drops out of the
    coefficients. ``P_dd`` follows from the truth curves: ``P_dd = P_eps r^2 / (T^2 (1 - r^2))``.
    """
    from ..physics import fit_even_polynomial
    out = {name: np.full((len(T_t), n_terms), np.nan) for name in ("bias_truth", "bias_pred", "stoch_truth", "stoch_pred")}
    n_ind = np.asarray(counts, float) / 2.0
    for c in range(len(T_t)):
        p_dd = e_t[c] * r_t[c] ** 2 / (T_t[c] ** 2 * (1.0 - r_t[c] ** 2))
        for which, T, e in (("truth", T_t[c], e_t[c]), ("pred", T_p[c], e_p[c])):
            out[f"bias_{which}"][c] = fit_even_polynomial(k, T, e / (n_ind * p_dd), k_max, n_terms)["coef"]
            out[f"stoch_{which}"][c] = fit_even_polynomial(k, e, 2.0 * e ** 2 / n_ind, k_max, n_terms)["coef"]
    return out


@figure("physical_eft", group="physical", needs=("curves",),
        help="EFT transfer, stochasticity, correlation and per-cone bias fits (fno-21cm physical_eft)")
def physical_eft(ctx):
    cat = ctx.catalog
    plt = pyplot()
    written = []
    markers = ["o", "s", "^", "D", "v", "P"]
    for target in cat.targets:
        if not cat.has(target, "eft_k"):
            continue
        k = np.asarray(cat.curves(target, cat.models[0])["eft_k"], float)
        counts = _mode_counts(k)
        k_max = float(_setting(cat, target, "eft_k_max", 0.25))
        labels, tokens = _stage_info(cat, target)
        active = tokens[-1]
        cmap = plt.get_cmap("viridis")
        n_st = len(labels) - 1
        stage_colors = [cmap(i / max(n_st - 1, 1)) for i in range(n_st)]
        unit = r"mK$^2$ Mpc$^2$" if target == "tb" else r"Mpc$^2$"
        pool = lambda m, key: _pool(cat, m, key, target)                               # noqa: E731
        with _quiet():
            fig, axes = plt.subplots(2, 3, figsize=(18, 9))
            for ax, key, title, yscale in ((axes[0][0], "transfer", r"$T(k) = P_{u\delta}/P_{\delta\delta}$", "linear"),
                                           (axes[0][1], "eps", rf"$P_\epsilon(k)$ [{unit}]", "log"),
                                           (axes[0][2], "r", r"$r_{u\delta}(k)$", "linear")):
                t = pool(cat.models[0], f"eft_{key}_truth_{active}")
                if t.size:
                    _band(ax, k, *med_band(t), "k", "truth", lw=2.0)
                for m in cat.models:
                    p = pool(m, f"eft_{key}_pred_{active}")
                    if p.size:
                        _band(ax, k, *med_band(p), cat.color(m), cat.label(m), ls=cat.ls(m))
                ax.axvline(k_max, color="grey", ls="--", lw=0.8)
                ax.set_xscale("log")
                ax.set_yscale(yscale)
                if key == "eps" and t.size:
                    floor = med_band(t)[0]
                    floor = floor[np.isfinite(floor) & (floor > 0)]
                    if floor.size:
                        ax.set_ylim(bottom=1e-3 * float(floor.min()))
                ax.set_xlabel(r"$k_\perp$ [Mpc$^{-1}$]")
                ax.set_title(f"{title} (active slices)")
                ax.legend(fontsize=8)

            fits = {}
            for i, m in enumerate(cat.models):
                for s, token in enumerate(tokens[:-1]):
                    arrays = [pool(m, f"eft_{key}_{token}") for key in
                              ("transfer_truth", "transfer_pred", "eps_truth", "eps_pred", "r_truth")]
                    if all(a.ndim == 2 and len(a) for a in arrays):
                        fits[(m, s)] = _eft_fits(k, counts, *arrays, k_max)
            b1_unit, bk2_unit = (" [mK]", r" [mK Mpc$^2$]") if target == "tb" else ("", r" [Mpc$^2$]")
            for ax, key, term, label in ((axes[1][0], "bias", 0, rf"$b_1${b1_unit}"),
                                         (axes[1][1], "bias", 1, rf"$b_{{k^2}}${bk2_unit}"),
                                         (axes[1][2], "stoch", 0, rf"$P_{{\epsilon,0}}$ [{unit}]")):
                values = []
                for i, m in enumerate(cat.models):
                    for s, stage in enumerate(labels[:-1]):
                        if (m, s) not in fits:
                            continue
                        t = fits[(m, s)][f"{key}_truth"][:, term]
                        p = fits[(m, s)][f"{key}_pred"][:, term]
                        values.extend([t, p])
                        ax.scatter(t, p, s=12, alpha=0.7, color=stage_colors[s], marker=markers[i % len(markers)],
                                   label=stage if i == 0 else None)
                _one_to_one(ax, *values)
                if key == "stoch":
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                ax.set_xlabel(f"truth {label}")
                ax.set_ylabel(f"prediction {label}")
                ax.set_title(f"{label}, k ≤ {k_max:g} Mpc⁻¹ (marker = model)")
                if values:
                    ax.legend(fontsize=7)
            if len(cat.models) > 1:
                note = ", ".join(f"{markers[i % len(markers)]} = {cat.label(m)}" for i, m in enumerate(cat.models))
                fig.text(0.5, 0.005, f"bottom row markers: {note}", ha="center", va="bottom", fontsize=8, color="0.3")
            if target != "xhi":
                fig.suptitle(f"EFT diagnostics of {field_name(target)} (u = {field_name(target)} − ⟨{field_name(target)}⟩)",
                             fontsize=11)
            fig.tight_layout(rect=(0, 0.02 if len(cat.models) > 1 else 0, 1, 1))
        written.append(save(fig, ctx.out, f"physical_eft_{target}", dpi=130, quiet=ctx.quiet))
    if not written:
        raise SkipFigure("no eft metric (needs density)")
    return written


# --------------------------------------------------------------------------------------------- #
# visualize_3d.py: per-cone physical diagnostics page
# --------------------------------------------------------------------------------------------- #
def _radial_fourier_statistics(truth, pred, n_bins=20):
    """fno-21cm util.metrics_21cm._radial_fourier_statistics: 3-D shells in grid units."""
    shape = truth.shape
    truth_fft = np.fft.rfftn(truth - truth.mean())
    pred_fft = np.fft.rfftn(pred - pred.mean())
    k_axes = [np.fft.fftfreq(shape[0]), np.fft.fftfreq(shape[1]), np.fft.rfftfreq(shape[2])]
    k = np.sqrt(sum(axis.reshape(tuple(len(axis) if i == j else 1 for i in range(3))) ** 2
                    for j, axis in enumerate(k_axes)))
    edges = np.linspace(0.0, float(k.max()), n_bins + 1)
    shell = np.digitize(k.ravel(), edges) - 1
    valid = (shell >= 0) & (shell < n_bins) & (k.ravel() > 0)
    shell = shell[valid]
    norm = float(np.prod(shape) ** 2)
    pt = (np.abs(truth_fft).ravel()[valid] ** 2) / norm
    pp = (np.abs(pred_fft).ravel()[valid] ** 2) / norm
    cross = np.real(truth_fft * np.conj(pred_fft)).ravel()[valid] / norm
    counts = np.bincount(shell, minlength=n_bins)
    sum_pt = np.bincount(shell, weights=pt, minlength=n_bins)
    sum_pp = np.bincount(shell, weights=pp, minlength=n_bins)
    sum_cross = np.bincount(shell, weights=cross, minlength=n_bins)
    nonempty = counts > 0
    power_truth = np.zeros_like(sum_pt)
    power_pred = np.zeros_like(sum_pp)
    np.divide(sum_pt, counts, out=power_truth, where=nonempty)
    np.divide(sum_pp, counts, out=power_pred, where=nonempty)
    coherence = sum_cross / np.sqrt(np.maximum(sum_pt * sum_pp, 1e-30))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers[nonempty], power_truth[nonempty], power_pred[nonempty], coherence[nonempty]


def _store_samples(ctx):
    store = ctx.store
    ids = [s for s in store.samples if not ctx.ids or s in ctx.ids]
    limit = ctx.opt("max_samples", None, int)
    return ids[:limit] if limit else ids


@figure("physical_3d", group="physical", needs=("store",),
        help="per-sample page: global field vs z, 3-D power, cross-correlation (fno-21cm physical_3d_*_cone*)")
def physical_3d(ctx):
    cat, store = ctx.catalog, ctx.store
    models = [m for m in cat.models if m in store.models]
    if not models:
        raise SkipFigure("no catalog model in the prediction store")
    plt = pyplot()
    written = []
    for sid in _store_samples(ctx):
        truth_all = store.truth(sid)
        z = truth_all["z_grid"]
        preds = {m: store.prediction(m, sid) for m in models}
        for target in cat.targets:
            truth = truth_all["targets"].get(target)
            have = [m for m in models if target in preds[m]]
            if truth is None or not have:
                continue
            truth = truth.astype(np.float64)
            with _quiet():
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                x = z if z is not None and len(z) == truth.shape[2] else np.arange(truth.shape[2])
                axes[0].plot(x, truth.mean(axis=(0, 1)), color=TRUTH, label="truth")
                for m in have:
                    p = preds[m][target].astype(np.float64)
                    kg, pt, pp, cc = _radial_fourier_statistics(truth, p)
                    if m == have[0]:
                        axes[1].loglog(kg, pt, color=TRUTH, label="truth")
                    axes[0].plot(x, p.mean(axis=(0, 1)), color=cat.color(m), ls=cat.ls(m), label=cat.label(m))
                    axes[1].loglog(kg, pp, color=cat.color(m), ls=cat.ls(m), label=cat.label(m))
                    axes[2].semilogx(kg, cc, color=cat.color(m), ls=cat.ls(m), label=cat.label(m))
                if target == "xhi":
                    axes[0].set(xlabel="redshift z" if z is not None else "LOS cell", ylabel="global x_HI", ylim=(0, 1))
                else:
                    axes[0].set(xlabel="redshift z" if z is not None else "LOS cell",
                                ylabel=f"global {target}{' [mK]' if target == 'tb' else ''}")
                axes[0].legend()
                axes[1].set(xlabel="k [grid units]", ylabel=f"P_{'xHI' if target == 'xhi' else target}(k)")
                axes[1].legend()
                axes[2].set(xlabel="k [grid units]", ylabel="Fourier cross-correlation", ylim=(-1.05, 1.05))
                if len(have) > 1:
                    axes[2].legend()
                fig.suptitle(f"Physical diagnostics - {ctx.title + ' ' if ctx.title else ''}{sid}"
                             + ("" if target == "xhi" else f" ({field_name(target)})"))
                fig.tight_layout()
            written.append(save(fig, ctx.out / "physical_3d", f"physical_3d_{target}_{sid}", dpi=150, quiet=ctx.quiet))
    if not written:
        raise SkipFigure("no stored sample has the catalog's targets")
    return written


# --------------------------------------------------------------------------------------------- #
# compare_xhi2d_to_3d.py: RMSE / R^2 per stage
# --------------------------------------------------------------------------------------------- #
def _r2(ss_res, sum_y, sum_y2, n):
    total = sum_y2 - sum_y ** 2 / n if n else 0.0
    return float("nan") if total <= 1e-12 else 1.0 - ss_res / total


def _slice_scores(ctx, target, models):
    """Per-slice RMSE / R^2 per stage from the stored cubes; None if the store cannot provide them."""
    store = ctx.store
    if store is None or not all(m in store.models for m in models):
        return None
    names = [s[0] for s in RMSE_STAGES] + ["overall"]
    pooled = {m: {n: np.zeros(4) for n in names} for m in models}           # ss_res, sum_y, sum_y2, n
    per_slice = {m: {n: {"rmse": [], "r2": []} for n in names[:-1]} for m in models}
    used = 0
    for sid in _store_samples(ctx):
        t_all = store.truth(sid)
        truth = t_all["targets"].get(target)
        stage_field = t_all["targets"].get("xhi", t_all["inputs"].get("xhi"))
        preds = {m: store.prediction(m, sid).get(target) for m in models}
        if truth is None or stage_field is None or any(p is None for p in preds.values()):
            continue
        used += 1
        truth = truth.astype(np.float64)
        xbar = stage_field.astype(np.float64).mean(axis=(0, 1))
        n_plane = truth.shape[0] * truth.shape[1]
        t_sum = truth.sum(axis=(0, 1))
        t_sum2 = (truth ** 2).sum(axis=(0, 1))
        for m in models:
            err = preds[m].astype(np.float64) - truth
            ss = (err ** 2).sum(axis=(0, 1))
            for name, _target, lo, hi in RMSE_STAGES:
                sel = (xbar >= lo) & (xbar < hi)
                if not sel.any():
                    continue
                pooled[m][name] += [ss[sel].sum(), t_sum[sel].sum(), t_sum2[sel].sum(), n_plane * sel.sum()]
                per_slice[m][name]["rmse"].extend(np.sqrt(ss[sel] / n_plane).tolist())
                per_slice[m][name]["r2"].extend(_r2(a, b, c, n_plane) for a, b, c in
                                                zip(ss[sel], t_sum[sel], t_sum2[sel]))
            pooled[m]["overall"] += [ss.sum(), t_sum.sum(), t_sum2.sum(), n_plane * truth.shape[2]]
    if not used:
        return None
    rows = []
    for name in names:
        row = {"stage": name}
        for m in models:
            a = pooled[m][name]
            row[f"{m}_rmse"] = float(np.sqrt(a[0] / a[3])) if a[3] else float("nan")
            row[f"{m}_r2"] = _r2(*a)
        rows.append(row)
    return rows, per_slice, used


def _rmse_r2_data(ctx, target):
    cat = ctx.catalog
    models = cat.models
    store_rows = None if ctx.opt("source", "store") == "curves" else _slice_scores(ctx, target, models)
    if store_rows is not None:
        rows, per, used = store_rows
        groups = [s[0] for s in RMSE_STAGES]
        dist = {m: {g: per[m][g] for g in groups} for m in models}
        return rows, dist, groups, f"{used} stored samples, all slices; stage = slice-mean true x_HI"
    if not cat.has(target, "rmse_per_sample"):
        return None
    row = {"stage": "overall"}
    for m in models:
        row[f"{m}_rmse"], row[f"{m}_r2"] = _pooled_voxel(cat, m, target)
    group = "overall (per cone)"
    dist = {m: {group: {"rmse": _pool(cat, m, "rmse_per_sample", target).tolist(),
                        "r2": _pool(cat, m, "r2_per_sample", target).tolist()}} for m in models}
    return [row], dist, [group], "voxel metric (no stored cubes: no stages)"


@figure("rmse_r2", group="physical", needs=("curves",),
        help="RMSE / R^2 per x_HI stage: bars and per-slice distributions (fno-21cm rmse_r2_comparison/_distributions)")
def rmse_r2(ctx):
    cat = ctx.catalog
    plt = pyplot()
    written = []
    models = cat.models
    n = len(models)
    for target in cat.targets:
        data = _rmse_r2_data(ctx, target)
        if data is None:
            continue
        rows, dist, groups, note = data
        unit = " [mK]" if target == "tb" else ""
        with _quiet():
            labels = [str(r["stage"]) for r in rows]
            x = np.arange(len(rows))
            width = 0.72 / n
            fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
            for i, m in enumerate(models):
                pos = x + (i - (n - 1) / 2) * width
                axes[0].bar(pos, [r[f"{m}_rmse"] for r in rows], width, label=cat.label(m), color=cat.color(m))
                axes[1].bar(pos, [r[f"{m}_r2"] for r in rows], width, label=cat.label(m), color=cat.color(m))
            for ax, title in zip(axes, (f"RMSE{unit}", "R^2")):
                ax.set_xticks(x, labels, rotation=25, ha="right")
                ax.set_title(title + ("" if target == "xhi" else f" ({field_name(target)})"))
                ax.grid(axis="y", alpha=0.25)
            axes[1].axhline(0, color="black", linewidth=0.8)
            axes[0].legend()
            fig.text(0.995, 0.005, note, ha="right", va="bottom", fontsize=7, color="0.4")
        written.append(save(fig, ctx.out, f"rmse_r2_comparison_{target}", dpi=170, quiet=ctx.quiet))

        with _quiet():
            fig, axes = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)
            step = n + 1
            for ax, metric, title in zip(axes, ("rmse", "r2"), (f"RMSE{unit}", "R^2")):
                positions, values, colors, ticks = [], [], [], []
                for gi, g in enumerate(groups):
                    center = gi * step
                    ticks.append(center)
                    for i, m in enumerate(models):
                        v = np.asarray(dist[m][g][metric], float)
                        v = v[np.isfinite(v)]
                        positions.append(center + i - (n - 1) / 2)
                        values.append(v if v.size else [np.nan])
                        colors.append(cat.color(m))
                boxes = ax.boxplot(values, positions=positions, widths=0.75, patch_artist=True, showfliers=False)
                for patch, color in zip(boxes["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.8)
                ax.set_xticks(ticks, groups, rotation=25, ha="right")
                per = "slice" if len(groups) > 1 else "cone"
                ax.set_title(f"Per-{per} {title} distributions" + ("" if target == "xhi" else f" ({field_name(target)})"))
                ax.grid(axis="y", alpha=0.25)
                if metric == "r2":
                    ax.axhline(0, color="black", linewidth=0.8)
            handles = [plt.Rectangle((0, 0), 1, 1, color=cat.color(m)) for m in models]
            axes[0].legend(handles, [cat.label(m) for m in models])
            fig.text(0.995, 0.005, note, ha="right", va="bottom", fontsize=7, color="0.4")
        written.append(save(fig, ctx.out, f"rmse_r2_distributions_{target}", dpi=170, quiet=ctx.quiet))
    if not written:
        raise SkipFigure("no voxel metric and no stored cubes")
    return written


# --------------------------------------------------------------------------------------------- #
# plot_params_vs_accuracy.py / plot_inference_speed.py
# --------------------------------------------------------------------------------------------- #
def _best_legend_loc(xs, ys) -> str:
    """Corner with the fewest points, preferring the one furthest from any point (original helper)."""
    if not xs:
        return "lower right"
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    nx = [(x - x0) / (x1 - x0) if x1 > x0 else 0.5 for x in xs]
    ny = [(y - y0) / (y1 - y0) if y1 > y0 else 0.5 for y in ys]
    boxes = {"lower left": (0.00, 0.40, 0.00, 0.34), "lower right": (0.60, 1.00, 0.00, 0.34),
             "upper left": (0.00, 0.40, 0.66, 1.00), "upper right": (0.60, 1.00, 0.66, 1.00)}
    scored = []
    for loc, (bx0, bx1, by0, by1) in boxes.items():
        inside = sum(1 for a, b in zip(nx, ny) if bx0 <= a <= bx1 and by0 <= b <= by1)
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        clearance = min(((a - cx) ** 2 + (b - cy) ** 2) ** 0.5 for a, b in zip(nx, ny))
        scored.append((inside, -clearance, loc))
    return min(scored)[2]


def _style(ax, minor=True):
    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    if minor:
        ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


def _legend(ax, xs, ys):
    leg = ax.legend(loc=_best_legend_loc(xs, ys), frameon=True, fontsize=9.5, facecolor=SURFACE,
                    edgecolor=GRID, borderpad=0.8)
    for t in leg.get_texts():
        t.set_color(INK_2)


def _scatter(ax, cat, pts, xkey, ykey):
    for p in sorted(pts, key=lambda p: p[xkey]):
        ax.plot(p[xkey], p[ykey], "o", ms=10, color=cat.color(p["model"]), mec=SURFACE, mew=2, zorder=3,
                linestyle="none", label=cat.label(p["model"]))
    for i, p in enumerate(sorted(pts, key=lambda p: p[xkey])):
        dy = 13 if i % 2 == 0 else -20
        ax.annotate(cat.label(p["model"]), (p[xkey], p[ykey]), textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=8.5, color=INK_2, zorder=4)


def _pad_axis(ax, vals, which, frac=0.16):
    lo, hi = min(vals), max(vals)
    pad = frac * (hi - lo) or 0.05 * (abs(hi) or 1.0)
    (ax.set_ylim if which == "y" else ax.set_xlim)(lo - pad, hi + pad)


def _points(cat, target):
    pts = []
    for m in cat.models:
        runs = cat.runs_for(target, m)
        eff = [r.metric(m, "efficiency") for r in runs]
        info = _model_info(cat, target, m)
        params = next((e.get("parameters") for e in eff if e.get("parameters")), None) or info.get("parameters")
        seconds = _pool(cat, m, "efficiency_seconds", target)
        if seconds.size:
            median = float(np.median(seconds))
            per_s = 1.0 / median if median > 0 else float("nan")
        else:
            vals = [e.get("samples_per_s") for e in eff if e.get("samples_per_s")]
            per_s = float(np.median(vals)) if vals else float("nan")
        slices = [e.get("slices_per_s") for e in eff if e.get("slices_per_s")]
        peak = [e.get("peak_memory_mib") for e in eff if e.get("peak_memory_mib")]
        rmse, r2 = _pooled_voxel(cat, m, target)
        pts.append({"model": m, "params": float(params) if params else float("nan"), "cubes_per_s": per_s,
                    "slices_per_s": float(np.mean(slices)) if slices else float("nan"),
                    "peak_mib": max(peak) if peak else None, "rmse": rmse, "r2": r2,
                    "timing": next((e.get("timing") for e in eff if e.get("timing")), None),
                    "n_timed": sum(int(e.get("n_timed") or 0) for e in eff),
                    "device": next((e.get("device") for e in eff if e.get("device")), info.get("device"))})
    return pts


def _finite(pts, *keys):
    return [p for p in pts if all(np.isfinite(p[k]) for k in keys)]


def _pareto(points, xkey, ykey, lower_is_better, x_low_better=True):
    front, best = [], float("inf") if lower_is_better else float("-inf")
    for p in sorted(points, key=lambda p: p[xkey] if x_low_better else -p[xkey]):
        v = p[ykey]
        if (v < best) if lower_is_better else (v > best):
            front.append(p)
            best = v
    return sorted(front, key=lambda p: p[xkey])


def _surface_fig(plt):
    fig, ax = plt.subplots(figsize=(10.2, 6.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def _save_surface(fig, out, name, quiet, footer=False):
    fig.tight_layout(rect=(0, 0.03, 1, 1) if footer else None)
    fig.set_facecolor(SURFACE)
    return save(fig, out, name, dpi=160, quiet=quiet)


@figure("params_vs_accuracy", group="physical", needs=("curves",),
        help="model size (parameters) vs R^2 and vs RMSE with Pareto front (fno-21cm params_vs_accuracy)")
def params_vs_accuracy(ctx):
    cat = ctx.catalog
    plt = pyplot()
    written = []
    for target in cat.targets:
        pts = _finite(_points(cat, target), "params", "rmse")
        if not pts:
            continue
        n_cones = _n_samples(cat, target)
        unit = " [mK]" if target == "tb" else ""
        fname = field_name(target)
        for metric, (ylabel, lower_better), name in (
                ("r2", (f"Test $R^2$  ({n_cones} cones, all voxels)", False), "params_vs_accuracy"),
                ("rmse", (f"Test RMSE{unit}  ({n_cones} cones, all voxels)", True), "params_vs_rmse")):
            use = _finite(pts, metric)
            if not use:
                continue
            with _quiet():
                fig, ax = _surface_fig(plt)
                front = _pareto(use, "params", metric, lower_better)
                ax.plot([p["params"] for p in front], [p[metric] for p in front], "-", color=INK_MUTED, lw=1.1,
                        alpha=0.7, zorder=1, label="Pareto front")
                _scatter(ax, cat, use, "params", metric)
                ax.set_xscale("log")
                ax.set_xlabel("Trainable parameters  (complex weights counted as two)", fontsize=10, color=INK_2)
                ax.set_ylabel(ylabel, fontsize=10, color=INK_2)
                ax.set_title(f"Model size vs accuracy ({fname})", fontsize=13, color=INK, pad=34, loc="left")
                best = (min if lower_better else max)(use, key=lambda p: p[metric])
                small = min(use, key=lambda p: p["params"])
                gap = abs(best[metric] - small[metric])
                gap_text = f"{gap:.5f} RMSE" if metric == "rmse" else f"{100 * gap:.2f} points of $R^2$"
                ax.text(0.0, 1.012, f"{ctx.title + ': ' if ctx.title else ''}colour is the model. "
                        f"{cat.label(small['model'])} is {best['params'] / small['params']:.0f}x smaller than "
                        f"{cat.label(best['model'])} for {gap_text}",
                        transform=ax.transAxes, fontsize=9, color=INK_MUTED, va="bottom")
                _pad_axis(ax, [p[metric] for p in use], "y")
                _style(ax)
                _legend(ax, [p["params"] for p in use], [p[metric] for p in use])
            written.append(_save_surface(fig, ctx.out, f"{name}_{target}", ctx.quiet))
    if not written:
        raise SkipFigure("no parameter counts (efficiency metric / adapter info) with voxel RMSE")
    return written


def _device_note(pts, ctx, cat, target):
    p = pts[0]
    run = cat.runs_for(target, cat.models[0])[0]
    shape = _results_root(run).get("dataset", {}).get("shape")
    cube = "x".join(str(s) for s in shape) if shape else "?"
    timing = {"forward_only": "forward-only benchmark", "per_call": "per predict() call"}.get(p["timing"], p["timing"])
    return (f"{p['device'] or 'unknown device'} | batch 1, whole {cube} cubes | median of {p['n_timed']} timed "
            f"passes ({timing}) after {run.settings.get('warmup', '?')} warmup")


def _pct(a, b):
    return f"{100 * (a / b - 1):+.1f}%"


@figure("inference_speed", group="physical", needs=("curves",),
        help="throughput vs RMSE and parameters vs throughput (fno-21cm inference_speed_vs_rmse/_vs_size)")
def inference_speed(ctx):
    cat = ctx.catalog
    plt = pyplot()
    written = []
    for target in cat.targets:
        pts = _points(cat, target)
        unit = " [mK]" if target == "tb" else ""
        acc = _finite(pts, "cubes_per_s", "rmse")
        if acc:
            with _quiet():
                fig, ax = _surface_fig(plt)
                front = _pareto(acc, "cubes_per_s", "rmse", True, x_low_better=False)
                ax.plot([p["cubes_per_s"] for p in front], [p["rmse"] for p in front], "-", color=INK_MUTED, lw=1.1,
                        alpha=0.7, zorder=1, label="Pareto front")
                _scatter(ax, cat, acc, "cubes_per_s", "rmse")
                ax.set_xlabel("Inference throughput  (cubes / s, higher is better)", fontsize=10, color=INK_2)
                ax.set_ylabel(f"Test RMSE{unit}  ({_n_samples(cat, target)} cones, all voxels)", fontsize=10,
                              color=INK_2)
                ax.set_title(f"Inference speed vs accuracy ({field_name(target)})", fontsize=13, color=INK, pad=34,
                             loc="left")
                fastest = max(acc, key=lambda p: p["cubes_per_s"])
                best = min(acc, key=lambda p: p["rmse"])
                claim = (f"{cat.label(fastest['model'])} is both the fastest and the most accurate"
                         if fastest is best else
                         f"{cat.label(fastest['model'])} is {fastest['cubes_per_s'] / best['cubes_per_s']:.1f}x "
                         f"faster than {cat.label(best['model'])} for {_pct(fastest['rmse'], best['rmse'])} RMSE")
                ax.text(0.0, 1.012, f"Lower right is better; colour is the model. {claim}",
                        transform=ax.transAxes, fontsize=9, color=INK_MUTED, va="bottom")
                _pad_axis(ax, [p["rmse"] for p in acc], "y")
                _pad_axis(ax, [p["cubes_per_s"] for p in acc], "x", 0.10)
                _style(ax)
                _legend(ax, [p["cubes_per_s"] for p in acc], [p["rmse"] for p in acc])
                fig.text(0.995, 0.006, _device_note(acc, ctx, cat, target), ha="right", va="bottom", fontsize=7.5,
                         color=INK_MUTED)
            written.append(_save_surface(fig, ctx.out, f"inference_speed_vs_rmse_{target}", ctx.quiet, footer=True))
        size = _finite(pts, "params", "slices_per_s")
        if size:
            with _quiet():
                fig, ax = _surface_fig(plt)
                _scatter(ax, cat, size, "params", "slices_per_s")
                ax.set_xscale("log")
                ax.set_xlabel("Trainable parameters  (complex weights counted as two)", fontsize=10, color=INK_2)
                ax.set_ylabel("Inference throughput  (LOS slices / s)", fontsize=10, color=INK_2)
                ax.set_title(f"Model size vs inference speed ({field_name(target)})", fontsize=13, color=INK, pad=34,
                             loc="left")
                big = max(size, key=lambda p: p["params"])
                small = min(size, key=lambda p: p["params"])
                ax.text(0.0, 1.012, f"{cat.label(big['model'])} is {big['params'] / small['params']:.0f}x the "
                        f"parameters of {cat.label(small['model'])} and "
                        f"{_pct(big['slices_per_s'], small['slices_per_s'])} the throughput; colour is the model",
                        transform=ax.transAxes, fontsize=9, color=INK_MUTED, va="bottom")
                _pad_axis(ax, [p["slices_per_s"] for p in size], "y")
                _style(ax)
                _legend(ax, [p["params"] for p in size], [p["slices_per_s"] for p in size])
                fig.text(0.995, 0.006, _device_note(size, ctx, cat, target), ha="right", va="bottom", fontsize=7.5,
                         color=INK_MUTED)
            written.append(_save_surface(fig, ctx.out, f"inference_speed_vs_size_{target}", ctx.quiet, footer=True))
    if not written:
        raise SkipFigure("no efficiency metric")
    return written


# --------------------------------------------------------------------------------------------- #
# report_truth_survey.py (+ survey_truth_sharpness.py)
# --------------------------------------------------------------------------------------------- #
def _survey_accumulate(blk, acc):
    """legacy/xhi2d/survey_truth_sharpness._accumulate, one cube at a time."""
    npl = blk.shape[2]
    area = blk.shape[0] * blk.shape[1]
    mean_xhi = blk.mean(axis=(0, 1))
    idx = np.clip((mean_xhi * N_XHI_BINS).astype(np.int64), 0, N_XHI_BINS - 1)
    gx = np.roll(blk, -1, axis=0) - blk
    gy = np.roll(blk, -1, axis=1) - blk
    gmag = np.sqrt(gx * gx + gy * gy)
    tv_xy = gmag.sum(axis=(0, 1))
    peak = np.percentile(gmag.reshape(-1, npl), 99.9, axis=0)
    gz = np.abs(np.diff(blk, axis=2))
    tv_z = np.zeros(npl)
    tv_z[:-1] = gz.sum(axis=(0, 1))
    band = ((blk > 0.1) & (blk < 0.9)).sum(axis=(0, 1)).astype(np.float64)
    width = band / np.maximum(tv_xy, 1e-9)
    np.add.at(acc["n_planes"], idx, 1)
    np.add.at(acc["sum_tv_xy"], idx, tv_xy / area)
    np.add.at(acc["sum_tv_z"], idx, tv_z / area)
    np.add.at(acc["sum_band"], idx, band / area)
    np.add.at(acc["sum_width"], idx, width)
    np.add.at(acc["sum_peak"], idx, peak)
    for b, (lo, hi) in enumerate(PARTIAL_BANDS):
        frac = ((blk > lo) & (blk < hi)).sum(axis=(0, 1)) / area
        np.add.at(acc["sum_partial"][b], idx, frac)
    acc["value_hist"] += np.histogram(blk, bins=N_VALUE_BINS, range=(0.0, 1.0))[0]
    acc["n_planes_total"] += npl


@figure("truth_sharpness_survey", group="physical", needs=("store",),
        help="truth x_HI survey: voxel population, partial fraction, front width, TV vs slice x_HI (fno-21cm)")
def truth_sharpness_survey(ctx):
    store = ctx.store
    acc = {"n_planes": np.zeros(N_XHI_BINS, dtype=np.int64), "sum_partial": np.zeros((len(PARTIAL_BANDS), N_XHI_BINS)),
           "sum_tv_xy": np.zeros(N_XHI_BINS), "sum_tv_z": np.zeros(N_XHI_BINS), "sum_band": np.zeros(N_XHI_BINS),
           "sum_width": np.zeros(N_XHI_BINS), "sum_peak": np.zeros(N_XHI_BINS),
           "value_hist": np.zeros(N_VALUE_BINS, dtype=np.int64), "n_cones": 0, "n_planes_total": 0}
    n_los = set()
    for sid in _store_samples(ctx):
        t = store.truth(sid)
        xhi = t["targets"].get("xhi", t["inputs"].get("xhi"))
        if xhi is None:
            continue
        _survey_accumulate(np.asarray(xhi, np.float32), acc)
        acc["n_cones"] += 1
        n_los.add(int(xhi.shape[2]))
    if not acc["n_cones"]:
        raise SkipFigure("no stored true x_HI")
    plt = pyplot()
    nb = N_XHI_BINS
    n = acc["n_planes"].astype(np.float64)
    centres = (np.arange(nb) + 0.5) / nb
    ok = n > 0
    mean = lambda k: np.where(ok, acc[k] / np.maximum(n, 1), np.nan)                 # noqa: E731
    partial = np.stack([np.where(ok, acc["sum_partial"][b] / np.maximum(n, 1), np.nan)
                        for b in range(len(PARTIAL_BANDS))])
    width, tv_xy, tv_z = mean("sum_width"), mean("sum_tv_xy"), mean("sum_tv_z")
    vh = acc["value_hist"].astype(np.float64)
    vtot = vh.sum()
    edges = np.linspace(0, 1, N_VALUE_BINS + 1)
    frac_ion, frac_neu, frac_mid = vh[0] / vtot, vh[-1] / vtot, vh[1:-1].sum() / vtot

    with _quiet():
        fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2))
        fig.patch.set_facecolor(SURFACE)
        for ax in axes.ravel():
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=GRID, lw=0.8, zorder=0)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            ax.tick_params(colors=INK_2, labelsize=9)

        ax = axes[0][0]
        c = 0.5 * (edges[1:] + edges[:-1])
        ax.bar(c, 100 * vh / vtot, width=1.0 / N_VALUE_BINS, color=SURVEY_HUES[0], zorder=3)
        ax.set_yscale("log")
        ax.set_xlabel("voxel $x_{HI}$", fontsize=10, color=INK_2)
        ax.set_ylabel("% of all voxels (log)", fontsize=10, color=INK_2)
        ax.set_title(f"Voxel population: {100 * frac_mid:.2f}% partially ionized", fontsize=11, color=INK, loc="left")

        ax = axes[0][1]
        for b, (lo, hi) in enumerate(PARTIAL_BANDS):
            ax.plot(centres[ok], 100 * partial[b][ok], "-", lw=2, color=SURVEY_HUES[b],
                    label=f"${lo:g} < x_{{HI}} < {hi:g}$", zorder=3)
        ax.set_xlabel("slice mean $x_{HI}$", fontsize=10, color=INK_2)
        ax.set_ylabel("% of voxels in the slice", fontsize=10, color=INK_2)
        ax.set_title("Partially ionized fraction vs ionization state", fontsize=11, color=INK, loc="left")
        leg = ax.legend(frameon=True, fontsize=9, facecolor=SURFACE, edgecolor=GRID)
        for t in leg.get_texts():
            t.set_color(INK_2)

        ax = axes[1][0]
        ax.plot(centres[ok], width[ok], "-", lw=2, color=SURVEY_HUES[0], zorder=3)
        ax.set_xlabel("slice mean $x_{HI}$", fontsize=10, color=INK_2)
        ax.set_ylabel("transition width (px)", fontsize=10, color=INK_2)
        ax.set_title("Front width vs ionization state", fontsize=11, color=INK, loc="left")

        ax = axes[1][1]
        ax.plot(centres[ok], tv_xy[ok], "-", lw=2, color=SURVEY_HUES[0], label="transverse", zorder=3)
        ax.plot(centres[ok], tv_z[ok], "-", lw=2, color=SURVEY_HUES[1], label="line of sight", zorder=3)
        ax.set_xlabel("slice mean $x_{HI}$", fontsize=10, color=INK_2)
        ax.set_ylabel("total variation per voxel", fontsize=10, color=INK_2)
        ax.set_title("Front strength, transverse vs LOS", fontsize=11, color=INK, loc="left")
        leg = ax.legend(frameon=True, fontsize=9, facecolor=SURFACE, edgecolor=GRID)
        for t in leg.get_texts():
            t.set_color(INK_2)

        los = "/".join(str(v) for v in sorted(n_los))
        fig.suptitle(f"{ctx.title or 'Stored'} ground truth: ionization-front survey over {acc['n_cones']} lightcones "
                     f"({acc['n_planes_total']:,} transverse planes, {los}-cell LOS)", fontsize=12, color=INK)
        fig.tight_layout()
    path = save(fig, ctx.out, "truth_sharpness_survey", dpi=160, quiet=ctx.quiet)
    (ctx.out / "truth_sharpness_survey.json").write_text(json.dumps(
        {"n_cones": int(acc["n_cones"]), "n_planes": int(acc["n_planes_total"]), "frac_ionized": float(frac_ion),
         "frac_neutral": float(frac_neu), "frac_partial": float(frac_mid),
         "slice_xhi_centres": centres.tolist(), "n_planes_per_bin": acc["n_planes"].tolist(),
         "partial_bands": [list(b) for b in PARTIAL_BANDS],
         "partial_fraction": [[None if not np.isfinite(v) else float(v) for v in row] for row in partial],
         "width_px": [None if not np.isfinite(v) else float(v) for v in width],
         "tv_xy": [None if not np.isfinite(v) else float(v) for v in tv_xy],
         "tv_z": [None if not np.isfinite(v) else float(v) for v in tv_z]}, indent=1))
    return [path]
