"""Comparative summary figures over models: scorecard, per-sample RMSE, spectra, bubbles, history,
fronts, EFT and morphology.

Generalized from rhizome-neural-operator's ``docs/figures/loreli/src/make_loreli_cm21eval_figs.py``.
Everything is pooled over the groups (runs of the same target on different data): per-sample
quantities are concatenated and summarized by the median (bands: 16-84 %), voxel RMSE is pooled
exactly (voxel-count weighted MSE), front profiles are count-weighted. T_b panels appear when a
``tb`` run is given. Figures: ``summary_scorecard``, ``summary_per_sample_rmse``,
``summary_spectra_<target>``, ``summary_bubbles``, ``summary_history``, ``summary_fronts``,
``summary_eft_morphology``; numbers in ``summary_scorecard.json``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .common import field_name, med_band, pyplot, save

SIGMA_TAU = 0.0073          # Planck 2018 1-sigma on tau_e


def _stage_keys(cat, target):
    run = cat.runs_for(target, cat.models[0])[0]
    edges = run.settings.get("stage_edges", [0.02, 0.2, 0.4, 0.6, 0.8, 0.98])
    lo, hi = run.settings.get("active_range", [0.05, 0.95])
    stages = [f"{a:.2f}_{b:.2f}" for a, b in zip(edges[:-1], edges[1:])]
    active = f"active_{lo:.2f}_{hi:.2f}".replace(".", "p")
    return stages, active


def plot_summary(cat, out: Path, title="", quiet=False):
    import matplotlib
    import matplotlib.ticker

    plt = pyplot()
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "legend.fontsize": 7, "figure.dpi": 100})
    out = Path(out)
    models, lab, col, ls, tick = cat.models, cat.label, cat.color, cat.ls, cat.tick
    if "xhi" not in cat.targets:
        raise SystemExit("summary figures need an x_HI run")
    has_tb = "tb" in cat.targets
    groups = cat.group_labels("xhi")
    sizes = cat.group_sizes("xhi")
    n_samples = sum(sizes)
    stages, active = _stage_keys(cat, "xhi")
    stage_label = {s: "$\\bar{x}_{HI}$ " + s.replace("_", "–") for s in stages}
    curves0 = cat.curves("xhi", models[0])
    k = curves0.get("spectrum_k")
    written = []

    def handles(where, **kw):
        hs = [plt.Line2D([], [], color=col(m), ls=ls(m), lw=1.6, label=lab(m)) for m in models]
        return where.legend(handles=hs, **kw)

    def pooled(m, key, target="xhi"):
        return cat.pooled(m, key, target)

    def pooled_rmse(m, target):
        n = mse = 0.0
        for run in cat.runs_for(target, m):
            v = run.metric(m, "voxel")
            if "mse" in v and "n_voxels" in v:
                n += float(v["n_voxels"])
                mse += float(v["mse"]) * float(v["n_voxels"])
        return math.sqrt(mse / n) if n else float("nan")

    def at_k(curve, k0=0.2):
        curve = np.atleast_2d(curve)
        return np.array([np.interp(np.log(k0), np.log(k), row) if np.isfinite(row).any() else np.nan
                         for row in curve]) if k is not None and curve.size else np.array([])

    # 1) scorecard ------------------------------------------------------------------------------
    score = {}
    for m in models:
        score[m] = {"x_HI RMSE (pooled)": pooled_rmse(m, "xhi"),
                    "x_HI RMSE per cone": pooled(m, "rmse_per_sample"),
                    "IoU (matched)": pooled(m, "morphology_iou_matched"),
                    "x_HI r(k=0.2/Mpc), active": at_k(pooled(m, f"spectrum_coherence_{active}")),
                    "x_HI P ratio (k=0.2/Mpc)": at_k(pooled(m, f"spectrum_ratio_{active}")),
                    "|Δτ| [σ_Planck]": np.abs(pooled(m, "history_tau_pred") - pooled(m, "history_tau_truth")) / SIGMA_TAU,
                    "Δz(x_HI=0.5)": pooled(m, "history_z_xhi50_pred") - pooled(m, "history_z_xhi50_truth"),
                    "front width ratio": pooled(m, "front_width_pred") / pooled(m, "front_width_truth")
                    if pooled(m, "front_width_truth").size else np.array([])}
        if has_tb:
            score[m].update({"T_b RMSE [mK] (pooled)": pooled_rmse(m, "tb"),
                             "T_b RMSE per cone [mK]": pooled(m, "rmse_per_sample", "tb"),
                             "T_b r(k=0.2/Mpc), active": at_k(pooled(m, f"spectrum_coherence_{_stage_keys(cat, 'tb')[1]}", "tb"))})
    panels = [("x_HI RMSE (pooled)", "low"), ("T_b RMSE [mK] (pooled)", "low"), ("IoU (matched)", "high"),
              ("x_HI r(k=0.2/Mpc), active", "high"), ("T_b r(k=0.2/Mpc), active", "high"),
              ("x_HI P ratio (k=0.2/Mpc)", "one"), ("|Δτ| [σ_Planck]", "low"), ("Δz(x_HI=0.5)", "zero"),
              ("front width ratio", "one")]
    panels = [p for p in panels if p[0] in score[models[0]] and np.size(score[models[0]][p[0]])]
    ncol = 3
    nrow = math.ceil(len(panels) / ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(13, 3.2 * nrow), constrained_layout=True, squeeze=False)
    y = np.arange(len(models))[::-1]
    for ax, (name, better) in zip(axs.ravel(), panels):
        for i, m in enumerate(models):
            v = score[m][name]
            if np.ndim(v) == 0:
                ax.barh(y[i], v, color=col(m), edgecolor="k", lw=0.4)
                ax.text(v, y[i], f" {v:.3g}", va="center", fontsize=7)
            else:
                md, lo, hi = med_band(v)
                ax.barh(y[i], md, color=col(m), edgecolor="k", lw=0.4)
                ax.errorbar(md, y[i], xerr=[[md - lo], [hi - md]], color="k", lw=0.8, capsize=2)
                jitter = (np.random.default_rng(i).random(len(v)) - 0.5) * 0.5
                ax.scatter(v, y[i] + jitter, s=4, color="k", alpha=0.35, lw=0, zorder=3)
                ax.text(hi, y[i] + 0.3, f" {md:.3g}", va="center", fontsize=7)
        if better == "one":
            ax.axvline(1, color="k", lw=0.6)
        if better == "zero":
            ax.axvline(0, color="k", lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([lab(m) for m in models] if ax in axs[:, 0] else [])
        hint = {"low": "lower is better", "high": "higher is better", "one": "1 is ideal", "zero": "0 is ideal"}[better]
        ax.set_title(f"{name}  ({hint})")
    for ax in axs.ravel()[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(f"21cmEVAL scorecard{': ' + title if title else ''}: {n_samples} lightcones in {len(groups)} group(s). "
                 "Bars: pooled value or median over cones (error bar 16–84 %, dots: cones)", fontsize=9.5)
    written.append(save(fig, out, "summary_scorecard", quiet=quiet))

    # 2) per-sample RMSE, paired across models ----------------------------------------------------
    keys = [("x_HI RMSE per cone", "")] + ([("T_b RMSE per cone [mK]", " [mK]")] if has_tb else [])
    fig, axs = plt.subplots(1, len(keys), figsize=(6.5 * len(keys), 4.8), constrained_layout=True, squeeze=False)
    gcol = dict(zip(groups, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"] * 4))
    sample_group = [g for g, n in zip(groups, sizes) for _ in range(n)]
    for ax, (key, unit) in zip(axs[0], keys):
        vals = np.array([score[m][key] for m in models])
        xs = np.arange(len(models))
        for c in range(vals.shape[1]):
            ax.plot(xs, vals[:, c], "-", color=gcol[sample_group[c]] if c < len(sample_group) else "0.5", alpha=0.45, lw=0.8)
        ax.boxplot(vals.T, positions=xs, widths=0.35, showfliers=False, medianprops={"color": "k"})
        ax.set_xticks(xs)
        ax.set_xticklabels([tick(m) for m in models], fontsize=7)
        ax.set_title(f"{key.replace(' per cone', '')} per lightcone: lines connect the same cone")
        ax.set_ylabel(key)
    if len(groups) > 1:
        hs = [plt.Line2D([], [], color=gcol[g], label=f"{g} ({n} cones)") for g, n in zip(groups, sizes)]
        axs[0, -1].legend(handles=hs, loc="upper right")
    written.append(save(fig, out, "summary_per_sample_rmse", quiet=quiet))

    # 3) power ratio and coherence per stage -------------------------------------------------------
    for target in [t for t in ("xhi", "tb") if t in cat.targets]:
        st_t, act_t = _stage_keys(cat, target)
        if not cat.has(target, f"spectrum_ratio_{act_t}"):
            continue
        kt = cat.curves(target, models[0])["spectrum_k"]
        cols = st_t + ["active"]
        fig, axs = plt.subplots(2, len(cols), figsize=(2.85 * len(cols), 6), constrained_layout=True, sharex=True)
        for j, st in enumerate(cols):
            key = act_t if st == "active" else "xbar_HI_" + st.replace(".", "p")
            for m in models:
                r = pooled(m, f"spectrum_ratio_{key}", target)
                c = pooled(m, f"spectrum_coherence_{key}", target)
                for row, a in ((0, r), (1, c)):
                    if not a.size:
                        continue
                    md, lo, hi = med_band(a)
                    axs[row, j].plot(kt, md, color=col(m), ls=ls(m), lw=1.4)
                    if m in (models[0], models[-1]):
                        axs[row, j].fill_between(kt, lo, hi, color=col(m), alpha=0.12, lw=0)
            r0 = pooled(models[0], f"spectrum_ratio_{key}", target)
            n = int(np.isfinite(r0).any(1).sum()) if r0.ndim == 2 else 0
            axs[0, j].set_title(("active $\\bar{x}_{HI}$ " + act_t[7:].replace("p", ".").replace("_", "–")
                                 if st == "active" else stage_label.get(st, st)) + f"  ({n} cones)")
            for row in (0, 1):
                axs[row, j].axhline(1, color="k", lw=0.6)
            axs[0, j].set_yscale("log")
            axs[0, j].set_ylim(0.1, 10)
            axs[1, j].set_ylim(-0.1, 1.02)
            axs[1, j].set_xscale("log")
            axs[1, j].set_xlabel("k$_\\perp$ [Mpc$^{-1}$]")
        axs[0, 0].set_ylabel("P$_{pred}$ / P$_{true}$")
        axs[1, 0].set_ylabel("cross-correlation r(k)")
        handles(axs[1, 0], loc="lower left")
        fig.suptitle(f"{field_name(target)}: transverse power ratio and coherence by reionization stage; median over cones "
                     f"(band 16–84 % for {lab(models[0])} and {lab(models[-1])})", fontsize=10)
        written.append(save(fig, out, f"summary_spectra_{target}", quiet=quiet))

    # 4) bubble sizes ---------------------------------------------------------------------------
    if "bubble_radius_mpc" in curves0:
        R = curves0["bubble_radius_mpc"]
        cols = stages + ["active"]
        fig, axs = plt.subplots(1, len(cols), figsize=(2.85 * len(cols), 3.6), constrained_layout=True, sharey=True)
        for j, st in enumerate(cols):
            key = "active" if st == "active" else "xbar" + st
            truth = pooled(models[0], f"bubble_mass_truth_{key}")
            if truth.size:
                axs[j].plot(R, np.nanmean(truth, 0), color="k", lw=2.2, label="truth")
            for m in models:
                p = pooled(m, f"bubble_mass_pred_{key}")
                if p.size:
                    axs[j].plot(R, np.nanmean(p, 0), color=col(m), ls=ls(m), lw=1.3)
            axs[j].set_xscale("log")
            axs[j].set_xlabel("mean-free-path size R [Mpc]")
            axs[j].set_title("active" if st == "active" else stage_label[st])
        axs[0].set_ylabel("ionized-area-weighted PDF (per bin)")
        hs = [plt.Line2D([], [], color="k", lw=2.2, label="truth")] + \
             [plt.Line2D([], [], color=col(m), ls=ls(m), label=lab(m)) for m in models]
        axs[0].legend(handles=hs, loc="upper left")
        fig.suptitle("Bubble-size distributions (mean-free-path rays through transverse slices), mean over cones",
                     fontsize=10)
        written.append(save(fig, out, "summary_bubbles", quiet=quiet))

    # 5) reionization history -------------------------------------------------------------------
    if "history_tau_truth" in curves0:
        fig = plt.figure(figsize=(16, 8.2), constrained_layout=True)
        gs = fig.add_gridspec(2, 4)
        ax_tau, ax_z50, ax_dz, ax_err = (fig.add_subplot(gs[0, i]) for i in range(4))
        for m in models:
            for ax, kp_, kt_ in ((ax_tau, "history_tau_pred", "history_tau_truth"),
                                 (ax_z50, "history_z_xhi50_pred", "history_z_xhi50_truth"),
                                 (ax_dz, "history_delta_z_pred", "history_delta_z_truth")):
                ax.scatter(pooled(m, kt_), pooled(m, kp_), s=10, color=col(m), alpha=0.8, lw=0, label=lab(m))
        for ax, name in ((ax_tau, "τ$_e$ over the window"), (ax_z50, "z(x$_{HI}$ = 0.5)"),
                         (ax_dz, "duration Δz = z(0.75) − z(0.25)")):
            lo, hi = ax.get_xlim()
            ax.plot([lo, hi], [lo, hi], "k-", lw=0.6)
            ax.set_xlabel(f"truth: {name}")
            ax.set_ylabel("prediction")
            ax.set_title(name)
        qs = [("history_tau_pred", "history_tau_truth", SIGMA_TAU, "Δτ / σ$_{Planck}$"),
              ("history_z_xhi25_pred", "history_z_xhi25_truth", 1, "Δz(0.25)"),
              ("history_z_xhi50_pred", "history_z_xhi50_truth", 1, "Δz(0.5)"),
              ("history_z_xhi75_pred", "history_z_xhi75_truth", 1, "Δz(0.75)")]
        w = 0.8 / len(models)
        for i, m in enumerate(models):
            for q, (kp_, kt_, s, _) in enumerate(qs):
                e = (pooled(m, kp_) - pooled(m, kt_)) / s
                if not e.size:
                    continue
                md, lo, hi = med_band(e)
                ax_err.errorbar(q + (i - len(models) / 2) * w, md, yerr=[[md - lo], [hi - md]], fmt="o", ms=3.5,
                                color=col(m), capsize=1.5, lw=0.9)
        ax_err.axhline(0, color="k", lw=0.6)
        ax_err.set_xticks(range(len(qs)))
        ax_err.set_xticklabels([q[3] for q in qs])
        ax_err.set_title("history errors: median and 16–84 % over cones")
        examples = []                                   # (group index, row) with the most evolution
        for gi in range(len(groups)):
            c0 = cat.curves("xhi", models[0], gi)
            if "observable_xbar_truth" in c0:
                spread = np.nanstd(c0["observable_xbar_truth"], 1)
                examples.append((gi, int(np.argmax(spread))))
        if len(examples) == 1 and "observable_xbar_truth" in curves0:
            spread = np.nanstd(curves0["observable_xbar_truth"], 1)
            examples = [(0, int(r)) for r in np.argsort(spread)[::-1][:4]]
        for slot, (gi, row) in enumerate(examples[:4]):
            ax = fig.add_subplot(gs[1, slot])
            c0 = cat.curves("xhi", models[0], gi)
            z = c0["history_z"]
            ids = c0.get("cyl_ids", c0.get("gallery_ids"))
            name = str(ids[row]) if ids is not None and row < len(ids) and len(ids) == len(c0["observable_xbar_truth"]) \
                else f"#{row}"
            ax.plot(z, c0["observable_xbar_truth"][row], color="k", lw=2.2, label="truth")
            for m in models:
                ax.plot(z, cat.curves("xhi", m, gi)["observable_xbar_pred"][row], color=col(m), ls=ls(m), lw=1.2)
            ax.set_xlabel("z")
            ax.set_title(f"{groups[gi]}: cone {name}")
            ax.set_ylim(-0.02, 1.02)
            if slot == 0:
                ax.set_ylabel("slice-mean x$_{HI}$")
        handles(ax_tau, loc="upper left")
        fig.suptitle("Reionization history recovered from the predicted maps (`history` / `observable`). "
                     "τ and crossings are computed inside each sample's z window, so they compare prediction and "
                     "truth, not the full reionization history", fontsize=9.5)
        written.append(save(fig, out, "summary_history", quiet=quiet))

    # 6) ionization fronts ----------------------------------------------------------------------
    if "front_d_mpc" in curves0:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)

        def front_pool(m, key):
            num = den = 0
            for gi in range(len(groups)):
                c = cat.curves("xhi", m, gi)
                n = c["front_counts"]
                v = c[key] ** 2 if key == "front_rmse" else c[key]
                num = num + n * np.nan_to_num(v)
                den = den + n
            res = num / np.maximum(den, 1)
            return np.sqrt(res) if key == "front_rmse" else res

        d = curves0["front_d_mpc"]
        axs[0].plot(d, front_pool(models[0], "front_mean_truth"), "k", lw=2.2, label="truth")
        for m in models:
            axs[0].plot(d, front_pool(m, "front_mean_pred"), color=col(m), ls=ls(m), lw=1.3)
            axs[1].plot(d, front_pool(m, "front_rmse"), color=col(m), ls=ls(m), lw=1.3)
            axs[2].scatter(pooled(m, "front_width_truth"), pooled(m, "front_width_pred"), s=10, color=col(m), lw=0,
                           alpha=0.85)
        axs[0].set_title("mean x$_{HI}$ vs signed distance to the true front")
        axs[0].set_xlabel("distance to true front [Mpc] (< 0: ionized side)")
        axs[1].set_title("RMSE vs distance to the true front")
        axs[1].set_xlabel("distance to true front [Mpc]")
        axs[2].set_title("10–90 % front width per cone")
        axs[2].set_xlabel("truth [Mpc]")
        axs[2].set_ylabel("prediction [Mpc]")
        lo, hi = axs[2].get_xlim()
        axs[2].plot([lo, hi], [lo, hi], "k-", lw=0.6)
        axs[2].text(hi, hi, "y = x ", ha="right", va="bottom", fontsize=7)
        axs[2].set_xlim(lo, hi)
        handles(axs[1], loc="upper right")
        written.append(save(fig, out, "summary_fronts", quiet=quiet))

    # 7) EFT and morphology ---------------------------------------------------------------------
    if "eft_k" in curves0:
        fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        ek_all = curves0["eft_k"]
        keep = ek_all <= 0.5            # the truth transfer function crosses zero near k ~ 1: ratios blow up there
        ek = ek_all[keep]
        eft_key = active
        for m in models:
            Tp, Tt = pooled(m, f"eft_transfer_pred_{eft_key}"), pooled(m, f"eft_transfer_truth_{eft_key}")
            Ep, Et = pooled(m, f"eft_eps_pred_{eft_key}"), pooled(m, f"eft_eps_truth_{eft_key}")
            rp, rt = pooled(m, f"eft_r_pred_{eft_key}"), pooled(m, f"eft_r_truth_{eft_key}")
            for ax, a in ((axs[0, 0], Tp / Tt if Tt.size else Tt), (axs[0, 1], Ep / Et if Et.size else Et),
                          (axs[0, 2], rp - rt if rt.size else rt)):
                if a.size:
                    ax.plot(ek, med_band(a)[0][keep], color=col(m), ls=ls(m), lw=1.3)
        for ax, t, ref in ((axs[0, 0], "transfer T(k) = P$_{x\\delta}$/P$_{\\delta\\delta}$: pred / truth", 1),
                           (axs[0, 1], "stochasticity P$_\\epsilon$(k): pred / truth", 1),
                           (axs[0, 2], "r$_{x\\delta}$(k): pred − truth", 0)):
            ax.axhline(ref, color="k", lw=0.6)
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlabel("k [Mpc$^{-1}$]")
            ax.set_title(t + "\n(active slices, median over cones, k ≤ 0.5 Mpc$^{-1}$)")
        axs[0, 1].set_yscale("log")
        fs = [0.1, 0.25, 0.5, 0.75, 0.9]
        for i, m in enumerate(models):
            md = [med_band(pooled(m, f"morphology_iou_f{f:g}"))[0] if pooled(m, f"morphology_iou_f{f:g}").size
                  else np.nan for f in fs]
            axs[1, 0].plot(fs, md, marker="o", color=col(m), ls=ls(m), ms=3.5, lw=1.2)
            tc, tr = pooled(m, "morphology_truth_count"), pooled(m, "morphology_truth_volume_weighted_radius_cells")
            if tc.size:
                for ax, a in ((axs[1, 1], pooled(m, "morphology_pred_count") / tc),
                              (axs[1, 2], pooled(m, "morphology_pred_volume_weighted_radius_cells") / tr)):
                    mdn, lo, hi = med_band(a)
                    ax.errorbar(i, mdn, yerr=[[mdn - lo], [hi - mdn]], fmt="o", color=col(m), capsize=2)
        axs[1, 0].set_xlabel("ionized volume fraction of the cone (threshold rank)")
        axs[1, 0].set_title("3-D ionized-region IoU at matched ionized fraction")
        for ax, t in ((axs[1, 1], "number of ionized regions: pred / truth"),
                      (axs[1, 2], "volume-weighted region radius: pred / truth")):
            ax.axhline(1, color="k", lw=0.6)
            ax.set_yscale("log")
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels([tick(m) for m in models], fontsize=6.5)
            ax.set_title(t + " (median, 16–84 %)")
        handles(axs[0, 0], loc="lower left")
        written.append(save(fig, out, "summary_eft_morphology", quiet=quiet))

    tab = {m: {kk: (float(v) if np.ndim(v) == 0 else {"median": float(med_band(v)[0]), "p16": float(med_band(v)[1]),
                                                      "p84": float(med_band(v)[2]), "n": int(np.isfinite(v).sum())})
               for kk, v in score[m].items() if np.size(v)} for m in models}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary_scorecard.json").write_text(json.dumps(
        {"models": models, "labels": {m: lab(m) for m in models}, "skipped_incomplete": cat.skipped,
         "groups": dict(zip(groups, sizes)), "scores": tab}, indent=1, ensure_ascii=False))
    return written
