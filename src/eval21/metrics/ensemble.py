"""Ensemble metrics for stochastic / generative models (``--draws N``).

The runner gives this metric every draw of a sample (``aux["draws"]``, shape (D, Nx, Ny, Nz)).
Two families:

Voxel calibration (meaningful for a probabilistic surrogate conditioned on the same fields as
the truth; ignored for a purely generative model, whose draws need not align with the truth):
    crps            fair CRPS estimator per voxel, overall and per stage
    spread_skill    sqrt((D+1)/D) * mean ensemble std / RMSE of the ensemble mean (1 = calibrated)
    rank_histogram  rank of the truth among the D draws (D+1 bins; ties broken at random, needed
                    for saturated x_HI = 0/1), with its chi^2 flatness and the coverage of the
                    central ~50 % / ~90 % of ranks. Coverage is rank based and reported with its
                    exact nominal value m/(D+1): percentile intervals of a few draws undercover
                    by construction (the range of 8 exchangeable draws holds 7/9 of truths).

Summary-statistic calibration (meaningful for both kinds of model): per sample, the same
statistics are computed for the truth and for each draw --
    xbar@z          slice-mean history at ``ensemble_n_los`` LOS positions
    history         z(x_HI = 0.25/0.5/0.75), duration (x_HI targets)
    power@stage     2-D power Delta^2(k) at ``ensemble_k`` (1/Mpc) in each reionization stage
-- and the truth is located in the draw distribution: randomised PIT = (rank + U)/(D+1), exactly
uniform for a calibrated model at any D; z = (truth - mean)/std. Over samples a calibrated model
gives uniform PITs (KS distance ~ 1/sqrt(n)), coverage at its nominal value and mean z^2 ~ 1. A PIT piled at 0 and 1 means draws that are too narrow; a hump at
0.5 means too broad.

Cosmic variance: when several samples share identical parameters (seed repeats), the spread of
their truths is the simulator's own variance; ``cosmic_variance`` compares it with the draw
spread at those parameters (ratio ~ 1 is right; << 1 means mode collapse). Without repeats the
entry says so.
"""
from __future__ import annotations

import json
import zlib

import numpy as np

from ..physics import KBinner, history_features, stage_labels, stage_masks
from ..registry import register
from .base import Metric, nanmedian

PIT_BINS = 10


def fair_crps(draws: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-voxel fair CRPS; draws (D, n), truth (n). Sorted-draw formula, O(D log D) per voxel."""
    d = draws.shape[0]
    skill = np.abs(draws - truth[None]).mean(axis=0)
    if d < 2:
        return skill
    s = np.sort(draws, axis=0)
    weights = (2.0 * np.arange(1, d + 1) - d - 1.0)[:, None]
    spread = (weights * s).sum(axis=0) * 2.0 / (d * (d - 1))   # E|X - X'| (unbiased)
    return skill - 0.5 * spread


@register("ensemble")
class EnsembleMetrics(Metric):
    pointwise = "partial"        # voxel calibration is pointwise, the statistic PITs are not
    pointwise_keys = ("voxel",)  # dropped in generative mode
    needs = ()
    wants_draws = True

    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        self.voxel = {"crps": [], "spread": [], "mse_mean": [], "n": 0}
        self.stage_rows = []      # per sample: (n_stage, 3) crps / spread / mse
        self.ranks = None
        self.stats: dict[str, list] = {}       # name -> list of (truth, draws[D]) per sample
        self.groups: dict[str, list] = {}      # params key -> list of (sample stats truth, draws)
        self.n_draws = None
        self.n_samples = 0
        self._binner = None
        self.single = 0
        self.identical = 0

    # ------------------------------------------------------------------ statistics
    def _bins(self, nx, ny, box):
        key = (nx, ny, round(box, 9))
        if self._binner is None or self._binner[0] != key:
            self._binner = (key, KBinner(nx, ny, box, self.settings.n_k_bins))
        return self._binner[1]

    def _summary(self, cube, z, masks, binner, k_index):
        """{name: value} summary statistics of one field (truth or one draw)."""
        out = {}
        xbar = cube.mean(axis=(0, 1))
        positions = np.linspace(0, cube.shape[2] - 1, self.settings.ensemble_n_los).round().astype(int)
        for p in positions:
            label = f"xbar@z{z[p]:.2f}" if z is not None else f"xbar@{p}"
            out[label] = float(xbar[p])
        if self.settings.target == "xhi" and z is not None:
            feats = history_features(z, xbar)
            for key in ("z_xhi25", "z_xhi50", "z_xhi75", "delta_z"):
                out[f"history_{key}"] = float(feats[key])
        sums = binner.slice_sums(cube - cube.mean(axis=(0, 1), keepdims=True))    # (nz, nb)
        power = sums / np.maximum(binner.counts, 1)[None, :]
        delta2 = power * binner.centers[None, :] ** 2 / (2 * np.pi)
        for label, mask in masks:
            if mask.sum() < self.settings.min_stage_slices:
                continue
            for k_target, kb in k_index:
                out[f"power_{label}_k{k_target:g}"] = float(np.log10(max(delta2[mask, kb].mean(), 1e-30)))
        return out

    # ------------------------------------------------------------------ update
    def update(self, sample, prediction, aux=None, runtime=None):
        draws = None if aux is None else aux.get("draws")
        if draws is None:
            draws = np.asarray(prediction[self.settings.target])[None]
        draws = np.asarray(draws, dtype=np.float64)
        d = draws.shape[0]
        if d < 2:
            self.single += 1
            return
        if all(np.array_equal(draws[0], draws[i]) for i in range(1, d)):
            self.identical += 1            # a deterministic model: no spread to calibrate
            return
        self.n_draws = d if self.n_draws is None else min(self.n_draws, d)
        truth = self.target(sample)
        nx, ny, nz = truth.shape
        self.n_samples += 1

        # ---- voxel calibration on a random voxel subset (deterministic per sample)
        rng = np.random.default_rng(zlib.crc32(str(sample.id).encode()) & 0xFFFFFFFF)
        n_vox = min(truth.size, self.settings.ensemble_voxels)
        flat_index = rng.choice(truth.size, n_vox, replace=False)
        t = truth.reshape(-1)[flat_index]
        x = draws.reshape(d, -1)[:, flat_index]
        crps = fair_crps(x, t)
        spread = x.std(axis=0, ddof=1)
        mse = (x.mean(axis=0) - t) ** 2
        self.voxel["crps"].append(float(crps.mean()))
        self.voxel["spread"].append(float(np.sqrt((spread ** 2).mean())))
        self.voxel["mse_mean"].append(float(mse.mean()))
        below = (x < t[None]).sum(axis=0)
        ties = (x == t[None]).sum(axis=0)
        rank = below + np.floor(rng.random(n_vox) * (ties + 1)).astype(int)
        if self.ranks is None or self.ranks.size != d + 1:
            self.ranks = np.zeros(d + 1, np.int64)
        self.ranks += np.bincount(np.minimum(rank, d), minlength=d + 1)[: d + 1]

        # ---- per-stage voxel scores (stages from the true x_HI)
        xbar = self.stage_xbar(sample, truth)
        masks = stage_masks(xbar, self.settings.stage_edges, self.settings.active_range,
                            self.settings.min_stage_slices)
        labels = stage_labels(self.settings.stage_edges, self.settings.active_range)
        z_of_voxel = flat_index % nz
        rows = []
        for mask in masks:
            sel = mask[z_of_voxel]
            rows.append([crps[sel].mean() if sel.any() else np.nan,
                         np.sqrt((spread[sel] ** 2).mean()) if sel.any() else np.nan,
                         mse[sel].mean() if sel.any() else np.nan])
        self.stage_rows.append(rows)

        # ---- summary statistics: truth vs each draw
        z = self.z_grid(sample)
        binner = self._bins(nx, ny, self.box_mpc(sample))
        k_index = [(k, int(np.argmin(np.abs(np.log(binner.centers / k))))) for k in self.settings.ensemble_k
                   if binner.centers[0] * 0.8 <= k <= binner.centers[-1] * 1.25]
        named_masks = list(zip(labels, masks))
        truth_stats = self._summary(truth, z, named_masks, binner, k_index)
        draw_stats = [self._summary(draws[i], z, named_masks, binner, k_index) for i in range(d)]
        for name, value in truth_stats.items():
            values = np.array([s.get(name, np.nan) for s in draw_stats])
            self.stats.setdefault(name, []).append((value, values))
        key = json.dumps({k: round(float(v), 9) for k, v in sorted((sample.meta.params or {}).items())})
        if sample.meta.params:
            self.groups.setdefault(key, []).append((truth_stats, draw_stats))
        self._labels = labels

    # ------------------------------------------------------------------ results
    @staticmethod
    def _pit_summary(rows, name=""):
        truth = np.array([r[0] for r in rows], dtype=float)
        draws = [np.asarray(r[1], float) for r in rows]
        rng = np.random.default_rng(zlib.crc32(name.encode()) & 0xFFFFFFFF)
        pit, zs, ranks, sizes = [], [], [], []
        for t, x in zip(truth, draws):
            x = x[np.isfinite(x)]
            if not np.isfinite(t) or x.size < 2:
                continue
            below, ties = int((x < t).sum()), int((x == t).sum())
            rank = below + int(rng.integers(0, ties + 1))
            ranks.append(rank)
            sizes.append(x.size)
            pit.append((rank + rng.random()) / (x.size + 1))
            sd = x.std(ddof=1)
            zs.append((t - x.mean()) / sd if sd > 0 else np.nan)
        if not pit:
            return None
        pit = np.sort(np.array(pit))
        n = pit.size
        ks = float(np.max(np.maximum(np.arange(1, n + 1) / n - pit, pit - np.arange(n) / n)))
        zs = np.array(zs, float)
        out = {"n": int(n), "pit_ks": ks,
               "mean_z2": float(np.nanmean(zs ** 2)) if np.isfinite(zs).any() else None,
               "pit_hist": np.histogram(pit, bins=PIT_BINS, range=(0, 1))[0].tolist()}
        d = int(np.median(sizes))
        for level in (68, 95):
            lo, hi, nominal = _central(d, level / 100)
            out[f"coverage{level}"] = float(np.mean([(lo <= r <= hi) for r in ranks]))
            out[f"coverage{level}_nominal"] = nominal
        return out

    def result(self):
        if not self.n_samples:
            if self.identical:
                return {"note": f"all draws identical in {self.identical} samples: the model is "
                                "deterministic, so there is no ensemble to calibrate"}
            return {"note": "single-draw predictions: run with --draws N (N >= 2) or return (N, Nx, Ny, Nz)"}
        d = self.n_draws
        crps = np.array(self.voxel["crps"])
        spread = np.array(self.voxel["spread"])
        rmse = np.sqrt(np.array(self.voxel["mse_mean"]))
        ratio = np.sqrt((d + 1) / d) * spread / np.maximum(rmse, 1e-30)
        expected = self.ranks.sum() / self.ranks.size
        chi2 = float(((self.ranks - expected) ** 2 / expected).sum() / max(self.ranks.size - 1, 1))
        stage = np.nanmean(np.array(self.stage_rows, float), axis=0) if self.stage_rows else None
        stages = []
        if stage is not None:
            for label, (c, s, m) in zip(self._labels, stage):
                stages.append({"label": label, "crps": _f(c),
                               "spread_skill": _f(np.sqrt((d + 1) / d) * s / np.sqrt(m)) if m > 0 else None})
        families = {}
        per_stat = {}
        for name, rows in self.stats.items():
            summary = self._pit_summary(rows, name)
            if summary is None:
                continue
            per_stat[name] = summary
            family = _family(name)
            families.setdefault(family, []).append(summary)
        family_summary = {}
        for family, items in families.items():
            pit_hist = np.sum([i["pit_hist"] for i in items], axis=0)
            family_summary[family] = {
                "n_stats": len(items),
                "pit_ks_med": _f(nanmedian([i["pit_ks"] for i in items])),
                "coverage68": _f(np.mean([i["coverage68"] for i in items])),
                "coverage68_nominal": items[0]["coverage68_nominal"],
                "coverage95": _f(np.mean([i["coverage95"] for i in items])),
                "coverage95_nominal": items[0]["coverage95_nominal"],
                "mean_z2_med": _f(nanmedian([i["mean_z2"] for i in items if i["mean_z2"] is not None])),
                "pit_hist": pit_hist.tolist(),
            }
        cv = self._cosmic_variance()
        return {
            "n_draws": int(d), "n_samples": int(self.n_samples), "single_draw_samples": int(self.single),
            "voxel": {"crps_med": _f(nanmedian(crps)), "spread_skill_med": _f(nanmedian(ratio)),
                      "rank_chi2": chi2, **self._rank_coverage(),
                      "rank_histogram": self.ranks.tolist()},
            "voxel_stages": stages,
            "statistics": family_summary,
            "per_statistic": per_stat,
            "cosmic_variance": cv,
        }

    def _rank_coverage(self):
        d = self.ranks.size - 1
        total = max(int(self.ranks.sum()), 1)
        out = {}
        for level in (50, 90):
            lo, hi, nominal = _central(d, level / 100)
            out[f"coverage{level}"] = _f(self.ranks[lo:hi + 1].sum() / total)
            out[f"coverage{level}_nominal"] = nominal
        return out

    def _cosmic_variance(self):
        repeated = {k: v for k, v in self.groups.items() if len(v) >= 2}
        if not repeated:
            return {"note": "no repeated parameter sets in the data; the truth spread at fixed parameters "
                            "(cosmic variance) cannot be measured, so draw spreads are judged by the PITs only"}
        ratios = {}
        for rows in repeated.values():
            names = set(rows[0][0])
            for name in names:
                truths = np.array([r[0].get(name, np.nan) for r in rows])
                draws = np.array([s.get(name, np.nan) for r in rows for s in r[1]])
                if np.isfinite(truths).sum() >= 2 and np.isfinite(draws).sum() >= 2:
                    st = np.nanstd(truths, ddof=1)
                    if st > 0:
                        ratios.setdefault(_family(name), []).append(np.nanstd(draws, ddof=1) / st)
        return {"n_groups": len(repeated),
                "draw_over_truth_std_med": {k: _f(nanmedian(v)) for k, v in ratios.items()}}

    def arrays(self):
        if not self.n_samples:
            return {}
        out = {"rank_histogram": self.ranks}
        for family, items in self.result()["statistics"].items():
            out[f"pit_hist_{family}"] = np.asarray(items["pit_hist"])
        return out


def _central(d: int, level: float):
    """Central rank window [lo, hi] of the D+1 ranks closest to ``level``; its nominal coverage."""
    m = int(np.clip(round(level * (d + 1)), 1, d + 1))
    lo = (d + 1 - m) // 2
    return lo, lo + m - 1, m / (d + 1)


def _family(name: str) -> str:
    """xbar@z7.12 -> xbar, history_z_xhi50 -> history, power_<stage>_k0.1 -> power."""
    for prefix in ("xbar", "history", "power"):
        if name.startswith(prefix):
            return prefix
    return name


def _f(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None

