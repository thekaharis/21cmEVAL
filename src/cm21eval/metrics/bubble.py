"""Bubble sizes from transverse mean-free paths.

For each reionization stage a handful of slices is sampled. Starting points
are uniform over ionized area and directions isotropic, so the distance to
the first neutral cell is an ionized-area-weighted mean-free-path sample --
the usual bubble-size estimator. Periodic transverse boundaries make a
percolating slice return censored rays instead of a fake edge.

Reported per stage: the histogram mass function, the restricted Wasserstein-1
distance (censored rays folded in at ``max_distance_mpc``), the Jensen-Shannon
divergence, and the relative bias of the mean bubble size. A model that blurs
fronts systematically over-produces small bubbles and under-produces large
ones; W1 catches that where voxel RMSE does not.
"""
from __future__ import annotations

import zlib

import numpy as np

from ..physics import jensen_shannon, mean_free_path_samples_2d, wasserstein_1d
from ..registry import register
from .base import Metric, median_band, nanmedian


def _selected(selection: np.ndarray, limit: int) -> np.ndarray:
    indices = np.flatnonzero(selection)
    if indices.size <= limit:
        return indices
    positions = np.linspace(0, indices.size - 1, limit)
    return indices[np.unique(np.rint(positions).astype(int))]


@register("bubble")
class BubbleSizeMetrics(Metric):
    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        self.per_sample: list[dict] = []
        self.edges = None
        self.centers = None

    def _stage_rows(self, truth):
        xbar = truth.mean(axis=(0, 1))
        stage = np.digitize(xbar, self.settings.stage_edges) - 1
        rows = [
            _selected(stage == index, self.settings.slices_per_stage)
            for index in range(len(self.settings.stage_edges) - 1)
        ]
        active = (xbar >= self.settings.active_range[0]) & (
            xbar <= self.settings.active_range[1])
        rows.append(_selected(active, self.settings.slices_per_stage))
        return rows

    def _sample_stage(self, field, slices, sample_id, stage_index, edges, cell):
        distances, requested, censored, underflow, ionized = [], 0, 0, 0, []
        token = zlib.crc32(str(sample_id).encode()) & 0xFFFFFFFF
        for slice_index in slices:
            mask = field[:, :, slice_index] < self.settings.threshold
            seed = np.random.SeedSequence(
                [self.settings.seed, token, int(stage_index), int(slice_index)]
            )
            result = mean_free_path_samples_2d(
                mask, self.settings.rays_per_slice, cell,
                float(edges[-1]), seed=seed,
            )
            distances.append(result["distances_mpc"])
            requested += result["n_requested"]
            censored += result["n_censored"]
            underflow += result["n_requested"] - result["n_started"]
            ionized.append(result["ionized_fraction"])
        hits = np.concatenate(distances) if distances else np.empty(0)
        if requested == 0:
            return None
        clipped = np.clip(hits, edges[0], np.nextafter(edges[-1], -np.inf))
        mass = np.histogram(clipped, bins=edges)[0].astype(np.float64) / requested
        return {
            "mass": mass,
            "distances": hits,
            "n_requested": requested,
            "n_censored": censored,
            "n_underflow": underflow,
            "censored_fraction": censored / requested,
            "underflow_fraction": underflow / requested,
            "ionized_fraction": float(np.mean(ionized)) if ionized else np.nan,
            # Underflow (no ionized starting cell) enters the size distribution
            # as zero-length paths, censored rays as the cap -- exactly the
            # two-sided restriction used by fno-21cm.
            "restricted_mean_mpc": float(
                (hits.sum() + censored * edges[-1]) / requested),
        }

    def update(self, sample, prediction, aux=None, runtime=None):
        truth = self.target(sample)
        pred = self.prediction(prediction)
        box = self.box_mpc(sample)
        cell = (box / truth.shape[0], box / truth.shape[1])
        cap = (self.settings.max_distance_mpc
               if self.settings.max_distance_mpc is not None else box)
        if self.edges is None:
            minimum = max(float(cell[0]) / 8.0, 1e-4)
            self.edges = np.geomspace(minimum, float(cap), self.settings.bubble_bins + 1)
            self.centers = np.sqrt(self.edges[:-1] * self.edges[1:])

        rows = self._stage_rows(truth)
        stages = []
        for stage_index, slices in enumerate(rows):
            t = self._sample_stage(truth, slices, sample.id, stage_index, self.edges, cell)
            p = self._sample_stage(pred, slices, sample.id, stage_index, self.edges, cell)
            if t is None or p is None:
                stages.append(None)
                continue
            restricted_t = np.concatenate([
                np.zeros(t["n_underflow"]), t["distances"],
                np.full(t["n_censored"], cap, dtype=np.float64)])
            restricted_p = np.concatenate([
                np.zeros(p["n_underflow"]), p["distances"],
                np.full(p["n_censored"], cap, dtype=np.float64)])
            mass_p = np.concatenate([[t["underflow_fraction"]], t["mass"],
                                     [t["censored_fraction"]]])
            mass_q = np.concatenate([[p["underflow_fraction"]], p["mass"],
                                     [p["censored_fraction"]]])
            stages.append({
                "n_slices": int(len(slices)),
                "w1": wasserstein_1d(restricted_t, restricted_p),
                "js": jensen_shannon(mass_p / mass_p.sum(), mass_q / mass_q.sum()),
                "truth_mean": t["restricted_mean_mpc"],
                "pred_mean": p["restricted_mean_mpc"],
                "censored_truth": t["censored_fraction"],
                "censored_pred": p["censored_fraction"],
                "underflow_truth": t["underflow_fraction"],
                "underflow_pred": p["underflow_fraction"],
                "ionized_truth": t["ionized_fraction"],
                "ionized_pred": p["ionized_fraction"],
                "mass_truth": t["mass"],
                "mass_pred": p["mass"],
            })
        self.per_sample.append({"sample": sample.id, "stages": stages})

    def result(self):
        if not self.per_sample:
            return {}
        n_rows = len(self.per_sample[0]["stages"])
        labels = [f"xbar_HI {lo:.2f}-{hi:.2f}" for lo, hi in
                  zip(self.settings.stage_edges[:-1], self.settings.stage_edges[1:])]
        labels.append(f"active {self.settings.active_range[0]:.2f}-"
                      f"{self.settings.active_range[1]:.2f}")

        def stacks(key):
            out = np.full((len(self.per_sample), n_rows), np.nan)
            for i, row in enumerate(self.per_sample):
                for j, stage in enumerate(row["stages"]):
                    if stage is not None:
                        out[i, j] = stage[key]
            return out

        w1 = stacks("w1")
        js = stacks("js")
        truth_mean = stacks("truth_mean")
        pred_mean = stacks("pred_mean")
        with np.errstate(invalid="ignore", divide="ignore"):
            bias = np.where(truth_mean > 0, pred_mean / truth_mean - 1.0, np.nan)
        w1_med, w1_lo, w1_hi = median_band(w1, axis=0)
        js_med, _, _ = median_band(js, axis=0)
        bias_med, bias_lo, bias_hi = median_band(bias, axis=0)

        stages = []
        for row, label in enumerate(labels):
            stages.append({
                "label": label,
                "n_slices": int(np.nansum([r["stages"][row]["n_slices"]
                                           for r in self.per_sample
                                           if r["stages"][row] is not None])),
                "wasserstein_med_mpc": _f(w1_med[row]),
                "wasserstein_p16_mpc": _f(w1_lo[row]),
                "wasserstein_p84_mpc": _f(w1_hi[row]),
                "js_divergence_med": _f(js_med[row]),
                "mean_size_truth_mpc": _f(nanmedian(truth_mean[:, row])),
                "mean_size_pred_mpc": _f(nanmedian(pred_mean[:, row])),
                "relative_bias_med": _f(bias_med[row]),
                "relative_bias_p16": _f(bias_lo[row]),
                "relative_bias_p84": _f(bias_hi[row]),
                "censored_truth": _f(nanmedian(
                    stacks("censored_truth")[:, row])),
                "censored_pred": _f(nanmedian(
                    stacks("censored_pred")[:, row])),
            })
        active = stages[-1]
        return {
            "radius_edges_mpc": self.edges,
            "radius_centers_mpc": self.centers,
            "stage_labels": labels,
            "stages": stages,
            "active": {
                "wasserstein_med_mpc": active["wasserstein_med_mpc"],
                "js_divergence_med": active["js_divergence_med"],
                "relative_bias_med": active["relative_bias_med"],
            },
            "n_samples": len(self.per_sample),
        }

    def arrays(self):
        if not self.per_sample:
            return {}
        out = {"bubble_radius_mpc": self.centers}
        for name, key in (("mass_truth", "mass_truth"), ("mass_pred", "mass_pred")):
            rows = [row["stages"][-1][key] for row in self.per_sample
                    if row["stages"][-1] is not None]
            if rows:
                out[f"bubble_{name}_active"] = np.stack(rows)
        return out


def _f(value):
    value = float(value)
    return value if np.isfinite(value) else None
