"""Numeric self-checks that need no data and no GPU.

Run with ``cm21eval selftest`` (also collected by pytest). Every check has an
analytic answer, so the suite itself stays honest: if a metric cannot
reproduce RMSE=0.1 for a constant offset, or coherence=1 for a perfect
prediction, the report is not trustworthy.
"""
from __future__ import annotations

import numpy as np

from . import physics
from .settings import Settings
from .spec import Meta, Sample


def _sample(pred=None, truth=None, density=None, shape=(32, 32, 8)):
    rng = np.random.default_rng(0)
    truth = rng.random(shape) if truth is None else truth
    density = 1.0 + 0.5 * rng.normal(size=shape) if density is None else density
    truth = truth.astype(np.float32)
    sample = Sample(
        id="check",
        inputs={"density": density.astype(np.float32)},
        targets={"xhi": truth},
        meta=Meta(box_mpc=100.0, z_grid=np.linspace(5, 25, shape[2]),
                  omega_m=0.30966),
    )
    prediction = {"xhi": (truth if pred is None else pred)}
    return sample, prediction


def _zero_rmse():
    sample, prediction = _sample()
    return float(np.sqrt(np.mean(
        (prediction["xhi"] - sample.targets["xhi"]) ** 2)))


def run(verbose: bool = True) -> int:
    failures = []

    def check(name, condition, detail=""):
        if condition:
            if verbose:
                print(f"  ok   {name}")
        else:
            failures.append(name)
            print(f"  FAIL {name} {detail}")

    if verbose:
        print("physics")

    k = np.logspace(-2, 0, 12)
    pt = np.full(12, 2.0)
    check("coherence of identical fields is 1",
          np.allclose(physics.safe_ratio(pt, pt, pt)[1], 1.0))
    check("power ratio of identical fields is 1",
          np.allclose(physics.safe_ratio(pt, pt, pt)[0], 1.0))

    z = np.linspace(5.0, 25.0, 200)
    xbar = np.clip((z - 6.0) / 12.0, 0.0, 1.0)
    crossings = physics.history_features(z, xbar)
    check("x_HI=0.5 crossing is where the linear ramp says",
          abs(crossings["z_xhi50"] - 12.0) < 0.15, crossings)
    check("duration delta_z is positive for a rising history",
          crossings["delta_z"] > 0, crossings)
    tau = physics.thomson_optical_depth(z, xbar)
    check("tau_e is finite and positive", np.isfinite(tau) and tau > 0, tau)

    # A fully ionized, very late history gives a small optical depth.
    tau_late = physics.thomson_optical_depth(z, np.zeros_like(z))
    check("tau_e falls as hydrogen becomes neutral", tau < tau_late, (tau, tau_late))

    check("1-D Wasserstein of shifted samples is the shift",
          abs(physics.wasserstein_1d(np.zeros(50), np.full(50, 0.3)) - 0.3) < 1e-9)
    edges = np.linspace(0, 1, 11)
    check("histogram W1 is zero for identical histograms",
          physics.histogram_w1(np.arange(10), np.arange(10), edges) == 0.0)

    crossing = physics.k_crossing(np.array([0.1, 0.2, 0.4]),
                                  np.array([1.0, 0.95, 0.5]), 0.9)
    check("k_crossing interpolates in log k",
          0.20 < crossing < 0.22, crossing)

    if verbose:
        print("metrics")

    from .metrics.bubble import BubbleSizeMetrics
    from .metrics.efficiency import EfficiencyMetrics
    from .metrics.front import FrontMetrics
    from .metrics.history import HistoryMetrics
    from .metrics.morphology import MorphologyMetrics
    from .metrics.observable import ObservableMetrics
    from .metrics.spectrum import SpectrumMetrics
    from .metrics.voxel import VoxelMetrics

    settings = Settings(box_mpc=100.0, n_k_bins=8, rays_per_slice=32,
                        slices_per_stage=2, bubble_bins=8)

    sample, prediction = _sample()
    voxel = VoxelMetrics(settings)
    voxel.update(sample, prediction)
    result = voxel.result()
    check("perfect prediction has RMSE 0", result["rmse"] < 1e-9, result["rmse"])
    check("perfect prediction has R^2 1", abs(result["r2"] - 1) < 1e-12)

    truth = np.asarray(sample.targets["xhi"])
    offset = {"xhi": truth + 0.1}
    voxel = VoxelMetrics(settings)
    voxel.update(sample, offset)
    check("constant +0.1 offset has RMSE 0.1",
          abs(voxel.result()["rmse"] - 0.1) < 1e-6, voxel.result()["rmse"])

    mean = np.full_like(truth, truth.mean())
    voxel = VoxelMetrics(settings)
    voxel.update(sample, {"xhi": mean})
    check("constant-mean predictor has R^2 0",
          abs(voxel.result()["r2"]) < 1e-6, voxel.result()["r2"])

    spectrum = SpectrumMetrics(settings)
    spectrum.update(sample, prediction)
    active = spectrum.result()["active"]
    check("perfect prediction has coherence 1",
          abs((active["coherence_ref"] or 0) - 1) < 1e-6, active)

    smoothed = {"xhi": _smooth2d(truth, 1.0)}
    spectrum = SpectrumMetrics(settings)
    spectrum.update(sample, smoothed)
    active = spectrum.result()["active"]
    check("smoothed prediction suppresses high-k power",
          (active["ratio_ref"] or 1) < 1.0, active["ratio_ref"])

    history = HistoryMetrics(settings)
    history.update(sample, prediction)
    check("perfect history has zero tau error",
          abs(history.result()["tau"]["error_med"]) < 1e-9)

    front = FrontMetrics(settings)
    front.update(sample, prediction)
    check("perfect front has zero excess width",
          abs(front.result()["front_width_excess_mpc"] or 0) < 1e-9)

    bubble = BubbleSizeMetrics(settings)
    bubble.update(sample, prediction)
    check("identical fields give zero bubble W1",
          abs(bubble.result()["active"]["wasserstein_med_mpc"] or 0) < 1e-12)

    morphology = MorphologyMetrics(settings)
    morphology.update(sample, prediction)
    check("identical fields give morphology ratio 1",
          abs((morphology.result()["count_ratio"] or 0) - 1) < 1e-9)

    observable = ObservableMetrics(settings)
    observable.update(sample, prediction)
    check("perfect observable has zero partial excess",
          abs(observable.result()["active"]["partial_fraction_excess"] or 0) < 1e-12)

    efficiency = EfficiencyMetrics(settings)
    efficiency.attach({"parameters": 123, "device": "cpu"})
    efficiency.update(sample, prediction, runtime={"seconds": 0.01})
    check("efficiency reports milliseconds",
          abs(efficiency.result()["ms_per_sample"] - 10.0) < 1e-9)

    if verbose:
        print()
        print(f"{len(failures)} failure(s)" if failures else "all checks passed")
    return 1 if failures else 0


def _smooth2d(field, sigma):
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(field, sigma=(sigma, sigma, 0))


if __name__ == "__main__":
    raise SystemExit(run())
