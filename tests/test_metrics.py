import numpy as np
import pytest

from eval21.metrics.anisotropy import AnisotropyMetrics
from eval21.metrics.bubble import BubbleSizeMetrics
from eval21.metrics.edge import EdgeMetrics
from eval21.metrics.efficiency import EfficiencyMetrics
from eval21.metrics.eft import EftMetrics
from eval21.metrics.front import FrontMetrics
from eval21.metrics.history import HistoryMetrics
from eval21.metrics.morphology import MorphologyMetrics
from eval21.metrics.observable import ObservableMetrics
from eval21.metrics.parity import ParityMetrics
from eval21.metrics.radial import RadialMetrics
from eval21.metrics.spectrum import SpectrumMetrics
from eval21.metrics.symmetry import SymmetryMetrics
from eval21.metrics.voxel import VoxelMetrics
from eval21.settings import Settings
from eval21.spec import Meta, Sample


def make_sample(shape=(24, 24, 8), seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.random(shape).astype(np.float32)
    density = (1.0 + 0.3 * rng.normal(size=shape)).astype(np.float32)
    return Sample(
        id="s0",
        inputs={"density": density},
        targets={"xhi": truth},
        meta=Meta(box_mpc=100.0, z_grid=np.linspace(5, 25, shape[2]),
                  omega_m=0.31),
    )


SETTINGS = Settings(box_mpc=100.0, n_k_bins=8, rays_per_slice=32,
                    slices_per_stage=2, bubble_bins=8, eft_k_max=0.5)


def test_voxel_analytic():
    sample = make_sample()
    truth = sample.targets["xhi"]
    metric = VoxelMetrics(SETTINGS)
    metric.update(sample, {"xhi": truth + 0.1})
    result = metric.result()
    assert result["rmse"] == pytest.approx(0.1, rel=1e-6)
    assert result["bias"] == pytest.approx(0.1, rel=1e-6)
    assert result["mae"] == pytest.approx(0.1, rel=1e-6)


def test_voxel_gradient_matches_reference_forward_difference():
    sample = make_sample()
    truth = sample.targets["xhi"].astype(np.float64)
    pred = truth + 0.05 * np.random.default_rng(1).normal(size=truth.shape)
    metric = VoxelMetrics(SETTINGS)
    metric.update(sample, {"xhi": pred})
    error = pred - truth
    dx = np.roll(error, -1, axis=0) - error
    dy = np.roll(error, -1, axis=1) - error
    expected = np.sqrt(np.sum(dx * dx + dy * dy) / (2 * error.size))
    assert metric.result()["gradient_rmse"] == pytest.approx(expected, rel=1e-9)


def test_voxel_mean_xhi_mae_matches_reference():
    sample = make_sample()
    truth = sample.targets["xhi"].astype(np.float64)
    pred = truth + 0.1
    metric = VoxelMetrics(SETTINGS)
    metric.update(sample, {"xhi": pred})
    expected = float(np.abs((pred.mean(axis=(0, 1)) - truth.mean(axis=(0, 1)))).sum())
    assert metric.result()["mean_xhi_mae"] == pytest.approx(expected, rel=1e-9)


def test_voxel_r2_against_constant_mean():
    sample = make_sample()
    truth = sample.targets["xhi"]
    metric = VoxelMetrics(SETTINGS)
    metric.update(sample, {"xhi": np.full_like(truth, truth.mean())})
    assert metric.result()["r2"] == pytest.approx(0.0, abs=1e-6)


def test_spectrum_perfect_and_smoothed():
    from scipy.ndimage import gaussian_filter

    sample = make_sample()
    truth = sample.targets["xhi"]

    metric = SpectrumMetrics(SETTINGS)
    metric.update(sample, {"xhi": truth})
    active = metric.result()["active"]
    assert active["coherence_ref"] == pytest.approx(1.0, abs=1e-9)

    smooth = {"xhi": gaussian_filter(truth, sigma=(1, 1, 0))}
    metric = SpectrumMetrics(SETTINGS)
    metric.update(sample, smooth)
    active = metric.result()["active"]
    assert active["ratio_ref"] < 1.0


def test_history_perfect_and_offset():
    z = np.linspace(5, 25, 16)
    truth = np.broadcast_to(
        np.clip((z - 5) / 20, 0, 1).reshape(1, 1, -1), (16, 16, 16)).astype(np.float32)
    sample = Sample(
        id="ramp",
        inputs={"density": np.ones_like(truth)},
        targets={"xhi": truth},
        meta=Meta(box_mpc=100.0, z_grid=z, omega_m=0.31),
    )
    metric = HistoryMetrics(SETTINGS)
    metric.update(sample, {"xhi": truth})
    result = metric.result()
    assert result["tau"]["error_med"] == pytest.approx(0.0, abs=1e-9)
    assert result["history"]["z_xhi50"]["error_med"] == pytest.approx(0.0, abs=1e-9)

    metric = HistoryMetrics(SETTINGS)
    metric.update(sample, {"xhi": np.clip(truth + 0.1, 0, 1)})
    shifted = metric.result()
    # +0.1 neutral fraction reaches 0.5 two redshift units earlier.
    assert shifted["history"]["z_xhi50"]["error_med"] == pytest.approx(-2.0, abs=1e-6)


def test_observable_perfect():
    sample = make_sample()
    truth = sample.targets["xhi"]
    metric = ObservableMetrics(SETTINGS)
    metric.update(sample, {"xhi": truth})
    active = metric.result()["active"]
    assert active["partial_fraction_excess"] == pytest.approx(0.0, abs=1e-12)
    assert active["w1_xhi"] == pytest.approx(0.0, abs=1e-12)
    assert active["dtb_variance_ratio"] == pytest.approx(1.0, rel=1e-9)


def test_observable_dxbar_and_xbar_d2_shapes():
    sample = make_sample()
    truth = sample.targets["xhi"]
    metric = ObservableMetrics(SETTINGS)
    metric.update(sample, {"xhi": np.clip(truth + 0.05, 0, 1)})
    result = metric.result()
    nz = truth.shape[2]
    assert len(result["dxbar"]["med"]) == nz
    assert result["dxbar"]["n_z"] == nz
    stage = metric.rows[0]
    assert stage["xbar_d2_truth"].shape == (20, len(SETTINGS.k_targets))


def test_observable_requires_density():
    assert ObservableMetrics.needs == ("density",)
    assert EftMetrics.needs == ("density",)


def test_eft_perfect_transfer():
    sample = make_sample()
    truth = sample.targets["xhi"]
    metric = EftMetrics(SETTINGS)
    metric.update(sample, {"xhi": truth})
    active = metric.result()["active"]
    assert active["b1_delta_sigma"] == pytest.approx(0.0, abs=1e-6)
    assert all(abs(value - 1.0) < 1e-6 for value in active["transfer_ratio"])


def test_front_perfect_excess_zero():
    x = np.arange(24).reshape(-1, 1, 1)
    truth = np.broadcast_to(
        1.0 / (1.0 + np.exp(-(x - 12.0) / 2.0)), (24, 24, 8)).astype(np.float32)
    sample = Sample(
        id="front",
        inputs={"density": np.ones_like(truth)},
        targets={"xhi": truth},
        meta=Meta(box_mpc=100.0, z_grid=np.linspace(5, 25, 8)),
    )
    metric = FrontMetrics(SETTINGS)
    metric.update(sample, {"xhi": truth})
    result = metric.result()
    assert result["front_width_excess_mpc"] == pytest.approx(0.0, abs=1e-9)
    assert 25 < result["front_width_truth_mpc"] < 45


def test_bubble_identical_is_zero():
    sample = make_sample()
    truth = sample.targets["xhi"]
    metric = BubbleSizeMetrics(SETTINGS)
    metric.update(sample, {"xhi": truth})
    active = metric.result()["active"]
    assert active["wasserstein_med_mpc"] == pytest.approx(0.0, abs=1e-12)
    assert active["js_divergence_med"] == pytest.approx(0.0, abs=1e-12)


def test_morphology_identical():
    sample = make_sample()
    truth = sample.targets["xhi"]
    metric = MorphologyMetrics(SETTINGS)
    metric.update(sample, {"xhi": truth})
    result = metric.result()
    assert result["count_ratio"] == pytest.approx(1.0)
    assert result["largest_fraction_ratio"] == pytest.approx(1.0)
    assert result["iou"]["matched"] == pytest.approx(1.0)
    assert result["iou"]["threshold"] == pytest.approx(1.0)


def test_anisotropy_perfect():
    sample = make_sample()
    truth = sample.targets["xhi"]
    metric = AnisotropyMetrics(SETTINGS, {"chunk_cells": 8})
    metric.update(sample, {"xhi": truth})
    headline = metric.result()["headline"]
    assert headline["coherence_high_kpar"] == pytest.approx(1.0, abs=1e-9)
    assert headline["anisotropy_median"] == pytest.approx(1.0, abs=1e-9)


def test_parity_counts_upper_edge():
    truth = np.ones((10, 10, 2), dtype=np.float32)
    sample = Sample(
        id="ones", inputs={"density": np.ones_like(truth)},
        targets={"xhi": truth},
        meta=Meta(box_mpc=100.0, z_grid=np.linspace(5, 25, 2)))
    metric = ParityMetrics(SETTINGS)
    metric.update(sample, {"xhi": truth})
    result = metric.result()
    assert result["n_voxels"] == truth.size
    assert result["headline"]["conditional_bias_max"] == pytest.approx(0.0, abs=1e-9)


def test_parity_percentiles_present():
    sample = make_sample()
    truth = sample.targets["xhi"]
    metric = ParityMetrics(SETTINGS)
    metric.update(sample, {"xhi": np.clip(truth + 0.05, 0, 1)})
    bins = metric.result()["bins"]
    assert bins and bins[0]["pred_p50"] is not None


def test_eft_normalisation_matches_manual():
    sample = make_sample()
    truth = sample.targets["xhi"]
    density = sample.inputs["density"].astype(np.float64)
    metric = EftMetrics(SETTINGS)
    metric.update(sample, {"xhi": truth})
    stage = metric.rows[0]["stages"][-1]
    binner = metric._binner[1]
    n = truth.shape[2]
    u = truth - truth.mean(axis=(0, 1), keepdims=True)
    delta = density - density.mean(axis=(0, 1), keepdims=True)
    dd = binner.slice_sums(delta).sum(axis=0)
    td = binner.cross_sums(u, delta).sum(axis=0)
    tt = binner.slice_sums(u).sum(axis=0)
    expected = (tt - td ** 2 / dd) / (binner.counts * n)
    expected = np.where(expected > 0, expected, np.nan)
    assert np.allclose(stage["eps_truth"], expected, equal_nan=True, rtol=1e-12)


def test_morphology_all_neutral_iou_is_one():
    truth = np.ones((12, 12, 4), dtype=np.float32)
    sample = Sample(
        id="neutral", inputs={"density": np.ones_like(truth)},
        targets={"xhi": truth},
        meta=Meta(box_mpc=100.0, z_grid=np.linspace(5, 25, 4)))
    metric = MorphologyMetrics(SETTINGS)
    metric.update(sample, {"xhi": truth})
    iou = metric.result()["iou"]
    assert iou["threshold"] == pytest.approx(1.0)
    assert iou["matched"] == pytest.approx(1.0)


def test_morphology_bimodal_iou():
    rng = np.random.default_rng(3)
    truth = (rng.random((16, 16, 8)) > 0.5).astype(np.float32)
    pred = (rng.random((16, 16, 8)) > 0.5).astype(np.float32)
    sample = Sample(
        id="bimodal", inputs={"density": np.ones_like(truth)},
        targets={"xhi": truth},
        meta=Meta(box_mpc=100.0, z_grid=np.linspace(5, 25, 8)))
    metric = MorphologyMetrics(SETTINGS)
    metric.update(sample, {"xhi": pred})
    result = metric.result()["iou"]
    assert 0 <= result["threshold"] <= 1
    assert 0 <= result["matched"] <= 1
    assert 0 <= result["f0.5"] <= 1


def test_front_skips_one_cell_axis():
    truth = np.zeros((1, 8, 4), dtype=np.float32)
    sample = Sample(
        id="thin", inputs={"density": np.ones_like(truth)},
        targets={"xhi": truth},
        meta=Meta(box_mpc=100.0, z_grid=np.linspace(5, 25, 4)))
    metric = FrontMetrics(SETTINGS)
    metric.update(sample, {"xhi": truth})
    assert metric.result()["skipped_samples"] == 1


def test_parity_perfect():
    sample = make_sample()
    truth = sample.targets["xhi"]
    metric = ParityMetrics(SETTINGS)
    metric.update(sample, {"xhi": truth})
    result = metric.result()
    assert result["headline"]["conditional_bias_max"] == pytest.approx(0.0, abs=1e-12)
    assert result["headline"]["partial_rmse"] == pytest.approx(0.0, abs=1e-12)


def test_radial_perfect():
    sample = make_sample()
    truth = sample.targets["xhi"]
    metric = RadialMetrics(SETTINGS)
    metric.update(sample, {"xhi": truth})
    result = metric.result()
    assert result["headline"]["coherence_ref"] == pytest.approx(1.0, abs=1e-9)
    assert result["headline"]["ratio_ref"] == pytest.approx(1.0, abs=1e-9)


def test_edge_metric_reports_active_window():
    sample = make_sample()
    truth = sample.targets["xhi"]
    metric = EdgeMetrics(SETTINGS)
    metric.update(sample, {"xhi": truth})
    result = metric.result()
    assert result["transverse_edge_rmse"] == pytest.approx(0.0, abs=1e-9)
    assert result["global_history_rmse"] == pytest.approx(0.0, abs=1e-9)


def test_efficiency_benchmark_uses_median_rate():
    metric = EfficiencyMetrics(SETTINGS)
    metric.observe_benchmark({"seconds": [1.0, 3.0], "slices": 4})
    result = metric.result()
    assert result["timing"] == "forward_only"
    assert result["slices_per_s"] == pytest.approx(2.0)


def test_symmetry_records_rerun():
    sample = make_sample()
    truth = sample.targets["xhi"]
    metric = SymmetryMetrics(SETTINGS)
    metric.update(sample, {"xhi": truth}, aux={"xy": {"xhi": truth}})
    assert metric.result()["xy_rms"] == pytest.approx(0.0)


def test_metric_with_missing_target_raises():
    sample = make_sample()
    metric = VoxelMetrics(SETTINGS)
    with pytest.raises(KeyError):
        metric.update(sample, {})
