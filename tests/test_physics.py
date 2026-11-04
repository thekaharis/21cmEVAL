import numpy as np
import pytest

from eval21 import physics


def test_wasserstein_shift():
    assert physics.wasserstein_1d(np.zeros(100), np.full(100, 0.4)) == pytest.approx(0.4)


def test_wasserstein_identical():
    values = np.linspace(0, 1, 50)
    assert physics.wasserstein_1d(values, values) == pytest.approx(0.0)


def test_histogram_w1_matches_sample_w1():
    edges = np.linspace(0, 2, 21)
    a = np.histogram(np.full(100, 0.5), bins=edges)[0]
    b = np.histogram(np.full(100, 1.5), bins=edges)[0]
    assert physics.histogram_w1(a, b, edges) == pytest.approx(1.0, abs=0.11)


def test_thomson_depth_ordering():
    z = np.linspace(5, 25, 200)
    neutral = np.clip((z - 6) / 12, 0, 1)
    ionized = np.zeros_like(z)
    assert physics.thomson_optical_depth(z, neutral) > 0
    assert physics.thomson_optical_depth(z, neutral) < physics.thomson_optical_depth(z, ionized)


def test_history_crossings_linear_ramp():
    z = np.linspace(5, 25, 401)
    x = np.clip((z - 10) / 10, 0, 1)
    features = physics.history_features(z, x)
    assert features["z_xhi50"] == pytest.approx(15.0, abs=0.06)
    assert features["z_xhi75"] - features["z_xhi25"] == pytest.approx(5.0, abs=0.1)


def test_history_crossing_unresolved_is_nan():
    z = np.linspace(5, 25, 10)
    assert np.isnan(physics.history_crossing(z, np.ones_like(z), 0.5))


def test_brightness_temperature_scaling():
    z = np.array([10.0])
    xhi = np.ones((2, 2, 1))
    delta = np.zeros((2, 2, 1))
    tb = physics.brightness_temperature_mk(xhi, delta, z)
    expected = 27.0 * (physics.OMEGA_B * physics.HUBBLE_H ** 2 / 0.023) * np.sqrt(
        0.15 / (physics.OMEGA_M * physics.HUBBLE_H ** 2) * 11.0 / 10.0)
    assert tb[0, 0, 0] == pytest.approx(expected, rel=1e-6)
    assert physics.brightness_temperature_mk(2 * xhi, delta, z)[0, 0, 0] == pytest.approx(
        2 * tb[0, 0, 0])


def test_kbinner_power_normalisation():
    index = np.arange(32)
    mode = np.cos(2 * np.pi * 4 * index / 32)
    field = np.broadcast_to(mode[:, None, None], (32, 32, 1)).copy()
    binner = physics.KBinner(32, 32, 100.0, 16)
    sums = binner.slice_sums(field - field.mean(axis=(0, 1), keepdims=True))
    # All power sits in one axis mode: sum over bins is variance times box^2.
    assert sums.sum() == pytest.approx(float(field.var()) * 100.0 ** 2, rel=1e-9)


def test_k_crossing():
    k = np.array([0.1, 0.2, 0.4])
    curve = np.array([1.0, 0.95, 0.5])
    value = physics.k_crossing(k, curve, 0.9)
    assert 0.20 < value < 0.22


def test_ray_tracer_half_plane():
    mask = np.zeros((64, 64), dtype=bool)
    mask[:32, :] = True
    distance = physics.trace_ray_to_neutral_2d(
        mask, (16.0, 16.0), (1.0, 0.0), (1.0, 1.0), 100.0)
    assert distance == pytest.approx(16.0)


def test_ray_tracer_censored():
    mask = np.ones((16, 16), dtype=bool)
    distance = physics.trace_ray_to_neutral_2d(
        mask, (8.0, 8.0), (1.0, 0.0), (1.0, 1.0), 20.0)
    assert np.isnan(distance)


def test_vectorised_ray_tracer_matches_scalar():
    rng = np.random.default_rng(5)
    mask = rng.random((64, 64)) > 0.45
    cells = np.argwhere(mask)
    starts = cells[rng.integers(0, len(cells), size=80)].astype(float) + rng.random((80, 2))
    angles = rng.uniform(0, 2 * np.pi, size=80)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    vector = physics.trace_rays_to_neutral_2d(
        mask, starts, directions, (1.5, 1.5), 200.0)
    scalar = np.array([
        physics.trace_ray_to_neutral_2d(
            mask, (float(s[0]), float(s[1])), (float(d[0]), float(d[1])),
            (1.5, 1.5), 200.0)
        for s, d in zip(starts, directions)
    ])
    assert np.array_equal(vector, scalar, equal_nan=True)


def test_mfp_distribution_mean_half_plane():
    mask = np.zeros((64, 64), dtype=bool)
    mask[:32, :] = True
    result = physics.mean_free_path_samples_2d(
        mask, 400, (1.0, 1.0), 100.0, seed=0)
    # Left half ionized: grazing rays travel far and are censored at 100 Mpc.
    assert 300 < result["n_hit"] < 400
    assert 20 < result["distances_mpc"].mean() < 35


def test_ionized_region_summary_single_sphere():
    field = np.ones((40, 40, 40))
    x, y, z = np.indices(field.shape)
    radius = 5.0
    field[(x - 20) ** 2 + (y - 20) ** 2 + (z - 20) ** 2 <= radius ** 2] = 0.0
    summary = physics.ionized_region_summary(field, threshold=0.5)
    assert summary["count"] == 1
    assert summary["mean_effective_radius_cells"] == pytest.approx(
        radius * (3 / 4) ** (1 / 3), rel=0.15)
    assert summary["volume_weighted_radius_cells"] == pytest.approx(
        radius * (3 / 4) ** (1 / 3), rel=0.15)


def test_front_width_of_sharp_step():
    d = np.linspace(-5, 5, 50)
    profile = 1.0 / (1.0 + np.exp(-d / 0.2))
    width = physics.front_width(d, profile)
    assert 0 < width < 1.0


def test_signed_distance_signs():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    distance = physics.signed_distance_2d(mask, (1.0, 1.0))
    assert distance[10, 10] > 0
    assert distance[0, 0] < 0
