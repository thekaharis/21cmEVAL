"""Agreement with the fno-21cm reference implementations.

These live in a sibling checkout that is not a dependency, so the module skips
unless ``FNO21CM_ROOT`` points at it::

    FNO21CM_ROOT=/work/fno-21cm python -m pytest tests/test_reference_agreement.py

They pin the physics primitives to the numbers the published evaluation used:
same tau_e, same crossing redshifts, same binned power, same WLS fit, and the
same mean-free-path ray tracing bit for bit (same RNG stream and seeds).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from cm21eval import physics

ROOT = os.environ.get("FNO21CM_ROOT")

pytestmark = pytest.mark.skipif(
    not ROOT or not os.path.isdir(ROOT),
    reason="set FNO21CM_ROOT to the fno-21cm checkout",
)


@pytest.fixture(scope="module", autouse=True)
def reference():
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from util import metrics_21cm, physical_stats
    from viz.power_spectrum_evaluation import KBinner

    return {"stats": physical_stats, "metrics": metrics_21cm, "binner": KBinner}


def test_optical_depth(reference):
    rng = np.random.default_rng(0)
    z = np.linspace(5, 25, 64)
    xbar = np.clip((z - 6) / 12, 0, 1) + 0.02 * rng.normal(size=z.size)
    assert physics.thomson_optical_depth(z, xbar) == pytest.approx(
        reference["stats"].thomson_optical_depth(z, xbar), rel=1e-12)


def test_history_features(reference):
    z = np.linspace(5, 25, 64)
    xbar = np.clip((z - 6) / 12, 0, 1)
    ours = physics.history_features(z, xbar)
    theirs = reference["stats"].history_features(z, xbar)
    for key in ours:
        assert ours[key] == pytest.approx(theirs[key], rel=1e-12)


def test_brightness_temperature(reference):
    rng = np.random.default_rng(1)
    shape = (4, 4, 32)
    z = np.linspace(5, 25, shape[2])
    xhi = rng.random(shape)
    delta = rng.normal(size=shape)
    ours = physics.brightness_temperature_mk(xhi, delta, z)
    theirs = reference["stats"].brightness_temperature_mk(xhi, delta, z)
    assert np.allclose(ours, theirs, rtol=1e-12, atol=0)


def test_binned_power(reference):
    rng = np.random.default_rng(2)
    field = rng.normal(size=(32, 32, 7))
    ours = physics.KBinner(32, 32, 200.0, 12)
    theirs = reference["binner"](32, 32, 200.0, 12)
    assert np.allclose(ours.slice_sums(field),
                       theirs.binned_slice_sums(field), rtol=1e-12, atol=0)


def test_polynomial_fit(reference):
    rng = np.random.default_rng(3)
    k = np.geomspace(0.02, 1.0, 20)
    y = rng.normal(size=k.size)
    variance = np.abs(rng.normal(size=k.size)) + 0.1
    ours = physics.fit_even_polynomial(k, y, variance, 0.3, 2)
    theirs = reference["stats"].fit_even_polynomial(k, y, variance, 0.3, 2)
    assert np.allclose(ours["coef"], theirs["coef"], rtol=1e-12)


def test_mean_free_path_ray_tracing(reference):
    rng = np.random.default_rng(4)
    mask = rng.random((64, 64)) > 0.5
    ours = physics.mean_free_path_samples_2d(
        mask, 128, (1.5, 1.5), 200.0, seed=7)
    theirs = reference["metrics"].mean_free_path_samples_2d(
        mask, 128, (1.5, 1.5), 200.0, seed=7)
    assert np.array_equal(ours["distances_mpc"], theirs["distances_mpc"])
    assert ours["n_censored"] == theirs["n_censored"]
