"""Shared test fixtures.

``viz_fixture`` builds, once per session, the small end-to-end data set the figure tests draw
from (the same pipeline a user runs): synthetic lightcones written as npz -> ``eval21 build`` ->
two models (``sharp``, ``blurry``) scored on x_HI and T_b with a shared prediction store, and a
generative model (``gen``) scored with ``--draws 4 --mode generative``.

    <root>/cubes.h5
    <root>/results/{sharp,blurry}/{xhi,tb}/   results/predictions/
    <root>/results_gen/gen/xhi/               results_gen/predictions/
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

MODELS = '''
from eval21.adapters.toy import SmoothThresholdModel
from eval21.physics import brightness_temperature_mk


def _make(sigma):
    model = SmoothThresholdModel(sigma)

    def predict(sample):
        xhi = model.predict(sample)["xhi"]
        tb = brightness_temperature_mk(xhi, sample.inputs["density"], sample.meta.z_grid,
                                       omega_m=sample.meta.omega_m or 0.31)
        return {"xhi": xhi, "tb": tb}

    return predict


sharp = _make(1.0)
blurry = _make(2.5)
'''


def build_viz_fixture(root: Path, n=4, shape=(32, 32, 64), box=64.0) -> Path:
    pytest.importorskip("h5py")
    from eval21.build import build
    from eval21.cli import main
    from eval21.data.synthetic import make_cone
    from eval21.physics import brightness_temperature_mk

    sims = root / "sims"
    sims.mkdir(parents=True)
    for i in range(n):
        density, xhi, z = make_cone(shape, box, seed=100 + i, z_min=5.0, z_max=15.0)
        tb = brightness_temperature_mk(xhi, density, z, omega_m=0.31)
        np.savez(sims / f"cone{1000 + i}.npz", density=density, xhi=xhi, tb=tb.astype(np.float32),
                 z_grid=z, box_mpc=box, omega_m=0.31,
                 params=json.dumps({"zeta": 20.0 + i, "T_vir": 4.5 + 0.1 * i}))
    cubes = build(sims, root / "cubes.h5", fmt="npz", splits="test", quiet=True)
    (root / "fixture_models.py").write_text(MODELS)
    common = ["--set", "rays_per_slice=8", "--set", f"gallery_samples={n}", "--set", "omega_m=0.31",
              "--set", "ensemble_voxels=2000", "--quiet"]
    for target in ("xhi", "tb"):
        for model in ("sharp", "blurry"):
            main(["run", "--data", str(cubes), "--target", target,
                  "--model", f"{model}=python:{root / 'fixture_models.py'}:{model}",
                  "--out", str(root / "results" / model / target),
                  "--save-predictions", str(root / "results" / "predictions"), "--save-n", "2", *common])
    main(["run", "--data", str(cubes), "--model", "gen=toygen:1.5", "--draws", "4", "--mode", "generative",
          "--out", str(root / "results_gen" / "gen" / "xhi"),
          "--save-predictions", str(root / "results_gen" / "predictions"), "--save-n", "2", *common])
    return root


@pytest.fixture(scope="session")
def viz_fixture(tmp_path_factory) -> Path:
    pytest.importorskip("matplotlib")
    return build_viz_fixture(tmp_path_factory.mktemp("viz_fixture"))
