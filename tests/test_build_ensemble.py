"""Build command, cube cache reader, prediction store, ensemble / generative mode."""
from __future__ import annotations

import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from eval21.adapters.toy import ToyGenerativeModel
from eval21.build import assign_splits, build, merge
from eval21.build.grid import LosInterpolator, parse_z_grid, resample_transverse
from eval21.data import CubeCacheDataset, open_data
from eval21.metrics.ensemble import fair_crps
from eval21.runner import evaluate, predict_draws
from eval21.settings import Settings
from eval21.spec import Meta, Sample
from eval21.store import PredictionStore


def _write_21cmfast(folder, n=3, nx=16, nlos=40, seed=0):
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    z = np.linspace(5.0, 12.0, nlos)
    for i in range(n):
        with h5py.File(folder / f"21cmfast_11d_sample{i:05d}.h5", "w") as f:
            f.attrs["box_len_mpc"] = 50.0
            lc = f.create_group("lightcone")
            lc["lightcone_redshifts"] = z
            lc["density"] = rng.normal(0, 0.3, (nx, nx, nlos)).astype(np.float32)
            frac = np.clip((z - 6.0) / 5.0, 0, 1)
            lc["neutral_fraction"] = (rng.random((nx, nx, nlos)) < frac).astype(np.float32)
            lc["brightness_temp"] = (20 * lc["neutral_fraction"][()]).astype(np.float32)
            p = f.create_group("params")
            p["names"] = np.array([b"F_ESC10", b"OMm"])
            p["values"] = np.array([-1.0 - 0.1 * i, 0.31])
    return z


def test_grid_parsing_and_interpolation(tmp_path):
    z = parse_z_grid("uniform:5,7,5")
    assert np.allclose(z, [5, 5.5, 6, 6.5, 7])
    np.save(tmp_path / "g.npy", z)
    assert np.allclose(parse_z_grid(str(tmp_path / "g.npy")), z)
    with h5py.File(tmp_path / "c.h5", "w") as f:
        f["target_z"] = z
    assert np.allclose(parse_z_grid(f"{tmp_path / 'c.h5'}:target_z"), z)
    assert np.allclose(parse_z_grid(str(tmp_path / "c.h5")), z)
    zc = parse_z_grid("comoving:6,8", cell_mpc=10.0)
    from eval21.physics import comoving_distance_mpc
    assert np.allclose(np.diff(comoving_distance_mpc(zc)), 10.0, atol=0.05)
    src = np.linspace(5, 7, 21)
    cube = np.broadcast_to(src, (2, 2, 21)).copy()           # field = z, linear -> exact
    interp = LosInterpolator(src[::-1], np.array([5.05, 6.33, 7.5]))   # descending source
    out = interp(cube[..., ::-1])
    assert np.allclose(out[0, 0, :2], [5.05, 6.33], atol=1e-5) and out[0, 0, 2] == 0
    assert interp.valid(None).tolist() == [True, True, False]
    plane = np.broadcast_to(np.arange(8.0)[:, None, None], (8, 8, 1))
    assert resample_transverse(plane, 4).shape == (4, 4, 1)


def test_splits():
    ids = [str(i) for i in range(100)]
    s = assign_splits(ids, "train=0.8,val=0.1,test=0.1", seed=1)
    assert sum(v == "test" for v in s.values()) == 10
    assert s == assign_splits(ids, "train=0.8,val=0.1,test=0.1", seed=1)
    assert set(assign_splits(ids, "test").values()) == {"test"}


def test_build_21cmfast_and_read(tmp_path):
    z = _write_21cmfast(tmp_path / "sims")
    out = build(tmp_path / "sims", tmp_path / "c.h5", fmt="21cmfast", fields=["density", "xhi", "tb"],
                z_grid="uniform:6,11,32", splits="test", quiet=True)
    ds = open_data(str(out), target="xhi")
    assert isinstance(ds, CubeCacheDataset) and len(ds) == 3
    sample = next(iter(ds))
    assert set(sample.inputs) == {"density"} and set(sample.targets) == {"xhi", "tb"}
    assert sample.shape == (16, 16, 32) and sample.meta.box_mpc == 50.0
    assert sample.meta.omega_m == pytest.approx(0.31) and "F_ESC10" in sample.meta.params
    # the interpolated planes match a direct interpolation of the raw field
    with h5py.File(tmp_path / "sims" / "21cmfast_11d_sample00000.h5") as f:
        raw = f["lightcone/density"][()]
    zt = np.linspace(6, 11, 32)
    ref = np.stack([np.interp(zt, z, raw[i, j]) for i in range(16) for j in range(16)]).reshape(16, 16, 32)
    assert np.allclose(sample.inputs["density"], ref, atol=1e-5)
    info = ds.describe()
    assert info["splits"] == {"test": 3} and info["targets"] == ["tb", "xhi"]


def test_build_shards_merge_and_z_range(tmp_path):
    _write_21cmfast(tmp_path / "sims", n=5)
    for k in range(2):
        build(tmp_path / "sims", tmp_path / "c.h5", fmt="21cmfast", shard=k, num_shards=2,
              z_grid="uniform:4,11,30", quiet=True)
    merged = merge(tmp_path / "c.h5", 2, quiet=True)
    ds = CubeCacheDataset(merged)
    ids = [r["id"] for r in ds._index()["rows"]]
    assert ids == ["00000", "00001", "00002", "00003", "00004"]
    # planes below z = 5 lie outside the source: invalid -> the full LOS is refused,
    # a window inside the source works
    with pytest.raises(ValueError, match="--z-range"):
        len(CubeCacheDataset(merged))
    window = CubeCacheDataset(merged, z_range=(5.5, 10.5))
    sample = next(iter(window))
    assert sample.meta.z_grid[0] >= 5.5 and sample.meta.z_grid[-1] <= 10.5 and len(window) == 5


def test_prediction_store_roundtrip(tmp_path):
    from eval21.data import SyntheticDataset
    from eval21.adapters.toy import SmoothThresholdModel

    ds = SyntheticDataset(n_samples=3, shape=(16, 16, 16))
    store = PredictionStore(tmp_path / "p", n=2)
    evaluate(SmoothThresholdModel(), ds, Settings(rays_per_slice=8), ["voxel"], store=store, model_name="toy")
    back = PredictionStore.open(tmp_path / "p")
    assert len(back.samples) == 2 and back.models == ["toy"]
    t = back.truth(back.samples[0])
    p = back.prediction("toy", back.samples[0])
    assert t["targets"]["xhi"].shape == p["xhi"].shape == (16, 16, 16)


def test_fair_crps_matches_definition():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(7, 50))
    y = rng.normal(size=50)
    brute = np.abs(x - y).mean(0) - 0.5 * np.abs(x[:, None] - x[None]).sum((0, 1)) / (7 * 6)
    assert np.allclose(fair_crps(x, y), brute)


def _exchangeable_dataset(n=10, shape=(32, 32, 24)):
    """Truths drawn from the same generator as the model's draws (calibrated by construction)."""
    model = ToyGenerativeModel(1.5)
    z = np.linspace(6, 12, shape[2])
    samples = []
    for i in range(n):
        base = Sample(id=f"s{i}", inputs={}, targets={"xhi": np.zeros(shape)},
                      meta=Meta(box_mpc=64.0, z_grid=z))
        truth = model.predict(base, draw=10_000 + i)["xhi"]
        samples.append(Sample(id=f"s{i}", inputs={}, targets={"xhi": truth}, meta=base.meta, split="test"))

    class Data:
        def __iter__(self):
            return iter(samples)

        def __len__(self):
            return len(samples)

        def describe(self):
            return {}
    return Data(), model


def test_generative_mode_and_calibrated_ensemble():
    data, model = _exchangeable_dataset()
    settings = Settings(draws=8, mode="generative", rays_per_slice=8, ensemble_voxels=5000)
    result = evaluate(model, data, settings, ["ensemble", "voxel", "spectrum", "history"])
    assert "voxel" in result.skipped and "generative" in result.skipped["voxel"]
    ens = result.metrics["ensemble"]
    assert ens["n_draws"] == 8 and "voxel" not in ens            # pointwise part dropped
    for family in ("xbar", "power"):
        stats = ens["statistics"][family]
        assert abs(stats["coverage95"] - stats["coverage95_nominal"]) < 0.2, (family, stats)
        assert stats["pit_ks_med"] < 0.45, (family, stats)
    assert "coherence_med" not in result.metrics["spectrum"]["stages"][0]
    assert result.info["draws"] == 8 and result.info["draws_identical"] is False
    # pointwise mode keeps the voxel calibration; exchangeable draws give spread/skill ~ 1
    pw = evaluate(model, data, Settings(draws=8, rays_per_slice=8, ensemble_voxels=5000), ["ensemble"])
    vox = pw.metrics["ensemble"]["voxel"]
    assert 0.8 < vox["spread_skill_med"] < 1.25
    assert abs(vox["coverage90"] - vox["coverage90_nominal"]) < 0.05, vox
    assert vox["rank_chi2"] < 50


def test_predict_draws_sources():
    data, model = _exchangeable_dataset(n=1, shape=(8, 8, 8))
    sample = next(iter(data))
    pred, draws = predict_draws(model, sample, Settings(draws=3))
    assert draws.shape == (3, 8, 8, 8) and np.array_equal(pred["xhi"], draws[0])
    pred, draws = predict_draws(model, sample, Settings(draws=3, score_draw="mean"))
    assert np.allclose(pred["xhi"], draws.mean(0))

    class Stacked:
        def predict(self, s):
            return {"xhi": np.stack([np.full(s.shape, v) for v in (0.0, 1.0)])}
    pred, draws = predict_draws(Stacked(), sample, Settings())
    assert draws.shape[0] == 2 and pred["xhi"].mean() == 0.0
    pred, draws = predict_draws(type("D", (), {"predict": lambda self, s: {"xhi": np.zeros(s.shape)}})(),
                                sample, Settings())
    assert draws is None


def test_target_specific_metrics_skip_tb(tmp_path):
    _write_21cmfast(tmp_path / "sims", n=2)
    out = build(tmp_path / "sims", tmp_path / "c.h5", fields=["density", "xhi", "tb"],
                z_grid="uniform:6,11,16", quiet=True)
    ds = open_data(str(out), target="tb")

    class Copy:
        def predict(self, s):
            return {"tb": s.targets["tb"] * 0.9}
    result = evaluate(Copy(), ds, Settings(target="tb", rays_per_slice=8), ["history", "bubble", "voxel"])
    assert "history" in result.skipped and "bubble" in result.skipped and "voxel" in result.metrics


def test_cli_target_reaches_the_metrics(tmp_path):
    from eval21.cli import main

    _write_21cmfast(tmp_path / "sims", n=2)
    build(tmp_path / "sims", tmp_path / "c.h5", fields=["density", "xhi", "tb"],
          z_grid="uniform:6,11,16", quiet=True)
    model = tmp_path / "m.py"
    model.write_text("def predict(s):\n    return {'tb': s.targets['tb'] * 0.5, 'xhi': s.targets['xhi']}\n")
    main(["run", "--data", str(tmp_path / "c.h5"), "--target", "tb", "--model", f"m=python:{model}",
          "--metrics", "voxel", "--out", str(tmp_path / "r"), "--legacy-dir", "--quiet"])
    r = json.loads((tmp_path / "r" / "results.json").read_text())
    assert r["settings"]["target"] == "tb"
    voxel = r["models"]["m"]["metrics"]["voxel"]
    assert voxel["truth_mean"] > 1.0 and voxel["rmse"] > 0     # scored T_b [mK], not x_HI
