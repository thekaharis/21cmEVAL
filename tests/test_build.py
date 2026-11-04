"""eval21 build: tiny fake raw files in the real 21cmFAST / LoReLi layouts, generic npz/h5."""
import json

import numpy as np
import pytest

from eval21.build import BuildError, BuildOptions, build, make_source
from eval21.build.loreli import grid_params
from eval21.cli import main
from eval21.data import NpzDataset
from eval21.physics import comoving_distance_mpc

h5py = pytest.importorskip("h5py")

NAMES = ["F_ESC10", "F_STAR10", "ALPHA_ESC", "ALPHA_STAR", "L_X", "NU_X_THRESH", "M_TURN",
         "t_STAR", "X_RAY_SPEC_INDEX", "OMm", "SIGMA_8"]
BASE = [-1.0, -1.3, -0.5, 0.5, 40.5, 500.0, 8.7, 0.5, 1.0, 0.31, 0.81]
FIELDS = ["brightness_temp", "neutral_fraction", "density", "los_velocity", "tau_21"]


# --------------------------------------------------------------------------- fake writers
_TABLES = {}


def _table(omm, h=0.6766):
    if (omm, h) not in _TABLES:
        z = np.linspace(0, 12, 120001)
        _TABLES[omm, h] = (z, comoving_distance_mpc(z, omm, h))
    return _TABLES[omm, h]


def _z_of_distance(d, omm, h=0.6766):
    z, dist = _table(omm, h)
    return np.interp(d, dist, z)


def fake_21cmfast(root, sample, omm, n=8, box=16.0, nlos=60, z_min=5.0, seed=None, spike=False):
    """One raw_lightcone_v2.0 file: uniform comoving LOS at the cell, delta density."""
    rng = np.random.default_rng(sample if seed is None else seed)
    cell = box / n
    zz, dist = _table(omm)
    d0 = float(np.interp(z_min, zz, dist))
    d = d0 + cell * np.arange(nlos)
    z = _z_of_distance(d, omm)
    shape = (n, n, nlos)
    delta = (0.3 * rng.normal(size=shape)).astype(np.float32)
    delta -= delta.mean()
    xhi = np.clip(rng.random(shape) * np.linspace(0.2, 1.4, nlos), 0, 1).astype(np.float32)
    tb = (27 * xhi * (1 + delta)).astype(np.float32)
    if spike:
        tb[0, 0, 3] = 2.0e5
    fields = {"brightness_temp": tb, "neutral_fraction": xhi, "density": delta,
              "los_velocity": (1e-13 * rng.normal(size=shape)).astype(np.float32),
              "tau_21": (1e-4 * xhi).astype(np.float32)}
    values = list(BASE)
    values[NAMES.index("OMm")] = omm
    values[0] = -1.0 - 0.01 * sample
    path = root / f"21cmfast_11d_sample{sample:06d}.h5"
    with h5py.File(path, "w") as f:
        lc = f.create_group("lightcone")
        for name, array in fields.items():
            lc.create_dataset(name, data=array)
            lc.create_dataset(f"{name}_with_rsds", data=array + np.float32(0.05))
        lc.create_dataset("lightcone_redshifts", data=z)
        lc.create_dataset("lightcone_distances", data=d)
        lc.create_dataset("node_redshifts", data=np.linspace(35, 4, 11))
        p = f.create_group("params")
        p.create_dataset("names", data=np.array(NAMES, dtype=object), dtype=h5py.string_dtype())
        p.create_dataset("values", data=np.array(values))
        p.create_dataset("fiducials", data=np.array(BASE))
        p.create_dataset("ranges", data=np.zeros((11, 2)))
        f.attrs.update({"box_len_mpc": box, "hii_dim": n, "lightcone_z_min": z_min,
                        "lightcone_z_max": float(z[-1]), "apply_rsds": True, "sample_id": sample,
                        "schema_version": "raw_lightcone_v2.0", "random_seed": 1000 + sample})
    return path, fields, z, d, values


def fake_design(root, rows):
    (root / "21cmfast_11d_design.json").write_text(json.dumps({"param_names": NAMES, "design": rows}))


def fake_loreli(root, sim, n=8, nlos=200, empty=0, kind="five", corrupt=None, seed=None):
    """LoReLi .dat: (n, n, nlos, 5) interleaved xa,tpHI,xion,dens,dtb (or dtb only); low z first."""
    rng = np.random.default_rng(sim if seed is None else seed)
    shape = (n, n, nlos)
    dens = (2.3e-9 * np.exp(0.1 * rng.normal(size=shape))).astype(np.float32)
    xion = np.clip(rng.random(shape) * np.linspace(1.5, -0.2, nlos), 1.3e-4, 1).astype(np.float32)
    dtb = (27 * (1 - xion) * dens / 2.3e-9 - 10).astype(np.float32)
    xa = rng.random(shape).astype(np.float32)
    tp = (100 * rng.random(shape)).astype(np.float32)
    if empty:
        for a in (dens, xion, xa, tp):
            a[:, :, :empty] = 0
        dtb[:, :, :empty] = np.nan
    if corrupt is not None:
        dtb[0, 0, corrupt] = 1e12
    cube = np.stack([xa, tp, xion, dens, dtb], -1)
    five = root / f"lightcone_xa_tpHI_xion_dens_dtb_no_rsd_simu{sim}.dat"
    cube.astype("<f4").tofile(five)
    dtb.astype("<f4").tofile(root / f"lightcone_dtb_no_rsd_simu{sim}.dat")
    return {"xa": xa, "tpHI": tp, "xion": xion, "dens": dens, "dtb": dtb}


def run_build(fmt, src, out, **kw):
    so = kw.pop("source_options", {})
    options = BuildOptions(out=out, quiet=True, source_options=so, **kw)
    return build(make_source(fmt, src, options), options)


def load(path):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


LORELI_SMALL = {"n_perp": 8, "n_los": 200}


# --------------------------------------------------------------------------- 21cmFAST
@pytest.fixture
def cmfast(tmp_path):
    src = tmp_path / "raw"
    src.mkdir()
    cones = {k: fake_21cmfast(src, k, omm) for k, omm in ((0, 0.30), (1, 0.32), (2, 0.28))}
    fake_design(src, [cones[k][4] for k in sorted(cones)])
    return src, cones


def test_21cmfast_native_mapping_and_density(cmfast, tmp_path):
    src, cones = cmfast
    out = tmp_path / "native"
    summary = run_build("21cmfast", src, out)
    assert summary["counts"] == {"written": 3, "existing": 0, "failed": 0}
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["target"] == "xhi"
    assert manifest["inputs"] == ["density", "los_velocity"]
    assert "z_grid" not in manifest           # native grids differ between cones (OMm)
    tb_manifest = json.loads((out / "manifest_tb.json").read_text())
    assert tb_manifest["target"] == "tb" and tb_manifest["inputs"] == ["density", "los_velocity", "xhi"]
    entry = manifest["samples"][1]
    assert entry["id"] == "cone1" and entry["cone_id"] == 1 and entry["omega_m"] == 0.32
    assert entry["params"]["F_ESC10"] == pytest.approx(-1.01)
    assert entry["source"]["files"][0]["size"] == (src / "21cmfast_11d_sample000001.h5").stat().st_size
    assert entry["source"]["design_max_abs_diff"] == 0.0
    prov = manifest["provenance"]
    assert prov["format"] == "21cmfast" and prov["source_folder"] == str(src.resolve())
    assert prov["builder"]["name"] == "eval21.build" and "options" in prov
    _, fields, z, _, _ = cones[1]
    data = load(out / "cone1.npz")
    n = data["xhi"].shape[-1]
    assert np.array_equal(data["xhi"], fields["neutral_fraction"][..., :n])
    assert np.array_equal(data["tb"], fields["brightness_temp"][..., :n])
    assert np.array_equal(data["density"], fields["density"][..., :n] + np.float32(1))
    assert abs(data["density"].mean() - 1) < 1e-2
    # the window is the shared one: native planes of cone 1 up to the lowest z_max of the three
    z_hi = min(c[2][-1] for c in cones.values())
    assert np.array_equal(data["z_grid"], z[z <= z_hi + 1e-9])
    assert float(data["omega_m"]) == 0.32
    assert json.loads(str(data["params"]))["OMm"] == 0.32


def test_21cmfast_rsd_and_field_selection(cmfast, tmp_path):
    src, cones = cmfast
    out = tmp_path / "rsd"
    run_build("21cmfast", src, out, fields=["xhi", "density"], ids="0", targets=["xhi"],
              source_options={"rsd": True})
    data = load(out / "cone0.npz")
    assert set(k for k in data if data[k].ndim == 3) == {"xhi", "density"}
    assert np.array_equal(data["xhi"], cones[0][1]["neutral_fraction"] + np.float32(0.05))
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["fields"]["xhi"]["source"] == "lightcone/neutral_fraction_with_rsds"
    assert not (out / "manifest_tb.json").exists()


def test_21cmfast_density_convention_check(cmfast, tmp_path):
    src, _ = cmfast
    with pytest.raises(BuildError, match="density"):
        run_build("21cmfast", src, tmp_path / "bad", ids="0",
                  source_options={"density_convention": "one_plus_delta"})


def test_21cmfast_los_modes(cmfast, tmp_path):
    src, cones = cmfast
    grid = "5.02,5.18,9"
    for mode in ("nearest", "linear"):
        run_build("21cmfast", src, tmp_path / mode, los=mode, z_grid=grid)
    zt = np.linspace(5.02, 5.18, 9)
    for k, (_, fields, z, d, _) in cones.items():
        near = load(tmp_path / "nearest" / f"cone{k}.npz")
        pick = np.abs(z[None, :] - zt[:, None]).argmin(1)
        assert np.array_equal(near["xhi"], fields["neutral_fraction"][..., pick])
        assert np.array_equal(near["z_grid"], zt) and np.array_equal(near["z_native"], z[pick])
        lin = load(tmp_path / "linear" / f"cone{k}.npz")
        dt = np.interp(zt, z, d)
        pos = np.interp(dt, d, np.arange(len(d)))
        i0 = np.floor(pos).astype(int)
        t = pos - i0
        a, b = fields["neutral_fraction"][..., i0], fields["neutral_fraction"][..., i0 + 1]
        assert np.allclose(lin["xhi"], a * (1 - t) + b * t, atol=1e-6)
        assert np.allclose(lin["density"], 1 + fields["density"][..., i0] * (1 - t)
                           + fields["density"][..., i0 + 1] * t, atol=1e-5)
    for mode in ("nearest", "linear"):
        manifest = json.loads((tmp_path / mode / "manifest.json").read_text())
        assert np.allclose(manifest["z_grid"], zt)      # shared grid -> one geometry
        assert manifest["geometry"]["uniform_geometry"]
    # comoving at the transverse cell == native for 21cmFAST (uniform LOS at the cell)
    run_build("21cmfast", src, tmp_path / "comoving", los="comoving")
    for k, (_, fields, z, d, _) in cones.items():
        a = load(tmp_path / "comoving" / f"cone{k}.npz")
        b = load(tmp_path / "native" / f"cone{k}.npz") if (tmp_path / "native").exists() else None
        n = a["xhi"].shape[-1]
        assert np.array_equal(a["xhi"], fields["neutral_fraction"][..., :n])
        assert np.allclose(a["z_grid"], z[:n], atol=1e-6)
        assert b is None


def test_nearest_rejects_repeated_planes(cmfast, tmp_path):
    src, _ = cmfast
    with pytest.raises(BuildError, match="repeat"):
        run_build("21cmfast", src, tmp_path / "x", los="nearest", z_grid="5.02,5.18,200")


def test_float16_with_safe_fallback(tmp_path):
    src = tmp_path / "raw"
    src.mkdir()
    _, fields, *_ = fake_21cmfast(src, 0, 0.31, spike=True)
    out = tmp_path / "f16"
    run_build("21cmfast", src, out, dtype="float16")
    data = load(out / "cone0.npz")
    assert data["xhi"].dtype == np.float16 and data["density"].dtype == np.float16
    assert np.abs(data["xhi"].astype(np.float32) - fields["neutral_fraction"]).max() < 1e-3
    assert data["tb"].dtype == np.float32           # 2e5 mK spike overflows float16
    assert data["los_velocity"].dtype == np.float32  # 1e-13 underflows float16
    entry = json.loads((out / "manifest.json").read_text())["samples"][0]
    assert "overflow" in entry["dtype_notes"]["tb"] and "underflow" in entry["dtype_notes"]["los_velocity"]


def test_crop_and_parallel_equal_serial(cmfast, tmp_path):
    src, cones = cmfast
    run_build("21cmfast", src, tmp_path / "serial", crop=4, los="nearest", z_grid="5.02,5.18,9")
    run_build("21cmfast", src, tmp_path / "parallel", crop=4, los="nearest", z_grid="5.02,5.18,9",
              workers=3)
    for k in cones:
        a, b = load(tmp_path / "serial" / f"cone{k}.npz"), load(tmp_path / "parallel" / f"cone{k}.npz")
        assert a["xhi"].shape == (4, 4, 9)
        assert all(np.array_equal(a[f], b[f]) for f in ("xhi", "tb", "density"))
    _, fields, z, _, _ = cones[0]
    pick = np.abs(z[None, :] - np.linspace(5.02, 5.18, 9)[:, None]).argmin(1)
    assert np.array_equal(load(tmp_path / "serial" / "cone0.npz")["xhi"],
                          fields["neutral_fraction"][2:6, 2:6][..., pick])
    manifest = json.loads((tmp_path / "serial" / "manifest.json").read_text())
    assert manifest["box_mpc"] == pytest.approx(8.0)
    assert manifest["geometry"]["crop"] == {"cells": 4, "x": [2, 6], "y": [2, 6]}


def test_splits(cmfast, tmp_path):
    src, _ = cmfast
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps({"train": [0], "val": ["cone1"], "test": [2]}))
    run_build("21cmfast", src, tmp_path / "file", split_file=split_file)
    m = json.loads((tmp_path / "file" / "manifest.json").read_text())
    assert [s["split"] for s in m["samples"]] == ["train", "val", "test"]
    meta = tmp_path / "run_metadata.json"
    meta.write_text(json.dumps({"split": {"train_cone_ids": [0, 1], "val_cone_ids": [], "test_cone_ids": [2]}}))
    run_build("21cmfast", src, tmp_path / "meta", split_from=meta, only_split="test")
    m = json.loads((tmp_path / "meta" / "manifest.json").read_text())
    assert [(s["id"], s["split"]) for s in m["samples"]] == [("cone2", "test")]
    a = run_build("21cmfast", src, tmp_path / "frac", split_fractions=(0.5, 0.25, 0.25), seed=3, dry_run=True)
    b = run_build("21cmfast", src, tmp_path / "frac", split_fractions=(0.5, 0.25, 0.25), seed=3,
                  dry_run=True, limit=2)
    assert sum(a["splits"].values()) == 3 and sum(b["splits"].values()) == 2
    run_build("21cmfast", src, tmp_path / "frac", split_fractions=(0.5, 0.25, 0.25), seed=3)
    full = {s["cone_id"]: s["split"] for s in json.loads((tmp_path / "frac" / "manifest.json").read_text())["samples"]}
    run_build("21cmfast", src, tmp_path / "frac2", split_fractions=(0.5, 0.25, 0.25), seed=3, ids="1-2")
    part = {s["cone_id"]: s["split"] for s in json.loads((tmp_path / "frac2" / "manifest.json").read_text())["samples"]}
    assert all(full[k] == v for k, v in part.items())   # per-id hash: independent of the selection


def test_resume_is_idempotent(cmfast, tmp_path):
    src, _ = cmfast
    out = tmp_path / "resume"
    first = run_build("21cmfast", src, out)
    assert first["counts"]["written"] == 3
    stamp = (out / "cone0.npz").stat().st_mtime_ns
    (out / "cone1.npz").write_bytes(b"truncated")       # a partial file is rebuilt
    (out / "cone2.npz").unlink()                        # a missing one is written
    second = run_build("21cmfast", src, out)
    assert second["counts"] == {"written": 2, "existing": 1, "failed": 0}
    assert (out / "cone0.npz").stat().st_mtime_ns == stamp
    assert np.load(out / "cone1.npz")["xhi"].shape[0] == 8
    third = run_build("21cmfast", src, out, dtype="float16")   # different content -> rewritten
    assert third["counts"]["written"] == 3
    assert not list(out.glob(".*.tmp*"))


def test_dry_run_writes_no_cubes(cmfast, tmp_path, capsys):
    src, _ = cmfast
    out = tmp_path / "dry"
    assert main(["build", "21cmfast", str(src), "--out", str(out), "--dry-run", "--los", "nearest",
                 "--z-grid", "5.02,5.18,9"]) == 0
    text = capsys.readouterr().out
    assert "group    all: 3 cones, shape (8, 8, 9)" in text and "GB uncompressed" in text
    assert not list(out.glob("*.npz")) if out.exists() else True


def test_eval21_describe_and_run_on_output(cmfast, tmp_path, capsys):
    src, _ = cmfast
    out = tmp_path / "data"
    meta = tmp_path / "run_metadata.json"
    meta.write_text(json.dumps({"split": {"test_cone_ids": [2, 0]}}))
    assert main(["build", "21cmfast", str(src), "--out", str(out), "--los", "linear",
                 "--z-grid", "5.02,5.18,12", "--dtype", "float16", "--quiet"]) == 0
    assert main(["describe", "--data", str(out)]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["n_total"] == 3 and info["shape"] == [8, 8, 12] and info["inputs"] == ["density", "los_velocity"]
    model = tmp_path / "toy.py"
    model.write_text("import numpy as np\n"
                     "def predict(sample):\n"
                     "    x = np.asarray(sample.inputs['density'], dtype=np.float64)\n"
                     "    return {'xhi': np.clip(1.5 - x, 0, 1)}\n")
    res = tmp_path / "res"
    assert main(["run", "--data", str(out / "manifest.json"), "--model", f"toy=python:{model}",
                 "--metrics", "voxel,history,spectrum", "--indices-json", str(meta), "--out", str(res),
                 "--quiet"]) == 0
    payload = json.loads((res / "results.json").read_text())
    assert payload["models"]["toy"]["n_samples"] == 2
    ds = NpzDataset(out / "manifest.json")
    assert ds.row_indices_for_ids([2, 0]) == [2, 0]


# --------------------------------------------------------------------------- LoReLi
@pytest.fixture
def loreli(tmp_path):
    src = tmp_path / "lc"
    src.mkdir()
    cones = {10001: fake_loreli(src, 10001, empty=0),
             10002: fake_loreli(src, 10002, empty=3),
             12000: fake_loreli(src, 12000, empty=60, corrupt=150)}
    return src, cones


def _valid(nlos, empty, corrupt=None):
    """Expected valid run of a fake five-field cone: drops the empty tail and a corrupt plane."""
    good = np.ones(nlos, bool)
    good[:empty] = False
    if corrupt is not None:
        good[corrupt] = False
        return (empty, corrupt - 1) if corrupt - empty > nlos - corrupt - 1 else (corrupt + 1, nlos - 1)
    return (empty, nlos - 1)


def test_loreli_five_field_auto_groups(loreli, tmp_path):
    src, cones = loreli
    out = tmp_path / "groups"
    run_build("loreli", src, out, group_tol=0.2, source_options=LORELI_SMALL)
    index = json.loads((out / "build.json").read_text())
    groups = index["groups"]
    assert len(groups) == 2                    # (10001, 10002) start together; 12000 much later
    (g1, m1), (g2, m2) = sorted(groups.items(), key=lambda kv: kv[1]["z"][0])
    assert m1["cones"] == [10001, 10002] and m2["cones"] == [12000]
    z = np.linspace(5, 15, 200)
    manifest = json.loads((out / g1 / "manifest.json").read_text())
    assert manifest["z_grid"][0] == pytest.approx(z[3])    # window starts at 10002's first valid plane
    assert manifest["box_mpc"] == pytest.approx(294.0) and manifest["omega_m"] == 0.308
    assert manifest["native"]["cyl_native_los_dz"] == pytest.approx(10 / 199)
    entry = manifest["samples"][0]
    assert entry["params"] == grid_params(10001)
    assert entry["params"] == {"f_x": 0.1, "tau": 8191.9, "rHS": 0.2, "logMmin": 8.0, "fesc": 0.275}
    assert "inferred" in manifest["provenance"]["params_inferred"]["status"]
    data = load(out / g1 / "cone10001.npz")
    raw = cones[10001]
    assert np.array_equal(data["xhi"], np.clip(1 - raw["xion"][..., 3:], 0, 1))
    assert np.array_equal(data["tb"], raw["dtb"][..., 3:])
    plane_mean = data["density"].mean((0, 1))
    assert np.allclose(plane_mean, 1, atol=1e-5)
    assert np.allclose(data["density"], raw["dens"][..., 3:] / raw["dens"][..., 3:].mean((0, 1)), rtol=1e-5)
    tb = json.loads((out / g1 / "manifest_tb.json").read_text())
    assert tb["target"] == "tb" and tb["inputs"] == ["density", "xhi"]
    # the corrupt plane splits cone 12000's valid run: the longer part is kept
    lo, hi = _valid(200, 60, 150)
    late = load(out / g2 / "cone12000.npz")
    assert np.allclose(late["z_grid"], z[lo:hi + 1])


def test_loreli_shared_window_and_groups_file(loreli, tmp_path):
    src, _ = loreli
    run_build("loreli", src, tmp_path / "shared", group="none", ids="10001,10002",
              source_options=LORELI_SMALL)
    m = json.loads((tmp_path / "shared" / "manifest.json").read_text())
    assert [s["cone_id"] for s in m["samples"]] == [10001, 10002] and len(m["z_grid"]) == 197
    groups = tmp_path / "groups.json"
    groups.write_text(json.dumps({"A": {"cones": [10001], "z": [6.0, 9.0]}, "B": {"cones": [10002]}}))
    run_build("loreli", src, tmp_path / "file", group=str(groups), source_options=LORELI_SMALL)
    a = json.loads((tmp_path / "file" / "A" / "manifest.json").read_text())
    assert 6.0 <= a["z_grid"][0] and a["z_grid"][-1] <= 9.0
    index = json.loads((tmp_path / "file" / "build.json").read_text())
    assert index["skipped"] == {"12000": "not listed in groups.json"}


def test_loreli_comoving_uniform_in_eval21_distance(loreli, tmp_path):
    src, cones = loreli
    out = tmp_path / "com"
    run_build("loreli", src, out, los="comoving", group="none", ids="10001",
              source_options=LORELI_SMALL)
    m = json.loads((out / "manifest.json").read_text())
    z = np.asarray(m["z_grid"])
    d = comoving_distance_mpc(z, 0.308, 0.678)
    assert np.allclose(np.diff(d), 294 / 8, rtol=1e-5)
    assert m["eval21_settings"]["hubble_h"] == 0.678
    assert m["eval21_settings"]["cyl_native_cell_mpc"] == pytest.approx(294 / 8)
    assert m["eval21_settings"]["cyl_native_los_dz"] == pytest.approx(10 / 199)
    # first plane is native plane 0 (z = 5): copied, not interpolated
    data = load(out / "cone10001.npz")
    assert np.array_equal(data["xhi"][..., 0], np.clip(1 - cones[10001]["xion"][..., 0], 0, 1))


def test_loreli_dtb_only(loreli, tmp_path):
    src, cones = loreli
    out = tmp_path / "dtb"
    run_build("loreli", src, out, group="none", ids="10001-10002",
              source_options={**LORELI_SMALL, "loreli_kind": "dtb"})
    m = json.loads((out / "manifest.json").read_text())
    assert m["target"] == "tb" and m["inputs"] == []
    data = load(out / "cone10002.npz")
    # loreli_globals rule: finite run minus one edge plane at each end
    assert np.array_equal(data["tb"], cones[10002]["dtb"][..., 4:199])


def test_loreli_params_file_crosscheck(loreli, tmp_path):
    src, _ = loreli
    table = {"sims": {str(s): {**grid_params(s), "status": "published"} for s in (10001, 10002, 12000)}}
    table["sims"]["10002"]["f_x"] = 99.0
    path = tmp_path / "loreli2_parameters.json"
    path.write_text(json.dumps(table))
    out = tmp_path / "p"
    run_build("loreli", src, out, group="none", ids="10001-10002",
              source_options={**LORELI_SMALL, "loreli_params": str(path)})
    s = json.loads((out / "manifest.json").read_text())["samples"]
    assert s[0]["source"]["param_status"] == "published"
    assert s[1]["params"]["f_x"] == 99.0 and "param_rule_mismatch" in s[1]["source"]


def test_loreli_eval21_run(loreli, tmp_path):
    out = tmp_path / "run"
    src, _ = loreli
    run_build("loreli", src, out, group="none", ids="10001,10002", los="comoving",
              split="test", dtype="float16", source_options=LORELI_SMALL)
    res = tmp_path / "res"
    assert main(["run", "--data", str(out), "--model", "toy=synthetic:1.5", "--metrics",
                 "voxel,history,cylindrical", "--split", "test", "--out", str(res), "--quiet",
                 "--set", "hubble_h=0.678", "--set", "cyl_chunk_cells=8"]) == 0
    assert json.loads((res / "results.json").read_text())["models"]["toy"]["n_samples"] == 2


# --------------------------------------------------------------------------- generic
def test_generic_npz_and_h5(tmp_path):
    src = tmp_path / "gen"
    src.mkdir()
    rng = np.random.default_rng(0)
    z = np.linspace(6, 9, 20)
    cubes = {}
    for k in (3, 4):
        x = rng.random((6, 20, 6)).astype(np.float32)          # LOS on axis 1
        rho = (5 * np.exp(0.2 * rng.normal(size=(6, 20, 6)))).astype(np.float32)
        cubes[k] = (x, rho)
        np.savez(src / f"run_{k}.npz", nf=x, rho=rho, redshift=z,
                 theta=np.array(json.dumps({"zeta": 30.0 + k, "Om": 0.3})))
    out = tmp_path / "out"
    run_build("generic", src, out, fields=["xhi=nf", "density=rho"], los="nearest", z_grid="6.5,8.5,5",
              min_planes=4,
              source_options={"z_key": "redshift", "params_key": "theta", "box_mpc": "60",
                              "omega_m_param": "Om", "los_axis": 1, "density_convention": "rho"})
    m = json.loads((out / "manifest.json").read_text())
    assert m["target"] == "xhi" and m["inputs"] == ["density"] and m["box_mpc"] == 60.0
    assert m["samples"][0]["params"] == {"zeta": 33.0, "Om": 0.3} and m["samples"][0]["omega_m"] == 0.3
    pick = np.abs(z[None, :] - np.linspace(6.5, 8.5, 5)[:, None]).argmin(1)
    data = load(out / "cone3.npz")
    assert np.array_equal(data["xhi"], np.moveaxis(cubes[3][0], 1, 2)[..., pick])
    # h5 with a params group and a box attribute
    src2 = tmp_path / "gen_h5"
    src2.mkdir()
    with h5py.File(src2 / "cone_7.h5", "w") as f:
        f["fields/xhi"] = cubes[3][0].transpose(0, 2, 1)
        f["fields/delta"] = np.log(cubes[3][1].transpose(0, 2, 1)) * 0 + 0.1 * rng.normal(size=(6, 6, 20)).astype(np.float32)
        f["z"] = z
        f["p/names"] = np.array(["a", "b"], dtype=object).astype(h5py.string_dtype())
        f["p/values"] = np.array([1.0, 2.0])
        f.attrs["L"] = 30.0
    run_build("generic", src2, tmp_path / "out2", fields=["xhi=fields/xhi", "density=fields/delta"],
              source_options={"z_key": "z", "params_key": "p", "box_mpc": "L", "omega_m": "0.31",
                              "density_convention": "delta"})
    m2 = json.loads((tmp_path / "out2" / "manifest.json").read_text())
    assert m2["box_mpc"] == 30.0 and m2["samples"][0]["params"] == {"a": 1.0, "b": 2.0}
    assert np.allclose(m2["z_grid"], z)
    assert NpzDataset(tmp_path / "out2").describe()["shape"] == [6, 6, 20]
