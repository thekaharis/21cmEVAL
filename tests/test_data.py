import json

import numpy as np
import pytest

from eval21.data import NpzDataset, SyntheticDataset, open_data


def write_cube(path, shape=(12, 12, 6), seed=0, with_density=True, with_z=True):
    rng = np.random.default_rng(seed)
    payload = {"xhi": rng.random(shape).astype(np.float32),
               "rho": rng.random(shape).astype(np.float32)}
    if with_density:
        payload["density"] = (1 + 0.1 * rng.normal(size=shape)).astype(np.float32)
    if with_z:
        payload["z_grid"] = np.linspace(5, 25, shape[2])
    np.savez(path, **payload)


def test_directory_dataset(tmp_path):
    write_cube(tmp_path / "a.npz", seed=0)
    write_cube(tmp_path / "b.npz", seed=1)
    dataset = NpzDataset(tmp_path)
    assert len(dataset) == 2
    sample = next(iter(dataset))
    assert sample.targets["xhi"].shape == (12, 12, 6)
    assert sample.meta.z_grid[0] == 5
    # Every 3-D array other than the target becomes an input.
    assert set(sample.inputs) == {"rho", "density"}


def test_manifest_dataset(tmp_path):
    (tmp_path / "cubes").mkdir()
    write_cube(tmp_path / "cubes" / "c0.npz", seed=0)
    manifest = {
        "box_mpc": 100.0,
        "z_grid": [5, 10, 15, 20, 25, 30],
        "target": "xhi",
        "inputs": ["density"],
        "samples": [
            {"id": "cone0", "path": "cubes/c0.npz", "split": "test",
             "omega_m": 0.3, "params": {"F_ESC10": -1.0}},
            {"id": "cone1", "path": "cubes/c0.npz", "split": "train"},
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    dataset = NpzDataset(path)
    assert len(dataset) == 2
    sample = next(iter(dataset))
    assert sample.meta.box_mpc == 100.0
    assert sample.meta.omega_m == 0.3
    assert sample.meta.params == {"F_ESC10": -1.0}
    assert set(sample.inputs) == {"density"}

    test_only = NpzDataset(path, split="test")
    assert len(test_only) == 1
    assert next(iter(test_only)).id == "cone0"

    limited = NpzDataset(path, limit=1)
    assert len(limited) == 1


def test_npz_box_override(tmp_path):
    write_cube(tmp_path / "a.npz")
    dataset = NpzDataset(tmp_path, box_mpc=250.0)
    assert next(iter(dataset)).meta.box_mpc == 250.0


def test_npz_describe_exposes_path(tmp_path):
    write_cube(tmp_path / "a.npz")
    info = NpzDataset(tmp_path).describe()
    assert info["path"] == str(tmp_path.resolve())


def test_missing_density_is_tolerated(tmp_path):
    write_cube(tmp_path / "a.npz", with_density=False)
    sample = next(iter(NpzDataset(tmp_path)))
    assert "density" not in sample.inputs


def test_open_data_synthetic():
    dataset = open_data("synthetic:2")
    assert isinstance(dataset, SyntheticDataset)
    assert len(dataset) == 2


def test_synthetic_history_is_monotone():
    sample = next(iter(SyntheticDataset(n_samples=1, shape=(16, 16, 16))))
    history = sample.targets["xhi"].mean(axis=(0, 1))
    # x_HI rises with redshift: ionized at low z, neutral at high z.
    assert history[0] < 0.2
    assert history[-1] > 0.8
    assert np.all(np.isfinite(history))


def test_sample_rejects_non_3d():
    from eval21.spec import Sample

    with pytest.raises(ValueError, match="3-D"):
        Sample(id="bad", inputs={}, targets={"xhi": np.zeros((4, 4))})
