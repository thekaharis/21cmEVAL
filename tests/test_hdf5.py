import numpy as np
import pytest

from cm21eval.data import Hdf5Dataset, open_data


@pytest.fixture()
def cube_file(tmp_path):
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(0)
    n, nx, ny, nz = 3, 8, 8, 5
    path = tmp_path / "cubes_3d.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("density", data=rng.random((n, nx, ny, nz)))
        handle.create_dataset("neutral_fraction", data=rng.random((n, nx, ny, nz)))
        handle.create_dataset("cone_id", data=np.arange(100, 100 + n))
        handle.create_dataset("target_z", data=np.linspace(5, 25, nz))
        handle.create_dataset("params", data=rng.normal(size=(n, 11)))
    return path


def test_analysis_layout(cube_file):
    dataset = Hdf5Dataset(cube_file, box_mpc=100.0)
    assert len(dataset) == 3
    sample = next(iter(dataset))
    assert sample.id == "100"
    assert sample.targets["xhi"].shape == (8, 8, 5)
    assert "density" in sample.inputs
    assert len(sample.meta.params) == 11
    assert "F_ESC10" in sample.meta.params
    assert sample.meta.box_mpc == 100.0


def test_indices(cube_file):
    dataset = Hdf5Dataset(cube_file, indices=[1])
    assert len(dataset) == 1
    assert next(iter(dataset)).id == "101"


def test_training_layout(tmp_path):
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(1)
    path = tmp_path / "train.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=rng.random((2, 3, 6, 6, 4)))
        handle.create_dataset("y", data=rng.random((2, 1, 6, 6, 4)))
    dataset = Hdf5Dataset(path, limit=1)
    sample = next(iter(dataset))
    assert sample.inputs["x"].shape == (3, 6, 6, 4)
    assert sample.targets["xhi"].shape == (6, 6, 4)


def test_split_is_rejected_without_a_map(cube_file):
    dataset = Hdf5Dataset(cube_file, split="test")
    with pytest.raises(ValueError, match="does not declare splits"):
        list(dataset)


def test_split_map_is_honoured(cube_file):
    dataset = Hdf5Dataset(cube_file, split="test",
                          split_map={"train": [0], "test": [1, 2]})
    assert len(dataset) == 2
    assert {sample.id for sample in dataset} == {"101", "102"}


def test_row_indices_for_ids(cube_file):
    dataset = Hdf5Dataset(cube_file)
    assert dataset.row_indices_for_ids([102, 100]) == [2, 0]
    with pytest.raises(KeyError, match="absent"):
        dataset.row_indices_for_ids([999])


def test_omega_m_from_params(cube_file):
    dataset = Hdf5Dataset(cube_file)
    assert next(iter(dataset)).meta.omega_m is not None


def test_open_data_detects_hdf5(cube_file):
    assert isinstance(open_data(str(cube_file), limit=1), Hdf5Dataset)
