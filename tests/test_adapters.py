import sys

import numpy as np
import pytest

from eval21.adapters import build
from eval21.adapters.command import CommandModelAdapter
from eval21.adapters.python import PythonModelAdapter
from eval21.adapters.toy import SmoothThresholdModel
from eval21.data import SyntheticDataset


@pytest.fixture(scope="module")
def sample():
    return next(iter(SyntheticDataset(n_samples=1, shape=(12, 12, 6))))


def test_python_file_adapter(tmp_path, sample):
    path = tmp_path / "model.py"
    path.write_text(
        "import numpy as np\n"
        "def predict(sample):\n"
        "    return {'xhi': np.full_like(sample.targets['xhi'], 0.5)}\n"
        "def describe():\n"
        "    return {'framework': 'test', 'parameters': 7}\n"
    )
    adapter = PythonModelAdapter(path)
    prediction = adapter.predict(sample)
    assert prediction["xhi"].shape == sample.targets["xhi"].shape
    assert adapter.describe()["parameters"] == 7
    assert build(f"python:{path}").describe()["framework"] == "test"


def test_python_adapter_missing_function(tmp_path):
    path = tmp_path / "empty.py"
    path.write_text("x = 1\n")
    with pytest.raises(AttributeError, match="predict"):
        PythonModelAdapter(path)


def test_command_adapter(tmp_path, sample):
    script = tmp_path / "infer.py"
    script.write_text(
        "import sys, numpy as np\n"
        "data = np.load(sys.argv[1])\n"
        "assert 'xhi' not in data.files, data.files\n"
        "xhi = np.full_like(data['density'], 0.25)\n"
        "np.savez(sys.argv[2], xhi=xhi)\n"
    )
    adapter = CommandModelAdapter(
        f"{sys.executable} {script} {{input}} {{output}}")
    prediction = adapter.predict(sample)
    assert np.allclose(prediction["xhi"], 0.25)


def test_command_adapter_does_not_leak_targets(tmp_path, sample):
    script = tmp_path / "copy.py"
    script.write_text(
        "import sys, numpy as np\n"
        "data = np.load(sys.argv[1])\n"
        "np.savez(sys.argv[2], xhi=data['density'] * 0 + 0.5)\n"
    )
    adapter = CommandModelAdapter(
        f"{sys.executable} {script} {{input}} {{output}}")
    prediction = adapter.predict(sample)
    assert np.allclose(prediction["xhi"], 0.5)


def test_command_adapter_reports_failure(tmp_path, sample):
    adapter = CommandModelAdapter("false {input} {output}")
    with pytest.raises(RuntimeError, match="command failed"):
        adapter.predict(sample)


def test_toy_adapter_is_bounded(sample):
    prediction = SmoothThresholdModel().predict(sample)
    values = prediction["xhi"]
    assert values.shape == sample.targets["xhi"].shape
    assert np.all(np.isfinite(values))
    assert values.min() >= 0 and values.max() <= 1


def test_fno_adapter_rejects_wrong_geometry(sample):
    from pathlib import Path

    from eval21.adapters.fno21cm import Fno21cmAdapter

    adapter = object.__new__(Fno21cmAdapter)
    adapter.spatial_shape = (8, 8, 99)
    adapter.expect_shape = None
    adapter.geometry_check = "verified"
    adapter._geometry_warned = False
    adapter.in_channels = 2
    adapter.channels = ("density/10", "1/(1+z)")
    adapter.checkpoint = Path("best.pt")
    with pytest.raises(ValueError, match="expects spatial shape"):
        adapter._check_geometry(sample)


def test_metadata_directory_uses_link_location(tmp_path):
    from eval21.adapters.fno21cm import metadata_directory

    target = tmp_path / "runA"
    target.mkdir()
    checkpoint = target / "best.pt"
    checkpoint.write_bytes(b"")
    link_dir = tmp_path / "runB"
    link_dir.mkdir()
    link = link_dir / "best.pt"
    link.symlink_to(checkpoint)
    assert metadata_directory(link) == link_dir


def test_unverified_geometry_warns(sample):
    from pathlib import Path

    from eval21.adapters.fno21cm import Fno21cmAdapter

    adapter = object.__new__(Fno21cmAdapter)
    adapter.spatial_shape = None
    adapter.expect_shape = None
    adapter.geometry_check = "unverified"
    adapter._geometry_warned = False
    adapter.in_channels = 1
    adapter.channels = ("density/10",)
    adapter.checkpoint = Path("old.pt")
    with pytest.warns(UserWarning, match="cannot be verified"):
        adapter._check_geometry(sample)


def test_cache_contract_validation(tmp_path):
    from eval21.adapters.fno21cm import validate_cache_contract

    metadata = {"dataset": {"cache_file": "cubes_3d.h5", "cache_rows": 6600,
                            "cache_shape": [140, 140, 256]}}
    info = {"path": "/data/cubes_3d.h5", "n_total": 6600,
            "shape": [140, 140, 256]}
    validate_cache_contract(metadata, info, tmp_path / "best.pt")
    with pytest.raises(ValueError, match="records cache"):
        validate_cache_contract(metadata, {**info, "path": "/data/other.h5"},
                                tmp_path / "best.pt")
    with pytest.raises(ValueError, match="cache rows"):
        validate_cache_contract(metadata, {**info, "n_total": 10},
                                tmp_path / "best.pt")


def test_cache_sha256_is_enforced(tmp_path):
    import hashlib

    from eval21.adapters.fno21cm import validate_cache_contract

    path = tmp_path / "cache.h5"
    path.write_bytes(b"cubes")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    info = {"path": str(path)}
    validate_cache_contract(
        {"dataset": {"cache_file": str(path), "cache_sha256": digest}},
        info, tmp_path / "best.pt")
    with pytest.raises(ValueError, match="hashes"):
        validate_cache_contract(
            {"dataset": {"cache_file": str(path), "cache_sha256": "0" * 64}},
            info, tmp_path / "best.pt")
    with pytest.raises(ValueError, match="records cache"):
        validate_cache_contract(
            {"dataset": {"cache_file": str(tmp_path / "wrong.h5")}},
            info, tmp_path / "best.pt")


def test_density_scale_helper():
    from eval21.adapters.fno21cm import cache_contract

    assert cache_contract({}) is None
    assert cache_contract({"dataset": {"cache_file": "x.h5"}}) == {
        "cache_file": "x.h5"}


def test_unknown_adapter():
    with pytest.raises(ValueError, match="unknown adapter"):
        build("nonsense:x")


def test_torch_adapter(sample):
    torch = pytest.importorskip("torch")
    from eval21.adapters.torch import TorchModelAdapter

    class MeanModel(torch.nn.Module):
        def forward(self, x):
            return x.mean(dim=1, keepdim=True)

    adapter = TorchModelAdapter(MeanModel())
    prediction = adapter.predict(sample)
    assert prediction["xhi"].shape == sample.targets["xhi"].shape
    assert adapter.describe()["parameters"] == 0

    stats = adapter.benchmark([sample], warmup=1, iters=3)
    assert len(stats["seconds"]) == 3
    assert stats["slices"] == sample.shape[2]

    from eval21.runner import evaluate
    from eval21.settings import Settings

    result = evaluate(adapter, SyntheticDataset(n_samples=1, shape=(12, 12, 6)),
                      Settings(box_mpc=100.0, warmup=1, iters=2),
                      ["efficiency"])
    assert result.metrics["efficiency"]["timing"] == "forward_only"
