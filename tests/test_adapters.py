import sys

import numpy as np
import pytest

from cm21eval.adapters import build
from cm21eval.adapters.command import CommandModelAdapter
from cm21eval.adapters.python import PythonModelAdapter
from cm21eval.adapters.toy import SmoothThresholdModel
from cm21eval.data import SyntheticDataset


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
        "xhi = np.full_like(data['xhi'], 0.25)\n"
        "np.savez(sys.argv[2], xhi=xhi)\n"
    )
    adapter = CommandModelAdapter(
        f"{sys.executable} {script} {{input}} {{output}}")
    prediction = adapter.predict(sample)
    assert np.allclose(prediction["xhi"], 0.25)


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


def test_unknown_adapter():
    with pytest.raises(ValueError, match="unknown adapter"):
        build("nonsense:x")


def test_torch_adapter(sample):
    torch = pytest.importorskip("torch")
    from cm21eval.adapters.torch import TorchModelAdapter

    class MeanModel(torch.nn.Module):
        def forward(self, x):
            return x.mean(dim=1, keepdim=True)

    adapter = TorchModelAdapter(MeanModel())
    prediction = adapter.predict(sample)
    assert prediction["xhi"].shape == sample.targets["xhi"].shape
    assert adapter.describe()["parameters"] == 0
