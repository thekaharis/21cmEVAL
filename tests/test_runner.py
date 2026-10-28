import csv
import json

import numpy as np
import pytest

from cm21eval import registry, report
from cm21eval.adapters.toy import SmoothThresholdModel
from cm21eval.cli import main
from cm21eval.data import SyntheticDataset
from cm21eval.runner import evaluate
from cm21eval.settings import Settings
from cm21eval.spec import Meta, Sample

SMALL = Settings(box_mpc=100.0, n_k_bins=8, rays_per_slice=24,
                 slices_per_stage=2, bubble_bins=8, eft_k_max=0.5)


def dataset(n=2, shape=(16, 16, 8)):
    return SyntheticDataset(n_samples=n, shape=shape, box_mpc=100.0)


def test_all_metrics_registered():
    assert set(registry.available()) >= {
        "voxel", "edge", "spectrum", "bubble", "morphology", "history",
        "observable", "eft", "front", "efficiency", "symmetry", "parity",
        "radial", "anisotropy",
    }


def test_evaluate_all_metrics():
    result = evaluate(SmoothThresholdModel(), dataset(), SMALL, ["all"])
    assert result.n_samples == 2
    assert not result.skipped
    for name in ("voxel", "spectrum", "history", "bubble", "front",
                 "morphology", "observable", "eft", "efficiency"):
        assert name in result.metrics
        assert result.metrics[name]
    assert result.metrics["voxel"]["rmse"] > 0
    assert np.isfinite(result.metrics["voxel"]["r2"])


def test_metrics_skip_without_density():
    sample = Sample(
        id="target-only",
        inputs={},
        targets={"xhi": np.random.default_rng(0).random((12, 12, 6)).astype(np.float32)},
        meta=Meta(box_mpc=100.0, z_grid=np.linspace(5, 25, 6)),
    )

    class Fixed:
        def predict(self, sample):
            return {"xhi": np.full_like(sample.targets["xhi"], 0.3)}

        def describe(self):
            return {}

    class One:
        def __iter__(self):
            yield sample

        def __len__(self):
            return 1

        def describe(self):
            return {}

    result = evaluate(Fixed(), One(), SMALL, ["all"])
    assert "observable" in result.skipped
    assert "eft" in result.skipped
    assert "voxel" in result.metrics


def test_report_files(tmp_path):
    result = evaluate(SmoothThresholdModel(), dataset(1), SMALL, ["all"])
    result.name = "toy"
    manifest = {"format": "synthetic", "n_samples": 1, "shape": [16, 16, 8],
                "box_mpc": 100.0}
    payload = report.write(tmp_path, [result], manifest, SMALL.to_dict())

    for name in ("results.json", "summary.csv", "report.md", "curves.npz"):
        assert (tmp_path / name).exists()

    # Strict JSON: no NaN/Infinity leaking into the report.
    loaded = json.loads((tmp_path / "results.json").read_text(),
                        parse_constant=lambda value: pytest.fail(value))
    assert loaded["schema"] == "cm21eval/1"
    assert loaded["models"]["toy"]["metrics"]["voxel"]["rmse"] > 0
    assert payload["models"]["toy"]["n_samples"] == 1

    with open(tmp_path / "summary.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and rows[0]["model"] == "toy"
    assert "voxel.rmse" in rows[0]

    text = (tmp_path / "report.md").read_text()
    assert "| model |" in text and "toy" in text


def test_flatten_skips_arrays():
    flat = report.flatten({"a": 1.0, "b": {"c": [1, 2]}, "d": float("nan"),
                           "e": np.array([1.0, 2.0])})
    assert flat == {"a": 1.0}


def test_cli_selftest_exit_code():
    assert main(["selftest"]) == 0


def test_cli_list_metrics(capsys):
    assert main(["list-metrics"]) == 0
    out = capsys.readouterr().out
    assert "spectrum" in out and "bubble" in out


def test_cli_demo(tmp_path, capsys):
    code = main(["demo", "--out", str(tmp_path), "--n-samples", "1",
                 "--shape", "16,16,8", "--quiet"])
    assert code == 0
    assert (tmp_path / "data" / "manifest.json").exists()
    assert (tmp_path / "results" / "results.json").exists()


def test_cli_run_with_config(tmp_path):
    from cm21eval.data.synthetic import write_demo

    manifest = write_demo(tmp_path / "data", n_samples=1, shape=(16, 16, 8))
    config = tmp_path / "eval.toml"
    config.write_text(
        "[run]\n"
        f'data = "{manifest}"\n'
        f'out = "{tmp_path / "out"}"\n'
        "limit = 1\n"
        "\n[models]\n"
        'toy = "synthetic:1.0"\n'
        "\n[settings]\n"
        "box_mpc = 100.0\n"
        "rays_per_slice = 16\n"
        "\n[metrics]\n"
        'enabled = ["voxel", "spectrum"]\n'
    )
    assert main(["run", "--config", str(config), "--quiet"]) == 0
    payload = json.loads((tmp_path / "out" / "results.json").read_text())
    assert set(payload["models"]["toy"]["metrics"]) == {"voxel", "spectrum"}
