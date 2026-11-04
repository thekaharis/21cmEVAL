"""A manifest whose target array has another name than ``--target`` (eval_cubes: ``truth``)."""
import json

import numpy as np

from eval21.data.npz import NpzDataset
from eval21.spec import check_prediction


def test_manifest_target_array_is_keyed_by_requested_target(tmp_path):
    rng = np.random.default_rng(0)
    truth = rng.random((8, 8, 6)).astype(np.float32)
    np.savez(tmp_path / "cone_1.npz", truth=truth, density=rng.random((8, 8, 6)).astype(np.float32),
             pred=truth)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "box_mpc": 16.0, "z_grid": list(np.linspace(6, 8, 6)), "target": "truth", "inputs": ["density"],
        "samples": [{"id": "cone_1", "path": "cone_1.npz", "split": "test"}]}))
    (sample,) = list(NpzDataset(tmp_path / "manifest.json", target="xhi"))
    assert set(sample.targets) == {"xhi"} and set(sample.inputs) == {"density"}
    check_prediction({"xhi": truth}, sample, names=["xhi"])          # shape (8, 8, 6), not (3,)


def test_true_xhi_follows_stage_field(tmp_path):
    from eval21.metrics.cylindrical import true_xhi

    rng = np.random.default_rng(1)
    truth = rng.random((8, 8, 6)).astype(np.float32)
    np.savez(tmp_path / "cone_1.npz", tb_truth=truth * 20, truth=truth,
             density=rng.random((8, 8, 6)).astype(np.float32))
    (tmp_path / "manifest_tb.json").write_text(json.dumps({
        "box_mpc": 16.0, "z_grid": list(np.linspace(6, 8, 6)), "target": "tb_truth",
        "inputs": ["density", "truth"], "samples": [{"id": "cone_1", "path": "cone_1.npz"}]}))
    (sample,) = list(NpzDataset(tmp_path / "manifest_tb.json", target="tb"))
    assert true_xhi(sample, "tb") is None                              # no field called xhi
    np.testing.assert_allclose(true_xhi(sample, "tb", "truth"), truth)
