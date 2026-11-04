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
