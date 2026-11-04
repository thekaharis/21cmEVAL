"""Ensemble figures from a toy generative run with draws and a prediction store."""
from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

from eval21.cli import main


def test_ensemble_figures(tmp_path):
    main(["demo", "--out", str(tmp_path / "demo"), "--n-samples", "3", "--shape", "24,24,32", "--quiet"])
    data = str(tmp_path / "demo" / "data" / "manifest.json")
    results = tmp_path / "results"
    for name, mode in (("gen", "generative"), ("gen_pw", "pointwise")):
        main(["run", "--data", data, "--model", f"{name}=toygen:1.5", "--draws", "4", "--mode", mode,
              "--metrics", "ensemble,spectrum", "--out", str(results / name / "xhi"),
              "--save-predictions", str(results / "predictions"), "--save-n", "2",
              "--set", "ensemble_voxels=2000", "--quiet"])
    out = tmp_path / "figs"
    main(["plot", "ensemble", "--run", str(results), "--out", str(out), "--quiet"])
    assert (out / "ensemble_calibration_xhi.png").exists()
    drawn = sorted(p.name for p in (out / "ensemble_draws").glob("*.png"))
    assert len(drawn) == 4, drawn            # 2 samples x 2 models
    # short ids resolve to stored 'synthetic000N' / 'coneN' ids
    main(["plot", "ensemble_draws", "--run", str(results), "--out", str(tmp_path / "one"),
          "--ids", drawn[0].split("_")[0], "--quiet"])
