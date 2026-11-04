"""Power-spectrum figures (visualization/spectra.py) on the synthetic fixture runs."""
from pathlib import Path

import numpy as np
import pytest

FIGURES = ("ps_overlay", "ps_stage_curves", "ps_cyl_r", "cyl_anisotropy_decomposition",
           "ps_coherence_highk", "ps_plane3d")

def _ctx(tmp_path, runs, options=None, models=None):
    from eval21.visualization import Catalog, PlotContext, find_runs

    cat = Catalog(find_runs(runs), models=models, quiet=True)
    return PlotContext(catalog=cat, out=tmp_path, options=options or {}, quiet=True)


def test_spectra_figures_on_fixture(tmp_path, viz_fixture):
    FIXTURE = viz_fixture
    pytest.importorskip("matplotlib")
    from eval21.visualization import load_all, run

    figures = load_all()
    assert {n for n, f in figures.items() if f.group == "spectra"} >= set(FIGURES)
    done = run(list(FIGURES), _ctx(tmp_path, [FIXTURE / "results"]))
    for name in FIGURES:
        assert isinstance(done[name], list) and done[name], (name, done[name])
        for target in ("xhi", "tb"):
            assert (tmp_path / f"{name}_{target}.png").stat().st_size > 0
            assert (tmp_path / f"{name}_{target}.pdf").exists()
    for name in ("cyl_anisotropy_decomposition", "ps_coherence_highk", "ps_plane3d"):
        assert (tmp_path / f"{name}_xhi.csv").read_text().count("\n") >= 3


def test_spectra_options_and_anisotropy_source(tmp_path, viz_fixture):
    FIXTURE = viz_fixture
    pytest.importorskip("matplotlib")
    from eval21.visualization import run

    ctx = _ctx(tmp_path, [FIXTURE / "results" / "sharp" / "xhi", FIXTURE / "results" / "blurry" / "xhi"],
               options={"cyl_source": "anisotropy", "model": "blurry", "zbin": 0})
    done = run(["ps_cyl_r", "cyl_anisotropy_decomposition"], ctx)
    assert (tmp_path / "cyl_anisotropy_decomposition_xhi.png").exists()
    assert isinstance(done["ps_cyl_r"], str) and "skipped" in done["ps_cyl_r"]   # no coherence maps
    rows = (tmp_path / "cyl_anisotropy_decomposition_xhi.csv").read_text().splitlines()
    assert rows[1].startswith("amplitude,blurry,sharp,z ≈ ")
    assert not any(r.startswith("decoherence") for r in rows)


def test_spectra_generative_run_drops_coherence(tmp_path, viz_fixture):
    FIXTURE = viz_fixture
    pytest.importorskip("matplotlib")
    from eval21.visualization import run

    if not (FIXTURE / "results_gen").exists():
        pytest.skip("generative fixture not available")
    done = run(["ps_overlay", "ps_cyl_r", "ps_plane3d", "cyl_anisotropy_decomposition"],
               _ctx(tmp_path, [FIXTURE / "results_gen"]))
    assert (tmp_path / "ps_overlay_xhi.png").exists()
    assert "skipped" in done["ps_cyl_r"] and "skipped" in done["cyl_anisotropy_decomposition"]
    assert "nan" in (tmp_path / "ps_plane3d_xhi.csv").read_text()


def test_decomposition_and_crossing_helpers():
    from eval21.visualization.spectra import coherence_scale, k_crossing, pretty_stage, two_way_decomposition

    par = np.outer(np.arange(4.0), np.ones(5))                  # varies along k_par only
    assert two_way_decomposition(par)["par_share"] == pytest.approx(1.0)
    perp = np.outer(np.ones(4), np.arange(5.0))
    assert two_way_decomposition(perp)["perp_share"] == pytest.approx(1.0)
    assert np.isnan(two_way_decomposition(np.zeros((3, 3)))["par_share"])
    k = np.array([0.1, 0.2, 0.4])
    assert k_crossing(k, np.array([1.0, 0.95, 0.85]), 0.9) == pytest.approx(np.exp(np.log(0.2) + 0.5 * np.log(2)))
    assert coherence_scale(k, np.array([1.0, 0.95, 0.85]), 0.9) == pytest.approx(0.3)
    assert np.isnan(k_crossing(k, np.ones(3), 0.9))
    assert pretty_stage("xbar_HI 0.02-0.20") == "x̄_HI 0.02–0.20"
