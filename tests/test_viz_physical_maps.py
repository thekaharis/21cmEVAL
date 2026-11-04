"""Physical-statistics figures (visualization/physical.py) and map figures (visualization/maps.py) on the fixture."""
from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

from eval21.store import PredictionStore  # noqa: E402
from eval21.visualization import registry  # noqa: E402
from eval21.visualization.common import Catalog, find_runs  # noqa: E402

# figures that may legitimately skip on the fixture, with the reason they must give
EXPECTED_SKIPS = {"params_vs_accuracy": "parameter counts"}


def _ctx(runs, out, store=None, **options):
    return registry.PlotContext(catalog=Catalog(find_runs(runs), quiet=True), out=out,
                                store=PredictionStore.open(store) if store else None,
                                options=options, quiet=True)


def _group(name):
    return sorted(n for n, f in registry.load_all().items() if f.group == name)


def _check_group(done, names, out):
    for name in names:
        result = done[name]
        if name in EXPECTED_SKIPS:
            assert isinstance(result, list) or EXPECTED_SKIPS[name] in result, (name, result)
            continue
        assert isinstance(result, list) and result, (name, result)
        for path in result:
            assert path.exists() and path.stat().st_size > 1000, path
    pngs = list(out.rglob("*.png"))
    assert pngs and all(p.with_suffix(".pdf").exists() for p in pngs)


def test_physical_figures(tmp_path, viz_fixture):
    names = _group("physical")
    assert {"physical_history", "physical_pdfs", "physical_21cm", "physical_eft", "physical_3d",
            "rmse_r2", "inference_speed", "truth_sharpness_survey"} <= set(names)
    results = viz_fixture / "results"
    done = registry.run(names, _ctx([results], tmp_path, results / "predictions"))
    _check_group(done, names, tmp_path)


def test_map_figures(tmp_path, viz_fixture):
    names = [n for n in _group("maps") if n != "lightcone_animation"]
    assert {"comparison_3d", "lightcone_3d", "scatter_3d", "lightcone_grid_3d", "multifield_slices",
            "multifield_strips", "slice_comparison", "representative_z_slices"} <= set(names)
    results = viz_fixture / "results"
    done = registry.run(names, _ctx([results], tmp_path, results / "predictions"))
    _check_group(done, names, tmp_path)
    # store-only figures skip with a reason when no cubes were stored
    done = registry.run(["comparison_3d"], _ctx([results], tmp_path / "nostore"))
    assert done["comparison_3d"].startswith("skipped")


def test_lightcone_animation(tmp_path, viz_fixture):
    results = viz_fixture / "results"
    done = registry.run(["lightcone_animation"],
                        _ctx([results], tmp_path, results / "predictions", frames=4, dpi=40))
    written = done["lightcone_animation"]
    assert isinstance(written, list) and written and written[0].suffix == ".gif"


def test_maps_on_generative_run(tmp_path, viz_fixture):
    gen = viz_fixture / "results_gen"
    done = registry.run(["comparison_3d", "lightcone_3d", "ensemble"],
                        _ctx([gen], tmp_path, gen / "predictions"))
    assert isinstance(done["comparison_3d"], list) and done["comparison_3d"]
    assert (tmp_path / "ensemble_calibration_xhi.png").exists()
    assert list((tmp_path / "ensemble_draws").glob("*.png"))
