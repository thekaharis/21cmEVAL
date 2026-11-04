"""Figures of viz/bubbles.py and viz/diagnostics.py built on the shared fixture."""
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from eval21.store import PredictionStore  # noqa: E402
from eval21.viz import registry  # noqa: E402
from eval21.viz.common import Catalog, find_runs  # noqa: E402


FIGURES = {
    "bubble_size_distribution": ["bubble_size_distribution"],
    "bubble_size_summary": ["bubble_size_summary"],
    "bsd_clean": ["bsd_clean"],
    "bsd_clean_summary": ["bsd_clean_summary"],
    "parity_overlay": ["parity_overlay"],
    "parity_comparison": ["parity_comparison"],
    "boundary_band_overlay": ["boundary_band_overlay"],
}


def _ctx(run_dirs, out, store_dir=None, **options):
    catalog = Catalog(find_runs(run_dirs), quiet=True)
    store = PredictionStore.open(store_dir) if store_dir else None
    return registry.PlotContext(catalog=catalog, out=out, store=store, options=options, quiet=True)


def _check(done, out, expected):
    for name, stems in expected.items():
        assert isinstance(done[name], list), f"{name}: {done[name]}"
        for stem in stems:
            for ext in ("png", "pdf"):
                f = out / f"{stem}.{ext}"
                assert f.exists() and f.stat().st_size > 1000, f


def test_registered():
    figures = registry.load_all()
    for name in FIGURES:
        assert name in figures
    assert figures["parity_comparison"].needs == ("store",)
    assert {figures[n].group for n in FIGURES} == {"bubbles", "diagnostics"}


def test_all_figures_on_fixture(tmp_path, viz_fixture):
    RESULTS, GEN = viz_fixture / "results", viz_fixture / "results_gen"
    out = tmp_path / "figs"
    done = registry.run(["bubbles", "diagnostics"], _ctx([RESULTS], out, RESULTS / "predictions"))
    _check(done, out, FIGURES)


def test_options_and_generative(tmp_path, viz_fixture):
    RESULTS, GEN = viz_fixture / "results", viz_fixture / "results_gen"
    out = tmp_path / "opt"
    done = registry.run(["bsd_clean", "bsd_clean_summary"],
                        _ctx([RESULTS], out, stat="med", stages=(1.0, 5.0), markers=0, ncols=2))
    _check(done, out, {"bsd_clean": ["bsd_clean"], "bsd_clean_summary": ["bsd_clean_summary"]})
    if not (GEN / "predictions" / "index.json").exists():
        pytest.skip("generative fixture missing")
    out = tmp_path / "gen"
    done = registry.run(["bubbles", "diagnostics"], _ctx([GEN], out, GEN / "predictions"))
    assert done["parity_overlay"].startswith("skipped")          # parity is pointwise: dropped
    _check(done, out, {k: v for k, v in FIGURES.items() if k != "parity_overlay"})


def test_two_groups_and_skip_without_xhi(tmp_path, viz_fixture):
    RESULTS, GEN = viz_fixture / "results", viz_fixture / "results_gen"
    # a second data set ("group") of the x_HI target: same runs, another dataset path
    root = tmp_path / "runs"
    for group in ("g1", "g2"):
        for model in ("sharp", "blurry"):
            dst = root / group / model / "xhi"
            shutil.copytree(RESULTS / model / "xhi", dst)
            res = json.loads((dst / "results.json").read_text())
            res["dataset"]["path"] = f"/data/{group}/manifest.json"
            (dst / "results.json").write_text(json.dumps(res))
    out = tmp_path / "two"
    done = registry.run(["bubbles", "parity_overlay", "boundary_band_overlay"], _ctx([root], out))
    _check(done, out, {**{k: v for k, v in FIGURES.items() if k not in ("parity_overlay", "parity_comparison")},
                       "parity_overlay": ["parity_overlay_g1", "parity_overlay_g2"]})

    # T_b only: every figure here is an x_HI figure and must skip with a reason
    done = registry.run(["bubbles", "diagnostics"], _ctx([RESULTS / "sharp" / "tb", RESULTS / "blurry" / "tb"],
                                                         tmp_path / "tb", RESULTS / "predictions"))
    for name in FIGURES:
        if name != "parity_comparison":
            assert isinstance(done[name], str) and done[name].startswith("skipped"), (name, done[name])
