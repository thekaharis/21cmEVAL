"""Round-trip tests for eval21-run/1 files and the optional eval21-cache/1."""
from __future__ import annotations

import numpy as np
import pytest

from eval21.adapters.toy import SmoothThresholdModel
from eval21.cli import main
from eval21.data import SyntheticDataset
from eval21.io import (CacheWriter, StoreGroup, default_path, open_cache, open_run,
                       validate_run, write_run)
from eval21.runner import evaluate
from eval21.settings import Settings

SMALL = Settings(box_mpc=100.0, n_k_bins=8, rays_per_slice=16,
                 slices_per_stage=2, bubble_bins=6, gallery_samples=2)


def _dataset(n=2):
    return SyntheticDataset(n_samples=n, shape=(16, 16, 8), box_mpc=100.0)


def _result(metrics=("voxel", "spectrum", "history")):
    result = evaluate(SmoothThresholdModel(), _dataset(), SMALL, list(metrics))
    result.name = "toy"
    return result


def test_write_read_run_file(tmp_path):
    data = _dataset()
    result = _result()
    path = write_run(tmp_path / "toy.xhi.h5", result, data.describe(), SMALL.to_dict(),
                     {"python": "3.13"}, command="eval21 run ...")
    assert path.exists()

    run = open_run(path)
    try:
        assert run.schema == "eval21-run/1"
        assert run.model == "toy"
        assert run.target == "xhi"
        assert run.n_samples == result.n_samples == 2
        assert run.sample_ids == ["synthetic0000", "synthetic0001"]
        assert set(run.metric_names()) == {"voxel", "spectrum", "history"}
        assert run.result("voxel")["rmse"] == pytest.approx(result.metrics["voxel"]["rmse"])
        assert run.metric_meta("voxel")["pointwise"] is True

        arrays = run.arrays("spectrum")
        assert arrays["spectrum_k"].shape == (SMALL.n_k_bins,)
        assert run.h5["metrics/spectrum/arrays/spectrum_k"].attrs["dims"] == "k"
        assert run.h5["metrics/spectrum/arrays/spectrum_k"].attrs["units"] == "Mpc^-1"
        assert "toy" in run.curves()
        assert run.validate(strict=True) == []
        payload = run.as_results()
        assert payload["schema"] == "eval21/1"
        assert payload["models"]["toy"]["n_samples"] == 2
        assert payload["dataset"]["n_samples"] == 2
    finally:
        run.close()

    with pytest.raises(FileExistsError):
        write_run(path, result, data.describe(), SMALL.to_dict())
    write_run(path, result, data.describe(), SMALL.to_dict(), overwrite=True)
    assert validate_run(path, strict=True) == []


def test_legacy_directory_reader(tmp_path):
    from eval21 import report

    result = _result()
    manifest = _dataset().describe()
    report.write(tmp_path / "legacy", [result], manifest, SMALL.to_dict())
    run = open_run(tmp_path / "legacy")
    try:
        assert run.schema == "eval21/1"
        assert run.model == "toy"
        assert run.as_results()["models"]["toy"]["n_samples"] == 2
        assert "voxel" in run.metric_names()
        assert run.curves()["toy"]["spectrum_k"].shape == (SMALL.n_k_bins,)
    finally:
        run.close()


def test_cli_writes_files_and_legacy_dirs(tmp_path):
    main(["demo", "--out", str(tmp_path / "demo"), "--n-samples", "2",
          "--shape", "16,16,8", "--quiet"])
    out = tmp_path / "demo" / "results"
    assert (out / "toy.xhi.h5").exists()
    assert not (out / "results.json").exists()
    assert main(["validate", str(out / "toy.xhi.h5")]) == 0

    data = str(tmp_path / "demo" / "data" / "manifest.json")
    single = tmp_path / "single.h5"
    main(["run", "--data", data, "--model", "toy=synthetic:1.5", "--metrics", "voxel",
          "--out", str(single), "--quiet"])
    assert single.exists()

    legacy = tmp_path / "legacy"
    main(["run", "--data", data, "--model", "toy=synthetic:1.5", "--metrics", "voxel",
          "--out", str(legacy), "--legacy-dir", "--quiet"])
    assert (legacy / "results.json").exists()
    assert not list(legacy.glob("*.h5"))


def test_two_models_one_directory_and_cache(tmp_path):
    main(["demo", "--out", str(tmp_path / "demo"), "--n-samples", "3",
          "--shape", "16,16,8", "--quiet"])
    data = str(tmp_path / "demo" / "data" / "manifest.json")
    out = tmp_path / "results"
    cache = tmp_path / "inv.cache.h5"
    for model, sigma in (("sharp", 1.0), ("blurry", 2.5)):
        main(["run", "--data", data, "--model", f"{model}=synthetic:{sigma}",
              "--metrics", "voxel,spectrum,gallery", "--out", str(out),
              "--cache", str(cache), "--save-n", "2",
              "--set", "gallery_samples=2", "--quiet"])
    files = sorted(p.name for p in out.glob("*.h5"))
    assert files == ["blurry.xhi.h5", "sharp.xhi.h5"]

    cached = open_cache(cache)
    try:
        assert cached.models == ["sharp", "blurry"]
        assert cached.samples == ["synthetic0000", "synthetic0001"]
        assert cached.targets == ["xhi"]
        truth = cached.truth("synthetic0000")
        assert truth["targets"]["xhi"].shape == (16, 16, 8)
        assert truth["inputs"]["density"].shape == (16, 16, 8)
        assert cached.prediction("blurry", "synthetic0000")["xhi"].shape == (16, 16, 8)
        assert cached.has("sharp", "synthetic0000")
        assert not cached.has("sharp", "synthetic9999")
    finally:
        cached.close()


def test_cache_writer_roundtrip(tmp_path):
    data = _dataset()
    sample = next(iter(data))
    path = tmp_path / "one.cache.h5"
    writer = CacheWriter(path, n=1, overwrite=True)
    writer.save_truth(sample)
    writer.save_prediction("toy", sample, {"xhi": np.asarray(sample.targets["xhi"])})
    writer.close()

    cache = open_cache(path)
    try:
        assert cache.models == ["toy"]
        assert cache.samples == [sample.id]
        np.testing.assert_allclose(cache.truth(sample.id)["targets"]["xhi"],
                                   sample.targets["xhi"], rtol=2e-3, atol=2e-3)
        np.testing.assert_allclose(cache.prediction("toy", sample.id)["xhi"],
                                   sample.targets["xhi"], rtol=2e-3, atol=2e-3)
    finally:
        cache.close()


def test_default_path_is_safe(tmp_path):
    assert default_path(tmp_path, "a/b", "xhi").name == "a_b.xhi.h5"


def test_plot_reads_a_file_and_cache(tmp_path):
    pytest.importorskip("matplotlib")
    main(["demo", "--out", str(tmp_path / "demo"), "--n-samples", "2",
          "--shape", "16,16,8", "--quiet"])
    data = str(tmp_path / "demo" / "data" / "manifest.json")
    out = tmp_path / "results"
    cache = tmp_path / "run.xhi.cache.h5"
    main(["run", "--data", data, "--model", "toy=synthetic:1.5",
          "--metrics", "voxel,gallery", "--out", str(out),
          "--cache", str(cache), "--save-n", "2", "--set", "gallery_samples=2", "--quiet"])
    run_file = out / "toy.xhi.h5"
    figs = tmp_path / "figs"
    assert main(["plot", "gallery", "scatter_3d", "--run", str(run_file),
                 "--store", str(cache), "--out", str(figs), "--quiet"]) == 0
    assert list((figs / "gallery").glob("*.png"))
    assert list((figs / "scatter_3d").glob("*.png"))


def test_store_group_merges_caches(tmp_path):
    data = _dataset()
    samples = list(data)[:2]
    paths = [tmp_path / "a.cache.h5", tmp_path / "b.cache.h5"]
    for path, sample, model in zip(paths, samples, ("a", "b")):
        writer = CacheWriter(path, n=1, overwrite=True)
        writer.save_truth(sample)
        writer.save_prediction(model, sample, {"xhi": np.asarray(sample.targets["xhi"])})
        writer.close()
    group = StoreGroup([open_cache(paths[0]), open_cache(paths[1])])
    try:
        assert group.samples == [samples[0].id, samples[1].id]
        assert group.models == ["a", "b"]
        assert group.has("a", samples[0].id) and not group.has("a", samples[1].id)
        assert group.truth(samples[1].id)["targets"]["xhi"].shape == (16, 16, 8)
        assert group.prediction("b", samples[1].id)["xhi"].shape == (16, 16, 8)
    finally:
        for store in group.stores:
            store.close()
