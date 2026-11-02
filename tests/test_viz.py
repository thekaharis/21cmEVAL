import numpy as np
import pytest

from cm21eval.metrics.cylindrical import CylindricalMetrics
from cm21eval.metrics.gallery import GalleryMetric
from cm21eval.physics import KBinner, comoving_distance_mpc, los_cell_size_mpc
from cm21eval.settings import Settings
from cm21eval.spec import Meta, Sample

BOX, N = 40.0, 16
CELL = BOX / N


def uniform_z_grid(nz, z0=7.0, cell=CELL, omega_m=0.30966, h=0.6766):
    """Redshifts whose comoving spacing is exactly ``cell``."""
    fine = np.linspace(z0, z0 + 3, 20001)
    d = comoving_distance_mpc(fine, omega_m, h)
    return np.interp(np.arange(nz) * cell, d, fine)


def cone(nz=64, seed=0, z=None, xbar=None):
    rng = np.random.default_rng(seed)
    field = rng.normal(size=(N, N, nz))
    if xbar is None:
        xbar = np.linspace(0.0, 1.0, nz)
    xhi = np.clip(xbar[None, None, :] + 0.2 * field, 0, 1)
    z = uniform_z_grid(nz) if z is None else z
    return Sample(id=f"c{seed}", inputs={"density": 1 + 0.1 * field}, targets={"xhi": xhi},
                  meta=Meta(box_mpc=BOX, z_grid=z, omega_m=0.30966))


SET = Settings(box_mpc=BOX, n_k_bins=6, cyl_chunk_cells=16, cyl_stages=(0.5, 0.95, 2.0))


def test_cylindrical_perfect_prediction_is_one():
    s = cone()
    m = CylindricalMetrics(SET)
    m.update(s, {"xhi": s.targets["xhi"]})
    a = m.arrays()
    assert a["cyl_ratio"].shape == (1, 3, 8, 6)
    finite = np.isfinite(a["cyl_ratio"][0, 0])
    assert finite.any() and np.allclose(a["cyl_ratio"][0, 0][finite], 1) and np.allclose(a["cyl_coherence"][0, 0][finite], 1)
    res = {e["label"]: e for e in m.result()["stages"]}
    assert res["xbar_0p5"]["within_dead_band"] == 1.0
    assert res["xbar_2"]["n_samples"] == 0                                 # stage never reached


def test_cylindrical_matches_brute_force_mode_sum():
    s = cone(seed=3)
    pred = s.targets["xhi"] + 0.3 * np.random.default_rng(9).normal(size=s.targets["xhi"].shape)
    m = CylindricalMetrics(SET)
    m.update(s, {"xhi": pred})
    j = int(np.argmin(np.abs(s.targets["xhi"].mean((0, 1)) - 0.5)))
    t, p = s.targets["xhi"][..., j - 8:j + 8], pred[..., j - 8:j + 8]
    w = np.hanning(16)
    ft = np.fft.fftn((t - t.mean((0, 1), keepdims=True)) * w)
    fp = np.fft.fftn((p - p.mean((0, 1), keepdims=True)) * w)
    edges = KBinner(N, N, BOX, 6).edges
    kx = 2 * np.pi * np.fft.fftfreq(N, CELL)
    kz = 2 * np.pi * np.fft.fftfreq(16, d=np.diff(comoving_distance_mpc(s.meta.z_grid)).mean())
    pt, pp = np.zeros((8, 6)), np.zeros((8, 6))
    for a in range(N):
        for b in range(N):
            kp = np.hypot(kx[a], kx[b])
            ib = np.digitize(kp, edges) - 1
            if kp == 0 or not 0 <= ib < 6:
                continue
            for c in range(1, 9):                                         # positive k_par rows, as rfft
                pt[c - 1, ib] += abs(ft[a, b, c]) ** 2
                pp[c - 1, ib] += abs(fp[a, b, c]) ** 2
    assert np.allclose(m.arrays()["cyl_ratio"][0, 0], pp / pt, rtol=1e-4)


def test_cylindrical_stage_shift_and_fixed_z():
    s = cone(nz=40)                                                       # x-bar = linspace(0, 1, 40)
    m = CylindricalMetrics(SET.updated(cyl_stages=(0.5, 0.9, 0.97)))
    m.update(s, {"xhi": s.targets["xhi"]})
    xc = m.arrays()["cyl_chunk_xbar"][0]
    assert abs(xc[0] - 0.5) < 0.05
    assert 0.8 < xc[1] < 0.85                                             # 0.9 shifted inside the cone, kept
    assert np.isnan(xc[2])                                                # 0.97 would move > 0.1: dropped
    fixed = CylindricalMetrics(SET.updated(cyl_z_centers=(float(s.meta.z_grid[20]), 99.0)))
    fixed.update(s, {"xhi": s.targets["xhi"]})
    a = fixed.arrays()
    assert str(a["cyl_mode"]) == "z" and a["cyl_have"][0].tolist() == [True, False]


def test_cylindrical_resamples_nonuniform_grid_and_marks_resolution():
    z = np.linspace(7.0, 7.6, 32)                                         # uniform in z, ~2.6 Mpc spacing
    s = cone(nz=32, z=z)
    m = CylindricalMetrics(SET.updated(cyl_native_cell_mpc=4.0, cyl_native_los_dz=0.02))
    m.update(s, {"xhi": s.targets["xhi"]})
    a = m.arrays()
    have = a["cyl_have"][0]
    assert have[0]
    spacing = np.diff(comoving_distance_mpc(z)).max()
    native = np.pi / los_cell_size_mpc(a["cyl_chunk_z"][0, 0], 0.02)
    assert a["cyl_kpar_max"][0, 0] == pytest.approx(min(np.pi / spacing, native), rel=0.05)
    assert float(a["cyl_kperp_max"]) == pytest.approx(np.pi / 4.0)
    assert np.allclose(np.diff(a["cyl_kpar_edges"]), 2 * np.pi / (16 * CELL))   # transverse cell along the LOS


def test_cylindrical_uses_the_sample_cosmology():
    z = uniform_z_grid(48, omega_m=0.308, h=0.6766)
    s = cone(nz=48, z=z)
    s = Sample(id="c", inputs=s.inputs, targets=s.targets, meta=Meta(box_mpc=BOX, z_grid=z, omega_m=0.308))
    m = CylindricalMetrics(SET)
    m.update(s, {"xhi": s.targets["xhi"]})
    assert m._uniform and m.cell_los == pytest.approx(CELL, rel=1e-6)


def test_cylindrical_needs_true_xhi_for_stages():
    s = cone()
    tb = Sample(id="t", inputs={"density": s.inputs["density"]}, targets={"tb": s.targets["xhi"] * 20},
                meta=s.meta)
    m = CylindricalMetrics(SET.updated(target="tb"))
    with pytest.raises(ValueError):
        m.update(tb, {"tb": tb.targets["tb"]})
    tb.inputs["xhi"] = s.targets["xhi"]                                   # T_b pass with true x_HI as input
    m.update(tb, {"tb": tb.targets["tb"]})
    assert m.arrays()["cyl_have"][0][0]


def test_gallery_stores_stage_planes_and_fields():
    s = cone()
    s.inputs["tb"] = 20 * s.targets["xhi"]
    g = GalleryMetric(SET.updated(gallery_samples=1))
    for i in range(2):
        g.update(s, {"xhi": s.targets["xhi"], "tb": s.inputs["tb"]})
    a = g.arrays()
    assert a["gallery_ids"].tolist() == ["c0"]
    assert a["gallery_xhi_truth"].shape == (1, 4, N, N) and "gallery_tb_pred" in a
    assert np.allclose(a["gallery_xbar"][0], [0.8, 0.6, 0.4, 0.2], atol=0.05)


def test_plot_all_on_synthetic_runs(tmp_path):
    pytest.importorskip("matplotlib")
    from cm21eval import report
    from cm21eval.cli import main
    from cm21eval.runner import evaluate

    class Data:
        def __init__(self, samples):
            self.samples = samples

        def __iter__(self):
            return iter(self.samples)

        def __len__(self):
            return len(self.samples)

        def describe(self):
            return {"path": str(tmp_path / "data" / "manifest.json"), "n_samples": len(self.samples)}

    data = Data([cone(seed=i) for i in range(3)])

    class Adapter:
        def __init__(self, noise):
            self.noise = noise

        def predict(self, sample):
            rng = np.random.default_rng(len(sample.id))
            return {"xhi": np.clip(sample.targets["xhi"] + self.noise * rng.normal(size=sample.shape), 0, 1)}

    settings = SET.updated(rays_per_slice=16, slices_per_stage=2, bubble_bins=8, eft_k_max=0.5)
    names = ["voxel", "spectrum", "history", "observable", "morphology", "front", "bubble", "eft",
             "cylindrical", "gallery"]
    results = []
    for name, noise in (("ref", 0.3), ("tuned", 0.1)):
        r = evaluate(Adapter(noise), data, settings, names)
        r.name = name
        results.append(r)
    report.write(tmp_path / "run", results, data.describe(), settings.to_dict())
    # a second run of the same models and data that only adds a metric is merged
    extra = []
    for name, noise in (("ref", 0.3), ("tuned", 0.1)):
        r = evaluate(Adapter(noise), data, settings, ["parity"])
        r.name = name
        extra.append(r)
    report.write(tmp_path / "run_extra", extra, data.describe(), settings.to_dict())
    assert main(["plot", "all", "--run", str(tmp_path / "run"), "--run", str(tmp_path / "run_extra"),
                 "--out", str(tmp_path / "figs"), "--quiet"]) == 0
    with pytest.raises(SystemExit):                                       # the same metric twice is ambiguous
        main(["plot", "summary", "--run", str(tmp_path / "run"), "--run", str(tmp_path / "run"),
              "--out", str(tmp_path / "figs2"), "--quiet"])
    made = {p.name for p in (tmp_path / "figs").rglob("*.png")}
    assert {"cyl_error_xhi.png", "summary_scorecard.png", "summary_spectra_xhi.png", "cone_c0.png",
            "cone_c0_deltas.png", "overview_deltas.png"} <= made
