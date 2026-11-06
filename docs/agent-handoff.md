# Agent handoff: 21cmEVAL changes in this session

**Repo:** `thekaharis/21cmEVAL` · **branch:** `main` · **HEAD:** `69da225` (= `origin/main`)
**Base:** `a62b109` (start of session) · **Diff:** 44 files, +4456 / −59
**Scope:** this session modified **one library, 21cmEVAL**, only. No changes were made to
`fno-21cm`, `rhizome-neural-operator`, or any other repository. Everything below is what another
agent needs to work on or consume this repo without re-deriving the session.

## TL;DR

| area | change | commit |
|---|---|---|
| Figure package | `src/eval21/viz/` renamed to `src/eval21/visualization/`; imports, tests, README updated | `fc16e5f` |
| Collage tool | `docs/readme_collage.py` moved to `src/eval21/visualization/collage.py`; run as a module | `fc16e5f` |
| Docs | 10 documentation files in `docs/` (~21k words) + generated API reference + generator; this handoff and later design notes are additional | `fc16e5f`, `e60c3e1` |
| Metrics | `feat/viz-suite` merged: new `dtb`, `front3d`, `survey` metrics and many per-sample arrays | `d22de94` |
| Figures | deduplicated 43 → 23: one canonical figure per statistic; removed the bubble replots, parity hexbin, 2-D/3-D slice duplicates, the summary dashboard and redundant spectra variants | `9806084` |
| Metric safety | `front3d`/`survey` restricted to x_HI targets; `front3d` marked pointwise | `e60c3e1` |
| Build branch | `feat/build` recorded as merged with `-s ours`; **its code is not in main** | `8503a9f` |
| README | `source` column removed from the metrics table | `69da225` |

`main` is clean and pushed; no worktrees or uncommitted files remain.

## Commit log, oldest first

| commit | kind | what it did |
|---|---|---|
| `fc16e5f` | change | Reorganised the figure code into `eval21.visualization`; added the documentation set |
| `f757e4b` | branch | (from `feat/viz-suite`) arrays for the fno-21cm suite; `dtb`, `front3d`, `survey` metrics |
| `d623a58` | branch | (from `feat/viz-suite`) `dtb` runs inside `all`, skips without a T_b target |
| `d22de94` | merge | `feat/viz-suite` → `main` (content applied, no conflicts) |
| `88f19fb`, `b35335e` | branch | (from `feat/build`) the npz-cubes + manifest builder, `BuildOptions` API |
| `8503a9f` | merge | `feat/build` → `main` with `-s ours`: ancestry recorded, tree unchanged |
| `e60c3e1` | change | Documented the three new metrics; class docstrings; x_HI guards; README/API rows |
| `69da225` | change | Removed the `source` column from the README metrics table |

## Breaking change: `eval21.viz` → `eval21.visualization`

All plotting code now lives in one clearly named package under the import package. The CLI, the
figure names, the `viz` install extra and the report schema are unchanged; only Python imports
move.

```python
# before                          # after
from eval21 import viz            from eval21 import visualization as viz   # shim-style
from eval21.viz.common import x   from eval21.visualization.common import x
from eval21.viz import Catalog    from eval21.visualization import Catalog
```

The public surface of `eval21.visualization` is `Catalog`, `Run`, `find_runs`,
`PlotContext`, `SkipFigure`, `figure`, `load_all`, `run`, `plot_cylindrical`, `plot_gallery`.
The `plot_deltas` and `plot_summary` helpers were removed with the duplicate figures.

Any agent holding a cached import, a patched `eval21.viz` target in tests, or a docs link to
`eval21/viz/` must update it. The test files are still named `tests/test_viz*.py` (they cover the
package; only their imports changed).

### Collage tool moved

```bash
# before
python docs/readme_collage.py figs/ --out docs/collage.png
# after (same CARDS, same output; Pillow required, matplotlib not)
python -m eval21.visualization.collage figs/ --out docs/collage.png
```

`--out` now defaults to `docs/collage.png` relative to the working directory.

## Metrics: 17 → 20 registered families

`--metrics all` now includes three more families. The registry names, declared gates and
generative-mode behaviour:

| metric | module | `targets` | `pointwise` | notes |
|---|---|---|---|---|
| `dtb` | `metrics/dtb.py` | `None` (self-gates) | `False` | scores the model's **own** `tb` field; needs `--target tb` and a true `xhi` input; on an x_HI run returns a note and counts `skipped_samples` |
| `front3d` | `metrics/front3d.py` | `("xhi",)` | `True` | 3-D distance-transform front error; skipped on T_b runs and in `--mode generative` |
| `survey` | `metrics/survey.py` | `("xhi",)` | `False` | truth/prediction front survey, not a score; skipped on T_b runs |

The guards on `front3d`/`survey` were added **after** the merge (`e60c3e1`): the branch code would
otherwise have thresholded a T_b field. Keep them when touching those modules.

`dtb` defaults to a −250…100 mK histogram; `front3d` assumes a 13 Mpc mean LOS cell. Both are
per-metric options, not `Settings` fields:

```toml
[metric_options.dtb]
dtb_range_mk = [-250, 100]
dtb_bins = 350

[metric_options.front3d]
los_cell_mpc = 13.0
```

Everything else about metric semantics (stages, result keys, settings) is in
[`docs/metrics.md`](metrics.md). The report schema is still `eval21/1`; no `Settings` fields were
added or renamed.

### New `curves.npz` arrays (from `feat/viz-suite`)

Agents that read `curves.npz` can now pool groups exactly instead of re-running metrics. Naming
is still `<model>__<metric>__<array>`.

| metric | added arrays |
|---|---|
| `anisotropy` | `anisotropy_cyl_ratio`, `anisotropy_cyl_r`, `anisotropy_cyl_z`, `anisotropy_cyl_kpar`, `anisotropy_cyl_z_target`, `anisotropy_kperp_edges` |
| `bubble` | `bubble_stage_<key>` for `n_slices`, `w1`, `js`, `truth_mean`, `pred_mean`, `censored_*`, `underflow_*`, `ionized_*`, `median_hit_*`; `bubble_radius_edges_mpc`, `bubble_sample_ids` |
| `eft` | `eft_fit_bias_truth`, `eft_fit_bias_pred`, `eft_fit_bias_sigma_truth`, `eft_fit_stoch_truth`, `eft_fit_stoch_pred`, `eft_fit_stoch_sigma_truth` |
| `front` | `front_sample_ids`, `front_n_front_vox`, `front_band_{L2,H1,errfrac}_<b>`; `errfrac` now divides by the error of every voxel of the front slices |
| `history` | `history_sample_ids`, `history_omega_m` |
| `observable` | per-stage `observable_*` suite arrays: histograms of x_HI/dT_b, W1s, partial fractions, significance curves, Δ² |
| `parity` | `parity_count`, `parity_sum_pred`, `parity_sum_pred2`, `parity_sum_truth`, `parity_sum_err2`, `parity_h2d` |
| `spectrum` | `spectrum_lin_k`, `spectrum_lin_power`, `spectrum_highk_slices`, `spectrum_highk_k_min`, `spectrum_sample_ids`, `spectrum_n_stage` |
| `voxel` | `voxel_sums` (11 pooled sums: n, ss_res, sum_y, sum_y2, grad_ss, grad_n, highk_pred, highk_truth, highk_cross, sum_history_abs, history_samples) |
| `dtb` / `front3d` / `survey` | their own `dtb_*`, `front3d_*`, `survey_*` arrays |

No `eval21 plot` figure consumes `dtb`, `front3d` or `survey` yet; the arrays are the contract for
anyone who wants to draw them. The figure registry is 23 figures in 6 groups.

## The `feat/build` situation (important)

`feat/build` is **not** part of main's code. It is an earlier, parallel `eval21 build`:

| | main | `feat/build` (not applied) |
|---|---|---|
| output | one HDF5 cube cache | one npz per lightcone + `manifest*.json` |
| API | `build(folder, out, fmt=..., fields=..., z_grid=...)` | `BuildOptions`, `make_source(fmt, root, options)`, `build(source, options)` |
| CLI | `eval21 build FOLDER --out cache.h5` | `eval21 build <format> FOLDER --out DIR` |
| extras | z-range, sharding/merge, dtype/compression | crop, group, resume, dry-run, RSD |

If someone wants the npz+manifest builder, it has to be ported deliberately (its 494-line test
suite lives only on the branch) and given a different command name; it cannot share
`eval21 build` with the HDF5 implementation. Do not assume the merge brought it in.

## Figure dedupe (43 → 23)

Policy: **one figure per statistic per model and dataset.** Alternate encodings of the same
quantity were removed; one canonical figure per statistic remains. The registry now has 23
figures in 6 groups (`bubbles`, `diagnostics`, `ensemble`, `maps`, `physical`, `spectra`).

| statistic | canonical figure | removed |
|---|---|---|
| bubble-size PDF | `bubble_size_distribution` | `bsd_clean` |
| bubble-size W1 / bias | `bubble_size_summary` | `bsd_clean_summary` |
| parity / calibration | `parity_overlay` | `parity_comparison` |
| field slices | `gallery` | `comparison_3d`, `multifield_slices`, `slice_comparison`, `representative_z_slices`, `representative_z_compare`, `deltas`, `regression_examples`, `multifield_detailed` |
| voxel scatter | `scatter_3d` | `slice_scatter` |
| line-of-sight planes | `multifield_los` | `lightcone_3d`, `lightcone_grid_3d`, `multifield_strips`, `lightcone_animation` |
| ensemble spread | `ensemble_draws` | `ebm_examples` |
| transverse P(k) ratio/coherence | `ps_overlay` | `ps_stage_curves` |
| high-k coherence vs power | `ps_coherence_highk` | `ps_plane3d` |
| RMSE / R² | `rmse_r2` (bars) | the distributions output of the same figure |
| cross-model dashboard | — | `summary` (scorecard, spectra, bubbles, history, fronts, EFT/morphology) |

Consequences: `morphology` now has no figure (its numbers remain in `results.json`), and the
`summary` group is gone. `docs/visualization.md`, the README figure table and the collage cards
describe the 23-figure set.

## Files added / moved / removed

| path | change |
|---|---|
| `src/eval21/visualization/` | renamed from `src/eval21/viz/` (12 modules, unchanged content apart from docstrings) |
| `src/eval21/visualization/collage.py` | moved from `docs/readme_collage.py`; lazy Pillow import; `python -m` entry point |
| `src/eval21/metrics/dtb.py` | new |
| `src/eval21/metrics/front3d.py` | new |
| `src/eval21/metrics/survey.py` | new |
| `src/eval21/metrics/{anisotropy,bubble,eft,front,history,observable,parity,spectrum,voxel}.py` | extended `arrays()`; `front` gained per-sample band arrays and `errfrac_all` |
| `docs/*.md` | 10 documentation files, plus this handoff (11) |
| `docs/gen_api.py` | new: regenerates `docs/api.md` |
| `README.md` | links to the docs, design tree updated, `source` column dropped |

`docs/readme_collage.py` no longer exists; `docs/collage.png` is unchanged.

## Documentation map

Entry point [`docs/index.md`](index.md). Contents:

`architecture.md` (control flow, package map, object model), `cli.md` (all commands, flags, TOML),
`data.md` (Sample/Meta, npz/HDF5/cube caches, `eval21 build`, prediction store), `metrics.md`
(all 20 metrics, keys, settings, caveats), `adapters.md` (all specs, draws, fno-21cm contracts),
`reports.md` (`results.json`, CSVs, `curves.npz`), `visualization.md` (package API, all 23
figures, layout conventions, adding figures), `api.md` (generated signatures), `development.md`
(tests, conventions, extension recipes, release checklist).

`docs/api.md` is generated — change the source docstring, then run:

```bash
python docs/gen_api.py
```

## Verification performed

| check | result |
|---|---|
| `pytest -p no:warnings` | **128 passed, 7 skipped** (the 7 need `FNO21CM_ROOT`) |
| `eval21 selftest` | all checks passed |
| `eval21 demo --out …` | all 20 metrics, report schema `eval21/1` |
| T_b smoke run (`--target tb`, 2 synthetic cones) | `dtb` produced stage tables; `front3d`/`survey` skipped with `defined for xhi targets, not tb` |
| `eval21 plot all` | 23 figures, PNG + PDF |
| `python -m eval21.visualization.collage` | builds a collage from the figure directory |
| docs links/anchors, markdown tables, `compileall` | clean |

The verifying environment was `/opt/data/cache/scratch/venv-eval21` (Python 3.13, numpy 2.5.4,
scipy 1.18.1, matplotlib 3.11.2, h5py 3.16.0). Figure tests need matplotlib; `import eval21` does
not import it.

Reproduce:

```bash
python -m pip install -e ".[all]"
python -m pytest
eval21 selftest
eval21 demo --out /tmp/demo
eval21 plot all --run /tmp/demo/results --out /tmp/demo/figs
```

## Notes for the next agent

- **Do not resurrect `eval21.viz`.** A compatibility shim was deliberately not added; fix imports.
- **Do not expect `feat/build` code in main.** `git branch --merged` says it is merged because
  `8503a9f` used `-s ours`; the tree is main's HDF5 builder.
- **`--metrics all` is now 20 metrics.** A full pass on a T_b target will skip `front3d` and
  `survey`; on an x_HI target `dtb` returns a note (it is not an error).
- **Metric options are the exception.** Built-ins read `Settings`; only `dtb` (`dtb_range_mk`,
  `dtb_bins`) and `front3d` (`los_cell_mpc`) read `[metric_options.*]`.
- **The example TOML is slightly misleading:** `examples/fno21cm_eval.toml` sets `n_k_bins` under
  `[metric_options.spectrum]`, but the built-in `spectrum` reads `Settings.n_k_bins`. Use
  `--set n_k_bins=20` or `[settings] n_k_bins = 20`.
- **When adding a metric:** register it, add a test with an analytic expectation, update
  `docs/metrics.md` and the README rows, then `python docs/gen_api.py`.
- **When adding a figure:** it goes in `eval21.visualization`, raise `SkipFigure` when inputs are
  missing, use `shared_legend`/`pane_values`, and update `docs/visualization.md`'s catalogue.
- **One statistic, one figure.** Do not add a second rendering (clean/replot/hexbin/alternative
  layout) of a statistic that already has a figure; extend the canonical one instead.
- **Shared branch discipline:** `main` is published; pull before working, and keep the docs tables
  in sync with `eval21 plot list` (23) and `eval21 list-metrics` (20).
- **Open cleanups:** remote branches `origin/feat/build` and `origin/feat/viz-suite` still exist;
  the 7 skipped reference tests need an fno-21cm checkout at `FNO21CM_ROOT`; there is no figure
  for the three new metrics yet.
