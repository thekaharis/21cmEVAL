# Visualization

All plotting lives in one package, `eval21.visualization` (`src/eval21/visualization/`). It
turns result directories into the figures of the fno-21cm and rhizome evaluations: statistics
figures read `results.json` and `curves.npz`, map figures read the prediction store written by
`eval21 run --save-predictions`.

```bash
python -m pip install -e ".[viz]"         # matplotlib
eval21 plot list                          # every figure, its group and what it needs
eval21 plot all --run results/ --out figs/
```

The CLI is unchanged by the package layout: `eval21 plot` is the only entry point, and every
figure is a function registered under a short name.

```text
eval21 plot cyl maps --run results/ --out figs/ --opt vmax=0.3
eval21 plot gallery  --run results/ --out figs/ --ids cone0001,cone0007
eval21 plot all      --run runA/ runB/ --out figs/ --models ref,tuned --style style.json
```

All figures are written twice, as PNG and PDF. `--out` defaults to nothing: `plot` requires it.

## The package

```text
src/eval21/visualization/
├── __init__.py        # public surface: Catalog, Run, find_runs, PlotContext, figure, load_all, run
├── registry.py        # @figure decorator, FIGURE_MODULES, PlotContext, SkipFigure
├── common.py          # Run/Merged/Catalog, style and label helpers, layout, save
├── bubbles.py         # bubble_size_distribution, bubble_size_summary
├── cylindrical.py     # cyl
├── diagnostics.py     # parity_overlay, boundary_band_overlay
├── ensemble.py        # ensemble_calibration, ensemble_draws
├── maps.py            # scatter_3d, multifield_los (stored cubes)
├── physical.py        # 9 physical-statistics figures (history, PDFs, 21cm, EFT, RMSE, speed, truth)
├── slices.py          # gallery
├── spectra.py         # 5 power-spectrum figures
└── collage.py         # docs/collage.png builder (Pillow; python -m eval21.visualization.collage)
```

`import eval21` does not import matplotlib. The plotting backend is forced to `Agg` inside
`common.pyplot()`, so figures render on a login node with no display.

## Data model

### `Run`

One `eval21 run` output directory: `results.json` plus optional `curves.npz`. It exposes
`settings`, `target`, `data`, `n_samples`, `models`, `group`, `metric(model, name)` and
`curves(model)`.

`Run.group` is a short label for the data a run scored: the manifest's directory name, or the
data path's name. It is what distinguishes redshift groups of the same target.

### `Merged`

Several runs of one `(target, data, model)` seen as one. This is how a later run that only adds
`--metrics cylindrical,gallery` is folded into the first one. A metric present in two of the
merged runs is an error, because there is no principled way to choose.

### `Catalog`

Runs indexed by target, group and model, with labels and line styles.

```python
catalog = Catalog(find_runs(["results/"]), models=None, style=None, quiet=False)
catalog.models                     # complete models, in order
catalog.targets                    # e.g. ["xhi", "tb"]
catalog.groups["xhi"]              # data keys (groups) of the x_HI runs
catalog.label(m), .color(m), .ls(m), .tick(m)
catalog.pooled(model, "rmse_per_sample", target="xhi")   # concatenated over groups
```

`find_runs` accepts result directories or trees: a directory without `results.json` is searched
recursively for it. A model must have results for **every** target and group to be shown;
incomplete models are printed as `incomplete (skipped): [...]` and left out. With no
`--models` filter, all complete models are used.

### Styling

`--style FILE` is JSON:

```json
{
  "ref":  {"label": "reference FNO", "color": "0.15", "ls": "--"},
  "tuned": {"label": "tuned, 8 blocks", "color": "#3182bd", "ls": "-"}
}
```

Defaults: the first model is dark grey dashed (the baseline); every later model takes the next
colour from a fixed ten-colour palette. A model with an explicit `color` does not consume a
palette slot. Labels may contain LaTeX; long labels are wrapped by panel titles and legends
(`wrap_label`, 22 characters).

Model labels can be long ("8 blk: pointwise, LoReLi v3, oracle history"); the layout is built
so that they never sit inside the data. See [Layout conventions](#layout-conventions).

### `PlotContext`

Everything a figure function receives:

```python
@dataclass
class PlotContext:
    catalog: Catalog
    out: Path
    store: PredictionStore | None = None
    ids: list[str] | None = None
    title: str = ""                       # --title
    options: dict = field(default_factory=dict)   # --opt key=value
    quiet: bool = False

    def opt(self, key, default=None, cast=None): ...
```

`ctx.opt("vmax", 0.4, float)` is the canonical way to read a figure option.

## The figure registry

```python
@figure("ps_overlay", group="spectra", needs=("curves",),
        help="active-stage Delta^2(k), P_pred/P_true and r(k) per model")
def ps_overlay(ctx: PlotContext) -> list[Path]:
    ...
```

| attribute | meaning |
|---|---|
| `name` | the string used on the command line; duplicates raise |
| `group` | `bubbles`, `diagnostics`, `ensemble`, `maps`, `physical`, `spectra` |
| `needs` | `("curves",)` (default) or `("store",)`; a store figure is skipped without a store |
| `help` | one line shown by `eval21 plot list` |

`load_all()` imports every module in `FIGURE_MODULES` lazily and returns the registry. A new
figure in an existing module needs nothing else; a new module must be added to that tuple.

`run(names, ctx)` selects figures by name or by group (`"all"` selects everything), skips store
figures when `ctx.store is None`, catches `SkipFigure` raised by a figure whose metric was not
run, and prints either the written files or `name: skipped: reason`. Explicit names print all
reasons; `all` hides the routine "needs stored cubes" noise.

## Figures

One statistic, one figure. Each figure may write several files (a per-sample page, a CSV
side-car), but there is no second rendering of the same statistic under a different name: the
legacy replots, the 2-D/3-D duplicates of the same slice diagnostics, and the summary dashboard
that repeated the dedicated figures were removed. The registry has **23 figures in 6 groups**.

Output stems below are relative to `--out`; `*` marks a target, field, model or sample suffix.

### bubbles

Bubble-size statistics from the `bubble` metric's mean-free-path samples.

| figure | needs | output | content |
|---|---|---|---|
| `bubble_size_distribution` | results | `bubble_size_distribution[_g*].png` | PDF per stage, truth vs models with 16–84 % bands and underflow/censored markers |
| `bubble_size_summary` | results | `bubble_size_summary[_g*].png` | per-stage restricted Wasserstein and mean-MFP bias bars |

### diagnostics

Calibration and front diagnostics.

| figure | needs | output | content |
|---|---|---|---|
| `parity_overlay` | results | `parity_overlay[_<group>].png` | E[pred x_HI \| true x_HI] with 16–84 % band and bin occupancy |
| `boundary_band_overlay` | results | `boundary_band_overlay.png` | RMSE, mean x_HI (front width) and gradient error vs distance to the true front |

### ensemble

Stochastic and generative models (`--draws N`).

| figure | needs | output | content |
|---|---|---|---|
| `ensemble_calibration` | results | `ensemble_calibration_<target>.png` | rank histogram, PIT histograms, coverage vs nominal, spread/skill per stage |
| `ensemble_draws` | store | `ensemble_draws/…` | truth, ensemble mean/std and draws at the gallery stages, per stored sample |

### maps

Field visualisations. `gallery` works from the `gallery` metric's slices in `curves.npz`; the
other two need the prediction store. One figure per view: transverse slices (`gallery`), voxel
scatter (`scatter_3d`), line-of-sight planes (`multifield_los`).

| figure | output (under `--out/`) | content |
|---|---|---|
| `gallery` | `gallery/cone_<id>.png` | stages × (x_HI \| T_b): truth / models / residuals, slice RMSE per panel |
| `scatter_3d` | `scatter_3d/scatter_3d_cone<id>_<field>.png` | voxel hexbin of truth vs prediction, one panel per model, R² / RMSE |
| `multifield_los` | `multifield_los/cone<id>_<field>_los.png` | LOS (x, z) planes at several y, cropped to the active band (`--opt full_z=1`, `--opt slices=`) |

### physical

Physical statistics and model-scale summaries.

| figure | needs | output | content |
|---|---|---|---|
| `physical_history` | results | `physical_history.png` | x_HI(z) error band, Δτ / σ histogram, midpoint and duration |
| `physical_pdfs` | results | `physical_pdfs.png` | stage-resolved x_HI and dT_b PDFs with W1 |
| `physical_21cm` | results | `physical_21cm.png` | 21-cm power, Δ² vs x_HI, dT_b moments, power-bias significance |
| `physical_eft` | results | `physical_eft_<target>.png` | EFT transfer, stochasticity, correlation, per-cone fits |
| `physical_3d` | store | `physical_3d/physical_3d_<target>_<id>.png` | per sample: global field vs z, 3-D power, cross-correlation |
| `rmse_r2` | results | `rmse_r2_comparison_<target>.png` | RMSE / R² per stage and overall, bars per model |
| `params_vs_accuracy` | results | `params_vs_accuracy_<target>.png` | parameters vs R² and vs RMSE with a Pareto front (`--opt point_labels=1`) |
| `inference_speed` | results | `inference_speed_vs_rmse_<target>.png`, `inference_speed_vs_size_<target>.png` | throughput vs RMSE and parameters vs throughput |
| `truth_sharpness_survey` | store | `truth_sharpness_survey.png`, `.json` | truth voxel population, partial fraction, front width, TV vs slice x_HI |

### spectra

Power-spectrum figures. Ratio and coherence are separate statistics: `cyl` shows the
cylindrical ratio error, `ps_cyl_r` the cylindrical coherence, `ps_overlay` the transverse
ratio/coherence at the active stage, `ps_coherence_highk` the high-k coherence/power relation,
and `cyl_anisotropy_decomposition` the k_par/k_perp decomposition of the cylindrical error.

| figure | needs | output | content |
|---|---|---|---|
| `cyl` | results | `cyl_error_<target>.png` | P_pred/P_true − 1 over (k_perp, k_par), white dead band, rows = models |
| `ps_overlay` | results | `ps_overlay_<target>.png` | active-stage Δ²(k), ratio and r(k) per model |
| `ps_cyl_r` | results | `ps_cyl_r_<target>.png` | coherence r(k_perp, k_par), rows = models |
| `cyl_anisotropy_decomposition` | results | `cyl_anisotropy_decomposition_<target>.png`, `.csv` | k_par / k_perp marginals plus the advantage map; needs ≥ 2 models |
| `ps_coherence_highk` | results | `ps_coherence_highk_<target>.png`, `.csv` | coherence r(k) vs small-scale power fidelity, placement/amount plane |

## Options

`--opt KEY=VALUE` values are parsed like `--set` (JSON, then float tuple, then string tuple).
The common options:

| option | figures | meaning |
|---|---|---|
| `vmax=X` | `cyl` | colour saturation (also `--vmax`) |
| `dead=X` | `cyl` | white dead band (also `--dead`) |
| `slices=N` | `multifield_los` | LOS planes per sample |
| `full_z=1` | `multifield_los` | show the whole LOS instead of the active band |
| `source=store\|curves` | `rmse_r2` | use per-slice scores from stored cubes, or the per-cone curves |
| `cyl_source=cylindrical\|anisotropy` | `ps_cyl_r`, `cyl_anisotropy_decomposition` | source of the cylindrical maps |
| `model=NAME`, `zbin=K` | `cyl_anisotropy_decomposition` | focus model and chunk column |
| `highk_k_min=X` | `ps_coherence_highk` | high-k cutoff, cycles/pixel |
| `point_labels=1` | scatter figures | also write the model name next to each point |

## Layout conventions

These are not cosmetic: they make figures from different runs comparable and keep long model
labels readable.

- **One model key per figure**, above the panels and outside the axes
  (`common.shared_legend`). Panels carry no model legends. The key holds truth, each model with
  its colour and line sample, then figure-wide style keys (bands, markers, reference lines,
  `common.style_key`).
- **Per-panel statistics** (W1, |Δτ|, front widths, k(r < 0.9), …) sit in one line under the
  panel's x label, drawn as a short line sample in each model's colour and style followed by
  the value (`common.pane_values`). `save` measures the text and re-runs the layout so rows do
  not collide.
- **Image grids** (maps, slices, cylindrical maps) name models in wrapped panel titles or row
  labels (`wrap_label`), not in a legend per panel.
- **Cylindrical (k⊥, k∥) maps are square**, with a tight colour bar fitted to the panel.
- **Scatter figures** name points through the shared key; `--opt point_labels=1` adds the name
  next to each point.
- **Colours are assigned once per catalog**, so a model has the same colour in every figure of
  a plotting session.

`save(fig, out, name, dpi=200)` applies the legend and per-panel values, writes `<name>.png`
and `<name>.pdf`, prints `wrote <path>` unless `--quiet`, and closes the figure.

## The README collage

`docs/collage.png` is built from the PNGs of an `eval21 plot all` run:

```bash
eval21 plot all --run results/ --out figs/
python -m eval21.visualization.collage figs/ --out docs/collage.png
```

It needs Pillow (not matplotlib). Cards are listed in `CARDS` in
`visualization/collage.py` as `(glob, width, centre x, centre y, angle)`; positions are
fractions of a 2800 × 1750 transparent canvas and angles are degrees. The first match of each
glob is used; a card that is not found is reported and skipped. The output is cropped to its
content and resized (default width 1600).

## Adding a figure

1. Write one decorated function in the module that matches its group (or a new module added to
   `FIGURE_MODULES` in `registry.py`).

```python
from .common import model_handles, pane_values, save, shared_legend
from .registry import SkipFigure, figure

@figure("my_curve", group="spectra", help="my new diagnostic")
def my_curve(ctx):
    plt = pyplot()
    cat = ctx.catalog
    rows = [cat.metric(m, "spectrum") for m in cat.models]
    if any(not r.get("stages") for r in rows):
        raise SkipFigure("no 'spectrum' results")
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, result in zip(cat.models, rows):
        stages = result["stages"]
        ax.plot([s["n_slices"] for s in stages], [s["coherence_med"] for s in stages],
                color=cat.color(model), ls=cat.ls(model), label=cat.label(model))
    ax.set(xlabel="slices", ylabel=r"$r(k)$")
    shared_legend(fig, handles=model_handles(cat), fontsize=8.5)
    return [save(fig, ctx.out, "my_curve", quiet=ctx.quiet)]
```

2. Use `ctx.opt("key", default)` for options, `ctx.ids` for `--ids`, `ctx.store` for cubes.
3. Raise `SkipFigure("reason")` when the inputs are absent; never draw an empty panel.
4. Return the list of written paths so `run()` can print them.
5. Add a row to the table above and, if the figure is user-facing, to the README.

The registry rejects duplicate names at import time, and `eval21 plot list` is the authority on
what exists — keep this document's tables in sync with it.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `no model has results for every target and group` | models differ across runs; use `--models` to pick a complete subset, or re-run the missing target |
| `incomplete (skipped): [...]` | as above, but some models are dropped automatically |
| `metric 'x' of model 'y' on <data> is in both …` | two runs for the same target/data/model report the same metric; merge or remove one |
| `skipped: needs stored cubes (run with --save-predictions)` | map figures need `eval21 run --save-predictions DIR`; `plot` looks for `predictions/` next to the results or takes `--store` |
| `skipped: no model was run with --draws N` | `ensemble_calibration` needs an `ensemble` metric |
| `skipped: needs at least two models` | `cyl_anisotropy_decomposition` compares models |
| `no 21cmEVAL results under <path>` | `--run` must point at a directory containing `results.json` or a tree of them |
| empty figures with one model and many targets | every model must exist in every target; run the second target or filter with `--models` |
| `ModuleNotFoundError: matplotlib` | `pip install -e ".[viz]"` |

## Import path

The package was renamed from `eval21.viz` to `eval21.visualization` so that all plotting lives
in one clearly named folder under `src/eval21/`. The CLI, the `viz` install extra and the
figure names are unchanged. Code that imported the old path must be updated:

```python
from eval21.visualization import Catalog, PlotContext, find_runs, run   # was eval21.viz
from eval21.visualization.common import model_handles, pane_values      # was eval21.viz.common
```
