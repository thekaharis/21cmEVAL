# Architecture

21cmEVAL is a pipeline with three seams: **data** produces samples, **adapters** turn samples
into predictions, and **metrics** turn (sample, prediction) pairs into numbers. Everything else
— the runner, the report writers, the figure package — sits on top of those seams and knows
nothing about training frameworks.

```text
                    ┌──────────────┐
   simulation files │   datasets   │  npz / manifest / HDF5 / eval21 cache / synthetic
   or result dirs   │ eval21.data  │
                    └──────┬───────┘
                           │ Sample(inputs, targets, meta)
                    ┌──────▼───────┐
                    │   adapter    │  python / torch / command / fno21cm / toy
                    │eval21.adapters│
                    └──────┬───────┘
                           │ {target: ndarray}  (+ draws when --draws)
                    ┌──────▼───────┐        ┌───────────────────────┐
                    │   runner     │───────▶│ cache (eval21/io)     │  --cache
                    │eval21.runner │        │ float16 cubes         │
                    └──────┬───────┘        └───────────┬───────────┘
                           │ per-metric dicts           │
                    ┌──────▼───────┐                    │
                    │ io/runfile   │  <model>.<target>.h5 (eval21-run/1)      │
                    │ report       │  legacy: results.json, CSVs, curves.npz   │
                    └──────┬───────┘                    │
                           │                            │
                    ┌──────▼────────────────────────────▼──────┐
                    │           eval21.visualization            │
                    │  figure registry → PNG + PDF per figure   │
                    └───────────────────────────────────────────┘
```

## Design principles

**One pass.** Every enabled metric updates from the same prediction, inside a single iteration
over the dataset, for each model. A sweep with 20 metrics costs the same forward passes as a
sweep with one; the metrics differ only in the numpy work after `predict`.

**Constant memory.** Accumulators are sums, histograms, per-slice arrays and per-sample scalars.
No metric keeps a cube after `update` returns. The exceptions are opt-in and bounded: the
`gallery` metric keeps a handful of slices (`gallery_samples`), and `--cache` writes
selected cubes to disk rather than holding them.

**Numpy core, optional extras.** `import eval21` imports numpy only (scipy is imported lazily
by the helpers that use `ndimage`). Torch appears
inside `eval21.adapters.torch` and `eval21.adapters.fno21cm`; h5py inside the HDF5 readers and
`eval21 build`; matplotlib inside `eval21.visualization`. Missing extras produce an actionable
`ModuleNotFoundError` at the point of use, not at import time.

**One geometry per run.** A run evaluates samples that share a shape, box size and redshift
grid. The runner compares `(shape, box_mpc, z_grid)` on every sample and stops on a mismatch,
because per-sample geometry would silently mix incompatible power spectra and histories.

**Fail loudly, skip with a reason.** Shape mismatches, missing channels, non-finite predictions,
unknown metrics and unknown settings raise. A metric that *cannot* run — no density field, a
T_b pass for an x_HI statistic, a figure whose metric was not enabled — is recorded in
`skipped` with a human-readable reason instead of disappearing.

**Same numbers, same figures.** Settings live in one frozen dataclass and are written into the
report, so two runs can be compared field by field. The figure package reads only the report
and the optional cube store; it never recomputes a metric.

## Control flow of `eval21 run`

1. `cli._run` merges the TOML config with CLI flags. Precedence: explicit CLI flag > config >
   default. `--target` and `--density` set `Settings.target` / `Settings.density` as well as the
   data fields.
2. `eval21.data.open_data` builds a dataset from the path (`npz`/`hdf5`/`cubes`/`synthetic`),
   applies the row selection (`--indices`, `--indices-json`) and the split filter.
3. For each model: `adapters.build(spec)` constructs the adapter, then `runner.evaluate`:
   - resolves metric names against `eval21.registry` and instantiates them;
   - reads the first sample, checks each metric's declared needs and target, drops the
     pointwise metrics in `--mode generative`, and records the reasons in `skipped`;
   - iterates the dataset: validate targets → check shape → predict (`predict_draws` handles
     `--draws`) → validate the prediction → optional x↔y re-run for `symmetry` → update every
     metric → optionally save cubes to the store → time the forward pass;
   - after the pass, runs the adapter's dedicated `benchmark()` if it has one, merges the
     post-loop `describe()` info, and collects `metric.arrays()` for the run file (or `curves.npz`);
   - closes the adapter (frees GPU memory between models).
4. `report.write` serialises the payload, flattens scalars to CSV, builds the long-form stage
   table and renders the Markdown report.

`eval21 build` and `eval21 plot` are separate entry points: `build` writes HDF5 caches from
simulation folders, `plot` reads result directories.

## Package map

```text
src/eval21/
├── __init__.py          # public surface: Settings, Sample, Meta, evaluate, __version__
├── __main__.py          # python -m eval21
├── cli.py               # argparse; run / build / describe / list-metrics / selftest / demo / plot
├── spec.py              # Sample, Meta, check_prediction, check_targets
├── settings.py          # one frozen dataclass for every knob
├── registry.py          # metric plugins by name
├── runner.py            # the streaming evaluation loop
├── report.py            # JSON / CSV / NPZ / Markdown writers
├── store.py             # PredictionStore: float16 truth + prediction cubes
├── physics.py           # cosmology, spectra, MFP rays, fronts, histograms
├── selftest.py          # analytic self-checks
├── adapters/            # python, torch, command, fno21cm, toy; BaseAdapter/Protocol
├── data/                # npz (+manifest), HDF5 fno-21cm cache, eval21 cube cache, synthetic
├── build/               # simulation folder → eval21 cube cache (HDF5)
├── metrics/             # one module per metric family, all streaming accumulators
└── visualization/       # figure registry, data loading, 23 figures, README collage
```

## Object model

| object | module | responsibility |
|---|---|---|
| `Settings` | `settings.py` | frozen dataclass of every physical and numerical knob; `from_dict`/`to_dict`/`updated` |
| `Sample` | `spec.py` | named inputs (3-D or `(C, Nx, Ny, Nz)`), named 3-D targets, `Meta`, `split` |
| `Meta` | `spec.py` | `box_mpc`, `z_grid`, `omega_m`, `params`; `None` means "not provided" |
| `Dataset` | `data/base.py` | protocol: `__iter__`, `__len__`, `describe()`; `BaseDataset` handles split/limit/indices |
| `BaseAdapter` | `adapters/base.py` | `predict(sample)`, optional `describe()`, `close()`, `benchmark()` |
| `Metric` | `metrics/base.py` | `update(sample, prediction, aux, runtime)`, `result()`, `arrays()`, declared `needs`/`targets`/`pointwise` |
| `ModelResult` | `runner.py` | `name`, `info`, `metrics`, `skipped`, `arrays`, `seconds`, `n_samples` |
| `PredictionStore` | `store.py` | writer during a run, reader for figures; `index.json` + float16 npz |
| `Run` / `Merged` / `Catalog` | `visualization/common.py` | result directories indexed by target, data group and model |
| `PlotContext` | `visualization/registry.py` | everything a figure function receives |

## Stage binning

Most metrics are stage-resolved. Stages are bins of the **true** x_HI slice mean
(`stage_field`, default `xhi`), with edges `stage_edges = (0.02, 0.2, 0.4, 0.6, 0.8, 0.98)`
plus a trailing **active** pseudo-stage (`active_range = (0.05, 0.95)`). Because staging uses
truth, truth and prediction are always paired, and an x_HI pass and a T_b pass of the same
cones use identical slices.

```text
stage labels:  xbar_HI 0.02-0.20 … 0.80-0.98, then "active 0.05-0.95"
masks:         slices whose true transverse mean falls in the bin
min slices:    min_stage_slices (default 2); thinner stages are dropped/marked NaN
```

A slice whose mean sits between two stage edges is not counted twice: the masks are disjoint
and derived from `stage_masks` in `physics.py`.

## Pointwise vs distributional

`Metric.pointwise` drives `--mode generative`:

| value | meaning |
|---|---|
| `True` | compares prediction and truth voxel by voxel; skipped entirely in generative mode |
| `"partial"` | mostly distributional, but some keys are pointwise; those keys are stripped in generative mode and listed under `_generative_dropped` |
| `False` | distributional only (histograms, power ratios, histories) |

`Metric.targets` restricts a metric to x_HI targets (`history`, `bubble`, `morphology`,
`front`, `observable`); a T_b run skips them with a reason. `Metric.needs` declares required
input fields (`density` for the physics metrics); missing fields skip the metric, so a suite
still runs on targets alone.

## Stochastic and generative models

`--draws N` makes the runner ask for N predictions per sample, in this order of preference:

1. `adapter.predict_draws(sample, n)` → `{target: (n, Nx, Ny, Nz)}`;
2. a target returned by `predict` with shape `(n, Nx, Ny, Nz)`;
3. N calls of `predict(sample, draw=i)` (or repeated `predict(sample)` for a sampler with
   internal state).

The standard metrics score one realisation (`score_draw="first"`) or the ensemble mean
(`"mean"`); the `ensemble` metric additionally receives every draw in `aux["draws"]`. `--mode
generative` disables the voxel-by-voxel comparisons and keeps the distributional diagnostics.
See [Metrics](metrics.md#ensemble) and [Adapters](adapters.md#draws).

## Extension points

| to add … | do this | documented in |
|---|---|---|
| a model | write `predict(sample)` in a file, or subclass `BaseAdapter` | [Adapters](adapters.md) |
| a metric | subclass `Metric`, decorate with `@register("name")`, import the module before the run | [Development](development.md#adding-a-metric) |
| a dataset | subclass `BaseDataset`, yield `Sample`s, add a branch to `open_data` | [Development](development.md#adding-a-dataset) |
| a figure | one `@figure(...)`-decorated function in `eval21.visualization`, listed in `FIGURE_MODULES` if it lives in a new module | [Visualization](visualization.md#adding-a-figure) |
| a build source | class with `items()` and `read(item, fields)`, referenced as `python:file.py:Class` | [Data](data.md#custom-build-sources) |
