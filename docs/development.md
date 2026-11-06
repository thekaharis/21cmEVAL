# Development

## Environment

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[all]"
python -m pytest              # ~3 minutes, no data, no GPU
eval21 selftest               # the same analytic checks, without pytest
```

Python 3.10+; the package uses `from __future__ import annotations` throughout, so annotations
are not evaluated at import time. Runtime dependencies are numpy and scipy (scipy only for
`ndimage`: connected components, distance transforms, Gaussian smoothing). Torch, h5py,
matplotlib and pytest are optional extras.

## Tests

| file | tests | covers |
|---|---|---|
| `test_adapters.py` | 14 | python/torch/command adapters, specs, draws, cache contracts |
| `test_build_ensemble.py` | 10 | `eval21 build`, sharding/merge, ensemble metric |
| `test_data.py` | 8 | npz/manifest/synthetic readers, splits, selection |
| `test_hdf5.py` | 11 | fno-21cm HDF5 layouts, lazy indexing, `--indices-json` |
| `test_metrics.py` | 26 | every metric family on analytic or synthetic fields |
| `test_npz_target_name.py` | 2 | manifest target naming, stage-field x_HI |
| `test_runfile.py` | 8 | run-file/cache round trip, legacy adapter, CLI defaults, validation, file plotting |
| `test_physics.py` | 16 | τ_e, crossings, dT_b, binning, W1/JS, ray tracing, fronts |
| `test_reference_agreement.py` | 6 | bit-level agreement with fno-21cm (needs `FNO21CM_ROOT`) |
| `test_runner.py` | 17 | one-pass loop, skips, geometry checks, report writers, CLI |
| `test_viz*.py` | 26 | registry, layout helpers, spectra, bubbles, diagnostics, maps, ensembles |

On a machine with numpy, scipy, matplotlib and h5py: **136 passed, 7 skipped**. The skipped
tests are the fno-21cm reference-agreement cases, which require a checkout:

```bash
FNO21CM_ROOT=/work/fno-21cm python -m pytest tests/test_reference_agreement.py
```

That file pins the physics primitives to the reference implementation: τ_e, crossing redshifts,
dT_b, binned power, the WLS fit and mean-free-path ray tracing (same RNG stream and seeds).

### The figure fixture

`tests/conftest.py` builds, once per session, the small end-to-end data set every figure test
draws from — the same pipeline a user runs:

```text
<tmp>/cubes.h5                                 4 synthetic cones, 32 x 32 x 64, box 64 Mpc
<tmp>/results/{sharp,blurry}/{xhi,tb}/        two models scored on x_HI and T_b (--legacy-dir)
<tmp>/results/predictions/                    shared legacy prediction store
<tmp>/results_gen/gen/xhi/                    generative model, --draws 4 --mode generative
```

`sharp` and `blurry` are `SmoothThresholdModel` with σ = 1.0 and 2.5, wrapped to also emit
dT_b; `gen` is `toygen`. The fixture sets `rays_per_slice=8`, `ensemble_voxels=2000` and
`gallery_samples=4` to keep the suite fast. Tests assert on structure and invariants (files
exist, figures are square, axes counts, values finite), not on pixel content.

### Test conventions

- Prefer analytic answers: closed-form RMSE/R², coherence = 1 for identical fields, a ray
  tracer against a half-plane, a sphere's effective radius, a known sigmoid front width.
- A new metric needs a test that distinguishes it from a plausible wrong implementation, not
  merely a smoke test.
- A new figure needs a test that it is registered, runs on the fixture, and writes PNG + PDF.
- Figure tests are slow (matplotlib); keep them on the fixture rather than new data sets.
- Failures should print the offending sample id, metric and field name.

## Code conventions

**Physics in `physics.py`.** Cosmology, spectra, ray tracing, histograms and front widths live
there and are unit-tested directly; metrics call them. A metric that re-derives an integral is
a bug waiting to happen.

**Settings, not constants.** A number that a user might want to change belongs in `Settings`
with a validation rule in `__post_init__` and, if it is a list, an entry in `_TUPLES` so TOML
lists become tuples. Add it to the table in [Metrics](metrics.md#settings).

**Streaming accumulators.** `Metric.update` may keep sums, histograms, per-slice vectors and
per-sample scalars — never a cube. If a metric needs the whole volume (connected components,
front distances), do the work inside `update` and keep the scalar summaries.

**Lazy optional imports.** Import torch/h5py/matplotlib inside the function or adapter that
needs them, with a message that says which extra to install. The core must import without any
of them.

**Fail loudly, skip explicitly.** Wrong shapes, missing fields, non-finite predictions and
unknown names raise with the sample id and the expected value. A metric that cannot run raises
nothing: the runner records it in `skipped` with a reason. Figures raise `SkipFigure`.

**Docstrings carry the physics.** Each metric module opens with what the statistic means, why
it is defined that way, and which reference script it reproduces. Keep that up to date; the
documentation is generated from these docstrings.

**One geometry per run.** The runner enforces it. A dataset that mixes shapes must be split
into separate runs and pooled by the visualization layer as groups.

## Adding a metric

1. Write the module in `src/eval21/metrics/`, subclass `Metric`, decorate with
   `@register("name")`.

```python
"""One-line summary of what this measures.

Why it matters, the formula, and the reference implementation it follows.
"""
from __future__ import annotations

import numpy as np

from ..registry import register
from .base import Metric, finite_or_none


@register("mystat")
class MyStat(Metric):
    needs = ("density",)          # optional
    targets = ("xhi",)            # optional: x_HI-only statistics
    pointwise = False             # True / "partial" / False

    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        self.total = 0.0
        self.count = 0

    def update(self, sample, prediction, aux=None, runtime=None):
        truth = self.target(sample)
        pred = self.prediction(prediction)
        self.total += float(((pred - truth) ** 2).sum())
        self.count += truth.size

    def result(self):
        return {"mse": self.total / max(self.count, 1), "n_voxels": self.count}

    def arrays(self):
        return {}                  # optional: per-sample arrays for curves.npz
```

2. Add the module to the import list in `metrics/__init__.py` so the decorator runs.
   Third-party metrics can instead be imported by a wrapper script that calls
   `eval21.cli.main`.
3. Add a test in `tests/test_metrics.py` with an analytic expectation.
4. Document it in [Metrics](metrics.md) and, if it needs a figure, add one in
   `eval21.visualization`.
5. If the metric declares `requires_rerun = True`, the runner re-runs the adapter on
   `sample.transposed_xy()` and passes the back-transposed prediction in `aux["xy"]` (this is
   how `symmetry` works).

Registration is global and duplicate names raise `ValueError` at import time.

## Adding a figure

See [Visualization § Adding a figure](visualization.md#adding-a-figure). Checklist:

- `@figure(name, group=..., needs=..., help=...)` with a unique name;
- new module → add it to `FIGURE_MODULES` in `visualization/registry.py`;
- raise `SkipFigure("reason")` when a metric or field is absent;
- use `shared_legend` + `model_handles` and `pane_values` for the layout conventions;
- return the written paths; write through `save` so PNG + PDF are produced;
- add a test to the matching `tests/test_viz*.py`;
- add a row to the figure tables in [Visualization](visualization.md) and, if user-facing, to
  the README.

## Adding a dataset

1. Subclass `BaseDataset`, implement `_samples()` (or `__iter__`/`__len__` for a lazy reader)
   and `describe()`; set `Sample.split` when the file declares splits.
2. Provide `row_indices_for_ids(ids)` if the dataset has ids, so `--indices-json` can map
   recorded cone ids rather than stale row numbers.
3. Add a prefix branch to `data/open_data` and a suffix/attribute detection rule if the format
   can be auto-detected.
4. Document the format in [Data](data.md) and add a reader test.

## Adding an adapter

Subclass `BaseAdapter` (or just write a `predict` function for `python:`). Implement
`describe()` with `framework`, `parameters`, `device` and anything else worth recording;
implement `benchmark(samples, warmup, iters)` if fair timings matter (pre-transfer inputs,
synchronise CUDA, discard warmup). `close()` is called between models. See
[Adapters](adapters.md#writing-an-adapter-class).

## Documentation workflow

- `docs/api.md` is generated from source signatures and docstrings:

  ```bash
  python docs/gen_api.py        # rewrites docs/api.md
  ```

  Re-run it whenever a public signature changes; do not hand-edit the generated file.
- The rest of `docs/` is hand-written and should be updated with the code that changes it:
  a new metric → `metrics.md`, a new figure → `visualization.md`, a new flag → `cli.md`.
- `eval21 plot list` is the authority on figure names and groups; keep the tables in sync.
- The README is the front page and the collage is its header
  (`python -m eval21.visualization.collage figs/ --out docs/collage.png`).

## Release checklist

```bash
python -m pytest -p no:warnings           # 136 passed, 7 skipped
eval21 selftest
eval21 demo --out /tmp/eval21-demo
eval21 plot list | wc -l                  # groups + figures, as documented
python docs/gen_api.py
git diff --stat docs/                     # documentation reflects the code
```

Bump `__version__` in `src/eval21/__init__.py` and `version` in `pyproject.toml` together. The
report schema (`eval21/1`) changes only when a reader would break, and then it changes in
`report.SCHEMA` with a note in [Reports](reports.md).
