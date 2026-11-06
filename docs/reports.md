# Reports and run files

`eval21 run` writes **one HDF5 file per model** in `--out`:

```text
results/
├── fno.xhi.h5            # model fno, target x_HI
└── ufno.xhi.h5
```

The file is schema **`eval21-run/1`**. It contains the metric results, the per-sample arrays the
figures need, the dataset description and the settings — but **no cubes, CSV tables or rendered
report**. Those are either derived on demand or stored in the optional inference cache.

```bash
eval21 run --data cubes.h5 --model fno=fno21cm:ckpt.pt --out results/
# results/fno.xhi.h5

eval21 validate results/fno.xhi.h5 --strict
```

Naming is `<sanitized model>.<sanitized target>.h5`. A single model may name the file directly
(`--out results/fno.h5`); that is an error with several `--model` entries. Existing files are kept
unless `--overwrite` is passed.

## Reading a run file

`open_run` accepts a run file and a legacy run directory, so downstream code does not branch on
the storage format:

```python
from eval21.io import open_run

run = open_run("results/fno.xhi.h5")
run.schema                    # "eval21-run/1"
run.model, run.target         # "fno", "xhi"
run.settings                  # Settings.to_dict()
run.dataset_info              # the dataset's describe()
run.n_samples, run.sample_ids
run.metric_names()
run.result("voxel")           # the metric's result() dict
run.metric_meta("voxel")      # needs / targets / pointwise
run.arrays("spectrum")        # {"spectrum_k": ..., "spectrum_ratio_...": ...}
run.curves()                  # {"fno": {array key: array}} (legacy-compatible shape)
run.as_results()              # the old results.json payload, one model
run.validate(strict=True)     # [] when the file is consistent
```

A raw `h5py` reader needs no eval21 code:

```python
import h5py, json

with h5py.File("results/fno.xhi.h5") as h5:
    manifest = json.loads(h5["manifest"][()])
    print(manifest["metric_groups"])                 # 20 metrics
    print(h5["metrics/spectrum/arrays/spectrum_k"].attrs["dims"])   # "k"
```

## Layout

```text
fno.xhi.h5
├── /                  attrs: schema, payload_schema, created, producer, eval21_version,
│                             model, target
├── /manifest          UTF-8 JSON: ordered list of every dataset (path, kind, dtype,
│                             shape, dims, units), metric groups, n_samples
├── /meta/
│   ├── environment    JSON: python, numpy, platform, host
│   ├── versions       JSON: eval21, numpy, python, model framework
│   ├── command        exact argv
│   └── settings       JSON: Settings.to_dict()
├── /dataset/
│   ├── info           JSON: the dataset's describe()
│   └── z_grid         (Nz,) float64, when the data declares one
├── /samples/
│   ├── ids            (N,) UTF-8
│   ├── splits         (N,) UTF-8, "" = undeclared
│   ├── omega_m        (N,) float64, NaN = not provided
│   └── params/
│       ├── names      (P,) UTF-8
│       └── values     (N, P) float64, NaN = absent
├── /model/
│   ├── info           JSON: adapter.describe() + runner post-processing
│   └── meta           JSON: n_samples, seconds, skipped metrics, metric metadata
└── /metrics/<name>/
    ├── result         UTF-8 JSON: the metric's result() dict (NaN -> null)
    ├── meta           UTF-8 JSON: needs, targets, pointwise
    └── arrays/
        └── <key>      dataset with attrs dims / units
```

Array keys are metric-local (`spectrum_k`, `gallery_xhi_truth`), not prefixed with the metric
name; the group carries that. `dims` is a comma-separated best-effort axis vocabulary
(`sample,stage,chunk,k,kperp,kpar,x,y,z,bin,radius,distance`), `units` is `Mpc^-1`, `mK`,
`redshift`, `count`, `s` or `""`. Arrays larger than 1 MB use gzip-4 + shuffle; everything uses
HDF5's stable subset (`libver="earliest"`) and writes to a temporary file that is atomically
renamed.

`result` uses the same JSON conventions as the legacy `results.json`: non-finite floats become
`null`, so strict JSON parsers work.

## Legacy directory (`--legacy-dir`)

During the transition, `--legacy-dir` writes the previous five-file layout:

| file | content |
|---|---|
| `results.json` | complete report, schema `eval21/1`, all models |
| `summary.csv` | every scalar leaf as one dotted column, one row per model |
| `stages.csv` | long-form stage table: model, metric, group, stage, key, value |
| `report.md` | headline table plus per-metric stage tables |
| `curves.npz` | per-sample and per-stage arrays for figures and custom analysis |

`stages.csv` and `curves.npz` are omitted when empty. `open_run(directory)` reads this layout;
`eval21 plot` accepts both. CSV tables and the Markdown report are **not** rendered for run
files yet — use `--legacy-dir` if you need them, or read `as_results()` and build your own.

## What is in `/metrics`

Every registered metric contributes one group. `result` is the same dict the metric's
`result()` returns (see [Metrics](metrics.md)); the arrays are the per-sample exports that
figures read. A metric that was skipped is absent from `/metrics` and listed in
`/model/meta/skipped` with the reason. A new metric is a new group: old readers ignore it and a
missing group is not an error.

## Comparing runs

Two runs are comparable when their settings agree on the fields that affect the metric:
`box_mpc`, `omega_m`, `stage_edges`, `active_range`, `threshold`, `n_k_bins`, `k_targets`,
`eft_k_max`, `coherence_level`, `highk_*`, and the front and bubble knobs. Every run file records
all of them:

```python
from eval21.io import open_run

a, b = open_run("run_a/fno.xhi.h5"), open_run("run_b/fno.xhi.h5")
diff = {k: (a.settings[k], b.settings[k]) for k in a.settings
        if a.settings[k] != b.settings[k]}
```

Redshift-window runs (LoReLi groups) are intentionally *not* comparable sample by sample; pass
several run files or directories to `eval21 plot` and let it pool them as groups.

## Optional cube cache

Map figures need cubes, which are deliberately not in the run file. `--cache` writes them to a
separate `eval21-cache/1` HDF5 file:

```bash
eval21 run ... --out results/ --cache results/fno.xhi.cache.h5 --save-n 8
eval21 plot maps --run results/ --out figs/ --store results/fno.xhi.cache.h5
```

The cache holds `float16` truth and prediction cubes for the selected samples; it is regenerable
and never needed to reproduce a statistic. The layout is documented in
[Data](data.md#inference-cache). `--save-predictions` still writes the legacy directory store;
an `.h5` path passed to it is treated as a cache.

## Legacy `results.json` payload

`run.as_results()` returns exactly this structure (one model per run file):

```json
{
  "schema": "eval21/1",
  "created": "2026-11-04T18:13:00+00:00",
  "environment": {"python": "3.13.5", "platform": "Linux-...", "numpy": "2.5.4", "host": "..."},
  "dataset": { "...": "the dataset's describe() dict" },
  "settings": { "...": "Settings.to_dict()" },
  "models": {
    "fno": {
      "info": {"framework": "fno-21cm", "parameters": 1234567, "device": "cuda:0"},
      "n_samples": 200,
      "seconds": 812.4,
      "skipped": {"morphology": "missing input fields: ..."},
      "metrics": {"voxel": {"...": "..."}, "spectrum": {"...": "..."}}
    }
  }
}
```

Numbers are strict JSON: NumPy scalars are converted, arrays become lists, and non-finite floats
become `null`. `settings` records the run, not the defaults.

The legacy `summary.csv`, `stages.csv`, `report.md` and `curves.npz` semantics are unchanged and
documented by the legacy writer (`eval21/report.py`). `curves.npz` keys are
`<model>__<metric>__<short name>`; `eval21.visualization.common.Run.curves(model)` strips the
prefixes and returns `{short name: array}`.
