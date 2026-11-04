# Reports

`eval21 run` writes five files to `--out`:

| file | content |
|---|---|
| `results.json` | the complete report, schema `eval21/1` |
| `summary.csv` | every scalar leaf as one dotted column, one row per model |
| `stages.csv` | long-form stage table: model, metric, group, stage, key, value |
| `report.md` | headline table plus per-metric stage tables |
| `curves.npz` | per-sample and per-stage arrays for figures and custom analysis |

`stages.csv` and `curves.npz` are omitted when their content is empty (and a stale file from an
earlier run is removed). Everything is written for every model in one pass, so comparing models
never means merging files by hand.

## `results.json`

```json
{
  "schema": "eval21/1",
  "created": "2026-11-04T18:13:00+00:00",
  "environment": {"python": "3.13.5", "platform": "Linux-...", "numpy": "2.5.4", "host": "..."},
  "dataset": { "...": "the dataset's describe() dict" },
  "settings": { "...": "Settings.to_dict()" },
  "models": {
    "fno": {
      "info": {"framework": "fno-21cm", "parameters": 1234567, "device": "cuda:0", "...": "..."},
      "n_samples": 200,
      "seconds": 812.4,
      "skipped": {"morphology": "missing input fields: ..."},
      "metrics": {"voxel": {"...": "..."}, "spectrum": {"...": "..."}}
    }
  }
}
```

| top-level key | contents |
|---|---|
| `schema` | report schema; currently `eval21/1` |
| `created` | ISO-8601 UTC timestamp (seconds) |
| `environment` | python, platform, numpy, host |
| `dataset` | `describe()` of the evaluated selection: format, path, shape, box, z range, fields, splits, excluded |
| `settings` | the full `Settings`, so two reports can be compared field by field |
| `models` | one entry per `--model`, in the order given |

Per model:

| key | contents |
|---|---|
| `info` | `describe()` merged with runner post-processing: `framework`, `parameters`, `device`, `checkpoint`, `draws`, `mode`, `score_draw`, `draws_identical`, `geometry_check`, … |
| `n_samples` | samples evaluated |
| `seconds` | wall-clock of the whole pass, including metric updates |
| `skipped` | metric → reason for metrics that could not run |
| `metrics` | metric name → the metric's `result()` dict, in registration order |

Numbers are strict JSON: NumPy scalars are converted, arrays become lists, and non-finite floats
become `null`. `settings` records the run, not the defaults — a reader can reproduce every
choice.

## `summary.csv`

One row per model. Columns are the dotted leaves of `info` and every scalar in every metric:
`model`, `info.framework`, `info.parameters`, `voxel.rmse`, `voxel.r2`,
`spectrum.active.coherence_ref`, `history.tau.error_sigma_med`, `efficiency.ms_per_sample`, …

The column set is the union over models, so a model missing a metric gets an empty cell rather
than shifting the table. Arrays, lists and non-finite values are skipped. This is the file to
load with pandas for a model-comparison table:

```python
import pandas as pd
df = pd.read_csv("results/compare/summary.csv", index_col="model")
df[["voxel.rmse", "voxel.r2", "spectrum.active.coherence_ref"]].sort_values("voxel.rmse")
```

## `stages.csv`

Long form, built from every metric's `stages` and `chunks` lists:

```text
model,metric,group,stage,key,value
fno,spectrum,stages,xbar_HI 0.40-0.60,coherence_med,0.87
fno,cylindrical,stages,x̄≈0.5,within_dead_band,0.42
fno,observable,stages,active 0.05-0.95,power_bias_significance_dtb,3.1
```

| column | meaning |
|---|---|
| `model` | model name |
| `metric` | metric name |
| `group` | `stages` or `chunks` |
| `stage` | the entry's `label`, else `z`, else its index |
| `key` | dotted leaf name inside the entry |
| `value` | scalar value |

Auxiliary arrays inside a stage entry (`k_targets`, `delta2_k_targets`, `kperp_centers`,
`kpar_centers`) are not exploded into rows; they live in `results.json` and `curves.npz`.

This is the file for "metric X versus stage" plots that are not part of the figure registry:

```python
df = pd.read_csv("results/stages.csv")
spec = df[(df.metric == "spectrum") & (df.key == "coherence_med")]
```

## `curves.npz`

Named arrays, no pickles (`allow_pickle=False`). Keys are

```text
<model>__<metric>__<short name>
```

where `<short name>` is the metric's array key with its own family prefix. Examples:

| key | shape | meaning |
|---|---|---|
| `fno__voxel__rmse_per_sample` | `(N,)` | per-sample RMSE |
| `fno__voxel__r2_per_sample` | `(N,)` | per-sample R² |
| `fno__spectrum__spectrum_k` | `(K,)` | k centres, 1/Mpc |
| `fno__spectrum__spectrum_ratio_active_0p05_0p95` | `(N, K)` | power ratio per sample, active stage |
| `fno__spectrum__spectrum_coherence_xbar_HI_0p40_0p60` | `(N, K)` | coherence per sample |
| `fno__bubble__bubble_mass_truth_active` | `(N, R)` | MFP histograms |
| `fno__history__history_z` | `(Nz,)` | redshift grid of the histories |
| `fno__gallery__gallery_xhi_truth` | `(S, G, H, W)` | stored truth slices |
| `fno__cylindrical__cyl_ratio` | `(N, C, Kpar, Kperp)` | per-sample cylindrical maps |
| `fno__cylindrical__cyl_ratio_median` | `(C, Kpar, Kperp)` | pooled median maps |
| `fno__ensemble__rank_histogram` | `(D+1,)` | rank histogram |

The exact set depends on the metrics that ran and on the data (stage labels encode the edges).
Print the inventory rather than guessing:

```python
import numpy as np
with np.load("results/curves.npz") as d:
    for key in d.files:
        print(f"{key:60s} {d[key].shape}")
```

`eval21.visualization.common.Run.curves(model)` loads the file and strips the model and metric
prefixes, giving `{short name: array}` per model. That is how the figures read it.

## `report.md`

A human-readable digest:

1. title, creation time, sample scope, shape, box, data path and z range;
2. headline table: RMSE, R², r(k) at 0.2 1/Mpc, k_r90, |Δτ|/σ, z50 error in σ, front-width
   excess, bubble W1, partial-fraction excess, ms/sample, parameter count (columns appear only
   when a model reports them);
3. the skipped metrics and reasons;
4. per model: framework/device/checkpoint line, then per metric either a stage table
   (`stage | n_slices | …`) or a key/value table.

Values are formatted compactly: integers with thousands separators, floats with four
significant digits, tiny/huge floats in scientific notation, long lists as
`med […] [min, max]`, `—` for undefined. The Markdown report is deliberately lossy; the JSON is
the record.

## Prediction store (optional, for figures)

With `--save-predictions DIR` the run also writes float16 truth and prediction cubes for a few
samples:

```text
DIR/index.json
DIR/truth/<id>.npz          t_<field>, i_<field>, z_grid, box_mpc, omega_m, params
DIR/<model>/<id>.npz        <field>, and with --draws <target>__draws
```

The layout and merging rules are documented in [Data](data.md#prediction-store). The store is
input to `eval21 plot` only; it is never needed to reproduce a number in the report.

## Comparing runs

Two runs are comparable when their `settings` agree on the fields that affect the metric:
`box_mpc`, `omega_m`, `stage_edges`, `active_range`, `threshold`, `n_k_bins`, `k_targets`,
`eft_k_max`, `coherence_level`, `highk_*`, the front and bubble knobs. `eval21 run` records all
of them, so a quick check is:

```python
import json
a, b = (json.load(open(p))["settings"] for p in ("run_a/results.json", "run_b/results.json"))
diff = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
```

Redshift-window runs (LoReLi groups) are intentionally *not* comparable sample by sample; pass
several result directories to `eval21 plot` and let it pool them as groups.
