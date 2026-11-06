# `eval21-run/1`: the single-file eval format

**Status:** proposed, not implemented. This is the spec to implement against.
**Scope:** one evaluation of **one model** on one target over one dataset. The inference cache is
a separate, optional file (`eval21-cache/1`).

## Decision

**HDF5** (`.h5`, h5py), one file per `(model, target, dataset)` evaluation, with a versioned
schema and a JSON manifest. Cubes are not in this file; they go to an optional
`eval21-cache/1` HDF5 file written with `--cache`.

Why, given the constraints:

- Python-only, local, single-writer: h5py is already a dependency and already used by
  `eval21 build` and the prediction store.
- No cubes in the run file: every payload is small-to-medium (metrics JSON, curves), so
  chunking is simple and files stay shareable (tens to a few hundred MB for a 200-cone sweep).
- One file per model: independent writes, no cross-model schema coupling, a file can be
  copied/archived alone, and the visualisation layer pools files exactly as it pools
  directories today.
- Long-lived and modular: groups are additive. A new metric is a new `/metrics/<name>` group;
  old readers ignore it, new readers handle its absence. No schema bump.
- HDF5's stable subset (groups, chunked datasets, gzip+shuffle, attributes, UTF-8 strings) is
  readable from Python forever and does not require plugins.

Rejected for this file: NetCDF-4 (adds CF rules that do not fit ragged metric dicts and a
heavier dependency for no interop gain in a Python-only setting), Zarr (parallel/cloud
advantages unused; zip store is a distribution format), Parquet bundle (multi-file, cubes and
per-sample arrays awkward), NPZ (flat namespace, no manifest, no longevity story), zip of the
old directory (no partial reads; kept only as an export).

## What is in the file, what is not

| payload | in `eval21-run/1` | why |
|---|---|---|
| dataset description (`describe()`) | yes, `/dataset/info` | identifies the data a run scored |
| `Settings` | yes, `/meta/settings` | makes two runs comparable |
| environment, command line, versions | yes, `/meta/*` | reproducibility |
| model info (`describe()`: framework, parameters, checkpoint, geometry check) | yes, `/model/info` | one model per file |
| metric results (ragged dicts, NaN) | yes, `/metrics/<name>/result` | canonical, exactly `results.json` semantics |
| per-sample arrays (spectra, PDFs, cyl maps, gallery, …) | yes, `/metrics/<name>/arrays/*` | the figures' input |
| sample ids, splits, Ω_m, parameters | yes, `/samples/*` | aligns every per-sample array |
| `summary.csv` / `stages.csv` | **no** — derived | `eval21 export` regenerates them |
| `report.md` | **no** — derived | `eval21 report FILE` renders it |
| truth/prediction cubes | **no** — separate cache | optional, large, regenerable |
| figures | no | outputs, not data |

Keeping only canonical data in the file avoids the current duplication between `results.json`,
`curves.npz`, `summary.csv` and `stages.csv` drifting apart.

## Layout

```text
run.h5
├── /                                  attrs: schema, payload_schema, created, producer,
│                                             eval21_version, git_commit, model, target
├── /manifest                          scalar UTF-8 JSON: index of every group/dataset
├── /meta/
│   ├── environment                    JSON: python, numpy, platform, host
│   ├── versions                       JSON: eval21 + model framework versions
│   ├── command                        string: exact argv
│   └── settings                       JSON: Settings.to_dict()
├── /dataset/
│   ├── info                           JSON: the dataset's describe() dict
│   └── z_grid                         (Nz,) float64
├── /samples/
│   ├── ids                            (N,) UTF-8
│   ├── splits                         (N,) UTF-8, "" = undeclared
│   ├── omega_m                        (N,) float64, NaN = not provided
│   └── params/
│       ├── names                      (P,) UTF-8
│       └── values                     (N, P) float64, NaN = absent
├── /model/
│   └── info                           JSON: adapter.describe() + runner post-processing
└── /metrics/
    └── <metric>/
        ├── result                     scalar UTF-8 JSON: the metric's result() dict
        └── arrays/
            ├── <array>                dataset, attrs dims / units / description
            └── …
```

### Root attributes

| attr | example | meaning |
|---|---|---|
| `schema` | `eval21-run/1` | container layout; readers refuse unknown **major** |
| `payload_schema` | `eval21/1` | metric-result semantics (same as `results.json` today) |
| `created` | `2026-11-04T20:00:00+00:00` | UTC, seconds |
| `producer` | `eval21 0.1.0` | writer |
| `eval21_version` | `0.1.0` | duplicated for convenience |
| `git_commit` | `01b3269` | of the eval21 checkout, if available |
| `model` | `fno` | the single model in this file |
| `target` | `xhi` | the scored field |

### `/manifest`

A single JSON string, written last, describing the file. It is the contract for generic readers:

```json
{
  "schema": "eval21-run/1",
  "payload_schema": "eval21/1",
  "model": "fno",
  "target": "xhi",
  "datasets": {
    "/dataset/z_grid": {"kind": "array", "dtype": "float64", "dims": ["z"], "units": "redshift"},
    "/metrics/spectrum/arrays/spectrum_k": {"kind": "array", "dtype": "float64",
                                            "dims": ["k"], "units": "Mpc^-1"},
    "/metrics/spectrum/arrays/spectrum_ratio_active_0p05_0p95": {"kind": "array",
        "dtype": "float64", "dims": ["sample", "k"], "units": ""}
  },
  "metric_groups": ["anisotropy", "bubble", "spectrum", "…"],
  "n_samples": 200,
  "n_metrics": 20
}
```

Readers can ignore `/manifest` and walk groups; it exists so tools that do not know the schema
can enumerate dimensions and units without heuristics.

### `/metrics/<name>`

- `result` is the exact `result()` dict, serialised with `json.dumps(..., allow_nan=False)`;
  non-finite floats are `null` (the current `report.jsonable` convention). This keeps the
  payload identical to today's `results.json`.
- `arrays/*` are the per-sample exports, with their existing key names
  (`spectrum_k`, `cylindrical__cyl_ratio`, `gallery_xhi_truth`, …) so the current figure code
  changes only where it opens the file.
- Every array carries:
  - `dims`: comma-separated axis names, e.g. `sample,k` or `sample,stage,x,y`;
  - `units`: `Mpc^-1`, `mK`, `""` for dimensionless;
  - `description`: one line for humans.

Axis names form a small controlled vocabulary: `sample`, `stage`, `chunk`, `k`, `kperp`,
`kpar`, `x`, `y`, `z`, `bin`, `draw`.

### Why no tables or report in the file

`summary.csv`, `stages.csv` and `report.md` are projections of `/metrics` and `/model`. Storing
them duplicates data and invites divergence. They are produced on demand:

```bash
eval21 export run.h5 --format csv --out tables/       # summary.csv + stages.csv
eval21 report run.h5 --out report.md
eval21 export run.h5 --format legacy-dir old_run/     # results.json + curves.npz + CSVs
```

The last command is the compatibility bridge: an old consumer that expects a run directory gets
one reconstructed from the file.

## Dataset rules

| data | dtype | chunking | compression |
|---|---|---|---|
| JSON blobs | variable-length UTF-8 | scalar | none (small) |
| ids, splits, names | variable-length UTF-8 | contiguous | none |
| z_grid, omega_m, params | float64 | contiguous | none |
| metric arrays | as computed (usually float64/float32) | contiguous | gzip 4 if > 1 MB |
| cache cubes | float16 | `(1, Nx, Ny, 16)` | gzip 4 + shuffle |

- Use only stable HDF5 features (`libver="earliest"`), no filter plugins, no virtual
  datasets, no SWMR. This is the longevity guarantee.
- One writer, many readers after close.
- Write to `<path>.tmp`, `flush()`, `os.replace()`; never write in place.
- Strings are UTF-8; no pickle anywhere.
- Sizes are small enough that per-dataset checksums are not required in v1; `fletcher32` is
  reserved as an option for archival exports.

## Reader API

```python
from eval21.io import open_run

run = open_run("results/fno.xhi.h5")     # also accepts a legacy run directory
run.schema            # "eval21-run/1"
run.model, run.target # "fno", "xhi"
run.info              # dict (model info)
run.settings          # dict
run.metric_names()    # ["anisotropy", "bubble", ...]
run.result("spectrum")            # dict, identical to results.json
run.arrays("spectrum")            # {"spectrum_k": ndarray, ...}
run.array("spectrum", "spectrum_k")
run.curves()                      # {metric: {array: ndarray}}
run.sample_ids()                  # (N,) str
run.n_samples
run.validate(strict=False)        # schema/manifest consistency
```

`open_run` dispatches on the path: an HDF5 file with a `schema` attribute opens the new reader;
a directory with `results.json` opens a legacy adapter exposing the same interface. That keeps
`eval21 plot` working on old results without branching in the figure code.

## Versioning and compatibility

Two versions, deliberately independent:

- **Container** `eval21-run/<major>.<minor>`: layout, required groups, dataset conventions.
- **Payload** `eval21/<major>`: the metric-result dict semantics (today `eval21/1`).

Policy:

1. **Additive changes do not bump the schema.** A new metric (`/metrics/mystat/...`), a new array
   under an existing metric, a new optional attribute: old readers ignore them; new readers
   handle their absence. This is the modularity requirement.
2. **A new required group or a changed dataset convention bumps the minor** (`1.0` → `1.1`).
   Readers accept any `1.x` and warn on a newer minor than they know.
3. **A renamed or removed path bumps the major** (`2.0`). The reader keeps an adapter for the
   previous major for at least one release.
4. **Readers must ignore unknown groups, datasets and root attributes.** Never infer semantics
   from file order or from the presence of an unrelated group.
5. **Writers always write a complete manifest** and the exact schema version they implement.
6. **Legacy directories remain readable** through the adapter; `--legacy-dir` keeps the old
   writer for one transition release, and `eval21 export --format legacy-dir` reconstructs one.

A metric is not allowed to change the meaning of an existing result key without a payload-major
bump; adding keys is fine.

## Inference cache: `eval21-cache/1`

Separate HDF5 file, optional, written only with `--cache` (or `--save-predictions` mapped to
it). It contains the cubes map figures need and nothing else.

```text
run.xhi.cache.h5
├── /                         attrs: schema="eval21-cache/1", model, target, created,
│                                    box_mpc, z_grid_len, n_samples
├── /ids                      (N,) UTF-8
├── /z_grid                   (Nz,) float64
├── /truth/<field>            (N, Nx, Ny, Nz) float16
├── /predictions/<field>      (N, Nx, Ny, Nz) float16
├── /draws/<field>            (N, D, Nx, Ny, Nz) float16, optional
└── /index                    JSON: samples, fields, draws, dtype, chunks
```

- float16 like today's store (20.07 MB per `140×140×512` field), chunked along z so one slice
  reads one chunk group, gzip + shuffle.
- Truth and prediction are both present; a cache is self-contained for map figures.
- Regenerable; it is not part of the archival record and not required for statistics.
- Samples are the current selection (`--save-n`, `--save-ids`), default 8.
- If several models should share truth, that is a future `/models/<name>` layout; v1 keeps one
  cache per eval for simplicity and independent writes.

## CLI surface

```bash
# one file per model; a directory --out receives <model>.<target>.h5
eval21 run --data cubes.h5 --model fno=fno21cm:ckpt.pt --out results/
#   -> results/fno.xhi.h5

# a single model may name the file directly
eval21 run --data cubes.h5 --model fno=fno21cm:ckpt.pt --out results/fno.h5

# optional cube cache next to it
eval21 run ... --out results/ --cache results/fno.xhi.cache.h5 --save-n 8

# old layout during the transition (one release)
eval21 run ... --out results/ --legacy-dir

# verification and migration
eval21 validate results/fno.xhi.h5
eval21 convert results_old/ --out results_new/
eval21 export results/fno.xhi.h5 --format csv --out tables/
eval21 export results/fno.xhi.h5 --format legacy-dir results_old_reconstructed/

# plotting accepts files, directories of files, and legacy directories
eval21 plot all --run results/ --out figs/
```

Naming rule: `<sanitized model>.<target>.h5`, e.g. `fno.xhi.h5`, `ufno.xhi.h5`,
`8blk_pointwise_rhizome.xhi.h5`. An explicit `--out file.h5` is an error with more than one
model. Existing files are not overwritten unless `--overwrite`.

## Validation

`eval21 validate FILE [--strict]` checks:

- `schema` major is known; `payload_schema` known;
- required groups exist (`manifest`, `meta`, `dataset`, `samples`, `model`, `metrics`);
- `len(/samples/ids) == /manifest n_samples`;
- every `/metrics/*/arrays/*` dataset has `dims`, `units` and a first axis of length `N` when
  `sample` is in its dims;
- every JSON blob parses and contains no NaN literals;
- `manifest.datasets` agrees with the actual tree (strict: no unlisted datasets).

The schema itself ships as `eval21/schemas/eval21-run-1.json` (JSON Schema for the manifest and
the required attributes), so external validators can use it too.

## Impact on the rest of the repo

| area | change |
|---|---|
| `runner.evaluate` | returns the same `ModelResult`; the CLI writes one file per result instead of one directory per multi-model run |
| `report.py` | becomes a serialiser: `write_run(path, result, ...)` plus `render_markdown` reading the file |
| `store.py` | becomes the `eval21-cache/1` writer/reader (`open_cache`, `slice`, `truth`, `prediction`) |
| `visualization/common.Run` | opens a file or a legacy directory via `open_run`; `find_runs` accepts `*.h5` and `results.json` |
| `eval21 plot` | `--run` accepts files/dirs; `--store` becomes `--cache` (alias kept) |
| docs | `reports.md` and `data.md` describe the file; `visualization.md` the reader; `cli.md` the new flags |

The figure code itself is untouched except for how it obtains `Run.curves` and the cache.

## Migration plan

1. **I/O layer** (`eval21/io/`): `runfile.py` (writer/reader), `cache.py`, `schema.py`,
   `convert.py`; round-trip, legacy-adapter and validator tests.
2. **Writer**: `eval21 run` writes `<model>.<target>.h5` per model; `--legacy-dir` preserves the
   old output; `eval21 validate`.
3. **Reader**: `open_run` dispatches file/directory; `visualization` uses it; `eval21 plot` reads
   both; `--cache` replaces `--store` (alias).
4. **Compatibility**: `eval21 convert`, `eval21 export --format legacy-dir`, docs and handoff.
5. **Default flip**: new file is the default; legacy directory behind the flag; announce in the
   README and handoff.

## Explicit non-goals

- Multiple models in one file. A collection is a directory of files; the visualisation layer
  already pools `(target, data, model)`.
- Cubes in the run file.
- Figures, reports or CSV tables in the run file (derived on demand).
- Cloud/object-store access, parallel writers, browser readers.
- Compression plugins beyond gzip/shuffle.
