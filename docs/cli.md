# Command line

The `eval21` console script is a thin front end over the library. Every subcommand is also
reachable as `python -m eval21 <subcommand>`.

```text
eval21 run          evaluate one or more models over a data set
eval21 build        build an eval21 cube cache from a folder of simulations
eval21 describe     summarise a data set (format, shape, splits, fields)
eval21 list-metrics list the registered metrics
eval21 selftest     analytic self-checks, no data required
eval21 demo         write synthetic data and score a toy model
eval21 plot         draw figures from result directories
```

Exit status is `0` on success, `1` for a runtime or validation error (the message is printed
without a traceback), and `2` for a usage error raised by `argparse`.

## Conventions

**`NAME=SPEC` for models.** `--model` is repeatable; the name is what appears in every report
and figure. Repeating a name replaces the earlier spec.

```bash
--model fno=fno21cm:ckpt.pt --model ufno=fno21cm:ckpt2.pt
```

**`KEY=VALUE` for settings and options.** `--set` and `--opt` parse the value as JSON first
(`1` → int, `0.5` → float, `true` → bool, `"x"` → str), then as a comma-separated tuple of
floats, then as a comma-separated tuple of strings. So:

```bash
--set stage_edges=0.02,0.2,0.4,0.6,0.8,0.98
--set n_k_bins=20
--set gallery_ids=cone0001,cone0007
--opt vmax=0.3 --opt full_z=1
```

**Settings precedence.** Explicit CLI flag > TOML value > default. `--target` and `--density`
set both the data field and `Settings.target` / `Settings.density`; an explicit
`--set target=...` conflicts with a differing `--target` and is rejected rather than silently
winning.

**Row selection.** `--indices` and `--indices-json` are mutually exclusive. Both bypass the
dataset's split declaration, because an explicit row list already encodes the split.

## `eval21 run`

Evaluate one or more models over one data set.

| option | meaning |
|---|---|
| `--config FILE` | TOML file with `[run]`, `[models]`, `[settings]`, `[metrics]`, `[metric_options.*]` |
| `--data SPEC` | data path, directory, manifest `.json`, `.npz`, `.h5`, or `synthetic:N` |
| `--model NAME=SPEC` | model adapter spec (repeatable); see [Adapters](adapters.md) |
| `--metrics LIST` | comma list of metric names; default all registered |
| `--out DIR` | output directory (default `results`) |
| `--title TEXT` | report title (default `21cmEVAL report`) |
| `--split NAME` | keep samples whose declared split equals `NAME` |
| `--limit N` | evaluate at most N samples |
| `--box-mpc X` | override the transverse box size in Mpc |
| `--expect-shape NX,NY,NZ` | enforce the sample shape before inference |
| `--expect-nz N` | enforce the LOS length; `NX,NY` come from the data |
| `--indices LIST` | row indices, e.g. `0-199,300` |
| `--indices-json FILE` | reuse a run's recorded split (`test_cone_ids`, else `test_indices`) |
| `--target NAME` | target field (default `xhi`) |
| `--density NAME` | density field (default `density`) |
| `--fno-root DIR` | fno-21cm checkout, for `fno21cm:` adapters |
| `--device DEV` | device string for torch-based adapters (`cpu`, `cuda`, `cuda:1`) |
| `--set KEY=VALUE` | override one `Settings` field (repeatable) |
| `--z-range ZMIN,ZMAX` | cube caches: evaluate this LOS window; drop samples not valid over it |
| `--draws N` | predictions per sample for stochastic/generative models |
| `--mode pointwise\|generative` | generative skips voxel-by-voxel metrics and pointwise keys |
| `--score-draw first\|mean` | what the standard metrics score when `--draws > 1` |
| `--save-predictions DIR` | write truth and prediction cubes for map figures |
| `--save-n N` | samples to store (default 8; `0` = all) |
| `--save-ids a,b` | store exactly these sample ids instead |
| `--quiet` | suppress progress and per-model summaries |

The run prints the dataset summary, the metric list, a per-model progress line (every tenth
sample), a five-number summary (RMSE, R², coherence at k≈0.2, |Δτ|/σ, ms per sample) and the
list of skipped metrics with reasons.

```bash
eval21 run --data cubes/ --model toy=synthetic:1.5 --out results/toy
eval21 run --config examples/fno21cm_eval.toml
eval21 run --data cache.h5 --model gen=python:gen.py --draws 16 --mode generative \
    --out results/gen --save-predictions results/predictions
```

### Selection and split resolution

`--indices-json FILE` reads the `split` block of an fno-21cm `run_metadata.json` (or a bare
map). The split name defaults to `test`, or to `--split`. Resolution order:

1. `<split>_cone_ids` — mapped through the dataset's ids (`row_indices_for_ids`). Datasets
   without ids fall back to `<split>_indices`, and fail if neither is present.
2. `<split>_indices` — row numbers.

A recorded cone id that is absent from the cache is an error, not a silent omission; merged or
reordered caches make positional indices stale, which is why ids win.

An HDF5 cache declares no splits, so `--split` is rejected (rather than scoring every row) and
the CLI prints a warning when no selection was given. `eval21 build` caches *do* declare
splits, so `--split test` works there.

### Cache and geometry checks

For `fno21cm:` models the CLI reads `run_metadata.json` next to the checkpoint *before*
loading the model and enforces any recorded cache identity: `dataset.cache_file` /
`cache_path`, `cache_rows`, `cache_shape`, and `cache_sha256` (which hashes the whole cache and
is slow for a 265 GB file). It also checks the recorded `input_features.spatial_shape` against
every sample. See [Adapters](adapters.md#fno21cm--fno-21cm-checkpoints).

## `eval21 build`

Turn a folder of simulations into one HDF5 cube cache that `eval21 run --data cache.h5` reads
natively.

| option | meaning |
|---|---|
| `FOLDER` | folder of simulation files (omit with `--merge`) |
| `--out FILE` | cache file (required) |
| `--format` | `auto` (default), `21cmfast`, `loreli`, `npz`, `python:FILE.py[:Class]` |
| `--fields` | comma list; default per source (`21cmfast`: `density,xhi`; `loreli`: `density,xhi,tb`) |
| `--targets` | fields stored as targets (default `xhi,tb`); the rest become inputs |
| `--z-grid` | `native` (default), `uniform:ZMIN,ZMAX,N`, `comoving:ZMIN,ZMAX[,N]`, or a file spec |
| `--transverse N` | resample the transverse plane to N cells on the same box |
| `--splits` | `test` (all) or `train=0.8,val=0.1,test=0.1` (seeded shuffle) |
| `--split-file FILE` | JSON `{split: [ids]}`, `{id: split}`, or a `run_metadata.json` with `<split>_cone_ids` |
| `--seed N` | seed for `--splits` fractions (default 0) |
| `--limit N` | build at most N samples |
| `--ids a,b` | build exactly these sample ids |
| `--shard K` | write shard K of `--num-shards` |
| `--num-shards N` | number of shards (default 1) |
| `--merge` | join `out.shardKKKofNNN.h5` files into `--out` |
| `--remove-shards` | with `--merge`, delete the shards afterwards |
| `--workers N` | worker processes for reading and encoding (default 1) |
| `--dtype float32\|float16` | storage precision (default `float32`) |
| `--compress gzip\|lzf` | HDF5 compression (default none) |
| `--option KEY=VALUE` | source-specific option, e.g. `--option l_perp=300` (repeatable) |
| `--quiet` | suppress per-sample output |

```bash
# 21cmFAST lightcones on fno-21cm's 512-plane envelope grid
eval21 build /work/data/data --out fast.h5 --format 21cmfast --fields density,xhi,tb \
    --z-grid /work/data/compressed/cubes_3d_envelope512.h5:target_z \
    --splits train=0.8,val=0.1,test=0.1

# LoReLi II five-field cones, native grid
eval21 build /work/data/loreli_lc128 --out loreli.h5 --format loreli --workers 4

# Slurm array: shards in parallel, one merge call
eval21 build sims/ --out big.h5 --shard $SLURM_ARRAY_TASK_ID --num-shards 32
eval21 build --out big.h5 --merge --num-shards 32 --remove-shards
```

The cache layout and source semantics are documented in [Data](data.md#eval21-build).

## `eval21 describe`

Print a JSON description of a data set: format, path, row counts, shape, box, redshift range,
input and target fields, split counts, and (for cube caches) the number of excluded samples.

| option | meaning |
|---|---|
| `--data SPEC` | data path (required) |
| `--target NAME` | target field used to interpret the file (default `xhi`) |
| `--limit N` | probe this many samples (default 3) |
| `--box-mpc X` | override the box size |

```bash
eval21 describe --data demo/data
```

## `eval21 list-metrics`

Print every registered metric with the first line of its docstring; `--json` emits a
name → description object. The same list is available programmatically through
`eval21.registry.available()`.

## `eval21 selftest`

Run the analytic checks in `eval21/selftest.py` with no data and no GPU: closed-form RMSE, R²,
coherence = 1 for identical fields, τ_e ordering, a ray tracer against a half-plane, a sphere's
effective radius, a known sigmoid front width, and strict JSON serialisation. Returns a
non-zero exit status on failure. The same checks are collected by pytest.

## `eval21 demo`

```bash
eval21 demo --out demo [--n-samples 4] [--shape 64,64,64] [--quiet]
```

Writes `demo/data/manifest.json` plus npz cubes, then runs `synthetic:1.5` over them with all
metrics and writes `demo/results/`. It is the installation smoke test and the smallest
end-to-end example of the report and figure pipeline.

## `eval21 plot`

Draw figures from one or more result directories. This command is documented in full in
[Visualization](visualization.md); the summary is:

```bash
eval21 plot list                                    # registry: name, group, needs, help
eval21 plot all --run results/ --out figs/
eval21 plot cyl maps --run results/ --out figs/      # groups or individual figures
eval21 plot gallery --run results/ --out figs/ --title "LoReLi II test"
```

| option | meaning |
|---|---|
| `FIGURE ...` | figure names, group names, `all`, or `list` |
| `--run DIR` | result directory or a tree containing them (repeatable) |
| `--out DIR` | output directory |
| `--store DIR` | prediction store (default: a `predictions/` directory near the results) |
| `--opt KEY=VALUE` | figure option (repeatable) |
| `--models a,b` | subset and order of models (default: all complete models) |
| `--style FILE` | JSON `{model: {"label": ..., "color": ..., "ls": ...}}` |
| `--ids a,b` | sample ids for the gallery figure |
| `--title TEXT` | data-set name used in figure titles |
| `--vmax X` | cylindrical maps: colour saturation (same as `--opt vmax=X`) |
| `--dead X` | cylindrical maps: white dead band (same as `--opt dead=X`) |
| `--quiet` | suppress the "wrote …" lines |

## TOML configuration

Everything on the command line can live in a file; CLI flags override it. Python 3.11+ is
required for `--config` (it uses `tomllib`).

```toml
[run]
data = "/work/data/compressed/cubes_3d.h5"
out = "results/fno"
split = "test"
limit = 200
target = "xhi"
density = "density"
fno_root = "/work/fno-21cm"
device = "cuda"
indices = "0-199"          # same syntax as --indices
z_range = "8,12"           # cube caches only

[models]
fno = "fno21cm:/work/fno-21cm/checkpoints/3d_xhi/fno/ckpt/best_model_state_dict.pt"
ufno = "fno21cm:/work/fno-21cm/checkpoints/3d_xhi/ufno/ckpt/best_model_state_dict.pt"

[settings]
box_mpc = 200.0
stage_edges = [0.02, 0.2, 0.4, 0.6, 0.8, 0.98]
rays_per_slice = 256
slices_per_stage = 6

[metrics]
enabled = ["voxel", "spectrum", "bubble", "history", "front"]

[metric_options.spectrum]
n_k_bins = 20
```

| section | contents |
|---|---|
| `[run]` | `data`, `out`, `target`, `density`, `split`, `limit`, `box_mpc`, `fno_root`, `device`, `indices`, `z_range` |
| `[models]` | name → adapter spec |
| `[settings]` | any field of `Settings` (see [Metrics](metrics.md#settings)) |
| `[metrics]` | `enabled` list; CLI `--metrics` overrides it |
| `[metric_options.<metric>]` | keyword arguments passed to that metric's constructor |

Unknown settings keys fail with a `KeyError` listing the unknown names; unknown metric names
fail with the list of registered metrics.

## Environment and cost

- On 16 CPU cores a `140² × 256` forward pass takes 30–60 s and a full metric pass is
  ≈4–5 min per cone; `voxel+history+parity` is ≈55 s per cone. Use a GPU for production sweeps.
- The HDF5 loader never materialises the cache: a 6600-cone, 265 GB `cubes_3d.h5` opens in
  seconds on a login node, but the sweep itself still needs the compute.
- `eval21 plot` needs matplotlib only; the core suite needs numpy and scipy.
