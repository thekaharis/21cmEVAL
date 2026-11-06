# Data

A **sample** is one lightcone. The suite reads samples from four kinds of data set and hands
them to adapters and metrics unchanged; nothing downstream knows where they came from.

```python
Sample(
    id="cone0000",
    inputs={"density": (Nx, Ny, Nz)},   # δ = ρ/ρ̄ − 1, mean 0
    targets={"xhi": (Nx, Ny, Nz)},      # neutral fraction in [0, 1]
    meta=Meta(box_mpc=200.0, z_grid=(Nz,), omega_m=0.30966),
    split="test",
)
```

## The `Sample` / `Meta` contract

| field | shape / type | meaning |
|---|---|---|
| `id` | `str` | sample identifier; names figure files and store entries |
| `inputs` | `{name: (Nx, Ny, Nz)}` or `{name: (C, Nx, Ny, Nz)}` | model inputs and fields metrics may need |
| `targets` | `{name: (Nx, Ny, Nz)}` | the truth fields a run scores |
| `meta.box_mpc` | `float \| None` | transverse comoving box size, Mpc |
| `meta.z_grid` | `(Nz,) float \| None` | redshift of each LOS cell |
| `meta.omega_m` | `float \| None` | matter density; falls back to `Settings.omega_m` |
| `meta.params` | `{str: float}` | simulation parameters (a parameters-only model reads these) |
| `split` | `str \| None` | `train`/`val`/`test` when the data declares it |

Orientation is `(Nx, Ny, Nz)`: `x` and `y` are the periodic transverse axes, `z` the lightcone
axis (increasing index = increasing redshift unless a dataset documents otherwise). Targets
must be 3-D; an input may be a ready-made channel stack `(C, Nx, Ny, Nz)` (the fno-21cm
adapter accepts this as `sample.inputs["x"]`).

`Sample.__post_init__` rejects empty arrays and wrong dimensionality. `check_targets` rejects
non-finite truth before any metric sees it; `check_prediction` rejects a missing field, a wrong
shape, or any NaN/inf in the prediction. A NaN-tolerant metric would turn a broken forward pass
into a plausible-looking report, so these are errors.

### Geometry defaults

`meta.box_mpc` and `meta.omega_m` may be `None`; metrics fall back to
`Settings.box_mpc` (200 Mpc) and `Settings.omega_m` (0.30966). `Settings.from_dict` never
fills them per sample. A dataset that knows the real values should always set them: power
spectra are quoted in physical units, and a wrong box silently rescales every k.

### What each metric needs

| field | used by |
|---|---|
| target (`xhi` or `tb`) | every metric |
| `xhi` as target **or** input | `history`, `bubble`, `morphology`, `front`, `observable`; also stages of a T_b run |
| `density` | `edge` (indirectly), `eft`, `observable` |
| `z_grid` | `history`, `observable`, `eft`, `front`, `cylindrical` stage chunks (z-based mode) |
| re-runnable adapter | `symmetry` (the runner calls `predict` on the x↔y-transposed sample) |

A metric whose needs are absent is skipped with a reason (`missing input fields: density`),
not failed.

## Data set formats

`open_data(spec, ...)` chooses a reader from the spec prefix or the file suffix.

| spec | reader | notes |
|---|---|---|
| `path/to/dir` | `NpzDataset` | every `*.npz`, or `manifest.json` inside |
| `path/manifest.json` | `NpzDataset` | explicit sample list |
| `path/cone.npz` | `NpzDataset` | a single sample |
| `path/cache.h5` | `CubeCacheDataset` if `attrs["format"] = eval21-cubes/1`, else `Hdf5Dataset` | `eval21 build` output vs fno-21cm cache |
| `cubes:path`, `npz:path`, `hdf5:path` | forced reader | disambiguates odd filenames |
| `synthetic:N` (or `toy:N`) | `SyntheticDataset` | deterministic artificial cones, N samples |

### npz cubes, no manifest

Every `*.npz` in the directory is one sample. The run's target field is the target; every other
3-D array is an input. Optional 1-D `z_grid` / `z` sets the redshift grid, `box_mpc` and
`omega_m` scalars set the geometry, and a JSON string array `params` sets sampled parameters.

```python
np.savez("cone_0000.npz", density=density, xhi=xhi, z_grid=z, box_mpc=200.0, omega_m=0.31,
         params=json.dumps({"F_ESC10": -1.2}))
```

### npz cubes with a manifest

Recommended when ids, splits or per-sample parameters matter. Paths are relative to the
manifest.

```json
{
  "box_mpc": 200.0,
  "z_grid": [5.0, 5.1, 25.0],
  "target": "xhi",
  "inputs": ["density"],
  "samples": [
    {"id": "cone0000", "path": "cubes/cone_0000.npz", "split": "test",
     "omega_m": 0.31, "params": {"F_ESC10": -1.2}}
  ]
}
```

`target` and `inputs` are optional; omitted, the run's target is used and every other 3-D
array becomes an input. If `target` is set it must agree with the run's `--target`: the
manifest field selects the truth array, while the metrics score `Settings.target`, and a
disagreement leaves every metric without its target field. Per-sample `omega_m`, `box_mpc`,
`params`, `split`, `inputs` and `target` override the manifest defaults.

### HDF5 cube caches (fno-21cm)

`Hdf5Dataset` auto-detects the two layouts that occur in practice:

| layout | keys |
|---|---|
| training | `x` `(N, C, Nx, Ny, Nz)`, `y` `(N, 1, Nx, Ny, Nz)`, optional `target_z`, `params` |
| analysis | `density`, `neutral_fraction` (or `xhi`), `cone_id`, `target_z`, `params` |

The target is `y`, `neutral_fraction` or `xhi` for an x_HI run, or a key equal to `--target`
otherwise. Density is read from `density` when present. Parameters come from
`params/values` + `params/names`, or from a bare `(N, 11)` array interpreted in the canonical
fno-21cm order (`F_ESC10`, `F_STAR10`, `ALPHA_ESC`, `ALPHA_STAR`, `L_X`, `NU_X_THRESH`,
`M_TURN`, `t_STAR`, `X_RAY_SPEC_INDEX`, `OMm`, `SIGMA_8`). `OMm` also supplies `omega_m` when
no dedicated field exists.

The loader indexes ids, parameters and geometry once and reads **one cube per iteration**; it
never materialises the cache. A 6600-cone `cubes_3d.h5` is about 265 GB, so opening it is cheap
even on a login node, while the sweep still wants a GPU.

A plain HDF5 cache declares no train/val/test splits. `--split test` is therefore rejected
rather than scoring every row; use `--indices 0-199` or `--indices-json run_metadata.json`.
For a different layout, construct `Hdf5Dataset` directly with explicit `x_key`, `y_key`,
`z_key`, `params_key`, `id_key`, `box_mpc`, `default_omega_m` or `split_map`.

### eval21 cube caches

Files written by `eval21 build` carry `attrs["format"] = "eval21-cubes/1"` and are read by
`CubeCacheDataset`. Every stored field is available; the fields listed in the file's `targets`
attribute (plus the run's `--target`) are targets, the rest are inputs.

`z_range=(zmin, zmax)` (CLI `--z-range`) crops the line of sight. Samples whose `valid` mask
does not cover the whole window are dropped and counted in `describe()["excluded"]`. Without a
window, every selected sample must be valid along its entire LOS; otherwise the dataset stops
and names the widest window that would work.

### Synthetic data

`SyntheticDataset(n)` and `eval21 demo` generate deterministic 21-cm-like cones: a log-normal
density field and a smoothed-threshold ionization field on a shared geometry. They exist for
tests, examples and smoke runs, not for science.

## `eval21 build`

`eval21 build FOLDER --out cache.h5` turns a folder of simulation files into one cache.

### Sources

| `--format` | reads | default fields |
|---|---|---|
| `21cmfast` | py21cmfast lightcone HDF5 (`lightcone/<field>`, `lightcone/lightcone_redshifts`, `params/{names,values}`, `box_len_mpc`) | `density,xhi` |
| `loreli` | LoReLi II `lightcone_*_simuN.dat` (five-field `xa,tpHI,xion,dens,dtb` or dT_b only) | `density,xhi,tb` |
| `npz` | one `.npz` per sample | every 3-D array |
| `python:FILE.py[:Class]` | a custom `Source` subclass | as declared |
| `auto` | picks `21cmfast`, `loreli` or `npz` from the folder contents | per source |

Canonical field names are `density` (overdensity δ), `xhi` (neutral fraction) and `tb`
(brightness temperature in mK); other names pass through (`xa`, `tpHI`, …). `--targets`
decides which fields are stored as targets (default `xhi,tb`); the rest are inputs. A target
that is not among `--fields` is ignored. Any stored field can still be chosen as a run's
`--target`.

**LoReLi specifics.** Files are `128² × 1536` planes uniform in z = 5–15, index 0 = low z.
`xhi = 1 − xion`; `density` is the particle count divided by the slice mean, minus 1. Empty or
corrupt planes (non-finite, |dT_b| > 1000 mK, partially filled first/last planes) are marked
invalid; only the longest contiguous valid run is kept. Parameters are derived from the
simulation number using `loreli2_parameters.json` (`RH`/`FX`/`FE` grids plus `PAIRS`); pass
`--option l_perp=294` (Mpc) and `--option omega_m=0.308` to override the defaults.

### Line-of-sight grids

| `--z-grid` | meaning |
|---|---|
| `native` (default) | keep each source's own planes; all files must agree |
| `uniform:ZMIN,ZMAX,N` | N planes uniform in redshift |
| `comoving:ZMIN,ZMAX[,N]` | uniform in comoving distance; without N the spacing equals the transverse cell (cubic cells) |
| `file:PATH`, `PATH.npy`, `PATH.txt`, `PATH.h5:key` | explicit grid; bare paths are accepted |
| `PATH.h5` (no key) | reads `target_z` |

Interpolation is linear per pixel along the LOS, the same scheme as fno-21cm's `build_cubes`,
implemented without scipy. Planes outside the source's redshift range are filled with 0 and
marked invalid. `--transverse N` resamples the transverse plane linearly between cell centres
on the same physical box.

### Splits

`--splits test` marks every sample `test`. `--splits train=0.8,val=0.1,test=0.1` shuffles the
sorted ids with `--seed` (default 0) and assigns contiguous fractions. `--split-file` accepts

- `{"train": ["cone1", ...], "test": [...]}` (ids per split),
- `{"cone1": "test", ...}` (split per id),
- an fno-21cm `run_metadata.json` with `<split>_cone_ids` keys.

Ids not mentioned get an empty split (`""`), which `--split` will not match.

### Cache layout

```text
fields/<name>   (N, Nx, Ny, Nz)  float32 | float16   one chunk per sample
z_grid          (Nz,)            float64             shared line of sight
valid           (N, Nz)          bool                planes usable for evaluation
id              (N,)             str
split           (N,)             str ("" = undeclared)
omega_m         (N,)             float64
params/values   (N, P) float64   params/names (P,) str
attrs: format, source, box_mpc, cell_mpc, fields, targets, units, created, command,
       eval21_version, skipped
```

`skipped` is a JSON list of `[id, error]` for samples the build could not read. `--dtype
float16` halves the size at some precision cost; `--compress gzip|lzf` compresses each chunk
(slower builds, smaller files). A build can be sharded over a Slurm array
(`--shard K --num-shards N`) and joined with `--merge`, which writes `out.shardKKKofNNN.h5`
files and restores the source order (ids are sorted per shard, then interleaved).

### Custom build sources

```python
# my_source.py
from eval21.build.sources import Source, SourceItem, RawSample

class MySource(Source):
    name = "mine"
    default_fields = ("density", "xhi")

    def items(self):
        return [SourceItem("sim1", path) for path in ...]

    def read(self, item, fields):
        return RawSample(fields={...}, z=..., valid=None, params={}, box_mpc=..., omega_m=...)
```

`read` must return each requested field as `(Nx, Ny, Nz)` on `raw.z`, plus an optional
`valid` mask over LOS planes. Use `--format python:my_source.py:MySource`.

## Inference cache

Map figures need cubes, so `eval21 run --cache PATH` writes an optional **`eval21-cache/1`**
HDF5 file. Cubes are float16, so a `140 × 140 × 512` field is about 20 MB. A run file never
contains cubes; this cache is regenerable and can be deleted.

```bash
eval21 run ... --out results/ --cache results/fno.xhi.cache.h5 --save-n 8
eval21 run ... --out results/ --cache results/ --save-n 8      # -> results/xhi.cache.h5
eval21 plot maps --run results/ --store results/fno.xhi.cache.h5
```

```text
cache.h5
├── /                  attrs: schema="eval21-cache/1", created, producer
├── /index             JSON: samples, models, fields, targets, draws
├── /ids               (N,) UTF-8
├── /splits            (N,) UTF-8
├── /omega_m           (N,) float64
├── /params/{names,values}
├── /z_grid            (Nz,) float64
├── /box_mpc           scalar float64
├── /truth/<field>     (N, Nx, Ny, Nz) float16, targets and inputs
├── /models/<model>/<field>          (N, Nx, Ny, Nz) float16
└── /models/<model>/draws/<field>    (N, D, Nx, Ny, Nz) float16, with --draws
```

Selection is `--save-n N` (first N wanted samples, default 8; `0` = all) or `--save-ids`.
Several models of one target can share one cache (the CLI opens it in append mode), which is
what map figures comparing models need. Truth is written once; each model adds its own group.

`open_cache(path)` reads it back; `resolve("1001")` finds `cone1001` and vice versa, so figure
file names can drop the prefix. `StoreGroup([...])` merges several caches (e.g. one per target)
behind the same interface.

`--save-predictions DIR` still writes the legacy **prediction store** (one `.npz` per sample,
`DIR/index.json`, `DIR/truth/`, `DIR/<model>/`); an `.h5` path passed to `--save-predictions`
is treated as the new cache. Legacy stores and caches both satisfy the interface the map
figures use, so `eval21 plot --store` accepts either.

## `describe()` fields

`eval21 describe --data ...` prints the dataset's own summary. Typical keys:

| key | meaning |
|---|---|
| `format` | reader format (`npz`, `eval21-cubes/1`, `hdf5`, …) |
| `path`, `root` | file and directory |
| `n_total`, `n_samples` | rows in the file / rows after selection |
| `split`, `splits` | requested split and per-split counts (`undeclared` when absent) |
| `indices`, `indices_requested` | explicit selection |
| `shape` | `[Nx, Ny, Nz]` |
| `box_mpc` | transverse box |
| `inputs`, `targets` | field names |
| `z_range` | cube caches: the evaluated LOS window |
| `excluded` | cube caches: samples dropped for invalid planes |

## Pitfalls

- **Density units.** `density` is the evolved overdensity with mean 0, not a count or a
  density contrast normalised per slice. A prediction built from raw counts scores nonsense.
- **Box size.** `box_mpc` is the transverse comoving size; a wrong value rescales every k.
- **One geometry per run.** Mixing shapes or redshift grids in one run raises; evaluate in
  separate runs and let `eval21 plot` pool them as groups.
- **x_HI polarity.** Targets are the *neutral* fraction `x_HI` (1 = neutral), matching
  `neutral_fraction`; `xion` fields must be converted at build time.
- **Splits.** A plain HDF5 cache has none: `--split` is an error by design.
