# API reference

Public objects and functions, generated from the source tree. Signatures are exact;
docstrings are the first paragraph of the real docstring. Everything not listed here is
private and may change without notice.

## `eval21`

21cmEVAL: end-to-end evaluation for 21cm field emulators.

```python
from eval21 import Settings, Sample, Meta, evaluate, report, __version__
```

The package re-exports the runner and the two data contracts. `__version__` is
`0.1.0`; the distribution name is `21cmeval`.

## `eval21.spec`

Data contract between datasets, model adapters and metrics.

```python
class Meta
```

Per-sample geometry and cosmology. ``None`` means "not provided".

| field | type | default |
|---|---|---|
| `box_mpc` | `float \| None` | `None` |
| `z_grid` | `np.ndarray \| None` | `None` |
| `omega_m` | `float \| None` | `None` |
| `params` | `dict[str, float]` | `field(default_factory=dict)` |

| method | description |
|---|---|
| `with_defaults(box_mpc: float, omega_m: float \| None=None)` |  |

```python
class Sample
```

One evaluation example: named inputs, named targets, geometry.

| field | type | default |
|---|---|---|
| `id` | `str` | `—` |
| `inputs` | `dict[str, np.ndarray]` | `—` |
| `targets` | `dict[str, np.ndarray]` | `—` |
| `meta` | `Meta` | `field(default_factory=Meta)` |
| `split` | `str \| None` | `None` |

| method | description |
|---|---|
| `shape` | property:  |
| `transposed_xy()` | Same sample with every field swapped in the transverse plane. |

```python
def check_prediction(prediction, sample: Sample, names=None) -> None
```

Fail loudly when an adapter returns a missing, wrong-shape or NaN field.

```python
def check_targets(sample: Sample) -> None
```

Targets must be finite 3-D fields before any metric sees them.

## `eval21.settings`

Suite-wide settings shared by every metric.

```python
class Settings
```

Physical and numerical configuration of the metric suite.

| field | type | default |
|---|---|---|
| `target` | `str` | `'xhi'` |
| `density` | `str` | `'density'` |
| `box_mpc` | `float` | `200.0` |
| `omega_m` | `float` | `0.30966` |
| `omega_b` | `float` | `0.04897` |
| `hubble_h` | `float` | `0.6766` |
| `stage_field` | `str` | `'xhi'` |
| `threshold` | `float` | `0.5` |
| `stage_edges` | `tuple[float, ...]` | `(0.02, 0.2, 0.4, 0.6, 0.8, 0.98)` |
| `active_range` | `tuple[float, float]` | `(0.05, 0.95)` |
| `min_stage_slices` | `int` | `2` |
| `n_k_bins` | `int` | `15` |
| `k_targets` | `tuple[float, ...]` | `(0.1, 0.2, 0.5)` |
| `eft_k_max` | `float` | `0.25` |
| `coherence_level` | `float` | `0.9` |
| `highk_k_min` | `float` | `0.2` |
| `highk_bins` | `int` | `12` |
| `edge_width_fraction` | `float` | `0.1` |
| `active_variance_fraction` | `float` | `0.05` |
| `rays_per_slice` | `int` | `256` |
| `slices_per_stage` | `int` | `6` |
| `bubble_bins` | `int` | `24` |
| `max_distance_mpc` | `float \| None` | `None` |
| `seed` | `int` | `0` |
| `front_bands_mpc` | `tuple[float, ...]` | `(2.0, 5.0)` |
| `front_d_max_mpc` | `float` | `30.0` |
| `front_bin_mpc` | `float` | `1.5` |
| `xhi_range` | `tuple[float, float]` | `(-0.2, 1.2)` |
| `xhi_bins` | `int` | `70` |
| `partial_range` | `tuple[float, float]` | `(0.1, 0.9)` |
| `dtb_range_mk` | `tuple[float, float]` | `(-5.0, 100.0)` |
| `dtb_bins` | `int` | `105` |
| `cyl_chunk_cells` | `int` | `140` |
| `cyl_stages` | `tuple[float, ...]` | `(0.9, 0.7, 0.5, 0.3, 0.1)` |
| `cyl_z_centers` | `tuple[float, ...]` | `()` |
| `cyl_dead_band` | `float` | `0.1` |
| `cyl_native_cell_mpc` | `float \| None` | `None` |
| `cyl_native_los_dz` | `float \| None` | `None` |
| `cyl_native_los_mpc` | `float \| None` | `None` |
| `gallery_samples` | `int` | `4` |
| `gallery_ids` | `tuple[str, ...]` | `()` |
| `gallery_stages` | `tuple[float, ...]` | `(0.8, 0.6, 0.4, 0.2)` |
| `gallery_fields` | `tuple[str, ...]` | `('xhi', 'tb')` |
| `draws` | `int` | `1` |
| `mode` | `str` | `'pointwise'` |
| `score_draw` | `str` | `'first'` |
| `ensemble_voxels` | `int` | `200000` |
| `ensemble_n_los` | `int` | `16` |
| `ensemble_k` | `tuple[float, ...]` | `(0.1, 0.3, 1.0)` |
| `store_draws` | `int` | `4` |
| `warmup` | `int` | `3` |
| `iters` | `int` | `10` |
| `benchmark_samples` | `int` | `4` |

| method | description |
|---|---|
| `updated(**values)` | Return a copy with the given fields replaced (unknown keys fail). |
| `to_dict()` |  |
| `from_dict(cls, values: dict)` |  |

## `eval21.registry`

Metric plugin registry.

```python
def register(name: str)
```

Class decorator: add a metric under ``name``.

```python
def available() -> dict[str, type]
```

```python
def create(name: str, settings: Settings, options: dict | None=None)
```

```python
def build(names: Iterable[str], settings: Settings, options: dict[str, dict] | None=None) -> list
```

Instantiate a metric list, preserving order and dropping duplicates.

## `eval21.runner`

Evaluation loop: one streaming pass over the data per model.

```python
class ModelResult
```

| field | type | default |
|---|---|---|
| `name` | `str` | `—` |
| `info` | `dict` | `—` |
| `metrics` | `dict` | `—` |
| `skipped` | `dict` | `—` |
| `arrays` | `dict` | `field(default_factory=dict)` |
| `seconds` | `float` | `0.0` |
| `n_samples` | `int` | `0` |
| `samples` | `list` | `field(default_factory=list)` |
| `z_grid` | `object` | `None` |
| `metric_meta` | `dict` | `field(default_factory=dict)` |

```python
def evaluate(adapter, dataset, settings: Settings, metric_names: Iterable[str]=('all',), metric_options: dict | None=None, progress=None, expect_shape: tuple[int, ...] | None=None, store=None, model_name: str='model') -> ModelResult
```

Run one adapter over ``dataset`` and return per-metric results.

```python
def predict_draws(adapter, sample, settings)
```

(prediction dict for the standard metrics, draws (D, Nx, Ny, Nz) or None).

```python
def strip_keys(value, patterns, dropped, path='')
```

Copy of a result with every key containing one of ``patterns`` removed (recursively).

```python
def environment() -> dict
```

## `eval21.report`

Report writers: JSON (complete), CSV (flat scalars), Markdown (readable).

```python
def jsonable(value)
```

Recursively convert numpy types and non-finite floats for strict JSON.

```python
def flatten(value, prefix: str='') -> dict
```

Dotted scalar leaves; arrays and lists are skipped.

```python
def write(directory, models, dataset_info: dict, settings: dict, environment: dict | None=None, title: str='21cmEVAL report') -> dict
```

Write ``results.json``, ``summary.csv``, ``report.md`` and ``curves.npz``.

```python
def render_markdown(models, dataset_info: dict, settings: dict, title: str='21cmEVAL report') -> str
```

## `eval21.store`

Prediction store: full truth and prediction cubes of selected samples, for map-based figures.

```python
class PredictionStore(directory, ids=None, n=None, dtype=np.float16)
```

Writer (during a run) and reader (for figures) of stored cubes.

| method | description |
|---|---|
| `wants(sample)` |  |
| `save_truth(sample)` |  |
| `save_prediction(model: str, sample, prediction: dict, draws: int=1)` | Add a model's fields for one sample; fields stored earlier (another target's run writing to the same store) are kept. |
| `flush()` |  |
| `open(cls, directory)` |  |
| `samples` | property:  |
| `models` | property:  |
| `draws(model: str)` |  |
| `resolve(sample_id)` | Stored id for ``sample_id``; '1001' finds 'cone1001' (and vice versa), as figure file names drop the 'cone' prefix. |
| `truth(sample_id)` | {"targets": {...}, "inputs": {...}, "z_grid", "box_mpc", "omega_m", "params"} as float32. |
| `prediction(model: str, sample_id)` |  |
| `has(model: str, sample_id)` |  |

## `eval21.io`

Storage for evaluation results: ``eval21-run/1`` files and ``eval21-cache/1`` cubes.

_No public names._

## `eval21.io.runfile`

The eval21 run file: one HDF5 file per ``(model, target, dataset)`` evaluation.

```python
def array_metadata(metric: str, key: str, shape: tuple, n_samples: int) -> tuple[list, str]
```

Best-effort ``(dims, units)`` for one exported array.

```python
def default_path(out: Path, model: str, target: str) -> Path
```

```python
def write_run(path, result, dataset_info: dict, settings: dict, environment: dict | None=None, command: str='', overwrite: bool=False) -> Path
```

Write one :class:`~eval21.runner.ModelResult` as an ``eval21-run/1`` file.

```python
class RunFile(path)
```

Reader for one ``eval21-run/1`` file.

| method | description |
|---|---|
| `model` | property:  |
| `target` | property:  |
| `dataset_info` | property:  |
| `settings` | property:  |
| `info` | property:  |
| `meta` | property:  |
| `n_samples` | property:  |
| `sample_ids` | property:  |
| `metric_names()` |  |
| `result(metric: str)` |  |
| `metric_meta(metric: str)` |  |
| `arrays(metric: str)` |  |
| `array(metric: str, key: str)` |  |
| `curves()` | ``{model: {array key: array}}`` — the same shape the legacy reader returns. |
| `as_results()` |  |
| `validate(strict: bool=False)` |  |
| `close()` |  |

```python
class LegacyRun(path)
```

Adapter over a legacy run directory (``results.json`` + ``curves.npz``).

| method | description |
|---|---|
| `model` | property:  |
| `target` | property:  |
| `dataset_info` | property:  |
| `settings` | property:  |
| `info` | property:  |
| `n_samples` | property:  |
| `sample_ids` | property:  |
| `metric_names()` |  |
| `result(metric)` |  |
| `metric_meta(metric)` |  |
| `arrays(metric)` |  |
| `array(metric, key)` |  |
| `curves()` |  |
| `as_results()` |  |
| `validate(strict=False)` |  |
| `close()` |  |

```python
def open_run(path)
```

Open an ``eval21-run/1`` file or a legacy run directory with one interface.

```python
def validate_run(path, strict: bool=False) -> list[str]
```

## `eval21.io.cache`

Optional HDF5 inference cache (``eval21-cache/1``).

```python
class CacheWriter(path, ids=None, n: int=DEFAULT_MAX_SAMPLES, dtype=np.float16, overwrite: bool=False)
```

Append truth and predictions to one cache file, model group by model group.

| method | description |
|---|---|
| `wants(sample)` |  |
| `save_truth(sample)` |  |
| `save_prediction(model: str, sample, prediction: dict, draws: int=1)` |  |
| `flush()` |  |
| `close()` |  |
| `dir` | property:  |

```python
class CacheFile(path)
```

Reader for one ``eval21-cache/1`` file.

| method | description |
|---|---|
| `dir` | property:  |
| `draws(model: str)` |  |
| `resolve(key)` | Accept a sample id (int/str) or a 0-based index. |
| `has(model, sid: str \| None=None)` | ``has(sid)`` or ``has(model, sid)`` (the legacy store's signature). |
| `truth(sid: str)` |  |
| `prediction(model: str, sid: str)` |  |
| `close()` |  |

```python
def resolve_sample(samples: list[str], key) -> str
```

```python
def open_cache(path) -> CacheFile
```

```python
class StoreGroup(stores)
```

Read several caches (or legacy stores) as one store.

| method | description |
|---|---|
| `dir` | property:  |
| `models` | property:  |
| `samples` | property:  |
| `draws(model: str)` |  |
| `resolve(key)` |  |
| `has(model, sid: str \| None=None)` | ``has(sid)`` or ``has(model, sid)``, matching the other stores. |
| `truth(sid: str)` |  |
| `prediction(model: str, sid: str)` |  |

## `eval21.physics`

Cosmology and field statistics shared by every metric.

```python
def hubble_rate_si(z, omega_m: float=OMEGA_M, h: float=HUBBLE_H)
```

H(z) in 1/s for flat LCDM; radiation neglected (<1% at z <= 25).

```python
def thomson_optical_depth(z_grid, xhi_mean, omega_m: float=OMEGA_M, omega_b: float=OMEGA_B, h: float=HUBBLE_H, y_he: float=Y_HE, z_he_double: float=Z_HE_DOUBLE, n_low: int=2000) -> float
```

CMB Thomson optical depth of a global reionization history.

```python
def history_crossing(z_grid, xhi_mean, level: float) -> float
```

Redshift where the global x_HI first reaches ``level`` from below.

```python
def history_features(z_grid, xhi_mean) -> dict
```

Crossing redshifts for x_HI = 0.25 / 0.5 / 0.75 and the duration.

```python
def brightness_temperature_mk(xhi, delta, z_grid, omega_m: float=OMEGA_M, omega_b: float=OMEGA_B, h: float=HUBBLE_H)
```

21-cm brightness temperature in mK, saturated spin temperature, no RSD.

```python
def slice_mean(field: np.ndarray) -> np.ndarray
```

```python
def stage_labels(stage_edges, active_range) -> list[str]
```

```python
def stage_masks(xbar, stage_edges, active_range, min_slices: int=1)
```

Boolean LOS masks per stage plus the trailing "active" pseudo-stage.

```python
def slice_moments(field) -> dict
```

Mean, variance and skewness of every transverse slice.

```python
def fit_even_polynomial(k, y, variance, k_max: float, n_terms: int=2) -> dict
```

Weighted least squares ``y(k) = sum_j c_j k^(2j)`` for ``k <= k_max``.

```python
def histogram_w1(counts_a, counts_b, edges) -> float
```

Wasserstein-1 distance between two histograms on shared ``edges``.

```python
def wasserstein_1d(a, b) -> float
```

Exact 1-D Wasserstein-1 distance between two sample sets.

```python
def jensen_shannon(p, q) -> float
```

Jensen-Shannon divergence (base e) between two discrete distributions.

```python
class KBinner(nx: int, ny: int, box_mpc: float, n_bins: int=15)
```

Radial k-binning of a 2-D transverse FFT grid, in physical 1/Mpc.

| method | description |
|---|---|
| `slice_sums(field: np.ndarray)` | Per-slice binned sums of \|FFT\|^2, shape (Nz, n_bins), P in Mpc^2. |
| `cross_sums(a: np.ndarray, b: np.ndarray)` | Per-slice binned sums of Re(A conj(B)), same layout. |

```python
def delta2(k, power)
```

Dimensionless 2-D power ``Delta^2(k) = k^2 P(k) / (2 pi)``.

```python
def safe_ratio(pt, pp, cross)
```

Power ratio ``pp/pt`` and coherence ``cross/sqrt(pt pp)``.

```python
def k_crossing(k, curve, level: float) -> float
```

Smallest k where ``curve`` first drops below ``level`` (log-interp).

```python
def trace_ray_to_neutral_2d(ionized_mask: np.ndarray, start_xy: tuple[float, float], direction_xy: tuple[float, float], cell_size_mpc: tuple[float, float], max_distance_mpc: float) -> float
```

Trace one periodic transverse ray to the first neutral cell.

```python
def mean_free_path_samples_2d(ionized_mask: np.ndarray, n_rays: int, cell_size_mpc: tuple[float, float], max_distance_mpc: float, seed=None) -> dict
```

Sample a transverse mean-free-path bubble-size distribution.

```python
def trace_rays_to_neutral_2d(mask: np.ndarray, starts: np.ndarray, directions: np.ndarray, cell_size_mpc: tuple[float, float], max_distance: float) -> np.ndarray
```

Vectorised DDA for many rays at once.

```python
def ionized_region_summary(field, threshold: float=0.5) -> dict
```

Connected-component summary of the ionized (``field < threshold``) mask.

```python
def signed_distance_2d(mask, sampling) -> np.ndarray
```

Signed distance to the mask boundary: positive inside, in Mpc.

```python
def front_width(d_mpc, mean_field, lo: float=0.1, hi: float=0.9) -> float
```

10-90 transition width (Mpc) of a mean x_HI-vs-signed-distance curve.

```python
def comoving_distance_mpc(z, omega_m: float=OMEGA_M, h: float=HUBBLE_H) -> np.ndarray
```

Comoving distance of each redshift from ``min(z)`` [Mpc], flat LCDM (trapezoid, 1e-6 accurate).

```python
def los_cell_size_mpc(z: float, dz_cell: float, omega_m: float=OMEGA_M, h: float=HUBBLE_H) -> float
```

Comoving size of one LOS cell at redshift z: dD = c dz / H(z).

## `eval21.selftest`

Numeric self-checks that need no data and no GPU.

```python
def run(verbose: bool=True) -> int
```

## `eval21.data`

Dataset selection from a ``format:path`` spec.

```python
def open_data(spec: str, target: str='xhi', density: str='density', split: str | None=None, limit: int | None=None, box_mpc: float | None=None, indices: list[int] | None=None, **kwargs) -> BaseDataset
```

Open ``path``, ``npz:path``, ``hdf5:path``, ``cubes:path`` or ``synthetic:N``.

## `eval21.data.base`

Dataset interface.

```python
class Dataset  # (Protocol)
```

| method | description |
|---|---|
| `describe()` |  |

```python
class BaseDataset(split: str | None=None, limit: int | None=None, indices: list[int] | None=None)
```

Common split/limit handling.

## `eval21.data.npz`

Saved-cube datasets (``.npz``), with or without a manifest.

```python
class NpzDataset(path, target: str='xhi', density: str='density', split: str | None=None, limit: int | None=None, default_omega_m: float | None=None, indices: list[int] | None=None, box_mpc: float | None=None)  # (BaseDataset)
```

| method | description |
|---|---|
| `row_indices_for_ids(ids)` |  |
| `describe()` |  |

## `eval21.data.hdf5`

HDF5 cube cache (the fno-21cm ``cubes_3d.h5`` layout and friends).

```python
class Hdf5Dataset(path, split: str | None=None, limit: int | None=None, target: str='xhi', density: str='density', x_key: str | None=None, y_key: str | None=None, z_key: str='target_z', params_key: str='params', id_key: str | None='cone_id', box_mpc: float | None=None, default_omega_m: float | None=None, indices: list[int] | None=None, split_map: dict[str, list[int]] | None=None)  # (BaseDataset)
```

| method | description |
|---|---|
| `row_indices_for_ids(ids)` |  |
| `describe()` |  |

## `eval21.data.cubes`

eval21 cube cache (``eval21 build`` output, ``attrs["format"] == "eval21-cubes/1"``).

```python
def is_cube_cache(path) -> bool
```

```python
class CubeCacheDataset(path, split: str | None=None, limit: int | None=None, target: str='xhi', density: str='density', indices: list[int] | None=None, box_mpc: float | None=None, z_range=None, fields=None, require_valid: bool=True, dtype=np.float32, **_ignored)  # (BaseDataset)
```

| method | description |
|---|---|
| `row_indices_for_ids(ids)` |  |
| `describe()` |  |

## `eval21.data.synthetic`

Synthetic 21-cm-like lightcones, for tests and demos without data.

```python
def make_cone(shape=(64, 64, 64), box_mpc: float=200.0, seed: int=0, z_min: float=5.0, z_max: float=25.0)
```

Return ``(density, xhi, z_grid)`` for one synthetic lightcone.

```python
class SyntheticDataset(n_samples: int=4, shape=(64, 64, 64), box_mpc: float=200.0, seed: int=0, split: str | None=None, limit: int | None=None, indices: list[int] | None=None)  # (BaseDataset)
```

Deterministic artificial lightcones with the same geometry per sample.

| method | description |
|---|---|
| `describe()` |  |

```python
def write_demo(directory, n_samples: int=4, shape=(64, 64, 64), box_mpc: float=200.0, seed: int=0) -> Path
```

Write a manifest + npz cubes, ready for ``eval21 run --data``.

## `eval21.build`

``eval21 build``: a folder of simulations -> one eval21 cube cache (HDF5).

```python
def assign_splits(ids: list[str], splits: str | None=None, split_file: str | None=None, seed: int=0) -> dict[str, str]
```

{id: split}. ``splits``: ``test`` (all), or ``train=0.8,val=0.1,test=0.1`` (seeded shuffle of the sorted ids). ``split_file``: JSON ``{split: [ids]}``, ``{id: split}`` or a training ``run_metadata.json`` with ``<split>_cone_ids``.

```python
class CacheWriter(path, fields, shape, z_grid, attrs, dtype='float32', compress=None)
```

| method | description |
|---|---|
| `append(sid, split, fields, valid, omega_m, params)` |  |
| `close(skipped=())` |  |

```python
def build(folder, out, fmt='auto', fields=None, targets=None, z_grid=None, transverse=None, splits=None, split_file=None, seed=0, limit=None, ids=None, shard=0, num_shards=1, workers=1, dtype='float32', compress=None, source_options=None, quiet=False, command='') -> Path
```

Build (one shard of) a cache; returns the written path.

```python
def merge(out, num_shards: int, quiet: bool=False, remove: bool=False) -> Path
```

Join ``out.shardKKKofNNN.h5`` files into ``out`` (rows in shard-interleaved source order).

## `eval21.build.grid`

Target grids and resampling for ``eval21 build``.

```python
def parse_z_grid(spec: str | None, cell_mpc: float | None=None, omega_m: float=0.31, hubble_h: float=0.678) -> np.ndarray | None
```

Grid from a spec string; ``None`` / ``native`` returns None (keep the source grid).

```python
class LosInterpolator(source_z, target_z)
```

Linear interpolation weights from one source grid to one target grid (reused per field).

| method | description |
|---|---|
| `valid(source_valid: np.ndarray \| None)` | Target planes whose two bracketing source planes are both valid. |

```python
def resample_transverse(cube: np.ndarray, n_out: int) -> np.ndarray
```

(Nx, Ny, Nz) -> (n_out, n_out, Nz), linear between cell centres on the same box.

## `eval21.build.sources`

Readers that turn one simulation file into fields on its native grid (``eval21 build``).

```python
class SourceItem
```

| field | type | default |
|---|---|---|
| `id` | `str` | `—` |
| `path` | `Path` | `—` |
| `extra` | `dict` | `field(default_factory=dict)` |

```python
class RawSample
```

| field | type | default |
|---|---|---|
| `fields` | `dict` | `—` |
| `z` | `np.ndarray` | `—` |
| `valid` | `np.ndarray \| None` | `None` |
| `params` | `dict` | `field(default_factory=dict)` |
| `box_mpc` | `float \| None` | `None` |
| `omega_m` | `float \| None` | `None` |

```python
class Source
```

| field | type | default |
|---|---|---|
| `default_fields` | `tuple` | `()` |
| `units` | `dict` | `{}` |

| method | description |
|---|---|
| `items()` |  |
| `read(item: SourceItem, fields)` |  |
| `describe()` |  |

```python
class Fast21cmSource(folder, pattern: str='*.h5', id_from: str='number')  # (Source)
```

py21cmfast lightcone HDF5 files.

| method | description |
|---|---|
| `items()` |  |
| `read(item, fields)` |  |

```python
class LoreliSource(folder, l_perp: float=294.0, omega_m: float=0.308, params_json: str | None=None, five_field_only: bool | None=None)  # (Source)
```

LoReLi II ``lightcone_*_simuN.dat`` files (see ``README_loreli_lc128.md``).

| method | description |
|---|---|
| `items()` |  |
| `params_of(sim: int)` |  |
| `read(item, fields)` |  |
| `describe()` |  |

```python
class NpzSource(folder, pattern: str='*.npz', z_key: str='z_grid')  # (Source)
```

A folder of per-sample ``.npz`` files (eval21's own cube format, or anything similar).

| method | description |
|---|---|
| `items()` |  |
| `read(item, fields)` |  |

```python
def open_source(kind: str, folder, **options) -> Source
```

``21cmfast``, ``loreli``, ``npz``, ``python:FILE.py[:Class]`` or ``auto``.

## `eval21.adapters`

Adapter selection from a ``name=adapter:argument`` CLI spec.

```python
def split_spec(spec: str) -> tuple[str, str]
```

```python
def build(spec: str, fno_root: str | None=None, device: str | None=None, target: str='xhi') -> BaseAdapter
```

## `eval21.adapters.base`

Model adapter interface.

```python
class ModelAdapter  # (Protocol)
```

| field | type | default |
|---|---|---|
| `framework` | `str` | `—` |

| method | description |
|---|---|
| `predict(sample: Sample)` |  |
| `describe()` |  |

```python
class BaseAdapter
```

| method | description |
|---|---|
| `predict(sample: Sample)` |  |
| `describe()` |  |
| `close()` | Release device memory between models (optional). |

## `eval21.adapters.python`

Load a model from a plain Python file.

```python
class PythonModelAdapter(path, attribute: str='predict')  # (BaseAdapter)
```

| method | description |
|---|---|
| `predict(sample: Sample, draw: int \| None=None)` |  |
| `describe()` |  |
| `close()` |  |

## `eval21.adapters.torch`

In-process PyTorch adapter.

```python
class TorchModelAdapter(module, channels=None, input_key: str='x', device=None, output_field: str='xhi', output_channels: int=1, half: bool=False)  # (BaseAdapter)
```

| method | description |
|---|---|
| `predict(sample: Sample)` |  |
| `benchmark(samples, warmup: int=3, iters: int=10)` | Time forward passes with inputs already on the device. |
| `describe()` |  |
| `close()` |  |

```python
def load_torch(model, checkpoint=None, **kwargs) -> TorchModelAdapter
```

Wrap an already-built module, optionally loading a raw state dict.

## `eval21.adapters.command`

Language-agnostic adapter: shell out to any inference program.

```python
class CommandModelAdapter(command: str, timeout: float | None=None, include_targets: bool=False)  # (BaseAdapter)
```

| method | description |
|---|---|
| `predict(sample: Sample)` |  |

## `eval21.adapters.fno21cm`

Adapter for fno-21cm checkpoints.

```python
class Fno21cmAdapter(checkpoint, project_root: str | Path | None=None, device: str | None=None, target: str='xhi')  # (BaseAdapter)
```

| method | description |
|---|---|
| `predict(sample: Sample)` |  |
| `benchmark(samples, warmup: int=3, iters: int=10)` | Reference-style forward timing with pre-transferred inputs. |
| `describe()` |  |
| `close()` |  |

```python
def metadata_directory(checkpoint) -> Path
```

Directory of run_metadata.json, following the path as given.

```python
def cache_contract(metadata: dict) -> dict | None
```

Return the cache identity a run recorded, if any.

```python
def validate_cache_contract(metadata: dict, dataset_info: dict, checkpoint: str | Path) -> None
```

Fail when a run's recorded cache identity does not match the data.

## `eval21.adapters.toy`

A deliberately simple density -> x_HI "model", for demos and smoke tests.

```python
class SmoothThresholdModel(sigma: float=1.5, width: float=0.1, threshold: float | None=None)  # (BaseAdapter)
```

| method | description |
|---|---|
| `predict(sample: Sample)` |  |
| `describe()` |  |

```python
class ToyGenerativeModel  # (SmoothThresholdModel)
```

A parameters-only "generative model": x_HI from a fresh random field per draw.

| method | description |
|---|---|
| `predict(sample: Sample, draw: int=0)` |  |

## `eval21.metrics.base`

Metric base class and shared helpers.

```python
class Metric(settings: Settings, options: dict[str, Any] | None=None)
```

One evaluation family.

| field | type | default |
|---|---|---|
| `needs` | `tuple[str, ...]` | `()` |
| `pointwise` | `bool \| str` | `False` |
| `targets` | `tuple[str, ...] \| None` | `None` |
| `pointwise_keys` | `tuple[str, ...]` | `()` |

| method | description |
|---|---|
| `attach(model_info: dict[str, Any])` | Receive static model information (parameter count, device, ...). |
| `update(sample, prediction, aux=None, runtime=None)` |  |
| `result()` |  |
| `arrays()` | Arrays worth exporting to ``curves.npz`` (empty by default). |
| `target(sample)` |  |
| `prediction(prediction)` |  |
| `stage_xbar(sample, truth=None)` | Slice means that define the reionization stages. |
| `stages(truth, sample=None)` |  |
| `require(sample, field: str)` |  |
| `box_mpc(sample)` |  |
| `omega_m(sample)` |  |
| `z_grid(sample)` |  |

```python
def median_band(rows, axis: int=0) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

NaN-aware median with 16-84 percentiles.

```python
def nanmedian(values) -> float
```

NaN-aware median without the all-NaN warning (empty stages are normal).

```python
def finite_or_none(values)
```

JSON-safe list: non-finite floats become ``None``.

## `eval21.visualization`

Figures from 21cmEVAL result directories (``eval21 plot``; needs matplotlib).

```python
from eval21.visualization import (
    Catalog, Run, find_runs, PlotContext, SkipFigure,
    figure, load_all, run,
    plot_cylindrical, plot_gallery,
)
```

The package re-exports the figure registry (`figure`, `load_all`, `run`,
`PlotContext`, `SkipFigure`), the result loader (`Run`, `Catalog`, `find_runs`)
and the hand-written figures (`plot_cylindrical`, `plot_gallery`).
Importing it does not import matplotlib.

## `eval21.visualization.registry`

Figure registry: every figure is a function registered under a short name.

```python
class Figure
```

| field | type | default |
|---|---|---|
| `name` | `str` | `—` |
| `func` | `Callable` | `—` |
| `group` | `str` | `'misc'` |
| `needs` | `tuple` | `('curves',)` |
| `help` | `str` | `''` |

```python
class PlotContext
```

Everything a figure function gets.

| field | type | default |
|---|---|---|
| `catalog` | `object` | `—` |
| `out` | `Path` | `—` |
| `store` | `object` | `None` |
| `ids` | `list \| None` | `None` |
| `title` | `str` | `''` |
| `options` | `dict` | `field(default_factory=dict)` |
| `quiet` | `bool` | `False` |

| method | description |
|---|---|
| `opt(key, default=None, cast=None)` |  |

```python
def figure(name: str, group: str='misc', needs=('curves',), help: str='')
```

```python
def load_all() -> dict[str, Figure]
```

```python
def run(names, ctx: PlotContext) -> dict
```

Run the named figures (``all``: every registered one). Returns {name: written | reason}.

```python
class SkipFigure  # (Exception)
```

Raised by a figure whose inputs are absent from the runs (e.g. a metric that was not run).

## `eval21.visualization.common`

Loading 21cmEVAL result directories for the figures.

```python
def field_name(field: str) -> str
```

```python
class Run(path: Path)
```

One run file (``eval21-run/1``) or legacy result directory, read through one interface.

| method | description |
|---|---|
| `group` | property: Short label of the data this run scored (manifest's directory, or the data path). |
| `curves(model: str)` |  |
| `metric(model: str, name: str)` |  |

```python
class Merged(model: str, runs)
```

Several runs of one (target, data, model), seen as one.

| method | description |
|---|---|
| `curves(model)` |  |
| `metric(model, name)` |  |

```python
def find_runs(paths) -> list[Run]
```

```python
class Catalog(runs, models=None, style=None, quiet=False)
```

Runs indexed by target, group and model, with labels and line styles.

| method | description |
|---|---|
| `label(m)` |  |
| `color(m)` |  |
| `ls(m)` |  |
| `tick(m)` |  |
| `runs_for(target, model)` |  |
| `group_labels(target)` |  |
| `group_sizes(target, key='rmse_per_sample', metric_prefix=None)` |  |
| `curves(target, model, group_index=0)` |  |
| `pooled(model, key, target='xhi')` | Concatenate a per-sample array over the groups (axis 0); empty if absent. |
| `has(target, key)` |  |

```python
def med_band(a, axis=0)
```

```python
def pyplot()
```

```python
def shared_legend(fig, axes=None, *, handles=None, labels=None, extra=(), drop=(), fontsize=8.5, ncol=None, title=None)
```

Collect the legend entries of ``axes`` (default: all axes of ``fig``), remove the per-axes legends and register one figure legend for :func:`save` to place above the panels.

```python
def model_handles(cat, models=None, truth=True, lw=1.6, markers=None, kind='line')
```

Proxy handles: ``truth`` (black, thick) followed by each model's colour and style.

```python
def style_key(entries, color='0.35', lw=1.4)
```

Grey key entries for a second encoding: ``[("k = 0.11", "-"), ("k = 0.22", "--")]`` for line styles, or ``[("median", {"ls": "-"}), ("16-84 %", {"kind": "band"})]``.

```python
def pane_values(ax, values, label=None, loc='below', fmt='{:.3g}', fontsize=7.5)
```

Per-panel statistics as one line under the panel: ``label  ── 0.127  - - 0.079``.

```python
def wrap_label(text: str, width: int=22) -> str
```

Break a long model label into lines of at most ``width`` characters (panel titles, row labels).

```python
def save(fig, out: Path, name: str, dpi=200, quiet=False)
```

## Registered metric classes

Each class is decorated with `@register("name")` in its module; the name is what
`--metrics` and `[metrics] enabled` accept.

| name | class | module | needs | targets | pointwise |
|---|---|---|---|---|---|
| `anisotropy` | `AnisotropyMetrics` | `eval21.metrics.anisotropy` | `()` | `None` | `partial` |
| `bubble` | `BubbleSizeMetrics` | `eval21.metrics.bubble` | `()` | `('xhi',)` | `False` |
| `cylindrical` | `CylindricalMetrics` | `eval21.metrics.cylindrical` | `()` | `None` | `partial` |
| `dtb` | `DtbMetrics` | `eval21.metrics.dtb` | `()` | `None` | `False` |
| `edge` | `EdgeMetrics` | `eval21.metrics.edge` | `()` | `None` | `True` |
| `efficiency` | `EfficiencyMetrics` | `eval21.metrics.efficiency` | `()` | `None` | `False` |
| `eft` | `EftMetrics` | `eval21.metrics.eft` | `('density',)` | `None` | `False` |
| `ensemble` | `EnsembleMetrics` | `eval21.metrics.ensemble` | `()` | `None` | `partial` |
| `front` | `FrontMetrics` | `eval21.metrics.front` | `()` | `('xhi',)` | `partial` |
| `front3d` | `Front3dMetrics` | `eval21.metrics.front3d` | `()` | `('xhi',)` | `True` |
| `gallery` | `GalleryMetric` | `eval21.metrics.gallery` | `()` | `None` | `False` |
| `history` | `HistoryMetrics` | `eval21.metrics.history` | `()` | `('xhi',)` | `False` |
| `morphology` | `MorphologyMetrics` | `eval21.metrics.morphology` | `()` | `('xhi',)` | `partial` |
| `observable` | `ObservableMetrics` | `eval21.metrics.observable` | `('density',)` | `('xhi',)` | `False` |
| `parity` | `ParityMetrics` | `eval21.metrics.parity` | `()` | `None` | `True` |
| `radial` | `RadialMetrics` | `eval21.metrics.radial` | `()` | `None` | `partial` |
| `spectrum` | `SpectrumMetrics` | `eval21.metrics.spectrum` | `()` | `None` | `partial` |
| `survey` | `SurveyMetrics` | `eval21.metrics.survey` | `()` | `('xhi',)` | `False` |
| `symmetry` | `SymmetryMetrics` | `eval21.metrics.symmetry` | `()` | `None` | `True` |
| `voxel` | `VoxelMetrics` | `eval21.metrics.voxel` | `()` | `None` | `True` |
