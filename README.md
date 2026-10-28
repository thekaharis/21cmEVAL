# 21cmEVAL

End-to-end evaluation for 21cm field emulators. Point it at a model and a set
of lightcones: it runs inference once and reports every diagnostic used to
compare the models in `fno-21cm` — voxel accuracy, scale-resolved power,
bubble sizes, reionization history, the 21-cm observable, EFT bias terms,
ionization-front sharpness, and inference cost.

The suite is model-agnostic by construction. A model is anything that maps a
sample to named 3-D arrays; adapters exist for plain Python functions,
PyTorch modules, external programs (read npz / write npz), and fno-21cm
checkpoints. Metrics are numpy-only, streaming accumulators, so a 200-cone
sweep costs one forward pass and constant memory.

```
cm21eval demo --out demo          # synthetic data + toy model, ~10 s
cm21eval selftest                 # analytic checks, no data needed
```

## Install

```bash
python -m pip install -e .            # numpy + scipy
python -m pip install -e ".[all]"     # + torch, h5py, pytest
```

Python 3.10+.

## Quick start

```bash
# 1. Score a toy model on generated data
cm21eval demo --out demo

# 2. Score an fno-21cm run on its own cube cache
cm21eval run \
    --data /work/data/compressed/cubes_3d.h5 \
    --model fno=fno21cm:/work/fno-21cm/checkpoints/3d_xhi/fno/ckpt/best_model_state_dict.pt \
    --fno-root /work/fno-21cm \
    --split test --limit 200 --out results/fno

# 3. Compare several models in one pass over the data
cm21eval run --data cubes/ \
    --model fno=fno21cm:.../best_model_state_dict.pt \
    --model ufno=fno21cm:.../best_model_state_dict.pt \
    --fno-root /work/fno-21cm \
    --metrics voxel,spectrum,bubble,history,front,efficiency \
    --out results/compare
```

`results/` then contains:

| file | content |
|---|---|
| `results.json` | complete report, schema `cm21eval/1`, all nested metrics |
| `summary.csv` | every scalar leaf as one column, one row per model |
| `report.md` | headline table plus per-stage tables |
| `curves.npz` | spectra, PDFs, profiles, per-sample records |

## The data contract

A sample is one lightcone: named inputs, named targets, geometry.

```python
Sample(
    id="cone0000",
    inputs={"density": (Nx, Ny, Nz)},   # evolved matter density, mean 1
    targets={"xhi": (Nx, Ny, Nz)},      # neutral fraction in [0, 1]
    meta=Meta(box_mpc=200.0, z_grid=(Nz,), omega_m=0.30966),
)
```

Arrays are `(Nx, Ny, Nz)`: `x`/`y` periodic transverse axes, `z` the
lightcone axis. Physics metrics use `density`; a sample without it is still
scored on every metric that needs only `xhi` (the others are listed under
`skipped` in the report).

### Saved cubes (`.npz`)

Without a manifest, every `*.npz` in a directory is one sample; the target is
`xhi`, every other 3-D array is an input, and 1-D `z_grid` / `omega_m` set the
geometry.

With a manifest (paths relative to it):

```json
{
  "box_mpc": 200.0,
  "z_grid": [5.0, 5.1, "…", 25.0],
  "target": "xhi",
  "inputs": ["density"],
  "samples": [
    {"id": "cone0000", "path": "cubes/cone_0000.npz", "split": "test",
     "omega_m": 0.31, "params": {"F_ESC10": -1.2}}
  ]
}
```

### HDF5 cube caches

`--data cubes_3d.h5` auto-detects the fno-21cm layout: `x` `(N, C, Nx, Ny,
Nz)` + `y`, or `density` + `neutral_fraction` + `target_z` + `params`.
For a different layout construct `Hdf5Dataset` directly with explicit keys —
`cm21eval.data.hdf5` documents every argument.

Rows are **not** split automatically. Pass `--indices 0-199` for explicit row
numbers, or `--indices-json path/to/run_metadata.json` to reuse the split the
checkpoint was trained with (`split.test_indices`); evaluating training cones
as "test" is the easiest way to produce a meaningless report.

## Model adapters

| spec | meaning |
|---|---|
| `python:model.py[:predict]` | file defining `predict(sample) -> dict` |
| `torch:model.py[:build[:ckpt]]` | file defining `build()` returning an `nn.Module` |
| `command:"python infer.py {input} {output}"` | any program: read npz, write npz |
| `fno21cm:path/best_model_state_dict.pt` | fno-21cm run, uses its `run_metadata.json` |
| `synthetic:1.5` | built-in smoothed-threshold toy model |

The Python adapter is the whole contract:

```python
# my_model.py
import torch

_model = torch.jit.load("model.pt")

def predict(sample):
    x = torch.from_numpy(sample.inputs["density"])[None, None].float()
    with torch.no_grad():
        return {"xhi": _model(x)[0, 0].numpy()}

def describe():
    return {"framework": "torchscript", "parameters": 1_234_567}
```

`describe()` is optional and feeds the efficiency report. For the fno-21cm
adapter the architecture, input channels and parameter normalization all come
from the checkpoint's own `run_metadata.json` — nothing is duplicated, so
legacy, local-operator, waveform and contrast-wrapped runs load through the
repository's canonical loader.

An external program gets one npz per sample containing every input and target
array plus a `meta` JSON string, and must write an npz with one array per
target name. That is the escape hatch for Julia, C++ or a cluster job.

## Metrics

Stage-resolved metrics bin LOS slices by the **truth** slice mean x_HI, so
truth and prediction are always paired. `active` is the last row: slices with
`0.05 <= xbar_HI <= 0.95`.

| metric | reports | source in `fno-21cm` |
|---|---|---|
| `voxel` | RMSE, MSE, MAE, bias, R², Pearson r, R² per sample (mean/p16/p84) | `viz/rmse_r2_eval.py`, `util/multifield_metrics.py` |
| `edge` | transverse edge vs interior RMSE, active-window RMSE, global-history RMSE | `util/metrics_21cm.py`, `viz/physical_evaluation.py` |
| `spectrum` | transverse P(k) ratio, coherence r(k), Δ²(k), k where r drops below 0.9 | `viz/power_spectrum_evaluation.py` |
| `bubble` | mean-free-path bubble-size PDFs, restricted Wasserstein-1, Jensen-Shannon, size bias, censored fraction | `viz/bubble_size_evaluation.py`, `util/metrics_21cm.py` |
| `morphology` | 3-D neutral-region count, mean and volume-weighted radius, largest-bubble fraction, ionized-mask IoU at matched ionized fraction | `util/metrics_21cm.py`, `tools_excursion_set_diagnostic.py` |
| `history` | τ_e, z(x_HI = 0.25/0.5/0.75), Δz, errors in Planck 2018 σ | `util/physical_stats.py`, `viz/physical_evaluation.py` |
| `observable` | x_HI and dT_b PDFs (W1), partially ionized fraction, dT_b variance/skewness, Δ²₂₁(k), power bias in units of single-slice sample variance | `viz/physical_evaluation.py` |
| `eft` | transfer T(k), stochasticity P_ε(k), r_uδ, fits b₁ + b_k²k² and P_ε,0 + P_ε,2k² | `viz/physical_evaluation.py`, `util/physical_stats.py` |
| `front` | RMSE and gradient RMS vs signed distance to the true front, band RMSE, 10–90 front width | `viz/boundary_band_diagnostic.py` |
| `parity` | conditional bias E[pred\|true] − true per truth bin, per-bin spread and RMSE, partial-range RMSE | `viz/parity_diagnostic.py` |
| `radial` | isotropic 3-D P(k) ratio and coherence (cross-check; non-periodic LOS) | `util/metrics_21cm.py` |
| `anisotropy` | cylindrical (k_perp, k_par) power ratio and coherence at z = 7/9/11 | `viz/power_spectrum_evaluation.py` |
| `efficiency` | parameters, ms per lightcone (median/p16/p84), samples/s, slices/s, peak memory | `viz/inference_speed_eval.py` |
| `symmetry` | RMS change under a transverse x↔y swap | `viz/visualize_3d.py` |

### Definitions worth stating

* **R²** uses the constant-mean predictor as reference. x_HI is bimodal, so
  this reference is poor and R² is flattering — a relative score, not an
  accuracy.
* **Spectra** are 2-D transverse spectra of individual slices, in Mpc². The
  LOS axis mixes geometry with evolution and the 256-cell grid discards most
  LOS structure, so a 3-D transform would not be interpretable. `k` therefore
  corresponds to 3-D wavenumbers of roughly 1–2 k_perp.
* **τ_e** integrates a global history with helium single ionization and He II
  → He III at z = 3; histories are clipped to [0, 1] so emulator overshoot
  cannot produce unphysical electron densities. Errors are quoted in units of
  the Planck 2018 σ (τ: 0.0073, z_re: 0.73).
* **dT_b** is the saturated-spin-temperature, no-RSD limit,
  `27 x_HI (1+δ) (Ω_b h²/0.023) sqrt(0.15/(Ω_m h²) (1+z)/10)` mK, with the
  *same* density for truth and prediction. It is not the 21cmFAST
  `brightness_temp` field (no T_S, no velocities); differences isolate the
  x_HI error.
* **Bubble sizes** are ionized-area-weighted mean free paths to the first
  neutral cell through periodic transverse slices, with censored rays folded
  in at `max_distance_mpc`. A blurry model over-produces small bubbles.
* **Power-bias significance** is `(P_pred − P_truth) / (P_truth sqrt(2/N_modes))`:
  the emulator bias in units of the Gaussian sample variance of one slice.
  `1/significance²` is how many independent slices an observer must average
  before the bias equals their error.
* **EFT** coefficients are fit per stage and compared cone by cone. A
  conditional-mean emulator usually reproduces T(k) but under-predicts P_ε.

## Configuration

Everything on the command line can live in a TOML file:

```toml
[run]
data = "/work/data/compressed/cubes_3d.h5"
out = "results/fno"
split = "test"
limit = 200
target = "xhi"
density = "density"
fno_root = "/work/fno-21cm"

[models]
fno = "fno21cm:/work/fno-21cm/checkpoints/3d_xhi/fno/ckpt/best_model_state_dict.pt"

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

CLI flags override the file. `--set key=value` overrides one setting.
`cm21eval list-metrics` prints the registry.

### Adding a metric

```python
from cm21eval.registry import register
from cm21eval.metrics.base import Metric

@register("mystat")
class MyStat(Metric):
    needs = ("density",)          # optional input fields

    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        self.total = 0.0

    def update(self, sample, prediction, aux=None, runtime=None):
        truth = self.target(sample)
        pred = self.prediction(prediction)
        self.total += float(((pred - truth) ** 2).mean())

    def result(self):
        return {"mean_mse": self.total / max(1, len(self.__dict__))}
```

Import the module before the run (a small wrapper script that imports it and
calls `cm21eval.cli.main`) and pass `--metrics mystat`.

## Design

```
src/cm21eval/
├── physics.py        # cosmology, spectra, MFP rays, front widths (numpy)
├── spec.py           # Sample / Meta contract
├── settings.py       # one frozen dataclass for the whole suite
├── registry.py       # metric plugins by name
├── metrics/          # one module per family, all streaming accumulators
├── adapters/         # python, torch, command, fno21cm, synthetic
├── data/             # npz + manifest, hdf5 cube cache, synthetic
├── runner.py         # one pass per model, skip reporting, timing
├── report.py         # JSON / CSV / Markdown / NPZ
├── cli.py
└── selftest.py       # analytic checks with known answers
```

* **One pass.** Every enabled metric updates from the same prediction.
* **Constant memory.** Accumulators are sums, histograms and per-cone scalars;
  no metric stores a cube after `update` returns.
* **Fail loudly.** Shape mismatches, missing channels and unknown metrics
  raise; a metric that cannot run is listed under `skipped` with the reason.
* **Numpy core.** `import cm21eval` does not import torch. Torch appears only
  inside the torch and fno-21cm adapters.

## What this suite does not do

* Train or fine-tune models, or convert checkpoints between architectures.
* Produce the PNG dashboards from `fno-21cm/viz` — metrics are numbers here;
  plotting belongs to the analysis notebook of the paper.
* Evaluate the LOS resampling grid itself (`viz/los_grid_evaluation.py`); that
  is a dataset diagnostic, not a model score.
* Know your units. Density must be the evolved field with mean 1 and the box
  size must be the transverse comoving size in Mpc; everything else follows.

## Tests

```bash
python -m pytest          # 57 tests, ~2 s, no data and no GPU
cm21eval selftest         # same analytic checks, no pytest

# Optional: pin the physics primitives to the fno-21cm reference code
FNO21CM_ROOT=/work/fno-21cm python -m pytest tests/test_reference_agreement.py
```

`tests/test_reference_agreement.py` asserts bit-level agreement with
`fno-21cm` for tau_e, crossing redshifts, dT_b, binned power, the WLS fit and
mean-free-path ray tracing (same RNG stream and seeds). It skips without
`FNO21CM_ROOT`.

The tests pin the definitions that are easy to get subtly wrong: RMSE and R²
against closed forms, coherence = 1 for identical fields, τ_e ordering, a ray
tracer against a half-plane, a sphere's effective radius, front width of a
known sigmoid, and a strict-JSON report.
