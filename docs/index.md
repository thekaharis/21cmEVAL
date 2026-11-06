# 21cmEVAL documentation

21cmEVAL (import package `eval21`, distribution `21cmeval`) is an end-to-end evaluation suite
for 21-cm field emulators. Point it at a model and a set of lightcones; it runs inference once
and reports every diagnostic used to compare neural-operator models of reionization — voxel
accuracy, scale-resolved and cylindrical power, bubble sizes, the reionization history, the
21-cm observable, EFT bias terms, ionization-front sharpness and inference cost — then draws
the figures of the fno-21cm and rhizome projects from those results.

The suite is model-agnostic by construction. A model is anything that maps a `Sample` to named
3-D arrays; adapters exist for plain Python functions, PyTorch modules, external programs
(anything that reads npz and writes npz) and fno-21cm checkpoints. Metrics are numpy-only
streaming accumulators, so a 200-cone sweep costs one forward pass and constant memory.

```text
eval21 demo --out demo          # synthetic data + toy model, ~10 s
eval21 selftest                 # analytic checks, no data needed
```

## Install

```bash
python -m pip install -e .            # numpy + scipy
python -m pip install -e ".[all]"     # + torch, h5py, matplotlib, pytest
```

Python 3.10 or newer. The core (`import eval21`) does not import torch or matplotlib; those
appear only inside the adapters and the visualization package.

| extra | pulls in | needed for |
|---|---|---|
| `torch` | `torch>=2.0` | `torch:` and `fno21cm:` adapters |
| `hdf5` | `h5py>=3.8` | HDF5 cube caches and `eval21 build` |
| `viz` | `matplotlib>=3.6` | `eval21 plot` |
| `dev` | `pytest>=7` | the test suite |
| `all` | all of the above | development |

## The five-minute tour

```bash
# 1. Synthetic data and a toy model: checks the installation end to end
eval21 demo --out demo

# 2. Score an fno-21cm run on its own cube cache
eval21 run \
    --data /work/data/compressed/cubes_3d.h5 \
    --indices-json /work/fno-21cm/checkpoints/3d_xhi/fno/ckpt/run_metadata.json \
    --model fno=fno21cm:/work/fno-21cm/checkpoints/3d_xhi/fno/ckpt/best_model_state_dict.pt \
    --fno-root /work/fno-21cm \
    --expect-shape 140,140,256 \
    --limit 200 --out results/fno

# 3. Compare several models in one pass over the data
eval21 run --data cubes/ \
    --model fno=fno21cm:.../best_model_state_dict.pt \
    --model ufno=fno21cm:.../best_model_state_dict.pt \
    --fno-root /work/fno-21cm \
    --metrics voxel,spectrum,bubble,history,front,efficiency \
    --out results/compare

# 4. Draw every figure the results support
eval21 plot all --run results/ --out figs/
```

A run writes `results.json`, `summary.csv`, `stages.csv`, `report.md` and `curves.npz` to
`--out`. `eval21 plot` reads those files back; it never re-runs a model.

## Documentation map

| document | contents |
|---|---|
| [Architecture](architecture.md) | design principles, control flow, package map, object model, extension points |
| [Command line](cli.md) | every subcommand and option, TOML configuration, selection rules |
| [Data](data.md) | `Sample`/`Meta` contract, dataset formats, `eval21 build`, cube-cache layout, prediction store |
| [Metrics](metrics.md) | stage binning, every metric's definition and result keys, settings, caveats |
| [Model adapters](adapters.md) | adapter protocol, `python:`/`torch:`/`command:`/`fno21cm:` specs, draws, cache contracts |
| [Reports](reports.md) | `results.json` schema, CSV/NPZ layouts, the Markdown report |
| [Visualization](visualization.md) | the `eval21.visualization` package, figure registry, all 23 figures, layout conventions |
| [API reference](api.md) | public functions and classes by module, with signatures |
| [Development](development.md) | test suite and fixture, conventions, adding metrics/figures/datasets |
| [Agent handoff](agent-handoff.md) | changelog of the visualization reorganisation, merges and new metrics, with migration notes |
| [Run file format](design/run-file.md) | design spec: `eval21-run/1` HDF5 files and the optional `eval21-cache/1` inference cache |

## The two contracts

**Data.** A sample is one lightcone: named inputs, named targets, geometry.

```python
Sample(
    id="cone0000",
    inputs={"density": (Nx, Ny, Nz)},   # overdensity δ = ρ/ρ̄ − 1 (mean 0), as in 21cmFAST
    targets={"xhi": (Nx, Ny, Nz)},      # neutral fraction in [0, 1]
    meta=Meta(box_mpc=200.0, z_grid=(Nz,), omega_m=0.30966),
)
```

Arrays are `(Nx, Ny, Nz)`: `x`/`y` are the periodic transverse axes, `z` the lightcone axis.

**Model.** An adapter maps a sample to named arrays:

```python
def predict(sample) -> dict[str, np.ndarray]:   # {"xhi": (Nx, Ny, Nz)}
    ...
```

Everything else — datasets, metrics, figures — is built around those two.

## What this suite does not do

- Train or fine-tune models, or convert checkpoints between architectures.
- Plot model internals (spectral weights, gates, training curves). `eval21 plot` draws
  evaluation figures only.
- Evaluate the LOS resampling grid itself; that is a dataset diagnostic, not a model score.
- Score several output fields in one pass. Multifield runs (x_HI, ΔT_b, velocity) are evaluated
  one `--target` at a time; metrics that are x_HI statistics are skipped with a reason on a
  T_b pass.
- Guess units. Density must be the evolved field with mean 1, and the box size must be the
  transverse comoving size in Mpc.

## Provenance

The metrics and figures reimplement, and are validated against, the evaluation code in
[fno-21cm](https://github.com/thekaharis/fno-21cm) and
[rhizome-neural-operator](https://github.com/thekaharis/rhizome-neural-operator). Source paths
quoted in [Metrics](metrics.md) and [Visualization](visualization.md) refer to those
repositories' original scripts, not to files in this one.
