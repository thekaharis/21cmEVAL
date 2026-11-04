# Model adapters

An adapter is the only place that knows about a training framework. It maps one `Sample` to
named prediction arrays:

```python
prediction = adapter.predict(sample)   # {"xhi": (Nx, Ny, Nz)}
```

Everything else in the suite is framework-agnostic. Adapters are selected on the command line
as `--model NAME=ADAPTER:ARGUMENT`, where `NAME` is the label used in every report and figure.

## Adapter specs

| spec | meaning |
|---|---|
| `python:model.py[:predict]` | a file defining `predict(sample) -> dict` (any function name after the path) |
| `torch:model.py[:build[:ckpt]]` | a file defining `build()` returning an `nn.Module`; optional state dict |
| `command:"prog {input} {output}"` | any program that reads npz and writes npz |
| `fno21cm:path/best_model_state_dict.pt` | an fno-21cm run, rebuilt from its `run_metadata.json` |
| `synthetic:1.5` | built-in smoothed-threshold toy model (density → x_HI) |
| `toygen:1.5` | built-in parameters-only generative toy: a fresh random field per draw |

Aliases: `py`/`numpy` for `python`, `toy` for `synthetic`, `synthetic-generative` for `toygen`,
`fno` for `fno21cm`, `h5`/`hdf5` for data specs. Anything after the first colon is
adapter-specific, so paths may contain colons.

## `python:` — a plain Python file

The smallest contract in the suite. The file must define the named function (default
`predict`), and may define `describe()` and `close()`.

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

```bash
eval21 run --data cubes/ --model mine=python:my_model.py --out results/mine
```

`describe()` is optional; its keys are merged into the model info block and feed `efficiency`
(`parameters`, `device`, `framework`, …). `close()` is called after the pass, between models,
so a file that allocates a CUDA context can release it.

For stochastic models the adapter looks for `predict_draws` and inspects `predict`'s signature
for a `draw` parameter; see [Draws](#draws).

## `torch:` — an in-process module

The file defines a builder that returns an `nn.Module`. The builder is called with the
checkpoint path when one is given, else with no arguments; if no checkpoint is given and the
module defines `load_checkpoint(model)`, that is called instead.

```python
# model_file.py
import torch.nn as nn

def build(checkpoint=None):
    model = MyUNet()
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model
```

```bash
eval21 run --data cubes.h5 --model unet=torch:model_file.py:build:weights.pt --device cuda
```

Input conventions, in order:

1. `sample.inputs["x"]` with shape `(C, Nx, Ny, Nz)` (the fno-21cm convention) is used as-is;
2. `channels=(...)` names input fields stacked in order; a 3-D field becomes one channel, a 1-D
   array of length `Nz` is broadcast per slice;
3. with no channel list and no `x`, a `density` input is used as one channel.

The module is called as `module(x=...)` first and `module(...)` second, so both wrapper-style
models and plain `nn.Module`s work. Output may be a tensor, a tuple/list (first item) or a
mapping (the `--target` key, else the first value); 4-D/5-D tensors are squeezed to
`(Nx, Ny, Nz)`. The adapter counts real and complex parameters for the efficiency report and
implements `benchmark()` with pre-transferred inputs, CUDA synchronisation and discarded
warmup — the same protocol as fno-21cm's `inference_speed_eval.py`.

From Python, `load_torch(model, checkpoint=None, **kwargs)` wraps an already-built module and
optionally loads a raw state dict (stripping a `module.` prefix and unwrapping a `"model"`
entry).

## `command:` — any program

The universal escape hatch: read npz, write npz. The command runs once per sample with
`{input}` and `{output}` replaced by temporary paths.

```bash
eval21 run --data cubes/ --model julia="command:julia infer.jl {input} {output}"
```

The input npz contains every sample input under its field name plus a 0-d string array `meta`
holding JSON:

```json
{"id": "cone0000", "box_mpc": 200.0, "omega_m": 0.31,
 "z_grid": [5.0, "...", 25.0], "params": {"F_ESC10": -1.2}}
```

Targets are **withheld** by default: writing them into the input file would let a model return
the truth unchanged and score RMSE = 0. A model that genuinely needs an auxiliary target field
must pass `include_targets=True` when constructing `CommandModelAdapter` directly.

The program must write an npz with one array per predicted target name. A non-zero exit
status raises with the last 4 kB of stderr; a missing output file raises with the last 2 kB of
stdout. Process startup per sample makes this slower than an in-process adapter, so use it for
correctness and portability, not for large sweeps.

## `fno21cm:` — fno-21cm checkpoints

```bash
eval21 run --data cubes_3d.h5 \
    --model fno=fno21cm:/work/fno-21cm/checkpoints/3d_xhi/fno/ckpt/best_model_state_dict.pt \
    --fno-root /work/fno-21cm \
    --indices-json /work/fno-21cm/checkpoints/3d_xhi/fno/ckpt/run_metadata.json
```

The adapter imports the fno-21cm repository (`--fno-root`, added to `sys.path`) and uses its
own `load_model` and `ModelConfig`, so legacy, local-operator, waveform and contrast-wrapped
runs all load through the canonical loader. Nothing about the architecture is duplicated here;
the checkpoint's `run_metadata.json` stays the single source of truth.

**Metadata resolution.** `run_metadata.json` is read from the checkpoint's *directory as given*
(not the symlink target), so a checkpoint copied or symlinked into a run directory is pinned to
that run's metadata, split and cache contract.

**Input channels.** `input_features.channel_names` (default `("density/10", "1/(1+z)")`)
decides the channel stack:

| channel | built from |
|---|---|
| `density`, `density/10`, … | `sample.inputs["density"] / density_scale` (default 10, from metadata) |
| `1/(1+z)` | `sample.meta.z_grid`, broadcast over the transverse plane |
| any other name | `sample.meta.params[name]`, optionally z-scored with the run's own `parameter_normalization` |

A ready-made `sample.inputs["x"]` with `(C, Nx, Ny, Nz)` and the right channel count is used
as-is.

**Geometry checks.** If `input_features.spatial_shape` is recorded, every sample's shape must
match it; a mismatch raises with the checkpoint name, the expected shape and the likely cause
(a 512-slice envelope cache for a 256-slice model). Older checkpoints predate the field: the
adapter warns once that geometry is unverified and the report records
`geometry_check: unverified`. `--expect-shape NX,NY,NZ` (or `--expect-nz N`) enforces it
explicitly and upgrades the report to `verified (--expect-shape)`.

**Cache contracts.** When the metadata records `dataset.cache_file` / `cache_path`,
`cache_rows`, `cache_shape` or `cache_sha256`, the CLI verifies them against the data *before*
loading the model. Relative paths compare by filename; `cache_sha256` hashes the whole file
(slow for a 265 GB cache) and should be recorded only when the binding matters more than the
startup time.

**Reported info.** `framework`, `parameters`, `complex_parameters` (a complex weight counts as
two reals), `device`, `checkpoint`, `channels`, `target`, `spatial_shape`, `in_channels`,
`density_scale`, `cache_contract`, `geometry_check`, `expected_shape`, `geometry_source`,
`model_kind`.

## Built-in toy models

`synthetic:σ` (`SmoothThresholdModel`) thresholds a Gaussian-smoothed density field; it is
deterministic and exists to exercise the plumbing, not to represent an emulator.
`toygen:σ` (`ToyGenerativeModel`) draws a fresh random field per draw and ignores the inputs —
the smallest model that exercises `--draws`, `--mode generative` and the `ensemble` metric.

## Draws

`--draws N` asks every model for N predictions per sample. The runner accepts, in order:

1. `adapter.predict_draws(sample, n)` → `{target: (n, Nx, Ny, Nz)}`;
2. a target array returned by `predict` with shape `(n, Nx, Ny, Nz)`;
3. N calls of `predict(sample, draw=i)` — seed your sampler with `i` for reproducibility — or
   N calls of `predict(sample)` for an adapter that samples internally.

The standard metrics score `stacked[0]` (`--score-draw first`, a realisation, right for
generative models) or `stacked.mean(0)` (`--score-draw mean`, the conditional mean). The
`ensemble` metric always receives all draws. The runner records `draws`, `score_draw`,
`mode`, and `draws_identical` (true when all draws of a sample are bit-identical, i.e. a
deterministic model under `--draws`).

## Writing an adapter class

For in-process use, subclass `BaseAdapter` and implement `predict`; `describe`, `close` and
`benchmark` are optional.

```python
from eval21.adapters.base import BaseAdapter

class MyAdapter(BaseAdapter):
    framework = "my-framework"

    def __init__(self, checkpoint, device=None):
        self.model = load(checkpoint, device=device)
        self.device = device
        self.parameters = count_parameters(self.model)

    def predict(self, sample):
        x = to_tensor(sample.inputs)
        return {"xhi": self.model(x)}

    def describe(self):
        info = super().describe()          # framework, parameters, device, peak_memory_bytes
        info["checkpoint"] = str(self.checkpoint)
        return info

    def close(self):
        release(self.model)
```

`describe()` feeds the report's model info and the efficiency metric. Recognised keys:
`framework`, `parameters`, `complex_parameters`, `device`, `peak_memory_bytes`, plus anything
else you want recorded (they are flattened into `summary.csv`).

## Failure modes

| error | cause |
|---|---|
| `adapter returned no fields` | `predict` returned an empty dict |
| `prediction … has no field 'xhi'` | the adapter's key does not match `--target` |
| `prediction … has shape (…), expected (…)` | wrong geometry; check the cache contract |
| `prediction … has N non-finite value(s)` | a diverged or uninitialised model; NaNs are never scored |
| `checkpoint … expects spatial shape …` | wrong cube cache for this checkpoint |
| `run expects parameter 'X'; sample … provides [...]` | the metadata needs a parameter the data lacks |
| `run expects a 1/(1+z) channel but the sample has no z_grid` | data without a redshift grid |
| `command failed (N)` / `command wrote no …` | external program error; stderr/stdout tail is included |
| `Expected one of …` / `unknown adapter` | typo in the adapter prefix |
