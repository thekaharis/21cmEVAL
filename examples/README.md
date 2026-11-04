# Examples

| file | purpose |
|---|---|
| `toy_model.py` | smallest possible model adapter (`python:examples/toy_model.py`) |
| `custom_metric.py` | registering a new metric and running it through the normal CLI |
| `fno21cm_eval.toml` | full config for comparing fno-21cm checkpoints |

```bash
eval21 demo --out /tmp/demo
eval21 run --data /tmp/demo/data \
    --model toy=python:examples/toy_model.py \
    --metrics voxel,spectrum,front --out /tmp/demo/custom
```

The toy model is a smoothed density threshold. It exists to exercise the
plumbing, not to represent an emulator; treat its scores as a smoke test.
