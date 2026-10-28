"""Adapter for fno-21cm checkpoints.

Reads the run's ``run_metadata.json`` next to the checkpoint, rebuilds the
architecture with the repository's own ``build_model`` (including legacy and
contrast-wrapped runs), and assembles the input channel stack from the
sample. Nothing about the architecture is duplicated here; the checkpoint
stays the single source of truth.

Usage::

    cm21eval run --data cached_cubes.npz \
        --model fno="fno21cm:checkpoints/3d_xhi/fno/ckpt/best_model_state_dict.pt" \
        --fno-root /path/to/fno-21cm

Input channel names come from the run metadata. Supported names are
``density/10``, ``1/(1+z)``, and the eleven sampled parameters, which are
z-scored with the run's own training normalization when present.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..spec import Sample
from .base import BaseAdapter


class Fno21cmAdapter(BaseAdapter):
    framework = "fno-21cm"

    def __init__(self, checkpoint, project_root: str | Path | None = None,
                 device: str | None = None, target: str = "xhi"):
        import torch

        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        root = Path(project_root).expanduser().resolve() if project_root else None
        if root is not None and str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from modeling import ModelConfig
        from util.run_metadata import load_run_metadata
        from viz.visualize_3d import load_model

        metadata = load_run_metadata(checkpoint.parent) or {}
        features = metadata.get("input_features", {})
        channels = tuple(features.get("channel_names") or ("density/10", "1/(1+z)"))
        in_channels = features.get("in_channels") or len(channels)
        config = (ModelConfig.from_dict(metadata["model_config"])
                  if "model_config" in metadata else None)
        self.model = load_model(
            in_channels=in_channels, checkpoint=checkpoint,
            model_config=config, device=device,
        )
        self.model.eval()
        self.device = str(next(self.model.parameters()).device)
        self.channels = channels
        self.metadata = metadata
        self.checkpoint = checkpoint
        self.target = target
        self.parameters, self.complex_parameters = _count(self.model, torch)

    def _channels(self, sample: Sample) -> np.ndarray:
        ready = sample.inputs.get("x")
        if ready is not None:
            array = np.asarray(ready, dtype=np.float32)
            if array.ndim == 4 and array.shape[0] == len(self.channels):
                return array
        stack = []
        shape = sample.shape
        z = sample.meta.z_grid
        for name in self.channels:
            if name.startswith("density"):
                density = np.asarray(sample.inputs["density"], dtype=np.float32)
                stack.append(density / 10.0)
            elif name == "1/(1+z)":
                if z is None:
                    raise ValueError(
                        "run expects a 1/(1+z) channel but the sample has no "
                        "z_grid")
                channel = (1.0 / (1.0 + np.asarray(z))).astype(np.float32)
                stack.append(np.broadcast_to(
                    channel.reshape(1, 1, -1), shape).astype(np.float32))
            else:
                value = sample.meta.params.get(name)
                if value is None:
                    raise KeyError(
                        f"run expects parameter {name!r}; sample {sample.id!r} "
                        f"provides {sorted(sample.meta.params)}")
                stack.append(np.full(shape, _normalize(self.metadata, name, value),
                                     dtype=np.float32))
        return np.stack(stack, axis=0)

    def predict(self, sample: Sample) -> dict:
        import torch

        stacked = self._channels(sample)
        tensor = torch.from_numpy(stacked).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(x=tensor)
        array = output.detach().float().cpu().numpy()
        while array.ndim > 3:
            array = array[0]
        return {self.target: array}

    def benchmark(self, samples, warmup: int = 3, iters: int = 10) -> dict:
        """Reference-style forward timing with pre-transferred inputs."""
        import time

        import torch

        tensors = []
        for sample in samples:
            stacked = self._channels(sample)
            tensors.append(torch.from_numpy(stacked).unsqueeze(0).to(self.device))
        with torch.no_grad():
            for index in range(max(0, warmup)):
                self.model(x=tensors[index % len(tensors)])
            cuda = str(self.device).startswith("cuda")
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            times = []
            for index in range(max(1, iters)):
                x = tensors[index % len(tensors)]
                if cuda:
                    torch.cuda.synchronize()
                start = time.perf_counter()
                self.model(x=x)
                if cuda:
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
        peak = torch.cuda.max_memory_allocated() if cuda else None
        return {"seconds": times, "peak_bytes": peak,
                "slices": int(tensors[0].shape[-1]), "n_timed": len(times)}

    def describe(self) -> dict:
        info = super().describe()
        info.update({
            "parameters": self.parameters,
            "complex_parameters": self.complex_parameters,
            "device": self.device,
            "checkpoint": str(self.checkpoint),
            "channels": list(self.channels),
            "target": self.target,
            "model_kind": (self.metadata.get("model_config") or {}).get("kind"),
        })
        return info

    def close(self) -> None:
        import torch

        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _normalize(metadata: dict, name: str, value: float) -> float:
    normalization = metadata.get("parameter_normalization")
    if not normalization:
        return float(value)
    names = list(normalization["names"])
    if name not in names:
        return float(value)
    index = names.index(name)
    mean = float(normalization["mean"][index])
    std = float(normalization["std"][index]) or 1.0
    return (float(value) - mean) / std


def _count(model, torch) -> tuple[int, int]:
    """(real parameters, complex parameters); a complex weight is two reals."""
    real, complex_ = 0, 0
    for parameter in model.parameters():
        if torch.is_complex(parameter):
            real += parameter.numel() * 2
            complex_ += parameter.numel()
        else:
            real += parameter.numel()
    return real, complex_
