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

        # Keep the path as given: a checkpoint may be symlinked into a run
        # directory whose own run_metadata.json is the contract. Resolving the
        # link would silently read the target directory's metadata instead.
        checkpoint = Path(checkpoint).expanduser()
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        root = Path(project_root).expanduser().resolve() if project_root else None
        if root is not None and str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from modeling import ModelConfig
        from util.run_metadata import load_run_metadata
        from viz.visualize_3d import load_model

        metadata = load_run_metadata(metadata_directory(checkpoint)) or {}
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
        self.in_channels = int(in_channels)
        shape = features.get("spatial_shape")
        self.spatial_shape = None if shape is None else tuple(int(v) for v in shape)
        self.expect_shape: tuple[int, ...] | None = None
        self.geometry_check = "verified" if self.spatial_shape else "unverified"
        self._geometry_warned = False
        dataset = metadata.get("dataset") if isinstance(metadata.get("dataset"), dict) else {}
        self.density_scale = float(
            metadata.get("density_scale",
                         dataset.get("density_scale", 10.0)))
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
                stack.append(density / self.density_scale)
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

    def _check_geometry(self, sample: Sample) -> None:
        expected = self.expect_shape or self.spatial_shape
        if expected is not None and tuple(sample.shape) != tuple(expected):
            source = ("--expect-shape" if self.expect_shape
                      else "run_metadata.json input_features.spatial_shape")
            raise ValueError(
                f"checkpoint {self.checkpoint.name} expects spatial shape "
                f"{tuple(expected)} ({source}), but sample {sample.id!r} has "
                f"{tuple(sample.shape)}. This usually means the cube cache is "
                f"the wrong one (e.g. a 512-slice envelope cache for a "
                f"256-slice model); pass the cache recorded for this run.")
        if expected is not None and tuple(sample.shape) == tuple(expected):
            self.geometry_check = ("verified (--expect-shape)" if self.expect_shape
                                   else "verified")
        if expected is None and not self._geometry_warned:
            self._geometry_warned = True
            import warnings

            warnings.warn(
                f"{self.checkpoint.name}: run_metadata.json has no "
                "input_features.spatial_shape, so the cache geometry cannot "
                "be verified. Pass --expect-shape NX,NY,NZ (or --expect-nz N) "
                "to enforce it.", stacklevel=2)
        channels = len(self.channels)
        if channels != self.in_channels:
            raise ValueError(
                f"checkpoint expects {self.in_channels} input channels, "
                f"metadata declares {channels}")

    def predict(self, sample: Sample) -> dict:
        import torch

        self._check_geometry(sample)
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
            "spatial_shape": None if self.spatial_shape is None
            else list(self.spatial_shape),
            "in_channels": self.in_channels,
            "density_scale": self.density_scale,
            "cache_contract": cache_contract(self.metadata),
            "geometry_check": self.geometry_check,
            "model_kind": (self.metadata.get("model_config") or {}).get("kind"),
        })
        return info

    def close(self) -> None:
        import torch

        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def metadata_directory(checkpoint) -> Path:
    """Directory of run_metadata.json, following the path as given.

    A symlinked checkpoint should use the metadata next to the link, not the
    target's: copying a checkpoint into a run directory is a common way to
    pin it to that run's split and cache contract.
    """
    return Path(checkpoint).expanduser().parent


def cache_contract(metadata: dict) -> dict | None:
    """Return the cache identity a run recorded, if any.

    Training currently records no dataset path, so this is often empty. When
    ``dataset.cache_file`` / ``cache_path`` / ``cache_rows`` / ``cache_shape``
    are present, the CLI enforces them before inference.
    """
    dataset = metadata.get("dataset")
    dataset = dataset if isinstance(dataset, dict) else {}
    contract = {}
    for key in ("cache_file", "cache_path", "cache_rows", "cache_shape",
                "cache_sha256"):
        value = dataset.get(key, metadata.get(key))
        if value is not None:
            contract[key] = value
    return contract or None


def validate_cache_contract(metadata: dict, dataset_info: dict,
                            checkpoint: str | Path) -> None:
    """Fail when a run's recorded cache identity does not match the data."""
    contract = cache_contract(metadata)
    if not contract:
        return
    declared = contract.get("cache_file") or contract.get("cache_path")
    actual = dataset_info.get("path")
    if declared and actual and Path(str(declared)).name != Path(str(actual)).name:
        raise ValueError(
            f"checkpoint {Path(checkpoint).name} records cache "
            f"{Path(str(declared)).name!r}, but the dataset is "
            f"{Path(str(actual)).name!r}")
    if ("cache_rows" in contract and dataset_info.get("n_total") is not None
            and int(contract["cache_rows"]) != int(dataset_info["n_total"])):
        raise ValueError(
            f"checkpoint {Path(checkpoint).name} records "
            f"{contract['cache_rows']} cache rows, dataset has "
            f"{dataset_info['n_total']}")
    if ("cache_shape" in contract and dataset_info.get("shape")
            and list(map(int, contract["cache_shape"]))
            != list(map(int, dataset_info["shape"]))):
        raise ValueError(
            f"checkpoint {Path(checkpoint).name} records cache shape "
            f"{contract['cache_shape']}, dataset has {dataset_info['shape']}")


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
