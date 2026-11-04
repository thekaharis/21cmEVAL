"""Adapter selection from a ``name=adapter:argument`` CLI spec.

Supported adapters
------------------
``python:PATH[:FUNCTION]``
    A file defining ``predict(sample) -> dict``. Default function ``predict``.
``torch:PATH[:BUILDER[:CHECKPOINT]]``
    A file defining ``BUILDER()`` that returns a ``torch.nn.Module``. If a
    checkpoint path is given it is loaded as a raw state dict.
``command:COMMAND``
    Any program with ``{input}`` / ``{output}`` npz placeholders.
``fno21cm:CHECKPOINT``
    An fno-21cm run (needs ``--fno-root``).

Anything after the first colon is adapter-specific, so paths may contain
colons on platforms that allow them only in the final segment.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .base import BaseAdapter
from .command import CommandModelAdapter
from .fno21cm import Fno21cmAdapter
from .python import PythonModelAdapter
from .torch import TorchModelAdapter


def split_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(
            f"model spec {spec!r} needs an adapter prefix, e.g. "
            f"'python:model.py' or 'fno21cm:ckpt.pt'")
    adapter, argument = spec.split(":", 1)
    return adapter.strip().lower(), argument


def build(spec: str, fno_root: str | None = None,
          device: str | None = None, target: str = "xhi") -> BaseAdapter:
    adapter, argument = split_spec(spec)
    if adapter in {"python", "py", "numpy"}:
        path, _, attribute = argument.partition(":")
        return PythonModelAdapter(path, attribute or "predict")
    if adapter == "torch":
        parts = argument.split(":")
        path = parts[0]
        builder = parts[1] if len(parts) > 1 and parts[1] else "build"
        checkpoint = parts[2] if len(parts) > 2 and parts[2] else None
        module = _import_file(Path(path))
        factory = getattr(module, builder, None)
        if factory is None:
            raise AttributeError(f"{path} defines no builder {builder!r}")
        model = factory(checkpoint) if checkpoint else factory()
        if checkpoint is None and hasattr(module, "load_checkpoint"):
            module.load_checkpoint(model)
        return TorchModelAdapter(model, device=device, output_field=target)
    if adapter == "command":
        return CommandModelAdapter(argument)
    if adapter in {"synthetic", "toy"}:
        from .toy import SmoothThresholdModel

        return SmoothThresholdModel(float(argument) if argument else 1.5)
    if adapter in {"fno21cm", "fno"}:
        return Fno21cmAdapter(argument, project_root=fno_root, device=device,
                              target=target)
    raise ValueError(
        f"unknown adapter {adapter!r}; use python, torch, command or fno21cm")


def _import_file(path: Path):
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"model file not found: {path}")
    name = f"_eval21_adapter_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


__all__ = ["BaseAdapter", "build", "split_spec"]
