"""In-process PyTorch adapter.

Two input conventions are accepted:

* the sample carries a ready-made channel stack under ``inputs["x"]`` with
  shape ``(C, Nx, Ny, Nz)`` -- this is the fno-21cm convention;
* ``channels=["density", "z"]`` names input fields that are stacked in order
  (each may be 3-D or a per-slice 1-D array of length Nz).

The module is called as ``module(x=...)`` first and ``module(...)`` second, so
both ``SilentFNO``-style wrappers and plain ``nn.Module``s work. Output may be
a tensor or a mapping; 4-D/5-D tensors are squeezed to ``(Nx, Ny, Nz)``.
"""
from __future__ import annotations

import numpy as np

from ..spec import Sample
from .base import BaseAdapter


class TorchModelAdapter(BaseAdapter):
    framework = "torch"

    def __init__(self, module, channels=None, input_key: str = "x",
                 device=None, output_field: str = "xhi",
                 output_channels: int = 1, half: bool = False):
        import torch

        self.torch = torch
        self.module = module
        self.channels = tuple(channels) if channels else None
        self.input_key = input_key
        self.output_field = output_field
        self.output_channels = int(output_channels)
        self.half = bool(half)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.module.to(self.device).eval()
        self.parameters, self.complex_parameters = _count(self.module, torch)

    def _stack(self, sample: Sample):
        torch = self.torch
        channels = self.channels
        if channels is None:
            ready = sample.inputs.get(self.input_key)
            if ready is not None and np.asarray(ready).ndim == 4:
                array = np.asarray(ready)
            elif "density" in sample.inputs:
                channels = ("density",)
            else:
                raise KeyError(
                    f"adapter has no input channels and sample {sample.id!r} "
                    f"provides neither {self.input_key!r} (C, Nx, Ny, Nz) nor "
                    f"'density'; inputs: {sorted(sample.inputs)}")
        if channels is not None:
            arrays = []
            shape = sample.shape
            for name in channels:
                if name not in sample.inputs:
                    raise KeyError(
                        f"adapter expects input {name!r}; sample {sample.id!r} "
                        f"has {sorted(sample.inputs)}")
                array = np.asarray(sample.inputs[name])
                if array.ndim == 3:
                    array = array[None]
                elif array.ndim == 1:
                    array = np.broadcast_to(
                        array.reshape(-1, 1, 1), (shape[2],) + tuple(shape[:2])
                    ).transpose(1, 2, 0)[None]
                arrays.append(array)
            array = np.concatenate(arrays, axis=0)
        tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
        tensor = tensor.unsqueeze(0)
        if self.half:
            tensor = tensor.half()
        return tensor.to(self.device)

    def predict(self, sample: Sample) -> dict:
        torch = self.torch
        x = self._stack(sample)
        with torch.no_grad():
            try:
                output = self.module(x=x)
            except TypeError:
                output = self.module(x)
        if isinstance(output, dict):
            output = (output[self.output_field] if self.output_field in output
                      else next(iter(output.values())))
        if isinstance(output, (tuple, list)):
            output = output[0]
        array = output.detach().float().cpu().numpy()
        while array.ndim > 4:
            array = array[0]
        if array.ndim == 4:
            if array.shape[1] == 1:
                array = array[0, 0]
            elif array.shape[0] == 1:
                array = array[0]
            else:
                raise ValueError(f"unexpected output shape {array.shape}")
        if array.ndim != 3:
            raise ValueError(f"adapter output must reduce to 3-D, got {array.shape}")
        return {self.output_field: array}

    def describe(self) -> dict:
        info = super().describe()
        info.update({
            "parameters": self.parameters,
            "complex_parameters": self.complex_parameters,
            "device": str(self.device),
        })
        return info

    def close(self) -> None:
        del self.module
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def _count(module, torch) -> tuple[int, int]:
    real = complex_ = 0
    for parameter in module.parameters():
        if torch.is_complex(parameter):
            real += parameter.numel() * 2
            complex_ += parameter.numel()
        else:
            real += parameter.numel()
    return real, complex_


def load_torch(model, checkpoint=None, **kwargs) -> TorchModelAdapter:
    """Wrap an already-built module, optionally loading a raw state dict."""
    if checkpoint is not None:
        import torch

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "_metadata" in state:
            state = state["model"] if "model" in state else state
        state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    return TorchModelAdapter(model, **kwargs)
