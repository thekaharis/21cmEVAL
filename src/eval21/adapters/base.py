"""Model adapter interface.

An adapter is the only place that knows about a training framework. It maps
one :class:`~eval21.spec.Sample` to named prediction arrays:

    model = MyAdapter(...)
    prediction = model.predict(sample)   # {"xhi": (Nx, Ny, Nz)}

``describe()`` is optional and feeds the efficiency report.
"""
from __future__ import annotations

from typing import Protocol

from ..spec import Sample


class ModelAdapter(Protocol):
    framework: str

    def predict(self, sample: Sample) -> dict[str, object]:
        ...

    def describe(self) -> dict:
        ...


class BaseAdapter:
    framework = "custom"

    def predict(self, sample: Sample) -> dict:
        raise NotImplementedError

    def describe(self) -> dict:
        return {
            "framework": self.framework,
            "parameters": getattr(self, "parameters", None),
            "complex_parameters": None,
            "device": getattr(self, "device", None),
            "peak_memory_bytes": None,
        }

    def close(self) -> None:
        """Release device memory between models (optional)."""
