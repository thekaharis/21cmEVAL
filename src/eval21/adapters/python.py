"""Load a model from a plain Python file.

The file must define ``predict(sample) -> dict[str, ndarray]`` (or the name
given after the path, e.g. ``python:model.py:forward``). It may also define
``describe() -> dict`` and ``close()``; the suite calls them when present.

This is the escape hatch for research code: no packaging, no registry, no
inheritance. The same file works with any framework the process can import.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from ..spec import Sample
from .base import BaseAdapter


def _load(path: Path, attribute: str):
    if not path.exists():
        raise FileNotFoundError(f"model script not found: {path}")
    module_name = f"_eval21_model_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    try:
        return module, getattr(module, attribute)
    except AttributeError:
        raise AttributeError(
            f"{path} defines no {attribute!r}; expected "
            f"`def {attribute}(sample) -> dict[str, ndarray]`"
        ) from None


class PythonModelAdapter(BaseAdapter):
    framework = "python"

    def __init__(self, path, attribute: str = "predict"):
        self.path = Path(path).expanduser()
        self.attribute = attribute
        self.module, self._predict = _load(self.path, attribute)
        describe = getattr(self.module, "describe", None)
        self._describe = describe() if callable(describe) else {}

    def predict(self, sample: Sample) -> dict:
        return self._predict(sample)

    def describe(self) -> dict:
        info = super().describe()
        info.update(self._describe)
        return info

    def close(self) -> None:
        close = getattr(self.module, "close", None)
        if callable(close):
            close()
