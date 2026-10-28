"""Evaluation loop: one streaming pass over the data per model.

Metrics are updated in the same pass, so a 200-cone sweep costs one forward
pass per model regardless of how many metrics are enabled. Nothing is stored
per voxel; every accumulator is a sum, a histogram or a per-cone scalar.
"""
from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from . import registry
from .settings import Settings
from .spec import check_prediction


@dataclass
class ModelResult:
    name: str
    info: dict
    metrics: dict
    skipped: dict
    arrays: dict = field(default_factory=dict)
    seconds: float = 0.0
    n_samples: int = 0


def evaluate(adapter, dataset, settings: Settings,
             metric_names: Iterable[str] = ("all",),
             metric_options: dict | None = None,
             progress=None) -> ModelResult:
    """Run one adapter over ``dataset`` and return per-metric results."""
    from .metrics import Metric  # noqa: F401  (registers built-ins)

    names = _resolve_names(metric_names)
    available = registry.available()
    metrics = registry.build(names, settings, metric_options)

    first = next(iter(dataset), None)
    if first is None:
        raise ValueError("dataset is empty; check --split and --limit")
    skipped = {}
    usable = []
    for metric in metrics:
        missing = [field for field in metric.needs if field not in first.inputs]
        if missing:
            skipped[metric.name] = f"missing input fields: {', '.join(missing)}"
            continue
        if settings.target not in first.targets:
            skipped[metric.name] = f"missing target field: {settings.target}"
            continue
        usable.append(metric)

    info = adapter.describe() if hasattr(adapter, "describe") else {}
    for metric in usable:
        metric.attach(info)

    rerun = any(getattr(metric, "requires_rerun", False) for metric in usable)
    started = time.perf_counter()
    n_samples = 0
    total = len(dataset) if hasattr(dataset, "__len__") else None
    try:
        for sample in dataset:
            t0 = time.perf_counter()
            prediction = adapter.predict(sample)
            elapsed = time.perf_counter() - t0
            check_prediction(prediction, sample, names=[settings.target])
            aux = None
            if rerun:
                swapped = adapter.predict(sample.transposed_xy())
                aux = {"xy": {key: np.asarray(value).transpose(1, 0, 2)
                              if np.asarray(value).ndim == 3
                              else np.asarray(value).transpose(0, 2, 1, 3)
                              for key, value in swapped.items()}}
            for metric in usable:
                metric.update(sample, prediction, aux=aux,
                              runtime={"seconds": elapsed})
            n_samples += 1
            if progress is not None:
                progress(n_samples, total, sample.id)

        arrays = {}
        for metric in usable:
            for key, value in metric.arrays().items():
                arrays[f"{metric.name}__{key}"] = np.asarray(value)
        results = {metric.name: metric.result() for metric in usable}
    finally:
        if hasattr(adapter, "close"):
            adapter.close()
    return ModelResult(
        name="", info=info, metrics=results, skipped=skipped, arrays=arrays,
        seconds=time.perf_counter() - started, n_samples=n_samples,
    )


def _resolve_names(names: Iterable[str]) -> list[str]:
    names = list(names)
    if not names or any(name in {"all", "*"} for name in names):
        return sorted(registry.available())
    expanded = []
    for name in names:
        if name in registry.available():
            expanded.append(name)
        else:
            raise KeyError(
                f"unknown metric {name!r}; available: "
                f"{', '.join(sorted(registry.available()))}")
    return expanded


def environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "host": platform.node(),
    }
