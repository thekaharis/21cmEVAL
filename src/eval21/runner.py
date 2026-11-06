"""Evaluation loop: one streaming pass over the data per model.

Metrics are updated in the same pass, so a 200-cone sweep costs one forward
pass per model regardless of how many metrics are enabled. Nothing is stored
per voxel; every accumulator is a sum, a histogram or a per-cone scalar.
"""
from __future__ import annotations

import itertools
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from . import registry
from .settings import Settings
from .spec import check_prediction, check_targets


@dataclass
class ModelResult:
    name: str
    info: dict
    metrics: dict
    skipped: dict
    arrays: dict = field(default_factory=dict)
    seconds: float = 0.0
    n_samples: int = 0
    #: id/split/omega_m/params/z_grid for every evaluated sample (for the run file).
    samples: list = field(default_factory=list)
    #: redshift grid of the run (all samples share it).
    z_grid: object = None
    #: needs/targets/pointwise of every built metric (for /metrics/<name>/meta).
    metric_meta: dict = field(default_factory=dict)


def evaluate(adapter, dataset, settings: Settings,
             metric_names: Iterable[str] = ("all",),
             metric_options: dict | None = None,
             progress=None,
             expect_shape: tuple[int, ...] | None = None,
             store=None, model_name: str = "model") -> ModelResult:
    """Run one adapter over ``dataset`` and return per-metric results.

    ``store`` (a :class:`~eval21.store.PredictionStore` or
    :class:`~eval21.io.cache.CacheWriter`) keeps the truth and prediction cubes of the samples it
    wants, under ``model_name``, for map figures."""
    from .metrics import Metric  # noqa: F401  (registers built-ins)

    names = _resolve_names(metric_names)
    metrics = registry.build(names, settings, metric_options)

    iterator = iter(dataset)
    first = next(iterator, None)
    if first is None:
        raise ValueError("dataset is empty; check --split, --indices and --limit")
    skipped = {}
    usable = []
    generative = settings.mode == "generative"
    for metric in metrics:
        missing = [field for field in metric.needs if field not in first.inputs]
        if missing:
            skipped[metric.name] = f"missing input fields: {', '.join(missing)}"
            continue
        if settings.target not in first.targets:
            skipped[metric.name] = f"missing target field: {settings.target}"
            continue
        if metric.targets is not None and settings.target not in metric.targets:
            skipped[metric.name] = (f"defined for {'/'.join(metric.targets)} targets, "
                                    f"not {settings.target}")
            continue
        if generative and metric.pointwise is True:
            skipped[metric.name] = ("pointwise (voxel-by-voxel) metric; a generative model's "
                                    "fields need not align with the truth (--mode generative)")
            continue
        usable.append(metric)

    info = adapter.describe() if hasattr(adapter, "describe") else {}
    for metric in usable:
        metric.attach(info)

    rerun = any(getattr(metric, "requires_rerun", False) for metric in usable)
    identical = None
    n_draws = 1
    started = time.perf_counter()
    n_samples = 0
    total = len(dataset) if hasattr(dataset, "__len__") else None
    geometry = None
    benchmark_pool = []
    sample_rows = []
    try:
        for sample in itertools.chain([first], iterator):
            check_targets(sample)
            if expect_shape is not None and tuple(sample.shape) != tuple(expect_shape):
                raise ValueError(
                    f"expected spatial shape {tuple(expect_shape)}, but sample "
                    f"{sample.id!r} has {tuple(sample.shape)}")
            z = sample.meta.z_grid
            z_key = None if z is None else (
                int(len(z)), round(float(z[0]), 9), round(float(z[-1]), 9),
                round(float(np.sum(z)), 6))
            current = (sample.shape,
                       round(float(sample.meta.box_mpc), 6)
                       if sample.meta.box_mpc is not None else None,
                       z_key)
            if geometry is None:
                geometry = current
            elif current != geometry:
                raise ValueError(
                    "the metric suite requires one geometry per run: got "
                    f"{geometry} then {current} at sample {sample.id!r}")
            t0 = time.perf_counter()
            prediction, draws = predict_draws(adapter, sample, settings)
            elapsed = time.perf_counter() - t0
            check_prediction(prediction, sample, names=[settings.target])
            if draws is not None:
                bad = int(np.count_nonzero(~np.isfinite(draws)))
                if bad:
                    raise ValueError(f"draws for sample {sample.id!r} have {bad} non-finite value(s)")
                n_draws = draws.shape[0]
                if n_draws > 1 and identical is None:
                    identical = bool(np.array_equal(draws[0], draws[-1]))
            aux = None
            if rerun:
                swapped = adapter.predict(sample.transposed_xy())
                aux = {"xy": {key: np.asarray(value).transpose(1, 0, 2)
                              if np.asarray(value).ndim == 3
                              else np.asarray(value).transpose(0, 2, 1, 3)
                              for key, value in swapped.items()}}
            if draws is not None:
                aux = dict(aux or {})
                aux["draws"] = draws
            for metric in usable:
                metric.update(sample, prediction, aux=aux,
                              runtime={"seconds": elapsed})
            if store is not None and store.wants(sample):
                store.save_truth(sample)
                saved = dict(prediction)
                if draws is not None and draws.shape[0] > 1:
                    saved[f"{settings.target}__draws"] = draws[:settings.store_draws]
                store.save_prediction(model_name, sample, saved,
                                      draws=1 if draws is None else draws.shape[0])
            if len(benchmark_pool) < settings.benchmark_samples:
                benchmark_pool.append(sample)
            sample_rows.append({
                "id": str(sample.id),
                "split": sample.split,
                "omega_m": sample.meta.omega_m,
                "params": dict(sample.meta.params or {}),
                "z_grid": sample.meta.z_grid,
            })
            n_samples += 1
            if progress is not None:
                progress(n_samples, total, sample.id)

        if benchmark_pool and hasattr(adapter, "benchmark"):
            stats = adapter.benchmark(benchmark_pool, settings.warmup,
                                      settings.iters)
            for metric in usable:
                observe = getattr(metric, "observe_benchmark", None)
                if callable(observe):
                    observe(stats)

        # Re-read the adapter description after the pass: geometry checks run
        # on the first sample and would otherwise be serialized as unverified.
        if hasattr(adapter, "describe"):
            post = adapter.describe()
            if isinstance(post, dict):
                info = {**info, **post}
        if expect_shape is not None:
            info["expected_shape"] = list(expect_shape)
            if info.get("geometry_check") in {None, "unverified"}:
                info["geometry_check"] = "verified (--expect-shape)"
            info.setdefault("geometry_source", "--expect-shape")

        arrays = {}
        for metric in usable:
            for key, value in metric.arrays().items():
                arrays[f"{metric.name}__{key}"] = np.asarray(value)
        results = {metric.name: metric.result() for metric in usable}
        if generative:
            for metric in usable:
                if metric.pointwise == "partial" and metric.pointwise_keys:
                    dropped = []
                    results[metric.name] = strip_keys(results[metric.name], metric.pointwise_keys, dropped)
                    if dropped:
                        results[metric.name]["_generative_dropped"] = sorted(set(dropped))
        info = dict(info)
        info["draws"] = int(n_draws)
        info["mode"] = settings.mode
        if n_draws > 1:
            info["score_draw"] = settings.score_draw
            info["draws_identical"] = bool(identical)
    finally:
        if hasattr(adapter, "close"):
            adapter.close()
    metric_meta = {
        metric.name: {
            "needs": list(metric.needs),
            "targets": None if metric.targets is None else list(metric.targets),
            "pointwise": metric.pointwise,
        }
        for metric in usable
    }
    return ModelResult(
        name="", info=info, metrics=results, skipped=skipped, arrays=arrays,
        seconds=time.perf_counter() - started, n_samples=n_samples,
        samples=sample_rows, z_grid=None if not sample_rows else sample_rows[0]["z_grid"],
        metric_meta=metric_meta,
    )


def _call_predict(adapter, sample, draw=None):
    if draw is None:
        return adapter.predict(sample)
    try:
        return adapter.predict(sample, draw=draw)
    except TypeError as exc:
        if "draw" not in str(exc):
            raise
        return adapter.predict(sample)


def predict_draws(adapter, sample, settings):
    """(prediction dict for the standard metrics, draws (D, Nx, Ny, Nz) or None).

    Draws come from, in this order: ``adapter.predict_draws(sample, n)``; a target array of
    shape (D, Nx, Ny, Nz) returned by ``predict``; ``settings.draws`` calls of
    ``predict(sample, draw=i)`` (or of ``predict(sample)`` for an adapter that samples
    internally). The standard metrics score the first draw (``score_draw="first"``, a
    realisation, right for generative models) or the ensemble mean (``"mean"``).
    """
    target = settings.target
    n = int(settings.draws)
    if n > 1 and hasattr(adapter, "predict_draws"):
        out = adapter.predict_draws(sample, n)
        stacked = np.asarray(out[target] if isinstance(out, dict) else out, dtype=np.float32)
        base = out if isinstance(out, dict) else {}
    else:
        first = _call_predict(adapter, sample, 0 if n > 1 else None)
        value = np.asarray(first[target]) if target in first else None
        if value is not None and value.ndim == 4:
            stacked = value.astype(np.float32)
            base = first
        elif n > 1:
            rest = [_call_predict(adapter, sample, i)[target] for i in range(1, n)]
            stacked = np.stack([value] + [np.asarray(r) for r in rest]).astype(np.float32)
            base = first
        else:
            return first, None
    if stacked.ndim != 4:
        raise ValueError(f"draws for sample {sample.id!r} must be (D, Nx, Ny, Nz), got {stacked.shape}")
    scored = stacked[0] if settings.score_draw == "first" else stacked.mean(axis=0)
    prediction = {k: v for k, v in base.items() if np.ndim(v) == 3}
    prediction[target] = scored
    return prediction, stacked


def strip_keys(value, patterns, dropped, path=""):
    """Copy of a result with every key containing one of ``patterns`` removed (recursively)."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if any(p in str(key) for p in patterns):
                dropped.append(f"{path}{key}")
                continue
            out[key] = strip_keys(item, patterns, dropped, f"{path}{key}.")
        return out
    if isinstance(value, list):
        return [strip_keys(item, patterns, dropped, f"{path}[].") for item in value]
    return value


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
