"""Metric plugin registry.

A metric is any class with ``update(sample, prediction)`` and ``result()``.
Registration is by short name, so ``--metrics voxel,spectrum,history`` and
``[metrics.spectrum]`` in a TOML file need no import paths.
"""
from __future__ import annotations

from typing import Iterable

from .settings import Settings

_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Class decorator: add a metric under ``name``."""

    def decorator(cls):
        if name in _REGISTRY:
            raise ValueError(f"metric {name!r} is already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def available() -> dict[str, type]:
    return dict(_REGISTRY)


def create(name: str, settings: Settings, options: dict | None = None):
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown metric {name!r}; available: {', '.join(sorted(_REGISTRY))}"
        ) from None
    return cls(settings, options or {})


def build(names: Iterable[str], settings: Settings,
          options: dict[str, dict] | None = None) -> list:
    """Instantiate a metric list, preserving order and dropping duplicates."""
    options = options or {}
    built, seen = [], set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        built.append(create(name, settings, options.get(name, {})))
    return built
