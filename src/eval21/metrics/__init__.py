"""Metric plugin imports.

Importing this package registers every built-in metric. Third-party metrics
are added with ``@register("name")`` in any module imported before the run: a
small wrapper script that imports that module and calls ``eval21.cli.main``,
or a plugin module imported from the caller.
"""
from . import (  # noqa: F401  (registration side effects)
    anisotropy,
    bubble,
    cylindrical,
    dtb,
    edge,
    eft,
    efficiency,
    ensemble,
    front,
    front3d,
    gallery,
    history,
    morphology,
    observable,
    parity,
    radial,
    spectrum,
    survey,
    symmetry,
    voxel,
)
from .base import Metric, median_band

__all__ = ["Metric", "median_band"]
