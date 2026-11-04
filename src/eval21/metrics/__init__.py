"""Metric plugin imports.

Importing this package registers every built-in metric. Third-party metrics
can be added with ``@register("name")`` in any module imported before the run
(``--metrics-package``).
"""
from . import (  # noqa: F401  (registration side effects)
    anisotropy,
    bubble,
    cylindrical,
    edge,
    eft,
    efficiency,
    front,
    gallery,
    history,
    morphology,
    observable,
    parity,
    radial,
    spectrum,
    symmetry,
    voxel,
)
from .base import Metric, median_band

__all__ = ["Metric", "median_band"]
