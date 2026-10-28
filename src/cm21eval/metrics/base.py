"""Metric base class and shared helpers."""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ..physics import stage_labels, stage_masks
from ..settings import Settings


class Metric:
    """One evaluation family.

    ``needs`` lists input fields beyond the target (``density`` for the
    physics metrics). The runner skips a metric whose needs are missing and
    records why, so the same suite runs on targets alone.
    """

    name = ""
    needs: tuple[str, ...] = ()

    def __init__(self, settings: Settings, options: dict[str, Any] | None = None):
        self.settings = settings
        self.options = options or {}

    # -- lifecycle --------------------------------------------------------- #
    def attach(self, model_info: dict[str, Any]) -> None:
        """Receive static model information (parameter count, device, ...)."""

    def update(self, sample, prediction, aux=None, runtime=None) -> None:
        raise NotImplementedError

    def result(self) -> dict[str, Any]:
        raise NotImplementedError

    def arrays(self) -> dict[str, np.ndarray]:
        """Arrays worth exporting to ``curves.npz`` (empty by default)."""
        return {}

    # -- helpers ----------------------------------------------------------- #
    def target(self, sample):
        return np.asarray(sample.targets[self.settings.target], dtype=np.float64)

    def prediction(self, prediction):
        return np.asarray(prediction[self.settings.target], dtype=np.float64)

    def stages(self, truth):
        masks = stage_masks(
            truth.mean(axis=(0, 1)),
            self.settings.stage_edges,
            self.settings.active_range,
            self.settings.min_stage_slices,
        )
        return masks, stage_labels(
            self.settings.stage_edges, self.settings.active_range
        )

    def require(self, sample, field: str) -> np.ndarray:
        if field not in sample.inputs:
            raise KeyError(f"sample {sample.id!r} has no input field {field!r}")
        return np.asarray(sample.inputs[field], dtype=np.float64)

    def box_mpc(self, sample) -> float:
        box = sample.meta.box_mpc
        return float(box if box is not None else self.settings.box_mpc)

    def omega_m(self, sample) -> float:
        omega = sample.meta.omega_m
        return float(omega if omega is not None else self.settings.omega_m)

    def z_grid(self, sample):
        z = sample.meta.z_grid
        return None if z is None else np.asarray(z, dtype=np.float64)


def median_band(rows, axis: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NaN-aware median with 16-84 percentiles."""
    rows = np.asarray(rows, dtype=np.float64)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return (np.nanmedian(rows, axis=axis),
                np.nanpercentile(rows, 16, axis=axis),
                np.nanpercentile(rows, 84, axis=axis))


def nanmedian(values) -> float:
    """NaN-aware median without the all-NaN warning (empty stages are normal)."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmedian(values))


def finite_or_none(values):
    """JSON-safe list: non-finite floats become ``None``."""
    out = []
    for value in np.asarray(values).ravel():
        value = float(value)
        out.append(value if np.isfinite(value) else None)
    return out
