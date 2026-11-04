"""Data contract between datasets, model adapters and metrics.

The suite never imports a training framework. Everything is exchanged as

    Sample  ->  adapter.predict(sample)  ->  {name: ndarray}
    Metric.update(sample, prediction)

so a model only has to produce named 3-D arrays on the sample's grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class Meta:
    """Per-sample geometry and cosmology. ``None`` means "not provided"."""

    box_mpc: float | None = None
    z_grid: np.ndarray | None = None
    omega_m: float | None = None
    params: dict[str, float] = field(default_factory=dict)

    def with_defaults(self, box_mpc: float, omega_m: float | None = None) -> "Meta":
        return replace(
            self,
            box_mpc=self.box_mpc if self.box_mpc is not None else box_mpc,
            omega_m=self.omega_m if self.omega_m is not None else omega_m,
        )


@dataclass
class Sample:
    """One evaluation example: named inputs, named targets, geometry."""

    id: str
    inputs: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]
    meta: Meta = field(default_factory=Meta)
    split: str | None = None

    def __post_init__(self):
        for name, value in self.inputs.items():
            array = np.asarray(value)
            # Inputs may be a ready-made channel stack (C, Nx, Ny, Nz); targets
            # are always one 3-D field.
            if array.ndim not in (3, 4):
                raise ValueError(
                    f"input {name!r} of sample {self.id!r} must be 3-D "
                    f"(Nx, Ny, Nz) or 4-D (C, Nx, Ny, Nz), got {array.shape}")
            if array.size == 0:
                raise ValueError(f"input {name!r} of sample {self.id!r} is empty")
        for name, value in self.targets.items():
            array = np.asarray(value)
            if array.ndim != 3:
                raise ValueError(
                    f"target {name!r} of sample {self.id!r} must be 3-D "
                    f"(Nx, Ny, Nz), got shape {array.shape}")
            if array.size == 0:
                raise ValueError(f"target {name!r} of sample {self.id!r} is empty")

    @property
    def shape(self) -> tuple[int, int, int]:
        first = next(iter(self.targets.values()), None)
        if first is None:
            first = next(iter(self.inputs.values()))
        return tuple(np.asarray(first).shape)

    def transposed_xy(self) -> "Sample":
        """Same sample with every field swapped in the transverse plane."""
        def swap(array):
            array = np.asarray(array)
            return (array.transpose(0, 2, 1, 3) if array.ndim == 4
                    else array.transpose(1, 0, 2))

        return replace(
            self,
            id=f"{self.id}#xy",
            inputs={k: swap(v) for k, v in self.inputs.items()},
            targets={k: swap(v) for k, v in self.targets.items()},
        )


def check_prediction(prediction, sample: Sample, names=None) -> None:
    """Fail loudly when an adapter returns a missing, wrong-shape or NaN field.

    A NaN-tolerant metric would turn a broken forward pass into a plausible
    report, so non-finite values are an error here rather than a masked value.
    """
    if not prediction:
        raise ValueError(f"adapter returned no fields for sample {sample.id!r}")
    wanted = names or sample.targets.keys()
    for name in wanted:
        if name not in prediction:
            raise KeyError(
                f"prediction for sample {sample.id!r} has no field {name!r}; "
                f"adapter returned {sorted(prediction)}"
            )
        array = np.asarray(prediction[name])
        expected = np.asarray(sample.targets.get(name, sample.shape)).shape
        if array.shape != expected:
            raise ValueError(
                f"prediction {name!r} for sample {sample.id!r} has shape "
                f"{array.shape}, expected {expected}"
            )
        bad = int(np.count_nonzero(~np.isfinite(array)))
        if bad:
            raise ValueError(
                f"prediction {name!r} for sample {sample.id!r} has {bad} "
                f"non-finite value(s)"
            )


def check_targets(sample: Sample) -> None:
    """Targets must be finite 3-D fields before any metric sees them."""
    for name, value in sample.targets.items():
        array = np.asarray(value)
        bad = int(np.count_nonzero(~np.isfinite(array)))
        if bad:
            raise ValueError(
                f"target {name!r} of sample {sample.id!r} has {bad} "
                f"non-finite value(s)"
            )
