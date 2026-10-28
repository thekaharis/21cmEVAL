"""Synthetic 21-cm-like lightcones, for tests and demos without data.

The generator is deliberately simple and physical only in its qualitative
behaviour: a lognormal density field, a correlated Gaussian field smoothed on
a growing scale, and a threshold that reionizes from low to high redshift.
That is enough to exercise every metric, to regression-test the suite, and to
give a new user a runnable end-to-end example in seconds.

It is **not** a 21cmFAST emulator and its numbers are not physical.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..spec import Meta, Sample
from .base import BaseDataset


def _power_law_field(rng, shape, slope: float = -2.5, k0: float = 1.0):
    white = rng.normal(size=shape)
    k = np.fft.fftfreq(shape[0])[:, None, None] ** 2
    k = k + np.fft.fftfreq(shape[1])[None, :, None] ** 2
    k = k + np.fft.fftfreq(shape[2])[None, None, :] ** 2
    amplitude = (k + (k0 / max(shape)) ** 2) ** (slope / 4.0)
    field = np.fft.ifftn(np.fft.fftn(white) * amplitude).real
    return (field - field.mean()) / (field.std() + 1e-12)


def _smooth(field, sigma):
    from scipy.ndimage import gaussian_filter

    if sigma <= 0:
        return field
    return gaussian_filter(field, sigma=sigma, mode="wrap")


def make_cone(shape=(64, 64, 64), box_mpc: float = 200.0, seed: int = 0,
              z_min: float = 5.0, z_max: float = 25.0):
    """Return ``(density, xhi, z_grid)`` for one synthetic lightcone."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = shape
    z = np.linspace(z_min, z_max, nz)

    delta = _power_law_field(rng, shape, slope=-2.5)
    density = np.exp(0.3 * delta - 0.5 * 0.3 ** 2)
    density = density.astype(np.float32)

    # Reionization proceeds from low to high z: the neutral fraction grows.
    progress = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)
    target_xbar = (0.5 * (1.0 - np.cos(np.pi * progress))) ** 1.4

    # Dense regions ionize first: x_HI falls where the smoothed density is high.
    xhi = np.empty(shape, dtype=np.float32)
    for k in range(nz):
        sigma = 0.6 + 3.0 * (1.0 - target_xbar[k])
        smoothed = _smooth(delta[:, :, k], sigma)
        threshold = np.quantile(smoothed, float(target_xbar[k]))
        width = 0.05
        xhi[:, :, k] = 1.0 / (1.0 + np.exp((smoothed - threshold) / width))
    return density, xhi, z


class SyntheticDataset(BaseDataset):
    """Deterministic artificial lightcones with the same geometry per sample."""

    def __init__(self, n_samples: int = 4, shape=(64, 64, 64),
                 box_mpc: float = 200.0, seed: int = 0,
                 split: str | None = None, limit: int | None = None,
                 indices: list[int] | None = None):
        super().__init__(split=split, limit=limit, indices=indices)
        self.n_samples = n_samples
        self.shape = tuple(shape)
        self.box_mpc = box_mpc
        self.seed = seed

    def _samples(self):
        samples = []
        for index in range(self.n_samples):
            density, xhi, z = make_cone(self.shape, self.box_mpc,
                                        self.seed + index)
            samples.append(Sample(
                id=f"synthetic{index:04d}",
                inputs={"density": density},
                targets={"xhi": xhi},
                meta=Meta(box_mpc=self.box_mpc, z_grid=z,
                          omega_m=0.30966),
                split="test",
            ))
        return samples

    def describe(self):
        visible = self._visible(self._samples())
        return {
            "format": "synthetic",
            "n_total": self.n_samples,
            "n_samples": len(visible),
            "split": self.split,
            "shape": list(self.shape),
            "box_mpc": self.box_mpc,
        }


def write_demo(directory, n_samples: int = 4, shape=(64, 64, 64),
               box_mpc: float = 200.0, seed: int = 0) -> Path:
    """Write a manifest + npz cubes, ready for ``cm21eval run --data``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dataset = SyntheticDataset(n_samples, shape, box_mpc, seed)
    entries = []
    first_z = None
    for sample in dataset:
        first_z = sample.meta.z_grid
        path = directory / f"{sample.id}.npz"
        np.savez(
            path,
            density=sample.inputs["density"].astype(np.float32),
            xhi=sample.targets["xhi"].astype(np.float32),
            z_grid=np.asarray(sample.meta.z_grid, dtype=np.float64),
            omega_m=np.array(sample.meta.omega_m),
        )
        entries.append({"id": sample.id, "path": path.name, "split": "test"})
    manifest = {
        "box_mpc": float(box_mpc),
        "z_grid": np.asarray(first_z).tolist(),
        "target": "xhi",
        "inputs": ["density"],
        "samples": entries,
    }
    path = directory / "manifest.json"
    import json

    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
