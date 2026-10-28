"""HDF5 cube cache (the fno-21cm ``cubes_3d.h5`` layout and friends).

Auto-detects the two layouts that occur in practice:

* training layout -- ``x`` (N, C, Nx, Ny, Nz) and ``y`` (N, 1, Nx, Ny, Nz);
* analysis layout -- ``density`` and ``neutral_fraction`` (N, Nx, Ny, Nz),
  plus ``cone_id``, ``target_z`` and ``params``.

Every key can be overridden from the CLI for other files. Input channels
beyond density are not reconstructed from ``x``: the whole stack is exposed
as an input named ``x``, which the torch adapter accepts directly. The
density cube is also exposed separately when the file stores it, so the
physics metrics work on both layouts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..spec import Meta, Sample
from .base import BaseDataset

# Canonical fno-21cm parameter order, used when the file stores a bare (N, 11)
# array without a names attribute.
LIGHTCONE_PARAM_NAMES = (
    "F_ESC10", "F_STAR10", "ALPHA_ESC", "ALPHA_STAR", "L_X", "NU_X_THRESH",
    "M_TURN", "t_STAR", "X_RAY_SPEC_INDEX", "OMm", "SIGMA_8",
)


class Hdf5Dataset(BaseDataset):
    def __init__(self, path, split: str | None = None, limit: int | None = None,
                 target: str = "xhi", density: str = "density",
                 x_key: str | None = None, y_key: str | None = None,
                 z_key: str = "target_z", params_key: str = "params",
                 id_key: str | None = "cone_id", box_mpc: float | None = None,
                 default_omega_m: float | None = None,
                 indices: list[int] | None = None):
        super().__init__(split=split, limit=limit, indices=indices)
        self.path = Path(path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"data path not found: {self.path}")
        self.target = target
        self.density = density
        self.x_key = x_key
        self.y_key = y_key
        self.z_key = z_key
        self.params_key = params_key
        self.id_key = id_key
        self.box_mpc = box_mpc
        self.default_omega_m = default_omega_m
        self._loaded: list[Sample] | None = None

    def _keys(self, handle) -> tuple[str | None, str | None]:
        keys = set(handle.keys())
        x_key = self.x_key or ("x" if "x" in keys else None)
        if self.y_key:
            y_key = self.y_key
        elif "y" in keys:
            y_key = "y"
        elif "neutral_fraction" in keys:
            y_key = "neutral_fraction"
        elif self.target in keys:
            y_key = self.target
        else:
            y_key = None
        return x_key, y_key

    def _samples(self) -> list[Sample]:
        if self._loaded is None:
            self._loaded = self._load()
        return self._loaded

    def _load(self) -> list[Sample]:
        import h5py

        with h5py.File(self.path, "r") as handle:
            x_key, y_key = self._keys(handle)
            if y_key is None:
                raise KeyError(
                    f"{self.path} has neither 'y' nor 'neutral_fraction' nor "
                    f"target {self.target!r}; keys: {sorted(handle.keys())}")
            target = handle[y_key]
            n = target.shape[0]
            x = handle[x_key] if x_key else None
            density = handle[self.density] if self.density in handle else None
            z = np.asarray(handle[self.z_key]) if self.z_key in handle else None
            params = None
            param_names = None
            if self.params_key in handle:
                node = handle[self.params_key]
                if isinstance(node, h5py.Group):
                    params = np.asarray(node["values"])
                    if "names" in node:
                        param_names = [
                            v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
                            for v in np.asarray(node["names"])]
                else:
                    params = np.asarray(node)
            ids = (np.asarray(handle[self.id_key])
                   if self.id_key and self.id_key in handle else None)
            if params is not None and param_names is None and self.params_key in handle:
                attrs = handle[self.params_key].attrs
                if "names" in attrs:
                    param_names = [str(v) for v in attrs["names"]]

            samples = []
            for i in range(n):
                y = np.asarray(target[i])
                if y.ndim == 4:
                    y = y[0]
                inputs = {}
                if x is not None:
                    inputs["x"] = np.asarray(x[i])
                if density is not None:
                    inputs["density"] = np.asarray(density[i])
                values = {}
                if params is not None and i < len(params):
                    if param_names:
                        names = param_names
                    elif len(params[i]) == len(LIGHTCONE_PARAM_NAMES):
                        names = LIGHTCONE_PARAM_NAMES
                    else:
                        names = [f"p{j}" for j in range(len(params[i]))]
                    values = {name: float(v) for name, v in zip(names, params[i])}
                samples.append(Sample(
                    id=str(ids[i]) if ids is not None else f"sample{i:04d}",
                    inputs=inputs,
                    targets={self.target: y},
                    meta=Meta(
                        box_mpc=self.box_mpc,
                        z_grid=None if z is None else np.asarray(z, dtype=np.float64),
                        omega_m=self.default_omega_m,
                        params=values,
                    ),
                ))
        return samples

    def describe(self) -> dict:
        samples = self._samples()
        first = samples[0] if samples else None
        return {
            "format": "hdf5",
            "path": str(self.path),
            "n_samples": len(samples),
            "shape": list(first.shape) if first else None,
            "box_mpc": first.meta.box_mpc if first else None,
            "inputs": sorted(first.inputs) if first else [],
            "targets": sorted(first.targets) if first else [],
        }
