"""HDF5 cube cache (the fno-21cm ``cubes_3d.h5`` layout and friends).

Auto-detects the two layouts that occur in practice:

* training layout -- ``x`` (N, C, Nx, Ny, Nz) and ``y`` (N, 1, Nx, Ny, Nz);
* analysis layout -- ``density`` and ``neutral_fraction`` (N, Nx, Ny, Nz),
  plus ``cone_id``, ``target_z`` and ``params``.

An HDF5 cache does not declare train/val/test splits. ``Sample.split`` is
``None`` unless an explicit ``split_map`` is supplied, and the dataset refuses
a ``--split`` filter rather than silently treating every row as test. For a
real evaluation select rows with ``--indices`` or with the checkpoint's own
``--indices-json`` (which prefers ``test_cone_ids``).

The whole channel stack is exposed as ``x`` for the torch/fno-21cm adapters;
the density cube is exposed separately so the physics metrics work on both
layouts.
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

_XHI_ALIASES = ("y", "neutral_fraction", "xhi")


class Hdf5Dataset(BaseDataset):
    def __init__(self, path, split: str | None = None, limit: int | None = None,
                 target: str = "xhi", density: str = "density",
                 x_key: str | None = None, y_key: str | None = None,
                 z_key: str = "target_z", params_key: str = "params",
                 id_key: str | None = "cone_id", box_mpc: float | None = None,
                 default_omega_m: float | None = None,
                 indices: list[int] | None = None,
                 split_map: dict[str, list[int]] | None = None):
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
        self.split_map = split_map
        self.cone_ids: list[str] = []
        self._loaded: list[Sample] | None = None

    # ------------------------------------------------------------------ keys
    def _target_key(self, handle) -> str | None:
        keys = set(handle.keys())
        if self.y_key:
            return self.y_key
        if self.target == "xhi":
            for key in _XHI_ALIASES:
                if key in keys:
                    return key
            return None
        return self.target if self.target in keys else None

    # ------------------------------------------------------------------ load
    def _samples(self) -> list[Sample]:
        if self._loaded is None:
            self._loaded = self._load()
        return self._loaded

    @staticmethod
    def _decode(value) -> str:
        return value.decode() if isinstance(value, (bytes, bytearray)) else str(value)

    def _load(self) -> list[Sample]:
        import h5py

        with h5py.File(self.path, "r") as handle:
            y_key = self._target_key(handle)
            if y_key is None:
                raise KeyError(
                    f"{self.path} has no target field for {self.target!r}; "
                    f"keys: {sorted(handle.keys())}. Pass y_key explicitly.")
            target = handle[y_key]
            n = target.shape[0]
            x_key = self.x_key or ("x" if "x" in handle else None)
            x = handle[x_key] if x_key else None
            density = handle[self.density] if self.density in handle else None
            z = np.asarray(handle[self.z_key]) if self.z_key in handle else None
            box = self.box_mpc
            if box is None and "box_mpc" in handle:
                box = float(np.asarray(handle["box_mpc"]))
            params = None
            param_names = None
            if self.params_key in handle:
                node = handle[self.params_key]
                if isinstance(node, h5py.Group):
                    params = np.asarray(node["values"])
                    if "names" in node:
                        param_names = [self._decode(v) for v in np.asarray(node["names"])]
                else:
                    params = np.asarray(node)
            ids = (np.asarray(handle[self.id_key])
                   if self.id_key and self.id_key in handle else None)
            if params is not None and param_names is None and self.params_key in handle:
                attrs = handle[self.params_key].attrs
                if "names" in attrs:
                    param_names = [self._decode(v) for v in attrs["names"]]

            split_of_row: dict[int, str] = {}
            for split_name, rows in (self.split_map or {}).items():
                for row in rows:
                    split_of_row[int(row)] = str(split_name)

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
                omega = self.default_omega_m
                if "OMm" in values and np.isfinite(values["OMm"]):
                    omega = float(values["OMm"])
                sample_id = (self._decode(ids[i]) if ids is not None
                             else f"sample{i:04d}")
                samples.append(Sample(
                    id=sample_id,
                    inputs=inputs,
                    targets={self.target: y},
                    meta=Meta(
                        box_mpc=box,
                        z_grid=None if z is None else np.asarray(z, dtype=np.float64),
                        omega_m=omega,
                        params=values,
                    ),
                    split=split_of_row.get(i),
                ))
        self.cone_ids = [sample.id for sample in samples]
        return samples

    # ------------------------------------------------------------- selection
    def row_indices_for_ids(self, ids) -> list[int]:
        """Map cone ids to cache rows; fail if any id is absent."""
        samples = self._samples()
        position = {sample.id: index for index, sample in enumerate(samples)}
        missing = [str(value) for value in ids if str(value) not in position]
        if missing:
            raise KeyError(
                f"{len(missing)} recorded cone id(s) are absent from "
                f"{self.path.name}: {missing[:5]}")
        return [position[str(value)] for value in ids]

    def describe(self) -> dict:
        samples = self._samples()
        visible = self._visible(samples)
        first = visible[0] if visible else None
        return {
            "format": "hdf5",
            "path": str(self.path),
            "n_total": len(samples),
            "n_samples": len(visible),
            "split": self.split,
            "indices": None if self.indices is None else len(self.indices),
            "shape": list(first.shape) if first else None,
            "box_mpc": first.meta.box_mpc if first else None,
            "inputs": sorted(first.inputs) if first else [],
            "targets": sorted(first.targets) if first else [],
            "splits": _split_counts(samples),
        }


def _split_counts(samples) -> dict:
    counts: dict[str, int] = {}
    for sample in samples:
        key = sample.split or "undeclared"
        counts[key] = counts.get(key, 0) + 1
    return counts
