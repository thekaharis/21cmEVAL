"""HDF5 cube cache (the fno-21cm ``cubes_3d.h5`` layout and friends).

The loader is **lazy**: it indexes ids, parameters and geometry once, then
reads one cube per iteration. A 6600-cone cache is ~265 GB, so materialising
every row (the behaviour this replaced) cannot run on a login node and wastes
most of a large node.

Auto-detects the two layouts that occur in practice:

* training layout -- ``x`` (N, C, Nx, Ny, Nz) and ``y`` (N, 1, Nx, Ny, Nz);
* analysis layout -- ``density`` and ``neutral_fraction`` (N, Nx, Ny, Nz),
  plus ``cone_id``, ``target_z`` and ``params``.

An HDF5 cache does not declare train/val/test splits. ``Sample.split`` is
``None`` unless an explicit ``split_map`` is supplied, and the dataset refuses
a ``--split`` filter rather than silently treating every row as test. For a
real evaluation select rows with ``--indices`` or with the checkpoint's own
``--indices-json`` (which prefers ``test_cone_ids``).
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
        self._index_cache: dict | None = None

    # ------------------------------------------------------------------ index
    @staticmethod
    def _decode(value) -> str:
        return value.decode() if isinstance(value, (bytes, bytearray)) else str(value)

    def _keys(self, handle) -> tuple[str | None, str | None, str | None]:
        keys = set(handle.keys())
        x_key = self.x_key or ("x" if "x" in keys else None)
        if self.y_key:
            y_key = self.y_key
        elif self.target == "xhi":
            y_key = next((key for key in _XHI_ALIASES if key in keys), None)
        else:
            y_key = self.target if self.target in keys else None
        density_key = self.density if self.density in keys else None
        return x_key, y_key, density_key

    def _index(self) -> dict:
        """Read ids, parameters and geometry only -- never the cubes."""
        if self._index_cache is not None:
            return self._index_cache
        import h5py

        with h5py.File(self.path, "r") as handle:
            x_key, y_key, density_key = self._keys(handle)
            if y_key is None:
                raise KeyError(
                    f"{self.path} has no target field for {self.target!r}; "
                    f"keys: {sorted(handle.keys())}. Pass y_key explicitly.")
            target = handle[y_key]
            n = int(target.shape[0])
            sample_shape = tuple(int(v) for v in target.shape[1:])
            if len(sample_shape) == 4 and sample_shape[0] == 1:
                sample_shape = sample_shape[1:]
            z = (np.asarray(handle[self.z_key], dtype=np.float64)
                 if self.z_key in handle else None)
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
                        param_names = [self._decode(v)
                                       for v in np.asarray(node["names"])]
                else:
                    params = np.asarray(node)
                if params is not None and param_names is None:
                    attrs = handle[self.params_key].attrs
                    if "names" in attrs:
                        param_names = [self._decode(v) for v in attrs["names"]]
            ids = (np.asarray(handle[self.id_key])
                   if self.id_key and self.id_key in handle else None)

        rows = []
        for i in range(n):
            values = {}
            if params is not None and i < len(params):
                if param_names:
                    names = param_names
                elif len(params[i]) == len(LIGHTCONE_PARAM_NAMES):
                    names = LIGHTCONE_PARAM_NAMES
                else:
                    names = [f"p{j}" for j in range(len(params[i]))]
                values = {name: float(value) for name, value in zip(names, params[i])}
            omega = self.default_omega_m
            if "OMm" in values and np.isfinite(values["OMm"]):
                omega = float(values["OMm"])
            rows.append({
                "id": self._decode(ids[i]) if ids is not None else f"sample{i:04d}",
                "params": values,
                "omega_m": omega,
            })

        split_of_row = {}
        for split_name, rows_in_split in (self.split_map or {}).items():
            for row in rows_in_split:
                split_of_row[int(row)] = str(split_name)

        self._index_cache = {
            "n": n,
            "x_key": x_key,
            "y_key": y_key,
            "density_key": density_key,
            "shape": sample_shape,
            "z": z,
            "box": box,
            "rows": rows,
            "splits": [split_of_row.get(i) for i in range(n)],
        }
        self.cone_ids = [row["id"] for row in rows]
        return self._index_cache

    # -------------------------------------------------------------- selection
    def _selection(self) -> list[int]:
        index = self._index()
        rows = list(self.indices) if self.indices is not None else list(range(index["n"]))
        rows = [row for row in rows if 0 <= row < index["n"]]
        if self.split:
            declared = index["splits"]
            if all(value is None for value in declared):
                raise ValueError(
                    f"this dataset does not declare splits, so --split "
                    f"{self.split!r} cannot be honoured. Pass --indices or "
                    f"--indices-json to select the evaluation rows explicitly.")
            rows = [row for row in rows if declared[row] == self.split]
        if self.limit is not None:
            rows = rows[:self.limit]
        return rows

    def __len__(self) -> int:
        return len(self._selection())

    def __iter__(self):
        index = self._index()
        rows = self._selection()
        if not rows:
            return
        import h5py

        with h5py.File(self.path, "r") as handle:
            y = handle[index["y_key"]]
            x = handle[index["x_key"]] if index["x_key"] else None
            density = handle[index["density_key"]] if index["density_key"] else None
            for i in rows:
                target = np.asarray(y[i])
                if target.ndim == 4 and target.shape[0] == 1:
                    target = target[0]
                inputs = {}
                if x is not None:
                    inputs["x"] = np.asarray(x[i])
                if density is not None:
                    inputs["density"] = np.asarray(density[i])
                row = index["rows"][i]
                yield Sample(
                    id=row["id"],
                    inputs=inputs,
                    targets={self.target: target},
                    meta=Meta(
                        box_mpc=index["box"],
                        z_grid=index["z"],
                        omega_m=row["omega_m"],
                        params=row["params"],
                    ),
                    split=index["splits"][i],
                )

    def row_indices_for_ids(self, ids) -> list[int]:
        index = self._index()
        position = {row["id"]: i for i, row in enumerate(index["rows"])}
        missing = [str(value) for value in ids if str(value) not in position]
        if missing:
            raise KeyError(
                f"{len(missing)} recorded cone id(s) are absent from "
                f"{self.path.name}: {missing[:5]}")
        return [position[str(value)] for value in ids]

    def describe(self) -> dict:
        index = self._index()
        rows = self._selection()
        z = index["z"]
        splits: dict[str, int] = {}
        for split in index["splits"]:
            key = split or "undeclared"
            splits[key] = splits.get(key, 0) + 1
        return {
            "format": "hdf5",
            "path": str(self.path),
            "n_total": index["n"],
            "n_samples": len(rows),
            "split": self.split,
            "indices": None if self.indices is None else len(self.indices),
            "shape": list(index["shape"]),
            "box_mpc": index["box"],
            "z_range": None if z is None else [float(z[0]), float(z[-1])],
            "inputs": [name for name in ("x", "density")
                       if index[f"{name}_key" if name == "x" else "density_key"]],
            "targets": [self.target],
            "splits": splits,
        }
