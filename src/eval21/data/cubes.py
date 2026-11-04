"""eval21 cube cache (``eval21 build`` output, ``attrs["format"] == "eval21-cubes/1"``).

Lazy like :class:`Hdf5Dataset`: ids, splits, parameters and validity are indexed once and one
sample is read per iteration. Every stored field is available; fields listed in the file's
``targets`` attribute (plus the run's ``--target``) are targets, the rest are inputs.

``z_range=(zmin, zmax)`` crops the line of sight to that window, the way the LoReLi runs were
split into redshift groups. Samples whose valid planes do not cover the window are left out
(and listed in ``describe()["excluded"]``). Without a window every selected sample must be valid
along the whole line of sight; otherwise the dataset stops with the widest window that would work.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..spec import Meta, Sample
from .base import BaseDataset

FORMAT = "eval21-cubes/1"


def is_cube_cache(path) -> bool:
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            return str(handle.attrs.get("format", "")).startswith("eval21-cubes/")
    except (OSError, ImportError):
        return False


def _text(value) -> str:
    return value.decode() if isinstance(value, (bytes, bytearray)) else str(value)


class CubeCacheDataset(BaseDataset):
    def __init__(self, path, split: str | None = None, limit: int | None = None,
                 target: str = "xhi", density: str = "density", indices: list[int] | None = None,
                 box_mpc: float | None = None, z_range=None, fields=None, require_valid: bool = True,
                 dtype=np.float32, **_ignored):
        super().__init__(split=split, limit=limit, indices=indices)
        self.path = Path(path).expanduser().resolve()
        self.target = target
        self.density = density
        self.box_override = box_mpc
        self.z_range = None if z_range is None else (float(z_range[0]), float(z_range[1]))
        self.only_fields = None if fields is None else list(fields)
        self.require_valid = require_valid
        self.dtype = dtype
        self._index_cache = None
        self.cone_ids: list[str] = []

    def _index(self) -> dict:
        if self._index_cache is not None:
            return self._index_cache
        import h5py

        with h5py.File(self.path, "r") as h:
            if not str(h.attrs.get("format", "")).startswith("eval21-cubes/"):
                raise ValueError(f"{self.path} is not an eval21 cube cache")
            fields = json.loads(h.attrs["fields"])
            targets = json.loads(h.attrs.get("targets", "[]"))
            ids = [_text(v) for v in h["id"][:]]
            splits = [_text(v) or None for v in h["split"][:]]
            omega = np.asarray(h["omega_m"][:], np.float64)
            valid = np.asarray(h["valid"][:], bool)
            z = np.asarray(h["z_grid"][:], np.float64)
            names = [_text(v) for v in h["params/names"][:]]
            values = np.asarray(h["params/values"][:], np.float64)
            box = self.box_override or float(h.attrs.get("box_mpc", np.nan))
            shape = tuple(int(v) for v in h[f"fields/{fields[0]}"].shape[1:])
            attrs = {k: (_text(v) if isinstance(v, (bytes, str)) else v) for k, v in h.attrs.items()}
        if self.target not in fields:
            raise KeyError(f"{self.path.name} has no field {self.target!r}; fields {fields}")
        lo, hi = 0, len(z)
        if self.z_range is not None:
            keep = np.flatnonzero((z >= self.z_range[0] - 1e-9) & (z <= self.z_range[1] + 1e-9))
            if keep.size < 2:
                raise ValueError(f"z_range {self.z_range} selects {keep.size} planes of {z[0]:.3f}..{z[-1]:.3f}")
            lo, hi = int(keep[0]), int(keep[-1]) + 1
        rows = []
        for i, sid in enumerate(ids):
            rows.append({"id": sid, "split": splits[i],
                         "omega_m": None if not np.isfinite(omega[i]) else float(omega[i]),
                         "params": {n: float(v) for n, v in zip(names, values[i]) if np.isfinite(v)},
                         "covers": bool(valid[i, lo:hi].all())})
        self._index_cache = {"n": len(ids), "fields": fields, "targets": targets, "rows": rows, "z": z,
                             "lo": lo, "hi": hi, "box": None if not np.isfinite(box) else box,
                             "shape": shape, "valid": valid, "attrs": attrs}
        self.cone_ids = ids
        return self._index_cache

    def _selection(self) -> list[int]:
        index = self._index()
        rows = list(self.indices) if self.indices is not None else list(range(index["n"]))
        bad = [r for r in rows if not 0 <= r < index["n"]]
        if bad:
            raise ValueError(f"row indices outside {self.path.name} (0..{index['n'] - 1}): {bad[:5]}")
        if self.split:
            if all(index["rows"][r]["split"] is None for r in rows):
                raise ValueError(f"{self.path.name} declares no splits; build with --splits or pass --indices")
            rows = [r for r in rows if index["rows"][r]["split"] == self.split]
        if self.require_valid:
            uncovered = [r for r in rows if not index["rows"][r]["covers"]]
            if uncovered and self.z_range is None:
                z, valid = index["z"], index["valid"][rows]
                both = valid.all(axis=0)
                hint = (f"; planes valid for all of them: z {z[both][0]:.3f}..{z[both][-1]:.3f}"
                        if both.any() else "")
                raise ValueError(
                    f"{len(uncovered)} of {len(rows)} selected samples have invalid planes "
                    f"(e.g. {index['rows'][uncovered[0]]['id']}); pass --z-range ZMIN,ZMAX to evaluate a "
                    f"window and drop samples that do not cover it{hint}")
            self.excluded = [index["rows"][r]["id"] for r in uncovered]
            rows = [r for r in rows if index["rows"][r]["covers"]]
        if self.limit is not None:
            rows = rows[:self.limit]
        return rows

    def __len__(self):
        return len(self._selection())

    def __iter__(self):
        import h5py

        index = self._index()
        lo, hi = index["lo"], index["hi"]
        z = index["z"][lo:hi]
        wanted = self.only_fields or index["fields"]
        if self.target not in wanted:
            wanted = [self.target] + list(wanted)
        targets = set(index["targets"]) | {self.target}
        with h5py.File(self.path, "r") as h:
            for i in self._selection():
                row = index["rows"][i]
                data = {name: np.asarray(h[f"fields/{name}"][i, :, :, lo:hi], dtype=self.dtype)
                        for name in wanted}
                yield Sample(
                    id=row["id"],
                    inputs={k: v for k, v in data.items() if k not in targets},
                    targets={k: v for k, v in data.items() if k in targets},
                    meta=Meta(box_mpc=index["box"], z_grid=z, omega_m=row["omega_m"], params=row["params"]),
                    split=row["split"],
                )

    def row_indices_for_ids(self, ids) -> list[int]:
        index = self._index()
        position = {row["id"]: i for i, row in enumerate(index["rows"])}
        out = []
        for value in ids:
            key = str(value)
            for candidate in (key, f"cone{key}", key.removeprefix("cone")):
                if candidate in position:
                    out.append(position[candidate])
                    break
            else:
                raise KeyError(f"id {key!r} is absent from {self.path.name}")
        return out

    def describe(self) -> dict:
        index = self._index()
        rows = self._selection()
        z = index["z"][index["lo"]:index["hi"]]
        splits: dict[str, int] = {}
        for r in index["rows"]:
            key = r["split"] or "undeclared"
            splits[key] = splits.get(key, 0) + 1
        targets = sorted(set(index["targets"]) | {self.target})
        return {
            "format": FORMAT, "path": str(self.path), "n_total": index["n"], "n_samples": len(rows),
            "split": self.split, "shape": [*index["shape"][:2], len(z)], "box_mpc": index["box"],
            "z_range": [float(z[0]), float(z[-1])],
            "inputs": [f for f in index["fields"] if f not in targets],
            "targets": targets, "splits": splits,
            "excluded": len(getattr(self, "excluded", [])),
            "source": index["attrs"].get("source"),
        }
