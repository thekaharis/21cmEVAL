"""Generic per-lightcone files: a folder of ``.npz`` or ``.h5`` files, one lightcone each.

Everything is declared on the command line, so another simulator needs no code::

    eval21 build generic /data/mysim --out cubes/mysim \\
        --fields xhi=neutral_fraction,tb=brightness_temp,density=density \\
        --z-key redshifts --params-key params --box-mpc 200 --omega-m-param OMm

* ``--fields name=key``: eval21 name = dataset key (h5 paths like ``lightcone/density`` work);
* ``--z-key``: 1-D redshift of every LOS plane (or ``--z-native z0,z1`` for planes uniform in z);
* ``--distance-key``: optional 1-D comoving distance [Mpc]; otherwise computed from z with
  ``--omega-m`` / ``--hubble-h`` (eval21 flat LCDM);
* ``--params-key``: npz: JSON string or 1-D values with ``--param-names``; h5: a group with
  ``names``/``values`` datasets, a group's attributes, or a 1-D dataset (+ ``--param-names``);
* ``--box-mpc``: a number, or a key / root attribute holding the transverse size;
* ``--omega-m``: a number; ``--omega-m-param NAME`` takes it from the parameters instead;
* ``--los-axis``: axis of the stored arrays that is the line of sight (default 2);
* ``--density-convention``: ``one_plus_delta`` (default), ``delta`` (+1 is added) or ``rho``
  (divided by the cone mean); the written density must have mean ~1.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ..physics import HUBBLE_H, OMEGA_M, comoving_distance_mpc
from .common import BuildError, Item, Source, file_record


class _Reader:
    """Uniform key access to one npz or h5 file."""

    def __init__(self, path: Path):
        self.path = path
        self.h5 = path.suffix.lower() in (".h5", ".hdf5")
        if self.h5:
            import h5py

            self.handle = h5py.File(path, "r")
        else:
            self.handle = np.load(path, allow_pickle=False)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.handle.close()

    def has(self, key) -> bool:
        if self.h5:
            return key in self.handle
        return key in self.handle.files

    def attr(self, key):
        if self.h5 and key in self.handle.attrs:
            value = self.handle.attrs[key]
            return value.item() if hasattr(value, "item") else value
        return None

    def array(self, key, index=None):
        node = self.handle[key]
        if self.h5:
            return np.asarray(node[()] if index is None else node[index])
        return np.asarray(node if index is None else node[index])

    def shape(self, key):
        return tuple(self.handle[key].shape)

    def node(self, key):
        return self.handle[key]


def _scalar(value):
    value = np.asarray(value)
    if value.dtype.kind in "SU" or value.dtype == object:
        return value.item().decode() if isinstance(value.item(), bytes) else value.item()
    return float(value.reshape(-1)[0])


class GenericSource(Source):
    format = "generic"

    def __init__(self, root, options):
        super().__init__(root, options)
        so = self.so
        self.pattern = so.get("pattern") or "*"
        self.z_key = so.get("z_key")
        self.z_native = so.get("z_native")
        self.distance_key = so.get("distance_key")
        self.params_key = so.get("params_key")
        self.param_names = so.get("param_names")
        self.box = so.get("box_mpc")
        self.omega_m = so.get("omega_m")
        self.omega_m_param = so.get("omega_m_param")
        self.h = float(so.get("hubble_h") or HUBBLE_H)
        self.hubble_h = self.h
        self.los_axis = int(so.get("los_axis") if so.get("los_axis") is not None else 2)
        self.id_regex = so.get("id_regex") or r"(\d+)\D*$"
        self.density_convention = so.get("density_convention") or "one_plus_delta"
        if self.density_convention not in ("one_plus_delta", "delta", "rho"):
            raise ValueError("--density-convention must be one_plus_delta, delta or rho")
        if not (self.z_key or self.z_native):
            raise ValueError("generic: pass --z-key (per-plane redshifts) or --z-native z0,z1")
        if self.box is None:
            raise ValueError("generic: pass --box-mpc (a number, a key or a root attribute)")

    def available_fields(self):
        return {name: {"source": key} for name, key in (self.field_map or {}).items()}

    def resolve_fields(self, requested):
        if not requested:
            raise ValueError("generic: --fields name=key[,name=key...] is required")
        out = []
        for spec in requested:
            name, _, key = str(spec).partition("=")
            out.append((name.strip(), (key or name).strip()))
        if len({n for n, _ in out}) != len(out):
            raise ValueError("--fields lists a field twice")
        self.field_map = dict(out)
        return out

    def default_fields(self):
        return []

    def scan_signature(self):
        return {"z_key": self.z_key, "z_native": self.z_native, "distance_key": self.distance_key,
                "params_key": self.params_key, "los_axis": self.los_axis, "box": self.box,
                "omega_m": self.omega_m, "omega_m_param": self.omega_m_param, "h": self.h}

    def read_signature(self):
        return {"density_convention": self.density_convention}

    def discover(self):
        items = []
        for path in sorted(self.root.glob(self.pattern)):
            if path.suffix.lower() not in (".npz", ".h5", ".hdf5") or not path.is_file():
                continue
            match = re.search(self.id_regex, path.stem)
            cone = int(match.group(1)) if match and match.group(1).isdigit() else (
                match.group(1) if match else path.stem)
            items.append(Item(cone, path))
        return items

    def _params(self, r: _Reader) -> dict:
        if not self.params_key:
            return {}
        key = self.params_key
        if r.h5 and key in r.handle and hasattr(r.node(key), "keys"):
            group = r.node(key)
            if "names" in group and "values" in group:
                names = [n.decode() if isinstance(n, bytes) else str(n) for n in group["names"][()]]
                return dict(zip(names, (float(v) for v in group["values"][()])))
            return {k: float(np.asarray(v).reshape(-1)[0]) for k, v in group.attrs.items()
                    if np.asarray(v).dtype.kind in "fiu"}
        if r.has(key):
            value = r.array(key)
            if value.dtype.kind in "SU" or value.dtype == object:
                text = value.item()
                return {k: float(v) for k, v in json.loads(text.decode() if isinstance(text, bytes) else text).items()}
            values = [float(v) for v in value.reshape(-1)]
            names = self.param_names or (
                [n.decode() if isinstance(n, bytes) else str(n) for n in r.node(key).attrs["names"]]
                if r.h5 and "names" in r.node(key).attrs else None)
            if not names:
                raise ValueError(f"{r.path}: {key} holds bare values; pass --param-names")
            return dict(zip(names, values))
        if r.h5 and r.attr(key) is not None:
            return json.loads(r.attr(key))
        raise KeyError(f"{r.path}: no params key {key!r}")

    def _number(self, r: _Reader, spec, what):
        try:
            return float(spec)
        except (TypeError, ValueError):
            pass
        if r.has(spec):
            return float(_scalar(r.array(spec)))
        if r.attr(spec) is not None:
            return float(r.attr(spec))
        raise KeyError(f"{r.path}: {what} key {spec!r} not found")

    def scan(self, item: Item) -> dict:
        with _Reader(item.path) as r:
            fields = [name for name, key in self.field_map.items() if r.has(key)]
            if not fields:
                raise KeyError(f"{item.path}: none of the keys {list(self.field_map.values())}")
            shape = list(r.shape(self.field_map[fields[0]]))
            if len(shape) != 3:
                raise ValueError(f"{item.path}: field {fields[0]} has shape {shape}, expected 3-D")
            los = shape.pop(self.los_axis)
            shape = shape + [los]
            params = self._params(r)
            box = self._number(r, self.box, "box")
            if self.z_key:
                z = np.asarray(r.array(self.z_key), dtype=np.float64).ravel()
            else:
                z0, z1 = (float(v) for v in str(self.z_native).split(","))
                z = np.linspace(z0, z1, los)
            d = (np.asarray(r.array(self.distance_key), dtype=np.float64).ravel()
                 if self.distance_key else None)
            omega = None
            if self.omega_m_param:
                omega = float(params[self.omega_m_param])
            elif self.omega_m is not None:
                omega = self._number(r, self.omega_m, "omega_m")
        if len(z) != los:
            raise ValueError(f"{item.path}: {len(z)} redshifts for {los} LOS planes")
        if z[0] > z[-1]:
            raise ValueError(f"{item.path}: redshift must increase along the LOS axis (flip the file)")
        if d is None:
            d = comoving_distance_mpc(z, omega if omega is not None else OMEGA_M, self.h)
        return {"cone_id": item.cone_id, "files": [file_record(item.path)], "shape": shape,
                "cell_mpc": box / shape[0], "z": z, "d": d, "valid": [0, len(z) - 1],
                "omega_m": omega, "params": params, "fields": fields,
                "info": {} if self.distance_key else {"distance": f"eval21 flat LCDM, h = {self.h}, "
                                                                  f"Omega_m = {omega if omega is not None else OMEGA_M}"}}

    def distance_table(self, scan):
        if self.distance_key:
            return scan["z"], scan["d"]
        omega = scan.get("omega_m") or OMEGA_M
        z = np.linspace(scan["z"][0], scan["z"][-1], 100001)
        return z, comoving_distance_mpc(z, omega, self.h) + scan["d"][0]

    def read(self, item, scan, names, lo, hi, crop):
        x0, x1, y0, y1 = crop
        out = {}
        with _Reader(item.path) as r:
            for name in names:
                index = [slice(None)] * 3
                index[self.los_axis] = slice(lo, hi)
                array = np.asarray(r.array(self.field_map[name], tuple(index)), dtype=np.float32)
                array = np.moveaxis(array, self.los_axis, 2)[x0:x1, y0:y1]
                if name == "density":
                    if self.density_convention == "delta":
                        array = array + np.float32(1)
                    elif self.density_convention == "rho":
                        full = np.asarray(r.array(self.field_map[name]), dtype=np.float64)
                        array = (array / full.mean()).astype(np.float32)
                out[name] = np.ascontiguousarray(array)
        return out

    def check_output(self, cone, stats):
        if "density" in stats and abs(stats["density"]["mean"] - 1.0) > 0.1:
            raise BuildError(f"cone {cone}: written density mean {stats['density']['mean']:.4f}, not ~1; "
                             f"set --density-convention (now {self.density_convention})")

    def native_resolution(self, scans):
        if not scans:
            return {}
        return {"cyl_native_cell_mpc": float(scans[0]["cell_mpc"]),
                "transverse_cells": scans[0]["shape"][:2], "native_los_planes": scans[0]["shape"][2]}

    def notes(self):
        return [f"density convention of the source: {self.density_convention}"]
