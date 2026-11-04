"""Readers that turn one simulation file into fields on its native grid (``eval21 build``).

A source discovers the samples of a folder and reads one at a time:

    source = open_source("21cmfast", folder)
    for item in source.items():                       # SourceItem(id, path)
        raw = source.read(item, fields=("density", "xhi"))
        raw.fields["xhi"]      # (Nx, Ny, Nlos) float32 on raw.z (1-D, any order)
        raw.valid              # (Nlos,) bool or None, planes usable for evaluation
        raw.params, raw.box_mpc, raw.omega_m

Canonical field names (what eval21 metrics expect): ``density`` = overdensity delta,
``xhi`` = neutral fraction, ``tb`` = 21-cm brightness temperature in mK. Other names are passed
through. Built-in sources:

``21cmfast``   py21cmfast lightcone files (``lightcone/<field>``, ``lightcone/lightcone_redshifts``,
               ``params/{names,values}``, attribute ``box_len_mpc``) -- the files fno-21cm's
               ``dataset/build_cubes.py`` reads (``21cmfast_11d_sample*.h5``).
``loreli``     LoReLi II ``.dat`` lightcones (five-field ``xa,tpHI,xion,dens,dtb`` or dT_b only),
               128^2 x 1536 planes uniform in z = 5..15, index 0 = low z; empty or corrupt
               planes are marked invalid; parameters from the simulation number.
``npz``        one ``.npz`` per sample with 3-D arrays, ``z_grid``, optional ``params`` (JSON
               string or a ``params_names``/``params_values`` pair), ``box_mpc``, ``omega_m``.
``python:FILE.py[:NAME]``  a custom class with ``items()`` and ``read(item, fields)``.
"""
from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SourceItem:
    id: str
    path: Path
    extra: dict = field(default_factory=dict)


@dataclass
class RawSample:
    fields: dict
    z: np.ndarray
    valid: np.ndarray | None = None
    params: dict = field(default_factory=dict)
    box_mpc: float | None = None
    omega_m: float | None = None


class Source:
    name = "source"
    default_fields: tuple = ()
    units: dict = {}

    def items(self) -> list[SourceItem]:
        raise NotImplementedError

    def read(self, item: SourceItem, fields) -> RawSample:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"source": self.name}


def _number(path: Path) -> str:
    found = re.findall(r"\d+", path.stem)
    return found[-1] if found else path.stem


# ------------------------------------------------------------------------------------- 21cmFAST
class Fast21cmSource(Source):
    """py21cmfast lightcone HDF5 files."""

    name = "21cmfast"
    default_fields = ("density", "xhi")
    aliases = {"density": "density", "xhi": "neutral_fraction", "tb": "brightness_temp",
               "x_e": "x_e", "ts": "spin_temperature", "tk": "kinetic_temp_neutral"}
    units = {"density": "overdensity", "xhi": "neutral fraction", "tb": "mK"}

    def __init__(self, folder, pattern: str = "*.h5", id_from: str = "number"):
        self.folder = Path(folder)
        self.pattern = pattern
        self.id_from = id_from

    def items(self):
        files = sorted(self.folder.glob(self.pattern))
        if not files:
            raise FileNotFoundError(f"no {self.pattern} files in {self.folder}")
        # cone_id = position in the sorted listing (fno-21cm's convention) unless the file
        # names carry numbers and id_from='number'.
        out = []
        for index, path in enumerate(files):
            sid = _number(path) if self.id_from == "number" else (
                str(index) if self.id_from == "index" else path.stem)
            out.append(SourceItem(sid, path, {"cone_id": index}))
        return out

    def read(self, item, fields):
        import h5py

        with h5py.File(item.path, "r") as handle:
            lc = handle["lightcone"]
            z = np.asarray(lc["lightcone_redshifts"], dtype=np.float64)
            out = {}
            for name in fields:
                key = self.aliases.get(name, name)
                if key not in lc:
                    raise KeyError(f"{item.path.name}: no lightcone field {key!r} (for {name!r}); "
                                   f"has {sorted(k for k in lc if lc[k].ndim == 3)}")
                out[name] = np.asarray(lc[key], dtype=np.float32)
            params = {}
            if "params" in handle and "names" in handle["params"]:
                names = [v.decode() if isinstance(v, bytes) else str(v)
                         for v in np.asarray(handle["params"]["names"])]
                values = np.asarray(handle["params"]["values"], dtype=np.float64).ravel()
                params = {n: float(v) for n, v in zip(names, values)}
            omega = params.get("OMm")
            if omega is None and "params/fixed_cosmo_params" in handle:
                attrs = handle["params/fixed_cosmo_params"].attrs
                omega = float(attrs["OMm"]) if "OMm" in attrs else None
            box = float(handle.attrs["box_len_mpc"]) if "box_len_mpc" in handle.attrs else None
        return RawSample(out, z, None, params, box, omega)


# --------------------------------------------------------------------------------------- LoReLi
LORELI_FIELDS = ("xa", "tpHI", "xion", "dens", "dtb")
LORELI_GRID = {"RH": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
               "FX": [0.1, 0.15, 0.22, 0.32, 0.46, 0.68, 1.0, 1.47, 2.15, 3.16, 4.64, 6.81, 10.0],
               "FE": [0.275, 0.05, 0.5]}


class LoreliSource(Source):
    """LoReLi II ``lightcone_*_simuN.dat`` files (see ``README_loreli_lc128.md``)."""

    name = "loreli"
    default_fields = ("density", "xhi", "tb")
    units = {"density": "overdensity (slice mean over valid cells)", "xhi": "neutral fraction",
             "tb": "mK", "xa": "x_alpha", "tpHI": "K"}
    n_native, n_los, z_range = 128, 1536, (5.0, 15.0)
    dtb_abs_max = 1.0e3

    def __init__(self, folder, l_perp: float = 294.0, omega_m: float = 0.308,
                 params_json: str | None = None, five_field_only: bool | None = None):
        self.folder = Path(folder)
        self.l_perp = float(l_perp)
        self.omega_m = float(omega_m)
        self.five_field_only = five_field_only
        grid = dict(LORELI_GRID)
        if params_json is None:
            for candidate in (self.folder / "loreli2_parameters.json",
                              self.folder.parent / "loreli_globals" / "loreli2_parameters.json"):
                if candidate.exists():
                    params_json = candidate
                    break
        self.pairs = None
        if params_json is not None:
            table = json.loads(Path(params_json).read_text())
            grid.update({k: table[k] for k in ("RH", "FX", "FE") if k in table})
            self.pairs = table.get("PAIRS")
        self.grid = grid

    def items(self):
        five = {_number(p): p for p in self.folder.glob("lightcone_xa_tpHI_xion_dens_dtb_no_rsd_simu*.dat")}
        dtb = {_number(p): p for p in self.folder.glob("lightcone_dtb_no_rsd_simu*.dat")}
        if not five and not dtb:
            raise FileNotFoundError(f"no LoReLi lightcone_*.dat files in {self.folder}")
        only_five = self.five_field_only if self.five_field_only is not None else bool(five)
        ids = sorted(five) if only_five else sorted(set(five) | set(dtb))
        return [SourceItem(i, five.get(i, dtb.get(i)), {"five_field": i in five}) for i in ids]

    def params_of(self, sim: int) -> dict:
        j = int(sim) - 10000          # the grid rule of loreli2_parameters.json
        out = {"r_HS": self.grid["RH"][j % 6], "f_X": self.grid["FX"][(j // 6) % 13],
               "f_esc": self.grid["FE"][min(j // 3276, 2)]}
        if self.pairs:
            tau, log_mmin = self.pairs[(j // 78) % len(self.pairs)]
            out.update(tau=float(tau), logMmin=float(log_mmin))
        return out

    def read(self, item, fields):
        z = np.linspace(*self.z_range, self.n_los)
        n = self.n_native
        if item.extra.get("five_field"):
            data = np.fromfile(item.path, dtype="<f4")
            expected = n * n * self.n_los * len(LORELI_FIELDS)
            if data.size != expected:
                raise ValueError(f"{item.path}: {data.size} values, expected {expected}")
            data = data.reshape(n, n, self.n_los, len(LORELI_FIELDS))
            get = {name: (lambda name=name: np.array(data[..., LORELI_FIELDS.index(name)], np.float32))
                   for name in LORELI_FIELDS}
        else:
            data = np.fromfile(item.path, dtype="<f4").reshape(n, n, self.n_los)
            get = {"dtb": lambda: np.array(data, np.float32)}
        dtb = get["dtb"]()
        finite = np.isfinite(dtb).all((0, 1))
        sane = np.where(np.isfinite(dtb), np.abs(dtb), 0).max((0, 1)) < self.dtb_abs_max
        valid = finite & sane
        dens = get["dens"]() if "dens" in get else None
        if dens is not None:
            mean = dens.mean((0, 1))
            valid &= mean > 0
            if valid.any():
                # partly filled first/last planes of a particle lightcone (see rhizome loreli.py)
                valid &= np.abs(mean / np.median(mean[valid]) - 1) <= 0.10
        idx = np.flatnonzero(valid)
        keep = np.zeros(self.n_los, bool)
        if idx.size:
            runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            keep[max(runs, key=len)] = True
        out = {}
        for name in fields:
            if name == "tb":
                value = np.where(keep[None, None, :], np.nan_to_num(dtb), 0.0)
            elif name == "xhi":
                if "xion" not in get:
                    raise KeyError(f"{item.path.name}: x_HI needs a five-field file")
                value = 1.0 - get["xion"]()
            elif name == "density":
                if dens is None:
                    raise KeyError(f"{item.path.name}: density needs a five-field file")
                good = dens > 0
                rho = np.where(good, dens, 0).sum((0, 1), keepdims=True) / np.maximum(
                    good.sum((0, 1), keepdims=True), 1)
                ratio = np.divide(dens, rho, out=np.zeros_like(dens), where=rho > 0)
                value = np.where(good, ratio - 1.0, 0.0)
            elif name in get:
                value = get[name]()
            else:
                raise KeyError(f"LoReLi has no field {name!r}; use density, xhi, tb or {LORELI_FIELDS}")
            out[name] = np.where(keep[None, None, :], value, 0.0).astype(np.float32)
        return RawSample(out, z, keep, self.params_of(int(item.id)), self.l_perp, self.omega_m)

    def describe(self):
        return {"source": self.name, "l_perp": self.l_perp, "omega_m": self.omega_m}


# ------------------------------------------------------------------------------------------ npz
class NpzSource(Source):
    """A folder of per-sample ``.npz`` files (eval21's own cube format, or anything similar)."""

    name = "npz"

    def __init__(self, folder, pattern: str = "*.npz", z_key: str = "z_grid"):
        self.folder = Path(folder)
        self.pattern = pattern
        self.z_key = z_key

    def items(self):
        files = sorted(self.folder.rglob(self.pattern))
        files = [f for f in files if f.name != "manifest.json"]
        if not files:
            raise FileNotFoundError(f"no {self.pattern} files under {self.folder}")
        return [SourceItem(f.stem, f) for f in files]

    def read(self, item, fields):
        with np.load(item.path, allow_pickle=False) as data:
            if not fields:
                fields = [k for k in data.files if data[k].ndim == 3]
            out = {name: np.asarray(data[name], np.float32) for name in fields}
            nz = next(iter(out.values())).shape[2]
            z = np.asarray(data[self.z_key], np.float64) if self.z_key in data.files else np.arange(nz, dtype=float)
            params = {}
            if "params" in data.files:
                params = json.loads(str(data["params"]))
            elif "params_names" in data.files:
                params = dict(zip([str(n) for n in data["params_names"]],
                                  map(float, data["params_values"])))
            valid = np.asarray(data["valid"], bool) if "valid" in data.files else None
            box = float(data["box_mpc"]) if "box_mpc" in data.files else None
            omega = float(data["omega_m"]) if "omega_m" in data.files else None
        return RawSample(out, z, valid, params, box, omega)


SOURCES = {"21cmfast": Fast21cmSource, "loreli": LoreliSource, "npz": NpzSource}


def open_source(kind: str, folder, **options) -> Source:
    """``21cmfast``, ``loreli``, ``npz``, ``python:FILE.py[:Class]`` or ``auto``."""
    folder = Path(folder)
    if kind == "auto":
        if any(folder.glob("lightcone_*_simu*.dat")):
            kind = "loreli"
        elif any(folder.glob("*.npz")) or any(folder.rglob("*.npz")):
            kind = "npz"
        elif any(folder.glob("*.h5")):
            kind = "21cmfast"
        else:
            raise ValueError(f"cannot tell the format of {folder}; pass --format")
    if kind.startswith("python:"):
        path, _, name = kind[len("python:"):].partition(":")
        spec = importlib.util.spec_from_file_location("_eval21_source", Path(path).expanduser())
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = getattr(module, name or "Source")
        return cls(folder, **options)
    if kind not in SOURCES:
        raise ValueError(f"unknown source {kind!r}; use {', '.join(SOURCES)}, python:FILE.py or auto")
    return SOURCES[kind](folder, **options)
