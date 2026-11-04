"""Shared machinery of ``eval21 build``: selection, splits, LOS plans, geometry groups, writer, manifests.

A *source* (one module per raw format) knows how to find lightcone files, how to read their
metadata cheaply (``scan``) and how to read a range of native LOS planes of named fields in the
eval21 conventions (``read``). Everything else is format independent and lives here:

1. **select** files (``--ids``, ``--limit``, ``--only-split``) and assign splits;
2. **scan** every selected file (cached under ``<out>/.eval21_build/scans`` so a resumed or
   repeated build does not re-read raw data for formats whose scan must read the cube);
3. **plan** geometry groups: a group is a set of cones that share one output geometry; its LOS
   window is the intersection of the members' valid native ranges and ``--z-range``;
4. **write** one npz per cone (atomic, skipped when an identical output already exists);
5. write one ``manifest.json`` per group (plus ``manifest_<target>.json`` per extra target).

LOS modes (``--los``):

``native``    keep the native planes inside the window (the cone's own redshifts);
``nearest``   for each redshift of ``--z-grid`` the nearest native plane (never interpolated);
``linear``    linear interpolation in comoving distance onto ``--z-grid``;
``comoving``  linear interpolation onto a uniform comoving grid of spacing ``--los-cell``
              (default: the transverse cell), anchored at the cone's first native plane, so cones
              with the same cosmology share planes.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BUILD_SCHEMA = "eval21-build/1"
LOS_MODES = ("native", "nearest", "linear", "comoving")
FLOAT16_MAX = 65504.0
FLOAT16_REL_RMS = 2e-3          # keep float32 when the float16 round trip is worse than this
Z_EPS = 1e-9


class BuildError(Exception):
    """A configuration or convention error that must stop the build (never a per-cone skip)."""


# --------------------------------------------------------------------------- options and items
@dataclass
class BuildOptions:
    out: Path
    fields: list[str] | None = None          # eval21 names, optionally name=source
    targets: list[str] | None = None         # first -> manifest.json, others -> manifest_<t>.json
    los: str = "native"
    z_grid: str | None = None                # "z0,z1,n" or a file
    z_range: tuple[float, float] | None = None
    los_cell: float | None = None
    allow_repeat: bool = False
    crop: int | None = None
    crop_offset: tuple[int, int] | None = None
    dtype: str = "float32"
    compress: bool = False
    workers: int = 1
    limit: int | None = None
    ids: str | None = None
    only_split: str | None = None
    split: str | None = None
    split_file: Path | None = None
    split_from: Path | None = None
    split_fractions: tuple[float, float, float] | None = None
    seed: int = 0
    group: str | None = None                 # none | auto | groups.json; None = source default
    group_tol: float = 0.15
    min_planes: int = 8
    id_format: str = "cone{id}"
    nonfinite: str = "skip"                  # skip | error
    overwrite: bool = False
    dry_run: bool = False
    quiet: bool = False
    argv: list[str] | None = None
    source_options: dict = field(default_factory=dict)

    def __post_init__(self):
        self.out = Path(self.out)
        if self.los not in LOS_MODES:
            raise ValueError(f"--los must be one of {LOS_MODES}, got {self.los!r}")
        if self.dtype not in ("float32", "float16"):
            raise ValueError("--dtype must be float32 or float16")
        if self.nonfinite not in ("skip", "error"):
            raise ValueError("--nonfinite must be skip or error")
        if self.los in ("nearest", "linear") and not self.z_grid:
            raise ValueError(f"--los {self.los} needs --z-grid z0,z1,n (or a file)")
        if self.los in ("native", "comoving") and self.z_grid:
            raise ValueError(f"--z-grid is only used by --los nearest|linear, not {self.los}")
        n_split = sum(x is not None for x in (self.split, self.split_file, self.split_from,
                                              self.split_fractions))
        if n_split > 1:
            raise ValueError("use one of --split, --split-file, --split-from, --split-fractions")
        if self.crop is not None and self.crop < 1:
            raise ValueError("--crop must be a positive number of cells")

    def public(self) -> dict:
        out = {}
        for key, value in asdict(self).items():
            if key in ("argv", "quiet", "dry_run", "workers"):
                continue
            out[key] = str(value) if isinstance(value, Path) else value
        return out


@dataclass
class Item:
    """One lightcone of the source: numeric (or string) id and its file(s)."""

    cone_id: int | str
    path: Path
    extra: dict = field(default_factory=dict)


def file_record(path: Path) -> dict:
    stat = Path(path).stat()
    return {"path": str(Path(path).resolve()), "size": int(stat.st_size),
            "mtime": round(float(stat.st_mtime), 3)}


# --------------------------------------------------------------------------- source interface
class Source:
    """Base class of a raw-format reader. Subclasses fill in discovery, scan and read."""

    format = "?"
    scan_reads_data = False       # True when scan() must read the cube (cached on disk)
    default_group = "none"
    hubble_h: float | None = None
    field_map: dict = {}          # eval21 name -> explicit source key (set by build())

    def __init__(self, root, options: BuildOptions):
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"source folder not found: {self.root}")
        self.options = options
        self.so = dict(options.source_options or {})

    # -- to implement ------------------------------------------------------ #
    def discover(self) -> list[Item]:
        raise NotImplementedError

    def available_fields(self) -> dict[str, dict]:
        """eval21 name -> {"source": ..., "units": ..., ...} for the fields this source can write."""
        raise NotImplementedError

    def default_fields(self) -> list[str]:
        raise NotImplementedError

    def scan(self, item: Item) -> dict:
        raise NotImplementedError

    def read(self, item: Item, scan: dict, names, lo: int, hi: int, crop) -> dict:
        """Native planes [lo, hi) of ``names`` as float arrays (nx', ny', hi-lo), eval21 conventions."""
        raise NotImplementedError

    # -- optional ---------------------------------------------------------- #
    def scan_signature(self) -> dict:
        """Options that change what scan() returns (part of the scan-cache key)."""
        return {}

    def default_targets(self, fields: list[str]) -> list[str]:
        out = [name for name in ("xhi", "tb") if name in fields]
        return out or [fields[0]]

    def distance_table(self, scan: dict):
        """(z, d) table used to convert redshift <-> comoving distance for this cone."""
        return scan["z"], scan["d"]

    def native_resolution(self, scans: list[dict]) -> dict:
        """Native resolution facts written to the manifest (``cyl_native_*`` hints)."""
        return {}

    def provenance(self, scans: list[dict]) -> dict:
        return {}

    def notes(self) -> list[str]:
        return []

    # -- shared ------------------------------------------------------------ #
    def resolve_fields(self, requested):
        """[(eval21 name, source key or None)] from --fields (``name`` or ``name=key``)."""
        available = self.available_fields()
        names = requested or self.default_fields()
        out = []
        for spec in names:
            name, _, key = str(spec).partition("=")
            name = name.strip()
            if not key and name not in available:
                raise ValueError(f"{self.format}: unknown field {name!r}; available: "
                                 f"{', '.join(sorted(available))} (or name=source_key)")
            out.append((name, key.strip() or None))
        if len({n for n, _ in out}) != len(out):
            raise ValueError("--fields lists a field twice")
        return out


# --------------------------------------------------------------------------- small helpers
def number_of(value):
    """Numeric id of 16, '16', 'cone16', 'cone_000016', 'simu16' (else the string itself)."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    match = re.search(r"(\d+)$", str(value))
    return int(match.group(1)) if match else str(value)


def parse_ids(text: str | None):
    """'1-5,9,cone12' -> set of ids (numbers where possible)."""
    if not text:
        return None
    out = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if re.fullmatch(r"\d+-\d+", part):
            a, b = (int(x) for x in part.split("-"))
            out.update(range(a, b + 1))
        else:
            out.add(number_of(part))
    return out


def parse_z_grid(spec: str) -> tuple[np.ndarray, str]:
    """``z0,z1,n`` -> linspace; or a file: .npy, .npz (z_grid), .json (list, z_grid or manifest), text."""
    text = str(spec).strip()
    parts = text.split(",")
    if len(parts) == 3:
        try:
            z0, z1, n = float(parts[0]), float(parts[1]), int(parts[2])
            return np.linspace(z0, z1, n), f"linspace({z0}, {z1}, {n})"
        except ValueError:
            pass
    path = Path(text).expanduser()
    if not path.exists():
        raise ValueError(f"--z-grid {text!r} is neither z0,z1,n nor an existing file")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        z = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as data:
            key = "z_grid" if "z_grid" in data.files else "z"
            z = data[key]
    elif suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            payload = payload.get("z_grid", payload.get("z"))
        z = np.asarray(payload, dtype=np.float64)
    else:
        z = np.loadtxt(path)
    z = np.asarray(z, dtype=np.float64).ravel()
    if z.size < 2 or not np.all(np.diff(z) > 0):
        raise ValueError(f"--z-grid from {path} must be 1-D and strictly increasing")
    return z, str(path.resolve())


def crop_window(nx: int, ny: int, crop, offset=None):
    """(x0, x1, y0, y1) of a centred (or offset) crop of ``crop`` cells; full plane when None."""
    if crop is None:
        return (0, nx, 0, ny)
    crop = int(crop)
    if crop > min(nx, ny):
        raise ValueError(f"--crop {crop} exceeds the transverse size {nx}x{ny}")
    if offset is not None:
        x0, y0 = (int(v) for v in offset)
    else:
        x0, y0 = (nx - crop) // 2, (ny - crop) // 2
    if x0 < 0 or y0 < 0 or x0 + crop > nx or y0 + crop > ny:
        raise ValueError(f"crop {crop} at offset ({x0}, {y0}) leaves the {nx}x{ny} plane")
    return (x0, x0 + crop, y0, y0 + crop)


def builder_info() -> dict:
    from .. import __version__

    info = {"name": "eval21.build", "version": __version__, "schema": BUILD_SCHEMA,
            "commit": None, "dirty": None}
    repo = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=10)
        if commit.returncode == 0:
            info["commit"] = commit.stdout.strip()
            status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", "src"],
                                    capture_output=True, text=True, timeout=10)
            info["dirty"] = bool(status.stdout.strip()) if status.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def _hash(payload) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


# --------------------------------------------------------------------------- splits
def assign_splits(ids, options: BuildOptions) -> tuple[dict, dict]:
    """{cone_id: split or None} and a provenance record of how splits were assigned."""
    ids = list(ids)
    if options.split is not None:
        return {i: options.split for i in ids}, {"method": "constant", "split": options.split}
    if options.split_fractions is not None:
        fractions = [float(v) for v in options.split_fractions]
        if len(fractions) != 3 or min(fractions) < 0 or abs(sum(fractions) - 1) > 1e-6:
            raise ValueError("--split-fractions expects train,val,test summing to 1")
        edges = np.cumsum(fractions)
        out = {}
        for cone in ids:
            digest = hashlib.sha256(f"{options.seed}:{cone}".encode()).hexdigest()
            u = int(digest[:13], 16) / float(16 ** 13)
            out[cone] = ("train", "val", "test")[min(2, int(np.searchsorted(edges, u, side="right")))]
        return out, {"method": "hash", "fractions": fractions, "seed": options.seed,
                     "rule": "u = int(sha256(f'{seed}:{cone_id}')[:13], 16) / 16**13; "
                             "train if u < f_train, val if u < f_train + f_val, else test"}
    path = options.split_file or options.split_from
    if path is None:
        return {i: None for i in ids}, {"method": "none"}
    payload = json.loads(Path(path).read_text())
    block = payload.get("split", payload) if isinstance(payload, dict) else {}
    table = {}
    for name in ("train", "val", "test"):
        for key in (f"{name}_cone_ids", name, f"{name}_ids"):
            if key in block and isinstance(block[key], list):
                for value in block[key]:
                    cone = number_of(value)
                    if cone in table and table[cone] != name:
                        raise ValueError(f"{path}: cone {cone} is in both {table[cone]} and {name}")
                    table[cone] = name
                break
    if not table:
        raise ValueError(f"{path}: no train/val/test lists (or split.<name>_cone_ids) found")
    method = "run_metadata" if options.split_from else "file"
    return ({i: table.get(i) for i in ids},
            {"method": method, "path": str(Path(path).resolve()),
             "counts_in_file": {s: sum(v == s for v in table.values()) for s in ("train", "val", "test")}})


# --------------------------------------------------------------------------- LOS plans
def _interp_plan(d_target, d_native, lo, hi):
    position = np.interp(d_target, d_native, np.arange(len(d_native), dtype=np.float64))
    i0 = np.floor(position + 1e-9).astype(np.int64)
    i0 = np.clip(i0, lo, max(lo, hi - 1))
    t = position - i0
    t[np.abs(t) < 1e-9] = 0.0
    t[np.abs(t - 1) < 1e-9] = 1.0
    if hi == lo:
        t[:] = 0.0
    return i0, t


def los_plan(source: Source, scan: dict, window, options: BuildOptions, z_target=None) -> dict:
    """Output LOS of one cone inside ``window`` = (z_lo, z_hi).

    Returns ``{"kind": "index", "idx"}`` or ``{"kind": "interp", "i0", "t"}`` plus ``z`` (output
    redshifts), ``d`` (comoving distance) and, for ``nearest``, ``z_native``.
    """
    z = np.asarray(scan["z"], dtype=np.float64)
    d = np.asarray(scan["d"], dtype=np.float64)
    lo, hi = (int(v) for v in scan["valid"])
    z_lo, z_hi = window
    mode = options.los
    if mode == "native":
        idx = np.arange(lo, hi + 1)
        idx = idx[(z[idx] >= z_lo - Z_EPS) & (z[idx] <= z_hi + Z_EPS)]
        return {"kind": "index", "idx": idx, "z": z[idx], "d": d[idx]}
    zt_table, dt_table = (np.asarray(a, dtype=np.float64) for a in source.distance_table(scan))
    if mode in ("nearest", "linear"):
        zt = np.asarray(z_target, dtype=np.float64)
        zt = zt[(zt >= z_lo - Z_EPS) & (zt <= z_hi + Z_EPS)]
        if mode == "nearest":
            right = np.clip(np.searchsorted(z, zt), 1, len(z) - 1)
            left = right - 1
            idx = np.where(np.abs(z[right] - zt) < np.abs(zt - z[left]), right, left)
            idx = np.clip(idx, lo, hi)
            if not options.allow_repeat and len(np.unique(idx)) != len(idx):
                raise BuildError(
                    f"--los nearest: the z grid is finer than the native spacing (cone "
                    f"{scan['cone_id']} would repeat {len(idx) - len(np.unique(idx))} native planes); "
                    f"use a coarser grid, --los linear, or --allow-repeat")
            return {"kind": "index", "idx": idx, "z": zt, "z_native": z[idx], "d": d[idx]}
        dt = np.interp(zt, zt_table, dt_table)
    else:
        cell = float(options.los_cell or scan["cell_mpc"])
        d_lo = max(float(np.interp(z_lo, zt_table, dt_table)), d[lo])
        d_hi = min(float(np.interp(z_hi, zt_table, dt_table)), d[hi])
        k0 = int(np.ceil((d_lo - d[0]) / cell - 1e-6))
        k1 = int(np.floor((d_hi - d[0]) / cell + 1e-6))
        dt = d[0] + np.arange(k0, k1 + 1) * cell
        zt = np.interp(dt, dt_table, zt_table)
    i0, t = _interp_plan(dt, d, lo, hi)
    return {"kind": "interp", "i0": i0, "t": t, "z": zt, "d": dt}


def plan_range(plan: dict) -> tuple[int, int]:
    """Native plane range [lo, hi) a plan reads."""
    if plan["kind"] == "index":
        return int(plan["idx"].min()), int(plan["idx"].max()) + 1
    i1 = plan["i0"] + (plan["t"] > 0)
    return int(plan["i0"].min()), int(i1.max()) + 1


def apply_plan(array: np.ndarray, plan: dict, lo: int) -> np.ndarray:
    """(nx, ny, hi-lo) native planes starting at plane ``lo`` -> (nx, ny, n_out) float32."""
    if plan["kind"] == "index":
        return np.ascontiguousarray(array[..., plan["idx"] - lo], dtype=np.float32)
    i0 = plan["i0"] - lo
    t = plan["t"].astype(np.float32)
    i1 = np.minimum(i0 + 1, array.shape[-1] - 1)
    a = array[..., i0].astype(np.float32)
    b = array[..., i1].astype(np.float32)
    out = a * (np.float32(1) - t) + b * t
    for mask, src in ((t == 0, a), (t == 1, b)):
        if mask.any():
            out[..., mask] = src[..., mask]   # copies on native planes (no 0 * NaN from a neighbour)
    return np.ascontiguousarray(out, dtype=np.float32)


# --------------------------------------------------------------------------- groups
@dataclass
class Group:
    name: str
    members: list          # cone ids
    window: tuple          # (z_lo, z_hi)
    directory: Path
    note: str = ""
    plans: dict = field(default_factory=dict)
    shape: tuple | None = None
    shared_z: np.ndarray | None = None


def valid_z(scan) -> tuple[float, float]:
    z = scan["z"]
    lo, hi = scan["valid"]
    return float(z[lo]), float(z[hi])


def plan_groups(source: Source, scans: dict, options: BuildOptions, z_target=None,
                log=print) -> tuple[list[Group], dict]:
    """Geometry groups and the cones that could not be placed ({cone: reason})."""
    spec = options.group or source.default_group
    skipped = {}
    z_range = options.z_range or (-np.inf, np.inf)
    ids = sorted(scans, key=lambda c: (str(type(c)), c))
    spans = {c: valid_z(scans[c]) for c in ids}
    raw_groups = []
    if spec == "none":
        raw_groups.append(("all", ids, None, ""))
    elif spec == "auto":
        order = sorted(ids, key=lambda c: spans[c][0])
        current, start = [], None
        for cone in order:
            first = spans[cone][0]
            if current and first - start > options.group_tol:
                raw_groups.append((None, current, None, ""))
                current = []
            if not current:
                start = first
            current.append(cone)
        if current:
            raw_groups.append((None, current, None, ""))
    else:
        path = Path(spec)
        if not path.exists():
            raise ValueError(f"--group must be none, auto or a groups.json file, got {spec!r}")
        payload = json.loads(path.read_text())
        seen = {}
        for name, entry in payload.items():
            cones = [number_of(c) for c in (entry["cones"] if isinstance(entry, dict) else entry)]
            zr = tuple(entry["z"]) if isinstance(entry, dict) and entry.get("z") else None
            for cone in cones:
                if cone in seen:
                    raise ValueError(f"{path}: cone {cone} is in groups {seen[cone]} and {name}")
                seen[cone] = name
            members = [c for c in cones if c in scans]
            if members:
                raw_groups.append((name, members, zr,
                                   entry.get("note", "") if isinstance(entry, dict) else ""))
        for cone in ids:
            if cone not in seen:
                skipped[cone] = f"not listed in {path.name}"
    groups = []
    single = spec == "none"
    for name, members, zr, note in raw_groups:
        z_lo = max(spans[c][0] for c in members)
        z_hi = min(spans[c][1] for c in members)
        z_lo, z_hi = max(z_lo, z_range[0]), min(z_hi, z_range[1])
        if zr is not None:
            z_lo, z_hi = max(z_lo, float(zr[0])), min(z_hi, float(zr[1]))
        if name is None:
            name = f"G{len(groups) + 1}_z{z_lo:.2f}"
        directory = options.out if single else options.out / name
        group = Group(name=name, members=list(members), window=(z_lo, z_hi), directory=directory,
                      note=note)
        if z_hi <= z_lo:
            for cone in members:
                skipped[cone] = f"group {name}: empty window z {z_lo:.3f}-{z_hi:.3f}"
            continue
        for cone in list(members):
            plan = los_plan(source, scans[cone], group.window, options, z_target)
            if len(plan["z"]) < options.min_planes:
                skipped[cone] = (f"group {name}: {len(plan['z'])} planes in window "
                                 f"z {z_lo:.3f}-{z_hi:.3f} < --min-planes {options.min_planes}")
                group.members.remove(cone)
                continue
            group.plans[cone] = plan
        if not group.members:
            continue
        zs = [group.plans[c]["z"] for c in group.members]
        if all(len(z) == len(zs[0]) and np.allclose(z, zs[0], rtol=0, atol=1e-7) for z in zs):
            group.shared_z = zs[0]
        groups.append(group)
    return groups, skipped


# --------------------------------------------------------------------------- writing
def cast_field(array: np.ndarray, dtype: str) -> tuple[np.ndarray, str | None]:
    """Cast to the storage dtype; float16 only when it is safe (else float32 and the reason)."""
    if dtype == "float32":
        return array.astype(np.float32, copy=False), None
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak >= FLOAT16_MAX:
        return array.astype(np.float32, copy=False), f"kept float32: max |x| = {peak:.4g} overflows float16"
    half = array.astype(np.float16)
    ref = float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))
    if ref > 0:
        err = float(np.sqrt(np.mean(np.square(half.astype(np.float64) - array))))
        if err > FLOAT16_REL_RMS * ref:
            return (array.astype(np.float32, copy=False),
                    f"kept float32: float16 round trip rms error {err / ref:.3g} of rms (underflow?)")
    return half, None


def _stats(array) -> dict:
    a = np.asarray(array, dtype=np.float64)
    return {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()),
            "std": float(a.std())}


_CTX: dict = {}


def _write_task(task):
    try:
        return _write_one(**task)
    except BuildError:
        raise
    except Exception as exc:  # noqa: BLE001  (reported per cone, never silently dropped)
        return {"cone_id": task["cone"], "error": f"{type(exc).__name__}: {exc}"}


def _write_one(cone, group_name, path, signature, entry):
    source: Source = _CTX["source"]
    options: BuildOptions = _CTX["options"]
    item = _CTX["items"][cone]
    scan = _CTX["scans"][cone]
    plan = _CTX["plans"][(group_name, cone)]
    names = _CTX["fields"]
    path = Path(path)
    if path.exists() and not options.overwrite:
        try:
            with np.load(path) as data:
                stored = json.loads(str(data["_build"].item())) if "_build" in data.files else {}
            if stored.get("signature") == signature:
                return {**entry, **stored.get("entry", {}), "status": "existing"}
        except Exception:  # noqa: BLE001  (unreadable partial file: rebuild)
            pass
    lo, hi = plan_range(plan)
    crop = _CTX["crop"][cone]
    raw = source.read(item, scan, [n for n, _ in names], lo, hi, crop)
    arrays, dtypes, notes, stats = {}, {}, {}, {}
    for name, _ in names:
        out = apply_plan(raw.pop(name), plan, lo)
        bad = int(np.count_nonzero(~np.isfinite(out)))
        if bad:
            message = f"field {name}: {bad} non-finite value(s) in the output window"
            if options.nonfinite == "error":
                raise BuildError(f"cone {cone}: {message}")
            return {"cone_id": cone, "error": message, "skip": True}
        stats[name] = _stats(out)
        stored, why = cast_field(out, options.dtype)
        arrays[name] = stored
        dtypes[name] = str(stored.dtype)
        if why:
            notes[name] = why
    check = getattr(source, "check_output", None)
    if callable(check):
        check(cone, stats)
    entry = dict(entry)
    entry["dtypes"] = dtypes
    entry["stats"] = stats
    if notes:
        entry["dtype_notes"] = notes
    if "z_native" in plan:
        entry["los_max_abs_dz"] = float(np.max(np.abs(plan["z_native"] - plan["z"])))
    meta = {"signature": signature, "entry": {k: entry[k] for k in
                                              ("dtypes", "stats", "dtype_notes", "los_max_abs_dz")
                                              if k in entry}}
    payload = dict(arrays)
    payload["z_grid"] = np.asarray(plan["z"], dtype=np.float64)
    if "z_native" in plan:
        payload["z_native"] = np.asarray(plan["z_native"], dtype=np.float64)
    payload["los_distance_mpc"] = np.asarray(plan["d"], dtype=np.float64)
    if scan.get("omega_m") is not None:
        payload["omega_m"] = np.float64(scan["omega_m"])
    payload["box_mpc"] = np.float64(entry["box_mpc"])
    payload["params"] = np.array(json.dumps(_jsonable(scan.get("params") or {})))
    payload["cone_id"] = np.array(cone)
    payload["_build"] = np.array(json.dumps(_jsonable(meta)))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    (np.savez_compressed if options.compress else np.savez)(tmp, **payload)
    os.replace(tmp, path)
    return {**entry, "status": "written"}


# --------------------------------------------------------------------------- scans
def _scan_task(args):
    cone, cache = args
    source: Source = _CTX["source"]
    item = _CTX["items"][cone]
    try:
        record = source.scan(item)
    except Exception as exc:  # noqa: BLE001
        return cone, {"error": f"{type(exc).__name__}: {exc}"}
    record = _jsonable(record)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(f".{cache.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(record))
        os.replace(tmp, cache)
    return cone, record


def _scan_key(source: Source, item: Item) -> str:
    files = [file_record(item.path)] + [file_record(p) for p in item.extra.get("files", [])]
    return _hash({"format": source.format, "files": files, "sig": source.scan_signature(),
                  "fields": getattr(source, "field_map", None), "schema": BUILD_SCHEMA})


def _pool(workers):
    return mp.get_context("fork").Pool(workers) if workers > 1 else None


def _map(fn, tasks, workers, progress=None):
    pool = _pool(workers)
    try:
        iterator = pool.imap_unordered(fn, tasks) if pool else map(fn, tasks)
        for k, result in enumerate(iterator):
            if progress:
                progress(k + 1, len(tasks))
            yield result
    finally:
        if pool:
            pool.close()
            pool.join()


# --------------------------------------------------------------------------- driver
def build(source: Source, options: BuildOptions, log=print) -> dict:
    """Run (or with ``options.dry_run`` only plan) a build; returns a summary dict."""
    say = (lambda *a: None) if options.quiet else log
    t_start = time.time()
    items = source.discover()
    if not items:
        raise FileNotFoundError(f"{source.format}: no lightcone files found in {source.root}")
    by_id = {}
    for item in items:
        if item.cone_id in by_id:
            raise ValueError(f"duplicate id {item.cone_id}: {by_id[item.cone_id].path} and {item.path}")
        by_id[item.cone_id] = item
    wanted = parse_ids(options.ids)
    ids = sorted(by_id, key=lambda c: (str(type(c)), c))
    n_found = len(ids)
    if wanted is not None:
        missing = sorted(wanted - set(ids), key=str)
        if missing:
            say(f"warning: {len(missing)} requested id(s) not found: {missing[:10]}")
        ids = [c for c in ids if c in wanted]
    splits, split_info = assign_splits(ids, options)
    if options.only_split:
        ids = [c for c in ids if splits.get(c) == options.only_split]
    if options.limit is not None:
        ids = ids[:options.limit]
    if not ids:
        raise ValueError("no lightcones selected")
    fields = source.resolve_fields(options.fields)
    source.field_map = dict(fields)
    field_names = [n for n, _ in fields]
    targets = list(options.targets or source.default_targets(field_names))
    for target in targets:
        if target not in field_names:
            raise ValueError(f"target {target!r} is not among the written fields {field_names}")
    z_target, z_grid_source = (parse_z_grid(options.z_grid) if options.z_grid else (None, None))
    say(f"{source.format}: {n_found} files in {source.root}; {len(ids)} selected")

    # -- scan ------------------------------------------------------------- #
    cache_dir = options.out / ".eval21_build" / "scans"
    _CTX.clear()
    _CTX.update(source=source, options=options, items=by_id)
    scans, failed, tasks = {}, {}, []
    for cone in ids:
        cache = cache_dir / f"{cone}_{_scan_key(source, by_id[cone])}.json" if source.scan_reads_data else None
        if cache is not None and cache.exists():
            scans[cone] = json.loads(cache.read_text())
        else:
            tasks.append((cone, cache))
    if tasks:
        if source.scan_reads_data:
            size = sum(by_id[c].path.stat().st_size for c, _ in tasks)
            say(f"scanning {len(tasks)} file(s) ({size / 1e9:.1f} GB) for valid planes "
                f"(cached in {cache_dir})")
        tick = max(1, len(tasks) // 10)
        for cone, record in _map(_scan_task, tasks, options.workers,
                                 progress=lambda k, n: say(f"  scanned {k}/{n}") if k % tick == 0 else None):
            if "error" in record:
                failed[cone] = record["error"]
            else:
                scans[cone] = record
    for cone, record in scans.items():
        record["z"] = np.asarray(record["z"], dtype=np.float64)
        record["d"] = np.asarray(record["d"], dtype=np.float64)
        record["cone_id"] = cone
    for cone in ids:
        if cone in scans and record_bad_fields(scans[cone], source, fields):
            failed[cone] = record_bad_fields(scans[cone], source, fields)
            scans.pop(cone)
    if not scans:
        raise ValueError(f"every selected file failed to scan: {dict(list(failed.items())[:3])}")

    # -- plan ------------------------------------------------------------- #
    groups, skipped = plan_groups(source, scans, options, z_target, log=say)
    skipped.update(failed)
    crops = {}
    for cone, scan in scans.items():
        nx, ny = scan["shape"][:2]
        crops[cone] = crop_window(nx, ny, options.crop, options.crop_offset)
    for group in groups:
        shapes = set()
        for cone in group.members:
            x0, x1, y0, y1 = crops[cone]
            shapes.add((x1 - x0, y1 - y0, len(group.plans[cone]["z"])))
        group.shape = shapes.pop() if len(shapes) == 1 else None
    plan_summary = summarize_plan(source, options, groups, scans, fields, targets, splits,
                                  skipped, crops, z_grid_source)
    if options.dry_run:
        print_plan(plan_summary, log)
        return {"dry_run": True, **plan_summary}

    # -- write ------------------------------------------------------------ #
    options.out.mkdir(parents=True, exist_ok=True)
    _CTX.update(scans=scans, fields=fields, crop=crops,
                plans={(g.name, c): g.plans[c] for g in groups for c in g.members})
    write_tasks = []
    for group in groups:
        for cone in group.members:
            scan = scans[cone]
            sample_id = options.id_format.format(id=cone)
            x0, x1, y0, y1 = crops[cone]
            box = (x1 - x0) * float(scan["cell_mpc"])
            plan = group.plans[cone]
            signature = _hash({
                "files": scan["files"], "fields": fields, "crop": crops[cone], "dtype": options.dtype,
                "compress": options.compress, "z": np.round(plan["z"], 10).tolist(),
                "kind": plan["kind"], "source": source.scan_signature(), "schema": BUILD_SCHEMA,
                "read": getattr(source, "read_signature", lambda: {})()})
            entry = {"id": sample_id, "cone_id": cone, "path": f"{sample_id}.npz",
                     "split": splits.get(cone), "omega_m": scan.get("omega_m"),
                     "params": scan.get("params") or {}, "box_mpc": box,
                     "n_planes": int(len(plan["z"])),
                     "z_range": [float(plan["z"][0]), float(plan["z"][-1])],
                     "source": {"files": scan["files"], **(scan.get("info") or {})}}
            if group.shared_z is None:
                entry["z_grid"] = np.asarray(plan["z"]).tolist()
            write_tasks.append({"cone": cone, "group_name": group.name,
                                "path": str(group.directory / entry["path"]),
                                "signature": signature, "entry": entry})
    say(f"writing {len(write_tasks)} cone(s) in {len(groups)} group(s) with {options.workers} worker(s)")
    results = {}
    tick = max(1, len(write_tasks) // 20)
    t0 = time.time()

    def progress(k, n):
        if k % tick == 0 or k == n:
            say(f"  {k}/{n} cones, {time.time() - t0:.0f} s")

    for result in _map(_write_task, write_tasks, options.workers, progress=progress):
        results[result["cone_id"]] = result       # a cone belongs to exactly one group
    by_cone = results
    counts = {"written": 0, "existing": 0, "failed": 0}
    manifests = []
    builder = builder_info()
    for group in groups:
        entries = []
        for cone in group.members:
            result = by_cone[cone]
            if "error" in result:
                skipped[cone] = result["error"]
                counts["failed"] += 1
                continue
            counts[result.pop("status")] += 1
            entries.append(result)
        if not entries:
            continue
        manifests.extend(write_manifests(source, options, group, entries, scans, fields, targets,
                                         split_info, skipped, builder, z_grid_source))
    index = {"schema": BUILD_SCHEMA, "format": source.format, "source_folder": str(source.root),
             "groups": {g.name: {"directory": str(g.directory.relative_to(options.out))
                                 if g.directory != options.out else ".",
                                 "cones": [c for c in g.members if "error" not in by_cone[c]],
                                 "z": list(g.window), "note": g.note} for g in groups},
             "manifests": [str(Path(m).relative_to(options.out)) for m in manifests],
             "skipped": {str(k): v for k, v in skipped.items()},
             "builder": builder, "options": options.public(), "argv": options.argv,
             "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "seconds": round(time.time() - t_start, 1), "counts": counts}
    (options.out / "build.json").write_text(json.dumps(_jsonable(index), indent=1) + "\n")
    say(f"done in {time.time() - t_start:.0f} s: {counts['written']} written, {counts['existing']} "
        f"up to date, {counts['failed']} failed, {len(skipped)} skipped")
    for m in manifests:
        say(f"  {m}")
    if skipped:
        for cone, why in list(skipped.items())[:10]:
            say(f"  skipped {cone}: {why}")
    hints = plan_summary.get("eval21_settings") or {}
    if hints:
        say("eval21 run hints: " + " ".join(f"--set {k}={v}" for k, v in hints.items()))
    return {"manifests": [str(m) for m in manifests], "counts": counts,
            "skipped": {str(k): v for k, v in skipped.items()}, "groups": len(groups)}


def record_bad_fields(scan: dict, source: Source, fields) -> str | None:
    """Reason a scanned file cannot provide the requested fields (``scan['fields']``), or None."""
    available = scan.get("fields")
    if available is None:
        return None
    missing = [name for name, _ in fields if name not in available]
    return f"missing field(s) {missing}" if missing else None


def eval21_hints(source: Source, scans, options: BuildOptions) -> dict:
    hints = {}
    if source.hubble_h is not None:
        hints["hubble_h"] = source.hubble_h
    native = source.native_resolution(list(scans.values()))
    if options.los != "native" or options.crop:
        for key in ("cyl_native_cell_mpc", "cyl_native_los_dz", "cyl_native_los_mpc"):
            if native.get(key) is not None:
                hints[key] = native[key]
    return hints


def summarize_plan(source, options, groups, scans, fields, targets, splits, skipped, crops,
                   z_grid_source) -> dict:
    itemsize = 2 if options.dtype == "float16" else 4
    info = []
    total = 0
    for group in groups:
        voxels = 0
        for cone in group.members:
            x0, x1, y0, y1 = crops[cone]
            voxels += (x1 - x0) * (y1 - y0) * len(group.plans[cone]["z"])
        size = voxels * len(fields) * itemsize
        total += size
        z = group.shared_z
        info.append({
            "name": group.name, "directory": str(group.directory), "n_cones": len(group.members),
            "window_z": [round(v, 5) for v in group.window], "shape": group.shape,
            "shared_z_grid": z is not None,
            "z": None if z is None else [float(z[0]), float(z[-1])],
            "n_planes": None if z is None else int(len(z)),
            "cones": group.members if len(group.members) <= 40 else group.members[:40] + ["..."],
            "bytes": int(size), "note": group.note})
    counts = {}
    for group in groups:
        for cone in group.members:
            key = splits.get(cone) or "undeclared"
            counts[key] = counts.get(key, 0) + 1
    available = source.available_fields()
    return {
        "format": source.format, "source": str(source.root), "out": str(options.out),
        "fields": {name: (key or available.get(name, {}).get("source")) for name, key in fields},
        "targets": targets, "los": options.los, "z_grid": z_grid_source,
        "los_cell_mpc": options.los_cell, "dtype": options.dtype, "crop": options.crop,
        "n_cones": sum(len(g.members) for g in groups), "groups": info,
        "splits": counts, "skipped": {str(k): v for k, v in skipped.items()},
        "estimated_bytes": int(total), "native": source.native_resolution(list(scans.values())),
        "eval21_settings": eval21_hints(source, scans, options),
    }


def print_plan(plan: dict, log=print) -> None:
    log(f"format   {plan['format']}  ({plan['source']})")
    log(f"out      {plan['out']}")
    log("fields   " + ", ".join(f"{k} <- {v}" for k, v in plan["fields"].items()))
    log(f"targets  {', '.join(plan['targets'])} (manifest.json = {plan['targets'][0]})")
    log(f"los      {plan['los']}" + (f" on {plan['z_grid']}" if plan["z_grid"] else "")
        + (f", cell {plan['los_cell_mpc']} Mpc" if plan["los_cell_mpc"] else ""))
    log(f"dtype    {plan['dtype']}" + (f", crop {plan['crop']}" if plan["crop"] else ""))
    log(f"splits   {plan['splits']}")
    for g in plan["groups"]:
        z = (f"z {g['z'][0]:.4f}-{g['z'][1]:.4f}, {g['n_planes']} planes" if g["shared_z_grid"]
             else f"window z {g['window_z'][0]:.4f}-{g['window_z'][1]:.4f}, per-cone z grids")
        log(f"group    {g['name']}: {g['n_cones']} cones, shape {g['shape']}, {z}, "
            f"{g['bytes'] / 1e9:.2f} GB -> {g['directory']}")
        log(f"           cones {g['cones']}")
    if plan["skipped"]:
        log(f"skipped  {len(plan['skipped'])}: " + "; ".join(
            f"{k}: {v}" for k, v in list(plan["skipped"].items())[:8]))
    log(f"total    {plan['n_cones']} cones, {plan['estimated_bytes'] / 1e9:.2f} GB uncompressed")
    if plan["native"]:
        log(f"native   {plan['native']}")
    if plan["eval21_settings"]:
        log("hints    " + " ".join(f"--set {k}={v}" for k, v in plan["eval21_settings"].items()))


def write_manifests(source, options, group, entries, scans, fields, targets, split_info, skipped,
                    builder, z_grid_source) -> list[Path]:
    entries = sorted(entries, key=lambda e: (str(type(e["cone_id"])), e["cone_id"]))
    field_names = [n for n, _ in fields]
    available = source.available_fields()
    boxes = {round(e["box_mpc"], 9) for e in entries}
    omegas = {e.get("omega_m") for e in entries}
    shared_box = boxes.pop() if len(boxes) == 1 else None
    first_scan = scans[entries[0]["cone_id"]]
    field_info = {}
    for name, key in fields:
        meta = dict(available.get(name, {}))
        if key:
            meta["source"] = key
        field_info[name] = meta
    los = {"mode": options.los, "window_z": list(group.window), "z_grid_source": z_grid_source,
           "cell_mpc": (options.los_cell or float(first_scan["cell_mpc"])) if options.los == "comoving" else None,
           "interpolation": {"native": "none (native planes)", "nearest": "nearest native plane in z",
                             "linear": "linear in comoving distance",
                             "comoving": "linear in comoving distance onto a uniform grid"}[options.los]}
    if group.shared_z is not None:
        los["n_planes"] = int(len(group.shared_z))
        los["z"] = [float(group.shared_z[0]), float(group.shared_z[-1])]
        d = group.plans[entries[0]["cone_id"]]["d"]
        if len(d) > 1:
            spacing = np.diff(d)
            los["spacing_mpc"] = [float(spacing.min()), float(spacing.max())]
    native = source.native_resolution([scans[e["cone_id"]] for e in entries])
    x0, x1, y0, y1 = crop_window(*first_scan["shape"][:2], options.crop, options.crop_offset)
    geometry = {"shape": list(group.shape) if group.shape else None,
                "cell_mpc": float(first_scan["cell_mpc"]),
                "native_shape": list(first_scan["shape"]),
                "crop": None if options.crop is None else {"cells": options.crop, "x": [x0, x1], "y": [y0, y1]},
                "los": los, "uniform_geometry": group.shared_z is not None and group.shape is not None}
    samples = []
    for e in entries:
        sample = {k: e[k] for k in ("id", "path", "split", "omega_m", "params", "cone_id") if k in e}
        for k in ("z_grid", "n_planes", "z_range", "box_mpc", "dtypes", "dtype_notes", "los_max_abs_dz",
                  "stats", "source"):
            if k in e and not (k == "box_mpc" and shared_box is not None):
                sample[k] = e[k]
        samples.append(sample)
    base = {
        "schema": BUILD_SCHEMA,
        "box_mpc": shared_box,
        "target": targets[0],
        "inputs": [n for n in field_names if n not in targets],
        "group": group.name,
        "fields": field_info,
        "geometry": geometry,
        "native": native,
        "eval21_settings": eval21_hints(source, {e["cone_id"]: scans[e["cone_id"]] for e in entries}, options),
        "splits": split_info,
        "provenance": {"source_folder": str(source.root), "format": source.format,
                       "builder": builder, "options": options.public(), "argv": options.argv,
                       "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "host": socket.gethostname(), "python": sys.version.split()[0],
                       **source.provenance([scans[e["cone_id"]] for e in entries])},
        "notes": source.notes() + ([group.note] if group.note else []),
        "skipped": {str(k): v for k, v in skipped.items()},
        "samples": samples,
    }
    if group.shared_z is not None:
        base["z_grid"] = np.asarray(group.shared_z).tolist()
    if len(omegas) == 1 and None not in omegas:
        base["omega_m"] = omegas.pop()
    if shared_box is None:
        base.pop("box_mpc")
    group.directory.mkdir(parents=True, exist_ok=True)
    written = []
    for k, target in enumerate(targets):
        manifest = dict(base)
        manifest["target"] = target
        # other targets are not inputs, except x_HI in a T_b pass (stage chunks need it, as in rhizome)
        manifest["inputs"] = [n for n in field_names if n not in targets] + (
            ["xhi"] if "xhi" in field_names and target != "xhi" else [])
        name = "manifest.json" if k == 0 else f"manifest_{target}.json"
        path = group.directory / name
        tmp = path.with_name(f".{name}.tmp")
        tmp.write_text(json.dumps(_jsonable(manifest), indent=1) + "\n")
        os.replace(tmp, path)
        written.append(path)
    return written
