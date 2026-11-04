"""``eval21 build``: a folder of simulations -> one eval21 cube cache (HDF5).

Layout (``attrs["format"] == "eval21-cubes/1"``; read by :class:`eval21.data.CubeCacheDataset`,
which ``eval21 run --data cache.h5`` picks automatically)::

    fields/<name>   (N, Nx, Ny, Nz)  float32 | float16   one chunk per sample
    z_grid          (Nz,)            float64             shared line of sight
    valid           (N, Nz)          bool                planes usable for evaluation
    id              (N,)             str
    split           (N,)             str ("" = undeclared)
    omega_m         (N,)             float64
    params/values   (N, P) float64,  params/names (P,) str
    attrs: format, source, box_mpc, cell_mpc, fields, targets, units, created, command,
           eval21_version, skipped (JSON list of [id, error])

Fields named in ``targets`` become ``Sample.targets``; every other field is an input. Any field
can still be chosen as the target of a run (``--target``). Large builds shard over a Slurm
array (``--shard K --num-shards N``) and are joined with ``--merge``.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from .grid import LosInterpolator, parse_z_grid, resample_transverse
from .sources import RawSample, SourceItem, open_source

FORMAT = "eval21-cubes/1"
DEFAULT_TARGETS = ("xhi", "tb")


# ------------------------------------------------------------------------------------- splits
def assign_splits(ids: list[str], splits: str | None = None, split_file: str | None = None,
                  seed: int = 0) -> dict[str, str]:
    """{id: split}. ``splits``: ``test`` (all), or ``train=0.8,val=0.1,test=0.1`` (seeded
    shuffle of the sorted ids). ``split_file``: JSON ``{split: [ids]}``, ``{id: split}`` or a
    training ``run_metadata.json`` with ``<split>_cone_ids``."""
    out = {i: "" for i in ids}
    if split_file:
        table = json.loads(Path(split_file).read_text())
        mapping = {}
        for key, value in table.items():
            name = key[:-len("_cone_ids")] if key.endswith("_cone_ids") else key
            if isinstance(value, list):
                for i in value:
                    mapping[str(i)] = name
            elif isinstance(value, str):
                mapping[str(key)] = value
        for i in ids:
            out[i] = mapping.get(i, mapping.get(i.lstrip("cone"), ""))
        return out
    if not splits:
        return out
    if "=" not in splits:
        return {i: splits for i in ids}
    parts = [(name, float(share)) for name, _, share in (p.partition("=") for p in splits.split(","))]
    total = sum(share for _, share in parts)
    order = sorted(ids)
    random.Random(seed).shuffle(order)
    edges = np.cumsum([share / total for _, share in parts]) * len(order)
    start = 0
    for (name, _), edge in zip(parts, edges):
        stop = int(round(edge))
        for i in order[start:stop]:
            out[i] = name
        start = stop
    return out


# ------------------------------------------------------------------------------------- writer
class CacheWriter:
    def __init__(self, path, fields, shape, z_grid, attrs, dtype="float32", compress=None):
        import h5py

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.h5 = h5py.File(self.path, "w")
        self.fields = list(fields)
        nx, ny, nz = shape
        self.shape = shape
        kwargs = {} if not compress else ({"compression": "lzf"} if compress == "lzf"
                                          else {"compression": "gzip", "compression_opts": 4})
        for name in self.fields:
            self.h5.create_dataset(f"fields/{name}", shape=(0, nx, ny, nz), maxshape=(None, nx, ny, nz),
                                   dtype=dtype, chunks=(1, nx, ny, nz), **kwargs)
        self.h5.create_dataset("z_grid", data=np.asarray(z_grid, np.float64))
        self.h5.create_dataset("valid", shape=(0, nz), maxshape=(None, nz), dtype=bool, chunks=(64, nz))
        str_dtype = h5py.string_dtype()
        for key in ("id", "split"):
            self.h5.create_dataset(key, shape=(0,), maxshape=(None,), dtype=str_dtype, chunks=(1024,))
        self.h5.create_dataset("omega_m", shape=(0,), maxshape=(None,), dtype=np.float64, chunks=(1024,))
        self.params: list[dict] = []
        for key, value in attrs.items():
            self.h5.attrs[key] = value if not isinstance(value, (dict, list, tuple)) else json.dumps(value)
        self.n = 0

    def append(self, sid, split, fields, valid, omega_m, params):
        i = self.n
        for name in self.fields:
            ds = self.h5[f"fields/{name}"]
            ds.resize(i + 1, axis=0)
            ds[i] = fields[name]
        for key, value in (("valid", valid), ("id", str(sid)), ("split", split or ""),
                           ("omega_m", np.nan if omega_m is None else float(omega_m))):
            ds = self.h5[key]
            ds.resize(i + 1, axis=0)
            ds[i] = value
        self.params.append(dict(params or {}))
        self.n += 1

    def close(self, skipped=()):
        import h5py

        names = []
        for p in self.params:
            names += [k for k in p if k not in names]
        values = np.array([[p.get(k, np.nan) for k in names] for p in self.params], np.float64).reshape(self.n, len(names))
        self.h5.create_dataset("params/values", data=values)
        self.h5.create_dataset("params/names", data=np.array(names, dtype=object), dtype=h5py.string_dtype())
        self.h5.attrs["skipped"] = json.dumps(list(skipped))
        self.h5.attrs["n_samples"] = self.n
        self.h5.close()


# -------------------------------------------------------------------------------------- build
def _process(job):
    source, item, fields, target_z, transverse, cell_native = job
    try:
        raw: RawSample = source.read(item, fields)
    except Exception as exc:  # noqa: BLE001 - recorded and reported, the build continues
        return item, None, f"{type(exc).__name__}: {exc}"
    z = np.asarray(raw.z, np.float64)
    if target_z is None:
        out = {k: np.asarray(v, np.float32) for k, v in raw.fields.items()}
        valid = np.ones(z.size, bool) if raw.valid is None else np.asarray(raw.valid, bool)
        if z[0] > z[-1]:
            out = {k: v[..., ::-1] for k, v in out.items()}
            valid = valid[::-1]
            z = z[::-1]
    else:
        interp = LosInterpolator(z, target_z)
        out = {k: interp(v) for k, v in raw.fields.items()}
        valid = interp.valid(raw.valid)
        z = np.asarray(target_z)
    if transverse:
        out = {k: resample_transverse(v, transverse) for k, v in out.items()}
    return item, (out, valid, z, raw), None


def build(folder, out, fmt="auto", fields=None, targets=None, z_grid=None, transverse=None,
          splits=None, split_file=None, seed=0, limit=None, ids=None, shard=0, num_shards=1,
          workers=1, dtype="float32", compress=None, source_options=None, quiet=False,
          command="") -> Path:
    """Build (one shard of) a cache; returns the written path."""
    from .. import __version__

    source = open_source(fmt, folder, **(source_options or {}))
    items = source.items()
    if ids:
        wanted = {str(i) for i in ids}
        items = [it for it in items if it.id in wanted]
    if limit:
        items = items[:int(limit)]
    if not items:
        raise SystemExit("eval21 build: no samples selected")
    split_of = assign_splits([it.id for it in items], splits, split_file, seed)
    mine = items[shard::num_shards] if num_shards > 1 else items
    fields = list(fields or source.default_fields)

    # geometry (and, for sources without defaults, the field list) from the first sample
    first, probe, error = _process((source, mine[0], fields, None, None, None))
    if probe is None:
        raise SystemExit(f"eval21 build: cannot read {first.path}: {error}")
    _, _, z_native, raw0 = probe
    if not fields:
        fields = list(raw0.fields)
        if not fields:
            raise SystemExit(f"eval21 build: no 3-D fields in {first.path}; pass --fields")
    targets = [f for f in (targets or DEFAULT_TARGETS) if f in fields]
    nx = next(iter(raw0.fields.values())).shape[0]
    box = raw0.box_mpc
    cell = None if not box else box / (transverse or nx)
    target_z = parse_z_grid(z_grid, cell_mpc=cell, omega_m=raw0.omega_m or 0.31)
    z_out = z_native if target_z is None else target_z
    n_out = transverse or nx

    out = Path(out)
    path = out if num_shards == 1 else out.with_name(f"{out.stem}.shard{shard:03d}of{num_shards:03d}{out.suffix}")
    attrs = {"format": FORMAT, "source": json.dumps(source.describe()), "folder": str(Path(folder).resolve()),
             "box_mpc": float(box) if box else np.nan, "cell_mpc": float(cell) if cell else np.nan,
             "fields": list(fields), "targets": list(targets),
             "units": {k: v for k, v in getattr(source, "units", {}).items() if k in fields},
             "z_grid_spec": z_grid or "native", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "command": command, "eval21_version": __version__, "shard": f"{shard}/{num_shards}"}
    writer = CacheWriter(path, fields, (n_out, n_out, len(z_out)), z_out, attrs, dtype=dtype, compress=compress)
    if not quiet:
        print(f"[build] {len(mine)}/{len(items)} samples ({source.name}) -> {path}  "
              f"shape ({n_out}, {n_out}, {len(z_out)}) fields {fields} targets {targets}", flush=True)
    jobs = [(source, it, fields, target_z, transverse, None) for it in mine]
    skipped = []
    started = time.time()
    if workers > 1:
        from multiprocessing import get_context

        pool = get_context("fork").Pool(workers)
        results = pool.imap(_process, jobs, chunksize=1)
    else:
        pool, results = None, map(_process, jobs)
    try:
        for count, (item, result, error) in enumerate(results, 1):
            if result is None:
                skipped.append([item.id, error])
                print(f"  [skip] {item.id}: {error}", file=sys.stderr, flush=True)
                continue
            data, valid, z, raw = result
            if len(z) != len(z_out) or not np.allclose(z, z_out, rtol=0, atol=1e-6):
                skipped.append([item.id, "line of sight differs from the first sample (use --z-grid)"])
                print(f"  [skip] {item.id}: line of sight differs (use --z-grid)", file=sys.stderr, flush=True)
                continue
            if raw.box_mpc and box and abs(raw.box_mpc - box) > 1e-6 * box:
                skipped.append([item.id, f"box {raw.box_mpc} != {box}"])
                continue
            writer.append(item.id, split_of.get(item.id, ""), data, valid, raw.omega_m, raw.params)
            if not quiet and (count % 25 == 0 or count == len(jobs)):
                rate = (time.time() - started) / count
                print(f"  ... {count}/{len(jobs)} ({rate:.1f} s/sample)", flush=True)
    finally:
        if pool is not None:
            pool.close()
        writer.close(skipped)
    if not quiet:
        print(f"[build] wrote {writer.n} samples to {path}" + (f", skipped {len(skipped)}" if skipped else ""))
    return path


def merge(out, num_shards: int, quiet: bool = False, remove: bool = False) -> Path:
    """Join ``out.shardKKKofNNN.h5`` files into ``out`` (rows in shard-interleaved source order)."""
    import h5py

    out = Path(out)
    shards = [out.with_name(f"{out.stem}.shard{k:03d}of{num_shards:03d}{out.suffix}") for k in range(num_shards)]
    missing = [str(s) for s in shards if not s.exists()]
    if missing:
        raise SystemExit(f"eval21 build --merge: missing shards {missing[:3]}")
    handles = [h5py.File(s, "r") for s in shards]
    try:
        first = handles[0]
        fields = json.loads(first.attrs["fields"])
        nz = first["z_grid"].shape[0]
        for h in handles[1:]:
            if not np.allclose(h["z_grid"][:], first["z_grid"][:]):
                raise SystemExit("eval21 build --merge: shards have different z grids")
        # interleave back to source order: shard k holds items k, k+N, k+2N, ...
        order = []
        counts = [h["id"].shape[0] for h in handles]
        for row in range(max(counts)):
            for k, h in enumerate(handles):
                if row < counts[k]:
                    order.append((k, row))
        nx, ny = first[f"fields/{fields[0]}"].shape[1:3]
        attrs = {k: v for k, v in first.attrs.items() if k not in {"skipped", "n_samples", "shard"}}
        attrs["fields"] = json.loads(attrs["fields"])
        attrs["targets"] = json.loads(attrs["targets"])
        attrs["units"] = json.loads(attrs["units"])
        attrs["shard"] = f"merged from {num_shards}"
        dtype = first[f"fields/{fields[0]}"].dtype
        comp = first[f"fields/{fields[0]}"].compression
        writer = CacheWriter(out, fields, (nx, ny, nz), first["z_grid"][:], attrs, dtype=dtype,
                             compress=None if comp is None else ("lzf" if comp == "lzf" else "gzip"))
        skipped = []
        for h in handles:
            skipped += json.loads(h.attrs.get("skipped", "[]"))
        for k, row in order:
            h = handles[k]
            names = [n.decode() if isinstance(n, bytes) else str(n) for n in h["params/names"][:]]
            values = h["params/values"][row]
            sid = h["id"][row]
            split = h["split"][row]
            writer.append(sid.decode() if isinstance(sid, bytes) else sid,
                          split.decode() if isinstance(split, bytes) else split,
                          {f: h[f"fields/{f}"][row] for f in fields}, h["valid"][row],
                          float(h["omega_m"][row]), dict(zip(names, map(float, values))))
        writer.close(skipped)
    finally:
        for h in handles:
            h.close()
    if remove:
        for s in shards:
            s.unlink()
    if not quiet:
        print(f"[merge] {sum(counts)} samples from {num_shards} shards -> {out}")
    return out
