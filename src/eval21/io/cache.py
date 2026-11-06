"""Optional HDF5 inference cache (``eval21-cache/1``).

Cubes are not part of a run file. When a caller needs to re-plot maps, galleries or LOS planes,
the run writes them here instead::

    /                      attrs: schema, target, created, producer
    /index                 JSON {samples, models, fields, targets, draws}
    /ids                   (N,) sample ids
    /splits                (N,)
    /omega_m               (N,)
    /params/{names,values}
    /truth/<field>         (N, Nx, Ny, Nz) float16, gzip
    /models/<model>/<field>          (N, Nx, Ny, Nz) scored prediction
    /models/<model>/draws/<field>    (N, D, Nx, Ny, Nz) optional samples

One file may hold several models of one target. A cache is optional and regenerable; it is never
needed to read a run file.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

SCHEMA = "eval21-cache/1"
DEFAULT_MAX_SAMPLES = 8
COMPRESSION = {"compression": "gzip", "compression_opts": 4, "shuffle": True}


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)


def _json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _string_dtype():
    import h5py

    return h5py.string_dtype(encoding="utf-8")


def _append_strings(dataset, index: int, value: str) -> None:
    if dataset.shape[0] <= index:
        dataset.resize((index + 1,))
    dataset[index] = str(value)


def _append_array(dataset, index: int, value: np.ndarray) -> None:
    if dataset.shape[0] <= index:
        dataset.resize((index + 1,) + dataset.shape[1:])
    dataset[index] = value


class CacheWriter:
    """Append truth and predictions to one cache file, model group by model group."""

    def __init__(self, path, ids=None, n: int = DEFAULT_MAX_SAMPLES, dtype=np.float16,
                 overwrite: bool = False):
        import h5py

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dtype = np.dtype(dtype)
        self.limit = None if n is None else int(n)
        self.ids = None if ids is None else {str(v) for v in ids}
        self.sample_ids: list[str] = []
        self.models: list[str] = []
        self.fields: list[str] = []
        self.targets: list[str] = []
        self.draws: dict[str, int] = {}
        self._truth_saved: set[str] = set()

        exists = self.path.exists() and self.path.stat().st_size > 0
        if exists and not overwrite:
            self.h5 = h5py.File(self.path, "r+")
            schema = _text(self.h5.attrs.get("schema", ""))
            if schema != SCHEMA:
                self.h5.close()
                raise ValueError(f"{self.path} is not an {SCHEMA} cache (schema {schema!r})")
            if "ids" in self.h5:
                self.sample_ids = [_text(v) for v in self.h5["ids"][:]]
            self._load_index()
        elif exists:
            self.path.unlink()
            self.h5 = h5py.File(self.path, "w", libver="earliest")
            self._init_file()
        else:
            self.h5 = h5py.File(self.path, "w", libver="earliest")
            self._init_file()
        self._truth_saved = set(self.sample_ids)

    def _init_file(self) -> None:
        from datetime import datetime, timezone

        self.h5.attrs["schema"] = SCHEMA
        self.h5.attrs["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.h5.attrs["producer"] = "eval21"

    def _load_index(self) -> None:
        if "index" not in self.h5:
            self.models = list(self.h5.get("models", {}))
            return
        index = json.loads(_text(self.h5["index"][()]))
        self.models = list(index.get("models", []))
        self.fields = list(index.get("fields", []))
        self.targets = list(index.get("targets", []))
        self.draws = dict(index.get("draws", {}))

    # -- selection --------------------------------------------------------------
    def wants(self, sample) -> bool:
        sid = str(sample.id)
        if self.ids is not None:
            return sid in self.ids
        return self.limit is None or len(self.sample_ids) < self.limit or sid in self.sample_ids

    def _index_of(self, sid: str) -> int:
        try:
            return self.sample_ids.index(sid)
        except ValueError:
            self.sample_ids.append(sid)
            if "ids" not in self.h5:
                self._create_sample_datasets()
            i = len(self.sample_ids) - 1
            _append_strings(self.h5["ids"], i, sid)
            if "splits" in self.h5:
                self.h5["splits"].resize((i + 1,))
            if "omega_m" in self.h5:
                self.h5["omega_m"].resize((i + 1,))
            self.h5.flush()
            return i

    def _create_sample_datasets(self) -> None:
        self.h5.create_dataset("ids", shape=(0,), maxshape=(None,), dtype=_string_dtype())
        self.h5.create_dataset("splits", shape=(0,), maxshape=(None,), dtype=_string_dtype())
        self.h5.create_dataset("omega_m", shape=(0,), maxshape=(None,), dtype="f8")
        params = self.h5.create_group("params")
        params.create_dataset("names", shape=(0,), maxshape=(None,), dtype=_string_dtype())
        params.create_dataset("values", shape=(0, 0), maxshape=(None, None), dtype="f8")

    def _save_sample_meta(self, sample, index: int) -> None:
        _append_strings(self.h5["splits"], index, getattr(sample, "split", "") or "")
        omega = getattr(getattr(sample, "meta", None), "omega_m", None)
        _append_array(self.h5["omega_m"], index,
                      np.array(np.nan if omega is None else float(omega), dtype="f8"))
        params = getattr(getattr(sample, "meta", None), "params", None) or {}
        names = [_text(v) for v in self.h5["params/names"][:]]
        new = [k for k in params if k not in names]
        if new:
            names += new
            self.h5["params/names"].resize((len(names),))
            for j, name in enumerate(names):
                self.h5["params/names"][j] = name
            self.h5["params/values"].resize((self.h5["params/values"].shape[0] or len(self.sample_ids), len(names)))
        values = self.h5["params/values"]
        if values.shape[0] <= index:
            values.resize((index + 1, len(names)))
        if names:
            values[index, :] = np.nan
            for key, value in params.items():
                if value is not None and np.isfinite(float(value)):
                    values[index, names.index(key)] = float(value)

    def _field_dataset(self, group, name: str, shape: tuple):
        if name in group:
            return group[name]
        if len(shape) != 3:
            raise ValueError(f"cache fields must be 3-D, got {name} {shape}")
        return group.create_dataset(name, shape=(0,) + shape, maxshape=(None,) + shape,
                                    dtype=self.dtype, chunks=(1,) + shape[:-1] + (min(16, shape[-1]),),
                                    **COMPRESSION)

    def _draw_dataset(self, group, name: str, shape: tuple):
        if name in group:
            return group[name]
        if len(shape) != 3:
            raise ValueError(f"draw fields must be 3-D per draw, got {name} {shape}")
        return group.create_dataset(name, shape=(0, 0) + shape, maxshape=(None, None) + shape,
                                    dtype=self.dtype,
                                    chunks=(1, 1) + shape[:-1] + (min(16, shape[-1]),), **COMPRESSION)

    # -- writing ----------------------------------------------------------------
    def save_truth(self, sample) -> None:
        sid = str(sample.id)
        index = self._index_of(sid)
        if sid in self._truth_saved:
            return
        self._save_sample_meta(sample, index)
        meta = getattr(sample, "meta", None)
        if meta is not None:
            if "z_grid" not in self.h5 and getattr(meta, "z_grid", None) is not None:
                self.h5.create_dataset("z_grid", data=np.asarray(meta.z_grid, np.float64))
            if "box_mpc" not in self.h5 and getattr(meta, "box_mpc", None) is not None:
                self.h5.create_dataset("box_mpc", data=np.float64(meta.box_mpc))
        targets = getattr(sample, "targets", {}) or {}
        inputs = getattr(sample, "inputs", {}) or {}
        group = self.h5.require_group("truth")
        for field, value in list(targets.items()) + list(inputs.items()):
            array = np.asarray(value)
            if array.ndim != 3:
                continue
            dataset = self._field_dataset(group, str(field), array.shape)
            _append_array(dataset, index, array.astype(self.dtype, copy=False))
            if str(field) not in self.fields:
                self.fields.append(str(field))
            if str(field) in targets and str(field) not in self.targets:
                self.targets.append(str(field))
        self._truth_saved.add(sid)
        self.h5.flush()

    def save_prediction(self, model: str, sample, prediction: dict, draws: int = 1) -> None:
        sid = str(sample.id)
        index = self._index_of(sid)
        model = str(model)
        if model not in self.models:
            self.models.append(model)
        group = self.h5.require_group(f"models/{model}")
        draw_group = group.require_group("draws") if any(k.endswith("__draws") for k in prediction) else None
        for field, value in prediction.items():
            array = np.asarray(value)
            if field.endswith("__draws"):
                if draw_group is None:
                    continue
                name = field[: -len("__draws")]
                dataset = self._draw_dataset(draw_group, name, array.shape[1:])
                if dataset.shape[0] <= index:
                    dataset.resize((index + 1, dataset.shape[1]) + dataset.shape[2:])
                if dataset.shape[1] < array.shape[0]:
                    dataset.resize((dataset.shape[0], array.shape[0]) + dataset.shape[2:])
                dataset[index, : array.shape[0]] = array.astype(self.dtype, copy=False)
                self.draws[model] = max(self.draws.get(model, 1), int(array.shape[0]))
                continue
            if array.ndim != 3:
                continue
            dataset = self._field_dataset(group, str(field), array.shape)
            _append_array(dataset, index, array.astype(self.dtype, copy=False))
            if str(field) not in self.fields:
                self.fields.append(str(field))
        self.h5.flush()

    def flush(self) -> None:
        index = {
            "schema": SCHEMA, "target": _text(self.h5.attrs.get("target", "")),
            "samples": self.sample_ids, "models": self.models, "fields": self.fields,
            "targets": self.targets, "draws": self.draws,
        }
        text = _json_text(index)
        if "index" in self.h5:
            del self.h5["index"]
        self.h5.create_dataset("index", data=np.array(text, dtype=_string_dtype()))
        self.h5.flush()

    def close(self) -> None:
        self.flush()
        self.h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def dir(self) -> Path:
        return self.path.parent


class CacheFile:
    """Reader for one ``eval21-cache/1`` file."""

    def __init__(self, path):
        import h5py

        self.path = Path(path)
        self.h5 = h5py.File(self.path, "r")
        schema = _text(self.h5.attrs.get("schema", ""))
        if schema != SCHEMA:
            self.h5.close()
            raise ValueError(f"{self.path} is not an {SCHEMA} cache (schema {schema!r})")
        self.index = _load_index(self.h5)
        self.samples = list(self.index.get("samples", []))
        self.models = list(self.index.get("models", []))
        self.targets = list(self.index.get("targets", []))
        self.fields = list(self.index.get("fields", []))
        self.draws_by_model = dict(self.index.get("draws", {}))

    @property
    def dir(self) -> Path:
        return self.path.parent

    def draws(self, model: str) -> int:
        return int(self.draws_by_model.get(model, 1))

    def resolve(self, key):
        """Accept a sample id (int/str) or a 0-based index."""
        return resolve_sample(self.samples, key)

    def has(self, model, sid: str | None = None) -> bool:
        """``has(sid)`` or ``has(model, sid)`` (the legacy store's signature)."""
        if sid is None:
            return str(model) in self.samples
        return str(sid) in self.samples and str(model) in self.models

    def truth(self, sid: str) -> dict:
        if "truth" not in self.h5:
            return {}
        i = self.samples.index(str(sid))
        out = {"targets": {}, "inputs": {}}
        for field in self.h5["truth"]:
            value = self.h5[f"truth/{field}"][i]
            bucket = out["targets"] if field in self.targets else out["inputs"]
            bucket[field] = value
        out["z_grid"] = self.h5["z_grid"][()] if "z_grid" in self.h5 else None
        out["box_mpc"] = float(self.h5["box_mpc"][()]) if "box_mpc" in self.h5 else None
        out["omega_m"] = float(self.h5["omega_m"][i]) if "omega_m" in self.h5 else None
        names = [_text(v) for v in self.h5["params/names"][:]] if "params/names" in self.h5 else []
        values = self.h5["params/values"][i] if "params/values" in self.h5 else []
        out["params"] = {n: float(v) for n, v in zip(names, values) if np.isfinite(v)}
        return out

    def prediction(self, model: str, sid: str) -> dict:
        node = f"models/{model}"
        if node not in self.h5:
            return {}
        i = self.samples.index(str(sid))
        out = {field: self.h5[f"{node}/{field}"][i] for field in self.h5[node] if field != "draws"}
        if f"{node}/draws" in self.h5:
            out.update({f"{field}__draws": self.h5[f"{node}/draws/{field}"][i]
                        for field in self.h5[f"{node}/draws"]})
        return out

    def close(self):
        self.h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _load_index(h5) -> dict:
    if "index" not in h5:
        return {}
    return json.loads(_text(h5["index"][()]))


_SAMPLE_RE = re.compile(r"(\d+)(?!.*\d)")


def resolve_sample(samples: list[str], key) -> str:
    if isinstance(key, (int, np.integer)):
        return samples[int(key)]
    text = str(key)
    if text in samples:
        return text
    match = _SAMPLE_RE.search(text)
    if match:
        number = match.group(1)
        for sid in samples:
            if _SAMPLE_RE.search(sid) and _SAMPLE_RE.search(sid).group(1) == number:
                return sid
    raise KeyError(f"no sample {key!r} in {samples}")


def open_cache(path) -> CacheFile:
    return CacheFile(path)


class StoreGroup:
    """Read several caches (or legacy stores) as one store."""

    def __init__(self, stores):
        self.stores = [s for s in stores if s is not None]
        if not self.stores:
            raise ValueError("no stores to group")

    @property
    def dir(self) -> Path:
        return self.stores[0].dir

    @property
    def models(self) -> list[str]:
        out = []
        for store in self.stores:
            for model in store.models:
                if model not in out:
                    out.append(model)
        return out

    @property
    def samples(self) -> list[str]:
        out = []
        for store in self.stores:
            for sid in store.samples:
                if sid not in out:
                    out.append(sid)
        return out

    def draws(self, model: str) -> int:
        return max((s.draws(model) for s in self.stores if model in getattr(s, "models", [])), default=1)

    def resolve(self, key):
        return resolve_sample(self.samples, key)

    def has(self, model, sid: str | None = None) -> bool:
        """``has(sid)`` or ``has(model, sid)``, matching the other stores."""
        if sid is None:
            return any(str(model) in s.samples for s in self.stores)
        return any(str(sid) in s.samples and model in getattr(s, "models", [])
                   for s in self.stores)

    def truth(self, sid: str) -> dict:
        merged = {"targets": {}, "inputs": {}}
        for store in self.stores:
            if str(sid) not in store.samples:
                continue
            truth = store.truth(sid)
            for bucket in ("targets", "inputs"):
                merged[bucket].update(truth.get(bucket, {}))
            for key in ("z_grid", "box_mpc", "omega_m", "params"):
                if merged.get(key) is None and truth.get(key) is not None:
                    merged[key] = truth[key]
        return merged

    def prediction(self, model: str, sid: str) -> dict:
        merged = {}
        for store in self.stores:
            if model in getattr(store, "models", []) and str(sid) in store.samples:
                merged.update(store.prediction(model, sid))
        return merged
