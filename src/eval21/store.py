"""Prediction store: full truth and prediction cubes of selected samples, for map-based figures.

``eval21 run --save-predictions DIR [--save-n N | --save-ids a,b]`` writes

    DIR/index.json                      {"samples": [...], "models": [...], "fields": [...], ...}
    DIR/truth/<sample id>.npz           targets (``t_<field>``), inputs (``i_<field>``), z_grid,
                                        box_mpc, omega_m, params (JSON)
    DIR/<model>/<sample id>.npz         predicted fields (``<field>``); generative / N-draw runs
                                        store (draws, Nx, Ny, Nz)

as float16 (cubes) to keep a 140 x 140 x 512 field at ~20 MB. Statistics-only figures never need
the store; slice, strip, scatter and animation figures do (``eval21 plot`` finds it next to the
results, or via ``--store``). ``PredictionStore.open`` reads one back.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

INDEX = "index.json"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


class PredictionStore:
    """Writer (during a run) and reader (for figures) of stored cubes."""

    def __init__(self, directory, ids=None, n=None, dtype=np.float16):
        self.dir = Path(directory)
        self.ids = None if ids is None else {str(i) for i in ids}
        self.n = n
        self.dtype = dtype
        self.index = {"samples": [], "models": [], "fields": [], "draws": {}}
        if (self.dir / INDEX).exists():
            self.index.update(json.loads((self.dir / INDEX).read_text()))

    # -- writing ------------------------------------------------------------------------------
    def wants(self, sample) -> bool:
        sid = str(sample.id)
        if self.ids is not None:
            return sid in self.ids
        if sid in self.index["samples"]:
            return True
        return self.n is None or len(self.index["samples"]) < self.n

    def save_truth(self, sample) -> None:
        sid = _safe(sample.id)
        path = self.dir / "truth" / f"{sid}.npz"
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"t_{k}": np.asarray(v).astype(self.dtype) for k, v in sample.targets.items()}
        for k, v in sample.inputs.items():
            v = np.asarray(v)
            arrays[f"i_{k}"] = v.astype(self.dtype) if v.ndim >= 3 else v
        meta = sample.meta
        if meta.z_grid is not None:
            arrays["z_grid"] = np.asarray(meta.z_grid, np.float64)
        arrays["box_mpc"] = np.float64(meta.box_mpc if meta.box_mpc is not None else np.nan)
        arrays["omega_m"] = np.float64(meta.omega_m if meta.omega_m is not None else np.nan)
        arrays["params"] = np.array(json.dumps({k: float(v) for k, v in (meta.params or {}).items()}))
        np.savez(path, **arrays)
        if str(sample.id) not in self.index["samples"]:
            self.index["samples"].append(str(sample.id))

    def save_prediction(self, model: str, sample, prediction: dict, draws: int = 1) -> None:
        """Add a model's fields for one sample; fields stored earlier (another target's run
        writing to the same store) are kept."""
        path = self.dir / _safe(model) / f"{_safe(sample.id)}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        if path.exists():
            with np.load(path) as old:
                arrays = {k: old[k] for k in old.files}
        arrays.update({k: np.asarray(v).astype(self.dtype) for k, v in prediction.items()
                       if np.asarray(v).ndim >= 3})
        np.savez(path, **arrays)
        if model not in self.index["models"]:
            self.index["models"].append(model)
        for k in arrays:
            if k not in self.index["fields"]:
                self.index["fields"].append(k)
        self.index["draws"][model] = int(draws)
        self.flush()

    def flush(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / INDEX).write_text(json.dumps(self.index, indent=1))

    # -- reading ------------------------------------------------------------------------------
    @classmethod
    def open(cls, directory) -> "PredictionStore":
        directory = Path(directory)
        if not (directory / INDEX).exists():
            raise FileNotFoundError(f"no prediction store at {directory} (run with --save-predictions)")
        return cls(directory)

    @property
    def samples(self) -> list[str]:
        return list(self.index["samples"])

    @property
    def models(self) -> list[str]:
        return list(self.index["models"])

    def draws(self, model: str) -> int:
        return int(self.index["draws"].get(model, 1))

    def resolve(self, sample_id) -> str:
        """Stored id for ``sample_id``; '1001' finds 'cone1001' (and vice versa), as figure
        file names drop the 'cone' prefix."""
        sid = str(sample_id)
        if sid in self.index["samples"]:
            return sid
        short = re.fullmatch(r"(?:cone_?)?(\d+)", sid)
        for stored in self.index["samples"]:
            other = re.fullmatch(r"(?:cone_?)?(\d+)", stored)
            if short and other and short.group(1) == other.group(1):
                return stored
        raise KeyError(f"sample {sid!r} is not in the prediction store {self.dir} "
                       f"(stored: {', '.join(self.index['samples'][:8])}...)")

    def truth(self, sample_id) -> dict:
        """{"targets": {...}, "inputs": {...}, "z_grid", "box_mpc", "omega_m", "params"} as float32."""
        sample_id = self.resolve(sample_id)
        with np.load(self.dir / "truth" / f"{_safe(sample_id)}.npz") as d:
            out = {"targets": {}, "inputs": {}, "z_grid": d["z_grid"] if "z_grid" in d.files else None,
                   "box_mpc": float(d["box_mpc"]), "omega_m": float(d["omega_m"]),
                   "params": json.loads(str(d["params"]))}
            for k in d.files:
                if k.startswith("t_"):
                    out["targets"][k[2:]] = d[k].astype(np.float32)
                elif k.startswith("i_"):
                    out["inputs"][k[2:]] = np.asarray(d[k]).astype(np.float32)
        return out

    def prediction(self, model: str, sample_id) -> dict:
        sample_id = self.resolve(sample_id)
        path = self.dir / _safe(model) / f"{_safe(sample_id)}.npz"
        if not path.exists():
            return {}
        with np.load(path) as d:
            return {k: d[k].astype(np.float32) for k in d.files}

    def has(self, model: str, sample_id) -> bool:
        try:
            sample_id = self.resolve(sample_id)
        except KeyError:
            return False
        return (self.dir / _safe(model) / f"{_safe(sample_id)}.npz").exists()
