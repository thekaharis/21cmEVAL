"""Saved-cube datasets (``.npz``), with or without a manifest.

Manifest (recommended) -- all paths relative to the manifest::

    {
      "box_mpc": 200.0,
      "z_grid": [5.0, 5.1, "...", 25.0],
      "target": "xhi",
      "inputs": ["density"],
      "samples": [
        {"id": "cone0000", "path": "cone_0000.npz", "split": "test",
         "omega_m": 0.31, "params": {"F_ESC10": -1.2}}
      ]
    }

Without a manifest, every ``*.npz`` in the directory is one sample and every
3-D array except the target becomes an input. One-dimensional arrays named
``z_grid`` / ``z`` set the redshift grid; a JSON string array ``params`` sets
sampled parameters. This makes the offline path a two-line contract: save
``density`` and ``xhi`` per cone, point the suite at the directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..spec import Meta, Sample
from .base import BaseDataset


class NpzDataset(BaseDataset):
    def __init__(self, path, target: str = "xhi", density: str = "density",
                 split: str | None = None, limit: int | None = None,
                 default_omega_m: float | None = None,
                 indices: list[int] | None = None):
        super().__init__(split=split, limit=limit, indices=indices)
        self.path = Path(path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"data path not found: {self.path}")
        self.target = target
        self.density = density
        self.default_omega_m = default_omega_m
        self._manifest = None
        if self.path.is_dir():
            self.root = self.path
            manifest_path = self.path / "manifest.json"
            if manifest_path.exists():
                self._manifest = json.loads(manifest_path.read_text())
                self.files = None
            else:
                self.files = sorted(self.path.glob("*.npz"))
                if not self.files:
                    raise FileNotFoundError(f"no .npz files in {self.path}")
        elif self.path.suffix.lower() == ".json":
            self._manifest = json.loads(self.path.read_text())
            self.root = self.path.parent
            self.files = None
        elif self.path.suffix.lower() == ".npz":
            self.root = self.path.parent
            self.files = [self.path]
        else:
            raise ValueError(
                f"expected a directory, a .npz file or a manifest .json, got {self.path}")
        self._loaded: list[Sample] | None = None

    def _samples(self) -> list[Sample]:
        if self._loaded is None:
            self._loaded = self._load()
        return self._loaded

    def _load(self) -> list[Sample]:
        if self._manifest is not None:
            return self._from_manifest(self._manifest)
        samples = []
        for path in self.files:
            samples.append(self._from_file(path, {}, len(samples)))
        return samples

    def _from_manifest(self, manifest: dict) -> list[Sample]:
        default_z = manifest.get("z_grid")
        default_box = manifest.get("box_mpc")
        default_omega = manifest.get("omega_m", self.default_omega_m)
        inputs = manifest.get("inputs")
        target = manifest.get("target", self.target)
        samples = []
        for index, entry in enumerate(manifest.get("samples", [])):
            path = self.root / entry["path"]
            context = {
                "id": entry.get("id", path.stem),
                "split": entry.get("split", "test"),
                "z_grid": entry.get("z_grid", default_z),
                "box_mpc": entry.get("box_mpc", default_box),
                "omega_m": entry.get("omega_m", default_omega),
                "params": entry.get("params", {}),
                "inputs": entry.get("inputs", inputs),
                "target": entry.get("target", target),
            }
            samples.append(self._from_file(path, context, index))
        return samples

    def _from_file(self, path: Path, context: dict, index: int) -> Sample:
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        target_name = context.get("target") or self.target
        if target_name not in arrays:
            raise KeyError(
                f"{path} has no target array {target_name!r}; found {sorted(arrays)}")
        target = np.asarray(arrays[target_name])
        if target.ndim == 4:
            target = target[0]

        inputs: dict[str, np.ndarray] = {}
        declared = context.get("inputs")
        for name, array in arrays.items():
            if name == target_name:
                continue
            if declared is not None and name not in declared:
                continue
            if declared is None and (array.ndim != 3 or array.dtype.kind not in "fiu"):
                continue
            inputs[name] = np.asarray(array)

        z_grid = context.get("z_grid")
        if z_grid is None and "z_grid" in arrays:
            z_grid = arrays["z_grid"]
        if z_grid is None and "z" in arrays and np.asarray(arrays["z"]).ndim == 1:
            z_grid = arrays["z"]
        box = context.get("box_mpc")
        if box is None:
            box = float(arrays["box_mpc"]) if "box_mpc" in arrays else None
        omega = context.get("omega_m")
        if omega is None and "omega_m" in arrays:
            omega = float(arrays["omega_m"])
        params = dict(context.get("params") or {})
        if "params" in arrays and np.asarray(arrays["params"]).dtype.kind in "US":
            try:
                params.update(json.loads(str(np.asarray(arrays["params"]).item())))
            except json.JSONDecodeError:
                pass
        return Sample(
            id=context.get("id", path.stem),
            inputs=inputs,
            targets={target_name: target},
            meta=Meta(
                box_mpc=None if box is None else float(box),
                z_grid=None if z_grid is None else np.asarray(z_grid, dtype=np.float64),
                omega_m=None if omega is None else float(omega),
                params=params,
            ),
            split=context.get("split", "test"),
        )

    def describe(self) -> dict:
        samples = self._samples()
        first = samples[0] if samples else None
        return {
            "format": "npz",
            "root": str(self.root),
            "n_samples": len(samples),
            "shape": list(first.shape) if first else None,
            "box_mpc": first.meta.box_mpc if first else None,
            "inputs": sorted(first.inputs) if first else [],
            "targets": sorted(first.targets) if first else [],
        }
