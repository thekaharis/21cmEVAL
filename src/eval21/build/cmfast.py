"""21cmFAST raw lightcones (``21cmfast_11d_sampleNNNNNN.h5``, schema ``raw_lightcone_v2.0``).

Layout of one file (as written by the fno-21cm generator)::

    lightcone/{brightness_temp, neutral_fraction, density, los_velocity, tau_21}   (140, 140, 2340) f4
    lightcone/<field>_with_rsds                                                     same, RSD applied
    lightcone/lightcone_redshifts, lightcone/lightcone_distances                    (2340,) f8
    lightcone/node_redshifts, lightcone/global_quantities/*
    params/{names, values, ranges, fiducials} (+ the values again as attrs)
    attrs: box_len_mpc=200, hii_dim=140, lightcone_z_min=5, lightcone_z_max=25, apply_rsds, ...

Conventions written by the builder:

* ``density`` is stored by 21cmFAST as the overdensity delta (cone mean ~ 0); eval21 wants the
  evolved matter density with mean 1, so the builder writes ``1 + delta`` (checked per cone:
  the written mean must lie within 0.1 of 1).
* ``xhi`` = ``neutral_fraction``, ``tb`` = ``brightness_temp`` [mK], ``los_velocity`` as stored.
  The stored ``brightness_temp`` includes the unbounded dv/dr term (spikes up to ~1e6 mK at
  velocity caustics; fno-21cm's ``tools_clean_brightness_temp.py``); it is copied unchanged.
* ``--rsd`` maps every field to its ``*_with_rsds`` variant.
* The sampled ``OMm`` is the cone's ``omega_m``; it changes the redshift of every LOS plane, so
  native grids differ between cones. ``lightcone_distances`` is uniform at the 200/140 Mpc cell.
* A ``*_design.json`` next to the files (the sampling design) is used to cross-check the stored
  parameters (``design_max_abs_diff`` per sample).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .common import BuildError, Item, Source, file_record

FIELDS = {
    "xhi": ("neutral_fraction", "neutral hydrogen fraction", "1"),
    "tb": ("brightness_temp", "21-cm brightness temperature (raw, incl. dv/dr term)", "mK"),
    "density": ("density", "evolved matter density 1 + delta (stored delta + 1)", "1"),
    "los_velocity": ("los_velocity", "line-of-sight peculiar velocity as stored by 21cmFAST", "21cmFAST"),
    "tau_21": ("tau_21", "21-cm optical depth", "1"),
}
DENSITY_CHECK = 0.1
_DESIGN: dict = {}


def _decode(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


class Cmfast21Source(Source):
    format = "21cmfast"
    hubble_h = 0.6766            # 21cmFAST v4 default (Planck 2018), same as eval21's default

    def __init__(self, root, options):
        super().__init__(root, options)
        self.rsd = bool(self.so.get("rsd", False))
        self.pattern = self.so.get("pattern") or "*.h5"
        self.density_convention = self.so.get("density_convention") or "delta"
        if self.density_convention not in ("delta", "one_plus_delta"):
            raise ValueError("21cmfast --density-convention must be delta or one_plus_delta")

    def key(self, name: str) -> str:
        explicit = getattr(self, "field_map", {}).get(name)
        if explicit:
            return explicit if "/" in explicit else f"lightcone/{explicit}"
        base = FIELDS[name][0]
        return f"lightcone/{base}_with_rsds" if self.rsd else f"lightcone/{base}"

    def available_fields(self):
        out = {}
        for name, (base, description, units) in FIELDS.items():
            src = f"{base}_with_rsds" if self.rsd else base
            out[name] = {"source": f"lightcone/{src}", "description": description, "units": units}
        out["density"]["convention"] = ("1 + delta (source: delta)" if self.density_convention == "delta"
                                        else "as stored (declared 1 + delta)")
        return out

    def default_fields(self):
        return ["xhi", "tb", "density", "los_velocity"]

    def scan_signature(self):
        return {"rsd": self.rsd}

    def read_signature(self):
        return {"density_convention": self.density_convention}

    def discover(self):
        items = []
        for path in sorted(self.root.glob(self.pattern)):
            match = re.search(r"sample(\d+)\.h5$", path.name) or re.search(r"(\d+)\D*$", path.stem)
            if match:
                items.append(Item(int(match.group(1)), path))
        return items

    def _design(self):
        """The sampling design next to the files (cached per folder), or {"rows": None}."""
        key = str(self.root)
        if key not in _DESIGN:
            files = sorted(self.root.glob("*_design.json"))
            entry = {"path": str(files[0]) if files else None, "rows": None, "names": None}
            if files:
                payload = json.loads(files[0].read_text())
                entry["rows"] = payload.get("design")
                entry["names"] = payload.get("param_names")
            _DESIGN[key] = entry
        return _DESIGN[key]

    def scan(self, item: Item) -> dict:
        import h5py

        with h5py.File(item.path, "r") as h:
            lc = h["lightcone"]
            z = np.asarray(lc["lightcone_redshifts"][:], dtype=np.float64)
            d = np.asarray(lc["lightcone_distances"][:], dtype=np.float64)
            fmap = getattr(self, "field_map", {}) or {}
            candidates = list(FIELDS) + [n for n in fmap if n not in FIELDS]
            available = [name for name in candidates if self.key(name) in h]
            probe = h[self.key(next(n for n in ("xhi", "density", "tb") if n in available))]
            shape = [int(v) for v in probe.shape]
            attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in h.attrs.items()}
            names = _decode(h["params/names"][:])
            values = [float(v) for v in h["params/values"][:]]
        params = dict(zip(names, values))
        box = float(attrs.get("box_len_mpc", np.nan))
        dim = int(attrs.get("hii_dim", shape[0]))
        cell = box / dim
        spacing = np.diff(d)
        info = {"random_seed": attrs.get("random_seed"), "sample_id": attrs.get("sample_id"),
                "los_spacing_mpc": [float(spacing.min()), float(spacing.max())]}
        if abs(float(np.median(spacing)) / cell - 1) > 1e-6:
            info["warning"] = f"LOS spacing {np.median(spacing):.6f} Mpc differs from the transverse cell {cell:.6f}"
        design = self._design()
        if design.get("rows") is not None and isinstance(item.cone_id, int) and item.cone_id < len(design["rows"]):
            row = dict(zip(design["names"], design["rows"][item.cone_id]))
            diffs = [abs(row[n] - params[n]) for n in names if n in row]
            info["design_max_abs_diff"] = float(max(diffs)) if diffs else None
        return {
            "cone_id": item.cone_id, "files": [file_record(item.path)], "shape": shape,
            "cell_mpc": cell, "box_mpc": box, "z": z, "d": d, "valid": [0, len(z) - 1],
            "omega_m": params.get("OMm"), "params": params, "fields": available, "info": info,
            "attrs": {k: attrs[k] for k in ("schema_version", "family_id", "p21cmfast_version",
                                            "apply_rsds", "include_dvdr_in_tau21", "source_model",
                                            "lightcone_z_min", "lightcone_z_max", "hii_filter")
                      if k in attrs},
        }

    def read(self, item, scan, names, lo, hi, crop):
        import h5py

        x0, x1, y0, y1 = crop
        out = {}
        with h5py.File(item.path, "r") as h:
            for name in names:
                array = np.asarray(h[self.key(name)][x0:x1, y0:y1, lo:hi], dtype=np.float32)
                if name == "density" and self.density_convention == "delta":
                    array += np.float32(1.0)
                out[name] = array
        return out

    def check_output(self, cone, stats):
        if "density" in stats and abs(stats["density"]["mean"] - 1.0) > DENSITY_CHECK:
            raise BuildError(
                f"cone {cone}: written density has mean {stats['density']['mean']:.4f}, not ~1; "
                f"the source convention is not {self.density_convention!r} (--density-convention)")

    def native_resolution(self, scans):
        if not scans:
            return {}
        cells = sorted({round(float(s["cell_mpc"]), 9) for s in scans})
        spacing = [s["info"]["los_spacing_mpc"] for s in scans]
        return {"cyl_native_cell_mpc": cells[0] if len(cells) == 1 else None,
                "cyl_native_los_mpc": float(np.median([np.mean(v) for v in spacing])),
                "transverse_cells": scans[0]["shape"][:2], "native_los_planes": scans[0]["shape"][2],
                "los_axis": "uniform in comoving distance (lightcone_distances)"}

    def provenance(self, scans):
        design = self._design()
        return {"source_attrs": scans[0].get("attrs", {}) if scans else {},
                "design_file": design.get("path"),
                "rsd": self.rsd}

    def notes(self):
        return [
            "density = 1 + delta (21cmFAST stores delta)" if self.density_convention == "delta"
            else "density copied as stored (declared 1 + delta)",
            "tb = raw brightness_temp including the unbounded dv/dr term (caustic spikes)",
            "native z grids differ between cones (sampled OMm); eval21 run needs one geometry per run",
        ]
