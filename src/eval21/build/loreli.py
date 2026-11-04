"""LoReLi II (Licorice) lightcones, ``lc_128`` release (Meriot, Semelin & Cornu 2024).

Files (raw little-endian float32, no header, C order)::

    lightcone_xa_tpHI_xion_dens_dtb_no_rsd_simuN.dat   (128, 128, 1536, 5)  fields interleaved per voxel
    lightcone_dtb_no_rsd_simuN.dat                      (128, 128, 1536)     dT_b only

Facts ported from rhizome-neural-operator ``ebm21cm/data/loreli.py`` / ``loreli_globals.py``
(verified there on the 228 five-field cones against the 21ssd box diagnostics):

* field order ``xa, tpHI, xion, dens, dtb``; index 0 of the LOS is the **low-redshift** end;
* the LOS is **uniform in redshift**, z = 5 ... 15 over 1536 planes (dz = 10/1535); comoving
  spacing 3.5 Mpc at z = 5 down to 0.81 Mpc at z = 15 (LoReLi I cosmology h = 0.678,
  Omega_m = 0.308);
* transverse: 128 cells over L_perp = 294 Mpc (LoReLi I table; LoReLi II quotes 300 Mpc,
  ``--l-perp``); **not periodic**;
* x_HI = 1 - xion; ``dens`` is in code units -> density = dens / mean over populated cells
  (per plane by default, ``--density-mean cone`` for the cone mean), i.e. 1 + delta; empty cells
  get density 1;
* cones stop at 99 % ionization: planes below the end redshift are empty (dens = 0, dT_b NaN);
  some planes are corrupt (|dT_b| up to 1e12 mK). Valid planes = the longest contiguous run of
  planes that are finite, sane (|dT_b| < 1e3 mK) and (five-field) populated, with the partly filled
  edge plane removed. Five-field: rhizome ``LoreliCone._validity``; dT_b-only:
  ``loreli_globals`` (``plane_run`` minus one edge plane, then the sanity cut).

Parameters are **inferred from the simulation number** with the LoReLi II grid rule (stated in
every manifest): ``j = N - 10000; r_HS = RH[j % 6]; f_X = FX[(j // 6) % 13];
(tau, log10 M_min) = PAIRS[(j // 78) % 42]; f_esc,post = FE[j // 3276]``; values from the 21ssd
``parameters.txt`` grid (``loreli2_parameters.json``; tau in units of 10 Myr). Pass
``--loreli-params loreli2_parameters.json`` to take values and publication status from that file
instead (the rule is then cross-checked).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ..physics import comoving_distance_mpc
from .common import BuildError, Item, Source, file_record

FIELDS5 = ("xa", "tpHI", "xion", "dens", "dtb")
L_PERP = 294.0
Z_RANGE = (5.0, 15.0)
N_LOS = 1536
N_PERP = 128
OMEGA_M = 0.308
HUBBLE_H = 0.678
DTB_ABS_MAX = 1.0e3
PATTERNS = {"five": "lightcone_xa_tpHI_xion_dens_dtb_no_rsd_simu*.dat",
            "dtb": "lightcone_dtb_no_rsd_simu*.dat"}

# LoReLi II grid (21ssd parameters.txt values, as in loreli2_parameters.json)
PARAM_NAMES = ("f_x", "tau", "rHS", "logMmin", "fesc")
RH = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
FX = (0.1, 0.15, 0.22, 0.32, 0.46, 0.68, 1.0, 1.47, 2.15, 3.16, 4.64, 6.81, 10.0)
FE = (0.275, 0.05, 0.5)
PAIRS = (
    (8191.9, 8.0), (6475.2, 8.27), (5118.3, 8.53), (4045.7, 8.8), (3197.9, 9.07), (2527.8, 9.33),
    (1998.1, 9.6), (3884.9, 8.0), (3070.8, 8.27), (2427.3, 8.53), (1918.7, 8.8), (1516.6, 9.07),
    (1198.8, 9.33), (947.6, 9.6), (6388.3, 8.0), (5049.6, 8.27), (3991.4, 8.53), (3155.0, 8.8),
    (2493.8, 9.07), (1971.2, 9.33), (1558.2, 9.6), (4981.8, 8.0), (3937.8, 8.27), (3112.6, 8.53),
    (2460.4, 8.8), (1944.8, 9.07), (1537.2, 9.33), (1215.1, 9.6), (10504.7, 8.0), (8303.4, 8.27),
    (6563.4, 8.53), (5188.0, 8.8), (4100.8, 9.07), (3241.5, 9.33), (2562.2, 9.6), (3029.6, 8.0),
    (2394.7, 8.27), (1892.9, 8.53), (1496.2, 8.8), (1182.7, 9.07), (934.8, 9.33), (738.9, 9.6),
)
GRID_RULE = ("j = N - 10000; rHS = RH[j % 6]; f_x = FX[(j // 6) % 13]; "
             "(tau, logMmin) = PAIRS[(j // 78) % 42]; fesc = FE[j // 3276]; tau in 10 Myr")
OFF_GRID = range(10488, 10500)    # parameters.txt labels off the published grid (loreli2_parameters.json)
PARAM_UNITS = {"f_x": "X-ray efficiency", "tau": "star-formation timescale [10 Myr]",
               "rHS": "hard/total X-ray energy", "logMmin": "log10 minimum halo mass [Msun]",
               "fesc": "post-reionization-onset UV escape fraction"}

_TABLE: dict = {}


def grid_params(sim: int) -> dict:
    """LoReLi II grid parameters of simulation number ``sim`` (inferred from the number)."""
    j = int(sim) - 10000
    if not 0 <= j < 6 * 13 * 42 * 3:
        raise ValueError(f"simulation {sim} is outside the LoReLi II grid (10000-19827)")
    tau, logm = PAIRS[(j // 78) % 42]
    return {"f_x": FX[(j // 6) % 13], "tau": tau, "rHS": RH[j % 6], "logMmin": logm,
            "fesc": FE[j // 3276]}


def loreli_redshifts(n: int = N_LOS, z_range=Z_RANGE) -> np.ndarray:
    return np.linspace(float(z_range[0]), float(z_range[1]), int(n))


def distance_table(z_range=Z_RANGE, omega_m=OMEGA_M, h=HUBBLE_H):
    """Fine (z, comoving distance from z_min) table, eval21's flat LCDM (no radiation)."""
    key = (tuple(z_range), omega_m, h)
    if key not in _TABLE:
        z = np.linspace(float(z_range[0]), float(z_range[1]), 100001)
        _TABLE[key] = (z, comoving_distance_mpc(z, omega_m, h))
    return _TABLE[key]


def longest_run(mask: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(mask)
    keep = np.zeros(len(mask), bool)
    if idx.size:
        keep[max(np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1), key=len)] = True
    return keep


def plane_run(plane_ok: np.ndarray, trim: int = 1) -> np.ndarray:
    """Longest contiguous run of good planes minus ``trim`` edge planes (loreli_globals)."""
    idx = np.flatnonzero(plane_ok)
    keep = np.zeros(len(plane_ok), bool)
    if idx.size == 0:
        return keep
    run = max(np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1), key=len)
    if len(run) > 2 * trim:
        keep[run[trim:len(run) - trim]] = True
    return keep


def validity_five(cube: np.ndarray) -> tuple[np.ndarray, dict]:
    """``LoreliCone._validity`` on a (nx, ny, n, 5) cube."""
    mean = cube[..., FIELDS5.index("dens")].mean((0, 1), dtype=np.float64)
    dtb = cube[..., FIELDS5.index("dtb")]
    finite = np.isfinite(dtb).all((0, 1))
    with np.errstate(invalid="ignore"):
        sane = np.where(np.isfinite(dtb), np.abs(dtb), 0).max((0, 1)) < DTB_ABS_MAX
    valid = finite & sane & (mean > 0)
    if valid.any():
        valid &= np.abs(mean / np.median(mean[valid]) - 1) <= 0.10
    keep = longest_run(valid)
    return keep, {"nonfinite_planes": int((~finite).sum()), "insane_planes": int((finite & ~sane).sum()),
                  "empty_planes": int((mean <= 0).sum())}


def validity_dtb(cube: np.ndarray) -> tuple[np.ndarray, dict]:
    """``loreli_globals`` rule on a (nx, ny, n) dT_b cube, reduced to one contiguous run."""
    finite = np.isfinite(cube).all((0, 1))
    with np.errstate(invalid="ignore"):
        absmax = np.where(np.isfinite(cube), np.abs(cube), 0).max((0, 1))
    geometry = plane_run(finite)
    sane = absmax < DTB_ABS_MAX
    keep = longest_run(geometry & sane)
    return keep, {"nonfinite_planes": int((~finite).sum()), "insane_planes": int((geometry & ~sane).sum())}


class LoreliSource(Source):
    format = "loreli"
    scan_reads_data = True
    default_group = "auto"
    hubble_h = HUBBLE_H

    def __init__(self, root, options):
        super().__init__(root, options)
        self.kind = self.so.get("loreli_kind") or "five"
        if self.kind not in PATTERNS:
            raise ValueError("--loreli-kind must be five or dtb")
        self.l_perp = float(self.so.get("l_perp") or L_PERP)
        self.density_mean = self.so.get("density_mean") or "slice"
        if self.density_mean not in ("slice", "cone"):
            raise ValueError("--density-mean must be slice or cone")
        # native sizes are fixed by the release; overridable for tests with tiny fake files
        self.n_perp = int(self.so.get("n_perp") or N_PERP)
        self.n_los = int(self.so.get("n_los") or N_LOS)
        self.params_file = self.so.get("loreli_params")
        self._param_table = None
        if self.params_file:
            payload = json.loads(Path(self.params_file).read_text())
            self._param_table = payload["sims"]

    # -- fields ------------------------------------------------------------ #
    def available_fields(self):
        if self.kind == "dtb":
            return {"tb": {"source": "lightcone_dtb_no_rsd (dtb)", "units": "mK",
                           "description": "21-cm brightness temperature, no RSD, no dv/dr term"}}
        return {
            "xhi": {"source": "1 - xion (clipped to [0, 1])", "units": "1",
                    "description": "neutral hydrogen fraction"},
            "tb": {"source": "dtb", "units": "mK",
                   "description": "21-cm brightness temperature, no RSD, no dv/dr term"},
            "density": {"source": "dens / mean(dens > 0)", "units": "1",
                        "convention": f"1 + delta, mean over populated cells per "
                                      f"{'plane' if self.density_mean == 'slice' else 'cone'}",
                        "description": "gas (SPH) density"},
            "xion": {"source": "xion", "units": "1", "description": "ionized fraction"},
            "x_alpha": {"source": "xa", "units": "1", "description": "Lyman-alpha coupling x_alpha"},
            "t_hi": {"source": "tpHI", "units": "K",
                     "description": "neutral-phase temperature (0 where fully ionized, capped at 1e4 K)"},
        }

    def default_fields(self):
        return ["tb"] if self.kind == "dtb" else ["xhi", "tb", "density"]

    def resolve_fields(self, requested):
        out = super().resolve_fields(requested)
        for name, key in out:
            if key:
                raise ValueError("loreli: name=key field mappings are not supported "
                                 f"(fields: {', '.join(self.available_fields())})")
        return out

    def scan_signature(self):
        return {"kind": self.kind, "l_perp": self.l_perp, "validity": "v1",
                "n": [self.n_perp, self.n_los]}

    def read_signature(self):
        return {"density_mean": self.density_mean}

    # -- files ------------------------------------------------------------- #
    def discover(self):
        items = []
        for path in sorted(self.root.glob(PATTERNS[self.kind])):
            match = re.search(r"simu(\d+)\.dat$", path.name)
            if match:
                items.append(Item(int(match.group(1)), path))
        return items

    def _load(self, path: Path) -> np.ndarray:
        np_, nl = self.n_perp, self.n_los
        n = np_ * np_ * nl * (5 if self.kind == "five" else 1)
        data = np.fromfile(path, dtype="<f4")
        if data.size != n:
            raise ValueError(f"{path}: {data.size} float32 values, expected {n}")
        shape = (np_, np_, nl, 5) if self.kind == "five" else (np_, np_, nl)
        return data.reshape(shape)

    def params_for(self, sim: int) -> tuple[dict, dict]:
        params = grid_params(sim)
        info = {"param_source": "inferred from the simulation number (LoReLi II grid rule)"}
        if sim in OFF_GRID:
            info["param_warning"] = "21ssd parameters.txt labels this id off the published grid"
        if self._param_table is not None:
            entry = self._param_table.get(str(sim))
            if entry is None:
                info["param_warning"] = f"simulation {sim} not in {self.params_file}"
            else:
                table = {k: float(entry[k]) for k in PARAM_NAMES}
                mismatch = {k: (params[k], table[k]) for k in PARAM_NAMES if abs(params[k] - table[k]) > 1e-9}
                if mismatch:
                    info["param_rule_mismatch"] = mismatch
                params = table
                info["param_source"] = f"{self.params_file} (rule cross-checked)"
                info["param_status"] = entry.get("status")
        return params, info

    def scan(self, item: Item) -> dict:
        cube = self._load(item.path)
        keep, quality = (validity_five if self.kind == "five" else validity_dtb)(cube)
        del cube
        if not keep.any():
            raise ValueError("no valid planes")
        idx = np.flatnonzero(keep)
        z = loreli_redshifts(self.n_los)
        zt, dt = distance_table()
        d = np.interp(z, zt, dt)
        params, pinfo = self.params_for(item.cone_id)
        return {
            "cone_id": item.cone_id, "files": [file_record(item.path)],
            "shape": [self.n_perp, self.n_perp, self.n_los], "cell_mpc": self.l_perp / self.n_perp,
            "z": z, "d": d, "valid": [int(idx[0]), int(idx[-1])], "omega_m": OMEGA_M,
            "params": params, "fields": list(self.available_fields()),
            "info": {"valid_planes": [int(idx[0]), int(idx[-1])],
                     "valid_z": [float(z[idx[0]]), float(z[idx[-1]])], **quality, **pinfo},
        }

    def distance_table(self, scan):
        return distance_table()

    def read(self, item, scan, names, lo, hi, crop):
        x0, x1, y0, y1 = crop
        if self.kind == "dtb":
            cube = np.memmap(item.path, dtype="<f4", mode="r", shape=(self.n_perp, self.n_perp, self.n_los))
            return {"tb": np.array(cube[x0:x1, y0:y1, lo:hi], dtype=np.float32)}
        cube = self._load(item.path)
        out = {}
        for name in names:
            if name == "xhi":
                out[name] = np.clip(1.0 - cube[x0:x1, y0:y1, lo:hi, 2], 0.0, 1.0).astype(np.float32)
            elif name == "density":
                if self.density_mean == "cone":
                    dens_all = cube[..., 3]
                    rho = np.float64(dens_all[dens_all > 0].mean())
                    dens = cube[x0:x1, y0:y1, lo:hi, 3].astype(np.float64)
                    ratio = np.where(dens > 0, dens / rho, 1.0)
                else:
                    dens = cube[:, :, lo:hi, 3].astype(np.float64)
                    good = dens > 0
                    rho = np.where(good, dens, 0).sum((0, 1)) / np.maximum(good.sum((0, 1)), 1)
                    ratio = np.divide(dens, rho[None, None, :], out=np.ones_like(dens),
                                      where=good & (rho[None, None, :] > 0))
                    ratio = ratio[x0:x1, y0:y1]
                out[name] = ratio.astype(np.float32)
            else:
                k = {"tb": "dtb", "xion": "xion", "x_alpha": "xa", "t_hi": "tpHI"}[name]
                out[name] = np.array(cube[x0:x1, y0:y1, lo:hi, FIELDS5.index(k)], dtype=np.float32)
        return out

    def check_output(self, cone, stats):
        if "density" in stats and abs(stats["density"]["mean"] - 1.0) > 0.1:
            raise BuildError(f"cone {cone}: written density mean {stats['density']['mean']:.4f}, not ~1")

    # -- manifest ---------------------------------------------------------- #
    def native_resolution(self, scans):
        return {"cyl_native_cell_mpc": self.l_perp / self.n_perp,
                "cyl_native_los_dz": (Z_RANGE[1] - Z_RANGE[0]) / (self.n_los - 1),
                "transverse_cells": [self.n_perp, self.n_perp], "l_perp_mpc": self.l_perp,
                "native_los_planes": self.n_los, "los_axis": "uniform in redshift, z 5-15",
                "cosmology": {"omega_m": OMEGA_M, "hubble_h": HUBBLE_H}}

    def provenance(self, scans):
        return {"loreli_kind": self.kind, "l_perp_mpc": self.l_perp, "density_mean": self.density_mean,
                "params_inferred": {
                    "names": list(PARAM_NAMES), "units": PARAM_UNITS, "rule": GRID_RULE,
                    "values": {"RH": RH, "FX": FX, "FE": FE, "PAIRS": PAIRS},
                    "source": self.params_file or "built-in LoReLi II grid (21ssd parameters.txt values)",
                    "status": "inferred from the simulation number; off-grid labels 10488-10499"},
                "validity": ("rhizome LoreliCone._validity (finite, |dTb| < 1e3 mK, populated, density within "
                             "10 % of the median plane mean; longest run)" if self.kind == "five" else
                             "loreli_globals: longest finite run minus one edge plane, |dTb| < 1e3 mK; "
                             "longest run")}

    def notes(self):
        notes = ["LoReLi is not periodic transversally; eval21 spectra assume periodic planes",
                 "parameters are inferred from the simulation number (provenance.params_inferred)",
                 "native LOS is uniform in z (dz = 10/1535); comoving/linear outputs interpolate linearly "
                 "in comoving distance (eval21 flat LCDM, h = 0.678, Omega_m = 0.308)"]
        if self.kind == "dtb":
            notes.append("dT_b-only cones: no density or x_HI inputs; density-dependent metrics are skipped")
        return notes
