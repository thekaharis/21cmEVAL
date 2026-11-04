"""Target grids and resampling for ``eval21 build``.

Line of sight
    ``native``                    keep each source's own planes (all files must agree)
    ``uniform:ZMIN,ZMAX,N``       N planes uniform in redshift
    ``comoving:ZMIN,ZMAX[,N]``    uniform in comoving distance; without N the spacing equals
                                  the transverse cell (cubic cells)
    ``file:PATH`` / ``PATH.npy``  explicit grid (``.npy``, ``.txt``, or ``FILE.h5:key``, e.g.
                                  ``cubes_3d_envelope512.h5:target_z`` to reuse fno-21cm's grid)

Interpolation along the LOS is linear per pixel, the same as fno-21cm's ``build_cubes``
(``scipy.interpolate.interp1d(kind="linear", fill_value=0)``) but without scipy; planes outside
the source's redshift range are filled with 0 and marked invalid.

Transverse resampling (``--transverse N``) is linear between cell centres on the same physical
box; it keeps the box size, so only the cell size changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..physics import comoving_distance_mpc


def parse_z_grid(spec: str | None, cell_mpc: float | None = None, omega_m: float = 0.31,
                 hubble_h: float = 0.678) -> np.ndarray | None:
    """Grid from a spec string; ``None`` / ``native`` returns None (keep the source grid)."""
    if spec is None or spec == "native":
        return None
    kind, sep, rest = spec.partition(":")
    if not sep:
        kind, rest = "file", spec
    if kind == "uniform":
        z0, z1, n = rest.split(",")
        return np.linspace(float(z0), float(z1), int(n))
    if kind == "comoving":
        parts = rest.split(",")
        z0, z1 = float(parts[0]), float(parts[1])
        fine = np.linspace(z0, z1, 20001)
        chi = comoving_distance_mpc(fine, omega_m=omega_m, h=hubble_h)
        if len(parts) > 2:
            target = np.linspace(chi[0], chi[-1], int(parts[2]))
        else:
            if not cell_mpc:
                raise ValueError("comoving:ZMIN,ZMAX without N needs the transverse cell size")
            n = int(np.floor((chi[-1] - chi[0]) / cell_mpc + 1e-9)) + 1
            target = chi[0] + np.arange(n) * float(cell_mpc)
        return np.interp(target, chi, fine)
    if kind == "file":
        path, colon, key = rest.rpartition(":") if rest.count(":") and not Path(rest).exists() else (rest, "", "")
        path = Path(path).expanduser()
        if path.suffix.lower() in {".h5", ".hdf5"}:
            import h5py

            with h5py.File(path, "r") as handle:
                return np.asarray(handle[key or "target_z"], dtype=np.float64)
        if path.suffix == ".npy":
            return np.load(path).astype(np.float64)
        if path.suffix == ".npz":
            with np.load(path) as data:
                return np.asarray(data[key or "z_grid"], dtype=np.float64)
        return np.loadtxt(path, dtype=np.float64).ravel()
    raise ValueError(f"unknown z grid {spec!r}; use native, uniform:, comoving: or a file")


class LosInterpolator:
    """Linear interpolation weights from one source grid to one target grid (reused per field)."""

    def __init__(self, source_z, target_z):
        source_z = np.asarray(source_z, dtype=np.float64)
        target_z = np.asarray(target_z, dtype=np.float64)
        order = np.argsort(source_z)
        self.flip = not np.all(order == np.arange(len(order)))
        self.order = order
        zs = source_z[order]
        i = np.clip(np.searchsorted(zs, target_z) - 1, 0, len(zs) - 2)
        w = (target_z - zs[i]) / np.maximum(zs[i + 1] - zs[i], 1e-300)
        self.inside = (target_z >= zs[0] - 1e-9) & (target_z <= zs[-1] + 1e-9)
        self.i, self.w = i, np.clip(w, 0.0, 1.0)

    def __call__(self, cube: np.ndarray) -> np.ndarray:
        cube = np.asarray(cube, dtype=np.float32)
        if self.flip:
            cube = cube[..., self.order]
        lo = cube[..., self.i]
        hi = cube[..., self.i + 1]
        out = lo + (hi - lo) * self.w.astype(np.float32)
        out[..., ~self.inside] = 0.0
        return out

    def valid(self, source_valid: np.ndarray | None) -> np.ndarray:
        """Target planes whose two bracketing source planes are both valid."""
        ok = self.inside.copy()
        if source_valid is not None:
            v = np.asarray(source_valid, bool)[self.order]
            ok &= v[self.i] & (v[self.i + 1] | (self.w < 1e-9))
        return ok


def resample_transverse(cube: np.ndarray, n_out: int) -> np.ndarray:
    """(Nx, Ny, Nz) -> (n_out, n_out, Nz), linear between cell centres on the same box."""
    cube = np.asarray(cube, dtype=np.float32)
    for axis in (0, 1):
        n_in = cube.shape[axis]
        if n_in == n_out:
            continue
        x = (np.arange(n_out) + 0.5) * n_in / n_out - 0.5
        x = np.clip(x, 0.0, n_in - 1.0)
        i = np.clip(np.floor(x).astype(int), 0, n_in - 2)
        w = (x - i).astype(np.float32)
        lo = np.take(cube, i, axis=axis)
        hi = np.take(cube, i + 1, axis=axis)
        shape = [1, 1, 1]
        shape[axis] = n_out
        cube = lo + (hi - lo) * w.reshape(shape)
    return cube
