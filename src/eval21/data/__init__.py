"""Dataset selection from a ``format:path`` spec."""
from __future__ import annotations

from pathlib import Path

from .base import BaseDataset, Dataset
from .cubes import CubeCacheDataset, is_cube_cache
from .hdf5 import Hdf5Dataset
from .npz import NpzDataset
from .synthetic import SyntheticDataset

__all__ = ["BaseDataset", "CubeCacheDataset", "Dataset", "Hdf5Dataset", "NpzDataset",
           "SyntheticDataset", "open_data"]


def open_data(spec: str, target: str = "xhi", density: str = "density",
              split: str | None = None, limit: int | None = None,
              box_mpc: float | None = None,
              indices: list[int] | None = None, **kwargs) -> BaseDataset:
    """Open ``path``, ``npz:path``, ``hdf5:path``, ``cubes:path`` or ``synthetic:N``.

    An ``.h5`` file written by ``eval21 build`` (attribute ``format = eval21-cubes/1``) opens as
    a :class:`CubeCacheDataset` (keyword ``z_range=(zmin, zmax)`` crops the line of sight);
    other ``.h5`` files are fno-21cm caches (:class:`Hdf5Dataset`)."""
    z_range = kwargs.pop("z_range", None)
    if ":" in spec and not Path(spec).exists():
        prefix, _, argument = spec.partition(":")
    else:
        prefix, argument = "", spec
    prefix = prefix.lower()

    if prefix in {"synthetic", "toy"}:
        n = int(argument) if argument else 4
        return SyntheticDataset(
            n_samples=n, box_mpc=box_mpc or 200.0,
            split=split, limit=limit, indices=indices, **kwargs)
    if prefix in {"cubes", "eval21"}:
        return CubeCacheDataset(argument, target=target, density=density, split=split,
                                limit=limit, box_mpc=box_mpc, indices=indices, z_range=z_range)
    if prefix in {"hdf5", "h5"}:
        return Hdf5Dataset(argument, target=target, density=density,
                           split=split, limit=limit, box_mpc=box_mpc,
                           indices=indices, **kwargs)
    if prefix in {"npz", "np"}:
        return NpzDataset(argument, target=target, density=density,
                          split=split, limit=limit, indices=indices,
                          box_mpc=box_mpc, **kwargs)

    path = Path(spec).expanduser()
    if path.suffix.lower() in {".h5", ".hdf5"} and is_cube_cache(path):
        return CubeCacheDataset(path, target=target, density=density, split=split,
                                limit=limit, box_mpc=box_mpc, indices=indices, z_range=z_range)
    if z_range is not None:
        raise ValueError("--z-range needs an eval21 cube cache (eval21 build)")
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return Hdf5Dataset(path, target=target, density=density,
                           split=split, limit=limit, box_mpc=box_mpc,
                           indices=indices, **kwargs)
    return NpzDataset(spec, target=target, density=density,
                      split=split, limit=limit, indices=indices,
                      box_mpc=box_mpc, **kwargs)
