"""``eval21 build``: raw simulation lightcones -> eval21 npz cubes + manifest(s).

    from eval21.build import BuildOptions, build, make_source
    options = BuildOptions(out="cubes/21cmfast_nearest", los="nearest", z_grid="5.001,24.97,512")
    build(make_source("21cmfast", "/data/21cmfast", options), options)

One module per source format (``cmfast``: 21cmFAST raw HDF5 lightcones, ``loreli``: LoReLi II
``.dat`` lightcones, ``generic``: a folder of per-lightcone npz/h5 files with a declared field
mapping) and a common planner/writer (``common``). See the README section "Building datasets".
"""
from __future__ import annotations

from .common import BUILD_SCHEMA, LOS_MODES, BuildError, BuildOptions, Source, build

FORMATS = ("21cmfast", "loreli", "generic")


def make_source(fmt: str, root, options: BuildOptions) -> Source:
    fmt = fmt.lower()
    if fmt in ("21cmfast", "cmfast", "21cm"):
        from .cmfast import Cmfast21Source

        return Cmfast21Source(root, options)
    if fmt in ("loreli", "loreli2", "licorice"):
        from .loreli import LoreliSource

        return LoreliSource(root, options)
    if fmt in ("generic", "npz", "h5", "hdf5"):
        from .generic import GenericSource

        return GenericSource(root, options)
    raise ValueError(f"unknown source format {fmt!r}; known: {', '.join(FORMATS)}")


__all__ = ["BUILD_SCHEMA", "FORMATS", "LOS_MODES", "BuildError", "BuildOptions", "Source", "build",
           "make_source"]
