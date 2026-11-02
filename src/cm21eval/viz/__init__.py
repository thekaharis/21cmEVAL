"""Figures from 21cmEVAL result directories (``cm21eval plot``; needs matplotlib).

    cm21eval plot cyl      --run results/ --out figs [--models a,b] [--style style.json]
    cm21eval plot gallery  --run results/xhi --run results/tb --out figs
    cm21eval plot deltas   --run results/ --out figs
    cm21eval plot summary  --run results/ --out figs
    cm21eval plot all      --run results/ --out figs

``--run`` takes result directories or trees containing them (searched for ``results.json``);
runs of one target on different data are pooled as groups. ``--style`` is a JSON file
``{model: {"label": ..., "color": ..., "ls": ...}}``.
"""
from .common import Catalog, Run, find_runs
from .cylindrical import plot_cylindrical
from .slices import plot_deltas, plot_gallery
from .summary import plot_summary

__all__ = ["Catalog", "Run", "find_runs", "plot_cylindrical", "plot_gallery", "plot_deltas", "plot_summary"]
