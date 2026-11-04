"""Figures from 21cmEVAL result directories (``eval21 plot``; needs matplotlib).

    eval21 plot cyl      --run results/ --out figs [--models a,b] [--style style.json]
    eval21 plot gallery  --run results/xhi --run results/tb --out figs
    eval21 plot deltas   --run results/ --out figs
    eval21 plot summary  --run results/ --out figs
    eval21 plot all      --run results/ --out figs

``--run`` takes result directories or trees containing them (searched for ``results.json``);
runs of one target on different data are pooled as groups. ``--style`` is a JSON file
``{model: {"label": ..., "color": ..., "ls": ...}}``.
"""
from .common import Catalog, Run, find_runs
from .registry import PlotContext, SkipFigure, figure, load_all, run
from .cylindrical import plot_cylindrical
from .slices import plot_deltas, plot_gallery
from .summary import plot_summary

__all__ = ["Catalog", "Run", "find_runs", "PlotContext", "SkipFigure", "figure", "load_all", "run", "plot_cylindrical", "plot_gallery", "plot_deltas", "plot_summary"]
