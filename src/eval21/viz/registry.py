"""Figure registry: every figure is a function registered under a short name.

    @figure("ps_overlay", group="spectra", needs=("curves",), help="...")
    def ps_overlay(ctx: PlotContext) -> list[Path]:
        ...

``eval21 plot <name> [<name> ...] --run results/ --out figs/`` runs the named figures; ``all``
runs every figure whose needs are met (skipping, with a reason, the others); ``list`` prints
the registry. ``needs`` declares what a figure reads: ``"curves"`` (metric results and
``curves.npz``, always present) or ``"store"`` (cubes written by ``run --save-predictions``).
Modules in this package are imported lazily by :func:`load_all`, so a new figure only has to
live in a module listed in ``FIGURE_MODULES``.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

FIGURE_MODULES = (
    "cylindrical", "slices", "summary",
    "spectra", "physical", "bubbles", "diagnostics", "maps", "ensemble",
)


@dataclass
class Figure:
    name: str
    func: Callable
    group: str = "misc"
    needs: tuple = ("curves",)
    help: str = ""
    default: bool = True          # False: drawn only when named explicitly (not by `all` or its group)


@dataclass
class PlotContext:
    """Everything a figure function gets."""

    catalog: object                      # viz.common.Catalog (results, labels, colours)
    out: Path                            # output directory for this figure set
    store: object = None                 # store.PredictionStore or None
    ids: list | None = None              # sample ids requested with --ids
    title: str = ""                      # data-set name for titles
    options: dict = field(default_factory=dict)   # --opt key=value
    quiet: bool = False

    def opt(self, key, default=None, cast=None):
        value = self.options.get(key, default)
        return cast(value) if (cast is not None and value is not None and key in self.options) else value


_FIGURES: dict[str, Figure] = {}


def figure(name: str, group: str = "misc", needs=("curves",), help: str = "", default: bool = True):
    def decorator(func):
        if name in _FIGURES:
            raise ValueError(f"figure {name!r} is already registered")
        _FIGURES[name] = Figure(name, func, group, tuple(needs), help or (func.__doc__ or "").strip().split("\n")[0],
                                default)
        return func
    return decorator


def load_all() -> dict[str, Figure]:
    for module in FIGURE_MODULES:
        try:
            importlib.import_module(f"{__package__}.{module}")
        except ModuleNotFoundError as exc:
            if exc.name != f"{__package__}.{module}":
                raise
    return dict(_FIGURES)


def run(names, ctx: PlotContext) -> dict:
    """Run the named figures (``all``: every default one). Returns {name: written | reason}.

    Figures registered with ``default=False`` run only when named themselves.
    """
    figures = load_all()
    if not names or "all" in names:
        selected = [n for n, f in figures.items() if f.default]
    else:
        unknown = [n for n in names if n not in figures and n not in {f.group for f in figures.values()}]
        if unknown:
            raise SystemExit(f"unknown figure(s) {unknown}; see `eval21 plot list`")
        selected = [n for n in figures if n in names or (figures[n].group in names and figures[n].default)]
    done = {}
    explicit = names and "all" not in names
    for name in selected:
        fig = figures[name]
        if "store" in fig.needs and ctx.store is None:
            done[name] = "skipped: needs stored cubes (run with --save-predictions)"
        else:
            try:
                done[name] = fig.func(ctx) or []
            except SkipFigure as reason:
                done[name] = f"skipped: {reason}"
        if not ctx.quiet and isinstance(done[name], str) and (explicit or "needs stored" not in done[name]):
            print(f"{name}: {done[name]}")
    return done


class SkipFigure(Exception):
    """Raised by a figure whose inputs are absent from the runs (e.g. a metric that was not run)."""
