"""Loading 21cmEVAL result directories for the figures.

A *run* is one ``eval21 run`` output directory (``results.json`` + ``curves.npz``), with one
target field and one or more models. Runs of the same model, target and data are merged (e.g. a
later run that only adds ``--metrics cylindrical,gallery``); a metric present in two of them is
an error. Several runs of the same target on different data (e.g.
LoReLi's redshift groups, one manifest each) are *groups*: per-sample curves are pooled over them
in the order given. A model must be present in every group of a target to be shown; models
missing somewhere are listed and skipped. A directory without ``results.json`` is searched
recursively, so ``--run results/`` picks up a whole tree (``<model>/<group>/<target>/``).
"""
from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path

import numpy as np

FIELD_NAMES = {"xhi": r"$x_{\rm HI}$", "tb": r"$T_b$"}
FIELD_UNITS = {"xhi": "", "tb": " [mK]"}


def field_name(field: str) -> str:
    return FIELD_NAMES.get(field, field)


class Run:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.results = json.loads((self.path / "results.json").read_text())
        self.settings = self.results.get("settings", {})
        self.target = self.settings.get("target", "xhi")
        dataset = self.results.get("dataset", {})
        self.data = str(dataset.get("path", self.path))
        self.n_samples = dataset.get("n_samples")
        self.models = list(self.results.get("models", {}))
        self._curves = None

    @property
    def group(self) -> str:
        """Short label of the data this run scored (manifest's directory, or the data path)."""
        path = Path(self.data)
        return path.parent.name if path.suffix else path.name

    def curves(self, model: str) -> dict:
        if self._curves is None:
            file = self.path / "curves.npz"
            self._curves = {}
            if file.exists():
                with np.load(file, allow_pickle=False) as d:
                    for key in d.files:
                        name, _, rest = key.partition("__")
                        _, _, short = rest.partition("__")
                        self._curves.setdefault(name, {})[short] = d[key]
        return self._curves.get(model, {})

    def metric(self, model: str, name: str) -> dict:
        return self.results["models"][model]["metrics"].get(name, {})


class Merged:
    """Several runs of one (target, data, model), seen as one."""

    def __init__(self, model: str, runs):
        self.runs = list(runs)
        seen = {}
        for run in self.runs:
            for name in run.results["models"][model]["metrics"]:
                if name in seen:
                    raise SystemExit(f"metric {name!r} of model {model!r} on {run.data} is in both "
                                     f"{seen[name]} and {run.path}")
                seen[name] = run.path
        first = self.runs[0]
        self.settings, self.target, self.data, self.n_samples = first.settings, first.target, first.data, first.n_samples
        self.group = first.group
        self.path = first.path

    def curves(self, model):
        out = {}
        for run in self.runs:
            out.update(run.curves(model))
        return out

    def metric(self, model, name):
        for run in self.runs:
            value = run.results["models"][model]["metrics"].get(name)
            if value is not None:
                return value
        return {}


def find_runs(paths) -> list[Run]:
    runs = []
    for path in paths:
        path = Path(path)
        if (path / "results.json").exists():
            runs.append(Run(path))
            continue
        found = sorted(p.parent for p in path.rglob("results.json"))
        if not found:
            raise SystemExit(f"no 21cmEVAL results under {path}")
        runs.extend(Run(p) for p in found)
    return runs


DEFAULT_COLORS = ["#3182bd", "#de2d26", "#31a354", "#756bb1", "#e6550d", "#08519c", "#a50f15",
                  "#636363", "#17becf", "#bcbd22"]


class Catalog:
    """Runs indexed by target, group and model, with labels and line styles."""

    def __init__(self, runs, models=None, style=None, quiet=False):
        self.runs = list(runs)
        style = style or {}
        seen = []
        for run in self.runs:
            for m in run.models:
                if m not in seen:
                    seen.append(m)
        wanted = [m for m in (models or seen)]
        unknown = [m for m in wanted if m not in seen]
        if unknown:
            raise SystemExit(f"models {unknown} are in none of the runs; available: {seen}")
        self.targets = []
        self.groups = {}                       # target -> [group data keys]
        parts = {}                             # (target, data, model) -> [runs]
        for run in self.runs:
            if run.target not in self.groups:
                self.targets.append(run.target)
                self.groups[run.target] = []
            if run.data not in self.groups[run.target]:
                self.groups[run.target].append(run.data)
            for m in run.models:
                parts.setdefault((run.target, run.data, m), []).append(run)
        self.by_key = {key: Merged(key[2], runs) for key, runs in parts.items()}
        complete = [m for m in wanted
                    if all((t, g, m) in self.by_key for t in self.targets for g in self.groups[t])]
        self.skipped = [m for m in wanted if m not in complete]
        if self.skipped and not quiet:
            print("incomplete (skipped):", self.skipped)
        if not complete:
            raise SystemExit("no model has results for every target and group")
        self.models = complete
        self.style = {}
        palette = itertools.cycle(DEFAULT_COLORS)
        for i, m in enumerate(self.models):
            given = style.get(m, {})
            if i == 0 and not given:
                default = {"label": m, "color": "0.15", "ls": "--"}
            else:
                default = {"label": m, "ls": "-"}
                if "color" not in given:
                    default["color"] = next(palette)
            self.style[m] = {**default, **given}

    # -- style -------------------------------------------------------------- #
    def label(self, m):
        return self.style[m]["label"]

    def color(self, m):
        return self.style[m]["color"]

    def ls(self, m):
        return self.style[m]["ls"]

    def tick(self, m):
        return self.label(m).replace(", ", ",\n").replace(" (", "\n(").replace(": ", ":\n")

    # -- data --------------------------------------------------------------- #
    def runs_for(self, target, model):
        return [self.by_key[(target, g, model)] for g in self.groups[target]]

    def group_labels(self, target):
        return [self.by_key[(target, g, self.models[0])].group for g in self.groups[target]]

    def group_sizes(self, target, key="rmse_per_sample", metric_prefix=None):
        sizes = []
        for run in self.runs_for(target, self.models[0]):
            c = run.curves(self.models[0])
            sizes.append(len(np.atleast_1d(c[key])) if key in c else int(run.n_samples or 0))
        return sizes

    def curves(self, target, model, group_index=0):
        return self.runs_for(target, model)[group_index].curves(model)

    def pooled(self, model, key, target="xhi"):
        """Concatenate a per-sample array over the groups (axis 0); empty if absent."""
        parts = []
        for run in self.runs_for(target, model):
            c = run.curves(model)
            if key in c:
                parts.append(np.atleast_1d(c[key]).astype(float))
        return np.concatenate(parts, 0) if parts else np.array([])

    def has(self, target, key):
        return target in self.groups and key in self.curves(target, self.models[0])


def med_band(a, axis=0):
    a = np.asarray(a, float)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis), np.nanpercentile(a, 16, axis), np.nanpercentile(a, 84, axis)


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# --------------------------------------------------------------------------------------------- #
# Legends: one shared legend above the panels instead of one per panel
# --------------------------------------------------------------------------------------------- #
# Model labels can be long ("8 blk: pointwise, LoReLi v3, oracle history"); a legend in every
# small panel hides the data. Figures therefore collect their legend entries once
# (``shared_legend``), and ``save`` places them in one row (wrapping into more rows as needed)
# above all panels, outside the axes, with the figure title above it. Per-panel numbers that used
# to sit in legend labels (W1, |dtau|, ...) go to the bottom of each panel (``pane_values``), as a
# short line sample in the model's colour/style followed by the value -- no names repeated.

def shared_legend(fig, axes=None, *, handles=None, labels=None, extra=(), drop=(), fontsize=8.5,
                  ncol=None, title=None):
    """Collect the legend entries of ``axes`` (default: all axes of ``fig``), remove the per-axes
    legends and register one figure legend for :func:`save` to place above the panels.

    Entries are deduplicated by label in order of first appearance (row-major over the axes).
    ``extra``: further ``(handle, label)`` pairs appended at the end (e.g. :func:`style_key`).
    ``drop``: labels to leave out. Explicit ``handles``/``labels`` replace the collection.
    """
    axes = list(fig.axes if axes is None else np.ravel(axes))
    if handles is None:
        seen = {}
        for ax in axes:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                if lab and not lab.startswith("_") and lab not in seen and lab not in drop:
                    seen[lab] = h
        labels, handles = list(seen), list(seen.values())
    else:
        handles = list(handles)
        labels = list(labels) if labels is not None else [h.get_label() for h in handles]
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None and (ax in axes):
            legend.remove()
    for h, lab in extra:
        handles.append(h)
        labels.append(lab)
    fig._eval21_legend = {"handles": handles, "labels": labels, "fontsize": fontsize, "ncol": ncol,
                          "title": title}
    return handles, labels


def model_handles(cat, models=None, truth=True, lw=1.6, markers=None, kind="line"):
    """Proxy handles: ``truth`` (black, thick) followed by each model's colour and style.

    ``markers``: one marker per model drawn on the line sample (a figure that shows the models
    as lines in some panels and as scatter markers in others needs only one entry per model).
    ``kind="patch"`` gives filled boxes (bar and box plots), ``kind="marker"`` points only.
    """
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    models = cat.models if models is None else models
    out = []
    if truth:
        out.append(mpatches.Patch(color="k", label="truth") if kind == "patch"
                   else mlines.Line2D([], [], color="k", lw=2.2, label="truth"))
    for i, m in enumerate(models):
        if kind == "patch":
            out.append(mpatches.Patch(color=cat.color(m), alpha=0.85, label=cat.label(m)))
        elif kind == "marker":
            out.append(mlines.Line2D([], [], color=cat.color(m), ls="none", marker="o", ms=7, label=cat.label(m)))
        else:
            marker = markers[i % len(markers)] if markers else None
            out.append(mlines.Line2D([], [], color=cat.color(m), ls=cat.ls(m), lw=lw, marker=marker, ms=5,
                                     label=cat.label(m)))
    return out


def style_key(entries, color="0.35", lw=1.4):
    """Grey key entries for a second encoding: ``[("k = 0.11", "-"), ("k = 0.22", "--")]`` for
    line styles, or ``[("median", {"ls": "-"}), ("16-84 %", {"kind": "band"})]``."""
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    out = []
    for label, style in entries:
        style = {"ls": style} if isinstance(style, str) else dict(style)
        if style.pop("kind", None) == "band":
            out.append((mpatches.Patch(color=style.pop("color", "0.75"), alpha=style.pop("alpha", 0.6),
                                       **style), label))
        elif "marker" in style and "ls" not in style:
            out.append((mlines.Line2D([], [], color=style.pop("color", color), ls="none", **style), label))
        else:
            out.append((mlines.Line2D([], [], color=style.pop("color", color), lw=style.pop("lw", lw),
                                      **style), label))
    return out


def pane_values(ax, values, label=None, loc="below", fmt="{:.3g}", fontsize=7.5):
    """Per-panel statistics as one line under the panel: ``label  ── 0.127  - - 0.079``.

    ``values``: ``[(color, linestyle, value), ...]`` (value a number or text; None/NaN skipped).
    ``loc="below"`` (default) puts the line under the x-axis label, outside the data; ``save``
    measures the label and re-runs the layout so rows do not collide. Any matplotlib location
    (``"lower right"``, ...) places it inside the panel instead.
    """
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea

    children = []
    if label:
        children.append(TextArea(label, textprops={"color": "0.25", "fontsize": fontsize}))
    for color, ls, value in values:
        if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
            continue
        sample = DrawingArea(16, fontsize * 0.8, 0, 0)
        sample.add_artist(Line2D([0, 16], [fontsize * 0.35] * 2, color=color, ls=ls, lw=1.4))
        text = fmt.format(value) if isinstance(value, (int, float, np.floating, np.integer)) else str(value)
        children += [sample, TextArea(text, textprops={"color": "0.1", "fontsize": fontsize})]
    if len(children) <= (1 if label else 0):
        return None
    pack = HPacker(children=children, align="center", pad=0, sep=3)
    if loc == "below":
        box = AnchoredOffsetbox(loc="upper center", child=pack, pad=0.1, borderpad=0.0, frameon=False,
                                bbox_to_anchor=(0.5, 0.0), bbox_transform=ax.transAxes)
        ax.__dict__.setdefault("_eval21_below", []).append(box)
    else:
        box = AnchoredOffsetbox(loc=loc, child=pack, pad=0.25, borderpad=0.35, frameon=True)
        box.patch.set(facecolor="white", alpha=0.85, edgecolor="none")
    ax.add_artist(box)
    return box


def _place_below_boxes(fig) -> bool:
    """Move ``pane_values(loc="below")`` lines under each panel's x-axis label; True if any."""
    from matplotlib.transforms import ScaledTranslation

    holders = [a for a in fig.axes if getattr(a, "_eval21_below", None)]
    if not holders:
        return False
    for _ in range(2):                        # place, re-layout (rows make room), place again
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in holders:
            bottom = ax.get_window_extent(renderer).y0
            parts = [t.get_window_extent(renderer).y0 for t in ax.get_xticklabels() if t.get_visible() and t.get_text()]
            if ax.xaxis.label.get_text():
                parts.append(ax.xaxis.label.get_window_extent(renderer).y0)
            drop_pt = (bottom - min(parts, default=bottom)) * 72.0 / fig.dpi + 2.0
            for box in ax._eval21_below:
                box.set_bbox_to_anchor((0.5, 0.0), ax.transAxes + ScaledTranslation(0, -drop_pt / 72.0,
                                                                                    fig.dpi_scale_trans))
        fig.canvas.draw()                     # anchored boxes update their offset only on draw
        engine = fig.get_layout_engine()
        if engine is None or type(engine).__name__ in ("TightLayoutEngine", "PlaceHolderLayoutEngine"):
            # figures that used one-shot fig.tight_layout(): redo it with the value lines counted
            # (they are axes artists) plus a little air before the next row's titles
            fig.tight_layout(h_pad=1.6, rect=getattr(fig, "_eval21_rect", None))
        else:
            engine.execute(fig)
    return True


def wrap_label(text: str, width: int = 22) -> str:
    """Break a long model label into lines of at most ``width`` characters (panel titles, row labels)."""
    import textwrap

    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False)) or str(text)


def _place_shared_legend(fig, spec) -> None:
    """Freeze the layout, then put the registered legend above all panels and the title above it."""
    fig.canvas.draw()
    try:
        fig.set_layout_engine("none")          # keep the positions the layout engine found
    except Exception:                          # pragma: no cover (matplotlib < 3.6)
        pass
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    axes = [a for a in fig.axes if a.get_visible()]
    if not axes:
        return
    boxes = [a.get_tightbbox(renderer).transformed(inv) for a in axes]
    top = max(b.y1 for b in boxes)
    x0 = min(a.get_position().x0 for a in axes)
    x1 = max(a.get_position().x1 for a in axes)
    n = len(spec["labels"])
    ncol = spec["ncol"] or n
    while True:
        legend = fig.legend(spec["handles"], spec["labels"], loc="lower center",
                            bbox_to_anchor=((x0 + x1) / 2, top + 0.004), bbox_transform=fig.transFigure,
                            ncol=max(1, ncol), frameon=False, fontsize=spec["fontsize"],
                            handlelength=2.4, columnspacing=1.6, handletextpad=0.6,
                            title=spec.get("title"), title_fontsize=spec["fontsize"])
        fig.canvas.draw()
        width = legend.get_window_extent(renderer).transformed(inv).width
        if width <= 0.98 or ncol <= 1 or spec["ncol"]:
            break
        legend.remove()
        ncol -= 1                          # wrap into more rows until it fits the figure width
    sup = getattr(fig, "_suptitle", None)
    if sup is not None:
        leg_top = legend.get_window_extent(renderer).transformed(inv).y1
        sup.set_y(leg_top + 0.006)
        sup.set_verticalalignment("bottom")


def save(fig, out: Path, name: str, dpi=200, quiet=False):
    out.mkdir(parents=True, exist_ok=True)
    spec = getattr(fig, "_eval21_legend", None)
    below = _place_below_boxes(fig)
    if spec and spec["handles"]:
        _place_shared_legend(fig, spec)
    for ext in ("png", "pdf"):
        # "tight" keeps the shared legend / title placed above the panels inside the file
        fig.savefig(out / f"{name}.{ext}", dpi=dpi if ext == "png" else None,
                    bbox_inches="tight" if (spec or below) else None, pad_inches=0.08)
    fig.clf()
    import matplotlib.pyplot as plt
    plt.close(fig)
    if not quiet:
        print("wrote", out / f"{name}.png")
    return out / f"{name}.png"
