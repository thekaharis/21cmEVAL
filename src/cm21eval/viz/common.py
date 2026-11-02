"""Loading 21cmEVAL result directories for the figures.

A *run* is one ``cm21eval run`` output directory (``results.json`` + ``curves.npz``), with one
target field and one or more models. Runs of the same model, target and data are merged (e.g. a
later run that only adds ``--metrics cylindrical,gallery``); a metric present in two of them is
an error. Several runs of the same target on different data (e.g.
LoReLi's redshift groups, one manifest each) are *groups*: per-sample curves are pooled over them
in the order given. A model must be present in every group of a target to be shown; models
missing somewhere are listed and skipped. A directory without ``results.json`` is searched
recursively, so ``--run results/`` picks up a whole tree (``<model>/<group>/<target>/``).
"""
from __future__ import annotations

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
        palette = iter(DEFAULT_COLORS)
        for i, m in enumerate(self.models):
            given = style.get(m, {})
            if i == 0 and not given:
                default = {"label": m, "color": "0.15", "ls": "--"}
            else:
                default = {"label": m, "color": next(palette), "ls": "-"}
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


def save(fig, out: Path, name: str, dpi=200, quiet=False):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=dpi if ext == "png" else None)
    fig.clf()
    import matplotlib.pyplot as plt
    plt.close(fig)
    if not quiet:
        print("wrote", out / f"{name}.png")
    return out / f"{name}.png"
