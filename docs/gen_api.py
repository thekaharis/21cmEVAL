"""Generate docs/api.md from the source tree (signatures + docstrings)."""
import ast
import pathlib
import textwrap

DOCS = pathlib.Path(__file__).resolve().parent
ROOT = DOCS.parent / "src" / "eval21"

MODULES = [
    ("eval21", "__init__.py"),
    ("eval21.spec", "spec.py"),
    ("eval21.settings", "settings.py"),
    ("eval21.registry", "registry.py"),
    ("eval21.runner", "runner.py"),
    ("eval21.report", "report.py"),
    ("eval21.store", "store.py"),
    ("eval21.io", "io/__init__.py"),
    ("eval21.io.runfile", "io/runfile.py"),
    ("eval21.io.cache", "io/cache.py"),
    ("eval21.physics", "physics.py"),
    ("eval21.selftest", "selftest.py"),
    ("eval21.data", "data/__init__.py"),
    ("eval21.data.base", "data/base.py"),
    ("eval21.data.npz", "data/npz.py"),
    ("eval21.data.hdf5", "data/hdf5.py"),
    ("eval21.data.cubes", "data/cubes.py"),
    ("eval21.data.synthetic", "data/synthetic.py"),
    ("eval21.build", "build/__init__.py"),
    ("eval21.build.grid", "build/grid.py"),
    ("eval21.build.sources", "build/sources.py"),
    ("eval21.adapters", "adapters/__init__.py"),
    ("eval21.adapters.base", "adapters/base.py"),
    ("eval21.adapters.python", "adapters/python.py"),
    ("eval21.adapters.torch", "adapters/torch.py"),
    ("eval21.adapters.command", "adapters/command.py"),
    ("eval21.adapters.fno21cm", "adapters/fno21cm.py"),
    ("eval21.adapters.toy", "adapters/toy.py"),
    ("eval21.metrics.base", "metrics/base.py"),
    ("eval21.visualization", "visualization/__init__.py"),
    ("eval21.visualization.registry", "visualization/registry.py"),
    ("eval21.visualization.common", "visualization/common.py"),
]


def first_para(node):
    doc = ast.get_docstring(node) or ""
    para = doc.strip().split("\n\n")[0]
    return " ".join(line.strip() for line in para.splitlines())


def esc(text):
    return str(text).replace("|", "\\|")


def dataclass_fields(node):
    out = []
    for s in node.body:
        if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name):
            default = ast.unparse(s.value) if s.value is not None else ""
            out.append((s.target.id, ast.unparse(s.annotation), default))
    return out


def sig(node, drop_self=True):
    args = ast.unparse(node.args)
    if drop_self and args.startswith("self, "):
        args = args[len("self, "):]
    elif drop_self and args == "self":
        args = ""
    return args


def is_public(name):
    return not name.startswith("_")


out = ["# API reference", "",
       "Public objects and functions, generated from the source tree. Signatures are exact;",
       "docstrings are the first paragraph of the real docstring. Everything not listed here is",
       "private and may change without notice.", ""]

for modname, rel in MODULES:
    path = ROOT / rel
    src = path.read_text()
    tree = ast.parse(src)
    out += [f"## `{modname}`", ""]
    mod_doc = first_para(tree)
    if mod_doc:
        out += [mod_doc, ""]
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and is_public(node.name):
            body.append(("fn", node))
        elif isinstance(node, ast.ClassDef) and is_public(node.name):
            body.append(("class", node))
    if not body:
        if modname == "eval21.visualization":
            out += ["```python",
                    "from eval21.visualization import (",
                    "    Catalog, Run, find_runs, PlotContext, SkipFigure,",
                    "    figure, load_all, run,",
                    "    plot_cylindrical, plot_gallery,",
                    ")",
                    "```", "",
                    "The package re-exports the figure registry (`figure`, `load_all`, `run`,",
                    "`PlotContext`, `SkipFigure`), the result loader (`Run`, `Catalog`, `find_runs`)",
                    "and the hand-written figures (`plot_cylindrical`, `plot_gallery`).",
                    "Importing it does not import matplotlib.", ""]
        elif modname == "eval21":
            out += ["```python",
                    "from eval21 import Settings, Sample, Meta, evaluate, report, __version__",
                    "```", "",
                    "The package re-exports the runner and the two data contracts. `__version__` is",
                    "`0.1.0`; the distribution name is `21cmeval`.", ""]
        else:
            out += ["_No public names._", ""]
        continue
    for kind, node in body:
        if kind == "fn":
            doc = first_para(node)
            body.append
            ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            out += ["```python", f"def {node.name}({sig(node)}){ret}", "```"]
            if doc:
                out += ["", doc]
            out += [""]
        else:
            init = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            args = f"({sig(init)})" if init and sig(init) else ""
            header = f"class {node.name}{args}" + (f"  # ({bases})" if bases else "")
            out += ["```python", header, "```"]
            doc = first_para(node)
            if doc:
                out += ["", doc]
            out += [""]
            fields = dataclass_fields(node)
            if fields:
                out += ["| field | type | default |", "|---|---|---|"]
                for name, ann, default in fields:
                    out.append(f"| `{name}` | `{esc(ann)}` | `{esc(default) or '—'}` |")
                out += [""]
            methods = [n for n in node.body if isinstance(n, ast.FunctionDef) and is_public(n.name) and n.name != "__init__"]
            if methods:
                out += ["| method | description |", "|---|---|"]
                for m in methods:
                    d = esc(first_para(m))
                    kind = "property" if any(getattr(dec, "id", "") == "property" for dec in m.decorator_list) else ""
                    name = f"`{m.name}`" if kind else f"`{esc(m.name)}({esc(sig(m))})`"
                    out.append(f"| {name} | {kind + ': ' if kind else ''}{d} |")
                out += [""]


# --------------------------------------------------------------------------- metric classes
out += ["## Registered metric classes", "",
        "Each class is decorated with `@register(\"name\")` in its module; the name is what",
        "`--metrics` and `[metrics] enabled` accept.", "",
        "| name | class | module | needs | targets | pointwise |",
        "|---|---|---|---|---|---|"]
for path in sorted((ROOT / "metrics").glob("*.py")):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        name = None
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and getattr(dec.func, "id", "") == "register":
                name = ast.literal_eval(dec.args[0])
        if not name:
            continue
        attrs = {}
        for s in node.body:
            if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name):
                try:
                    attrs[s.targets[0].id] = ast.literal_eval(s.value)
                except Exception:
                    attrs[s.targets[0].id] = ast.unparse(s.value)
        needs = attrs.get("needs", "()")
        targets = attrs.get("targets", "None")
        pointwise = attrs.get("pointwise", "False")
        out.append(f"| `{name}` | `{node.name}` | `eval21.metrics.{path.stem}` | "
                   f"`{esc(needs)}` | `{esc(targets)}` | `{esc(pointwise)}` |")
out += [""]

(DOCS / "api.md").write_text("\n".join(out).rstrip() + "\n")
print("wrote", DOCS / "api.md", len(out), "lines")
