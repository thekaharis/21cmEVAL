"""Command line interface.

    eval21 build sims/ --out cubes.h5 --format 21cmfast --z-grid uniform:5,25,512
    eval21 run --data cubes/ --model toy=synthetic:1.5 --out results/
    eval21 run --data cubes.h5 --model gen=python:gen.py --draws 16 --mode generative
    eval21 run --config eval.toml
    eval21 describe --data cubes/
    eval21 list-metrics
    eval21 selftest
    eval21 demo --out demo/
    eval21 plot all --run results/ --out figs/
    eval21 plot list
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import adapters, registry, report, selftest
from . import metrics as _metrics  # noqa: F401  (registers built-ins)
from .data import open_data
from .runner import environment, evaluate
from .settings import Settings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval21",
        description="End-to-end evaluation for 21cm field emulators.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="evaluate one or more models")
    run.add_argument("--config", type=Path, help="TOML file with run settings")
    run.add_argument("--data", help="data path (dir, manifest.json, .npz, .h5)")
    run.add_argument("--model", action="append", default=[],
                     metavar="NAME=SPEC",
                     help="model spec, e.g. fno=fno21cm:ckpt.pt (repeatable)")
    run.add_argument("--metrics", help="comma list; default all")
    run.add_argument("--out", type=Path, help="output directory")
    run.add_argument("--title", default="21cmEVAL report")
    run.add_argument("--split", default=None)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--box-mpc", type=float, default=None,
                     help="transverse box size; default: read from the data "
                          "file, else the settings value")
    run.add_argument("--expect-shape", default=None, metavar="NX,NY,NZ",
                     help="enforce the sample shape (needed for checkpoints "
                          "whose run_metadata.json predates spatial_shape)")
    run.add_argument("--expect-nz", type=int, default=None,
                     help="enforce the LOS length; NX,NY come from the data")
    run.add_argument("--indices", default=None,
                     help="row indices to evaluate, e.g. '0-199,300'")
    run.add_argument("--indices-json", type=Path, default=None,
                     help="run_metadata.json: use its split.<split>_indices")
    run.add_argument("--target", default=None)
    run.add_argument("--density", default=None)
    run.add_argument("--fno-root", default=None)
    run.add_argument("--device", default=None)
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                     help="override a settings field (repeatable)")
    run.add_argument("--z-range", default=None, metavar="ZMIN,ZMAX",
                     help="cube caches: evaluate this LOS window; samples not valid over it are dropped")
    run.add_argument("--draws", type=int, default=None,
                     help="predictions per sample for stochastic/generative models (enables 'ensemble')")
    run.add_argument("--mode", choices=["pointwise", "generative"], default=None,
                     help="generative: skip voxel-by-voxel metrics and pointwise keys (r, IoU, RMSE)")
    run.add_argument("--score-draw", choices=["first", "mean"], default=None,
                     help="what the standard metrics score with --draws: one draw or the ensemble mean")
    run.add_argument("--legacy-dir", action="store_true",
                     help="write the legacy results/ directory (results.json, curves.npz) "
                          "instead of one eval21-run/1 HDF5 file per model")
    run.add_argument("--overwrite", action="store_true",
                     help="replace existing run files / cache entries")
    run.add_argument("--cache", type=Path, default=None, metavar="PATH",
                     help="optional HDF5 inference cache for cubes; a directory gets "
                          "<target>.cache.h5, a .h5 path is used directly "
                          "(default: no cubes are stored)")
    run.add_argument("--save-predictions", type=Path, default=None, metavar="DIR",
                     help="legacy prediction store directory (or an .h5 path for the new "
                          "cache); stores the first --save-n samples")
    run.add_argument("--save-n", type=int, default=8, help="samples to store (default 8; 0 = all)")
    run.add_argument("--save-ids", default=None, help="comma list of sample ids to store instead")
    run.add_argument("--quiet", action="store_true")

    build = sub.add_parser("build", help="build an eval21 cube cache (HDF5) from a folder of simulations")
    build.add_argument("folder", type=Path, nargs="?", help="folder of simulation files")
    build.add_argument("--out", type=Path, required=True, help="cache file (.h5)")
    build.add_argument("--format", default="auto",
                       help="21cmfast, loreli, npz, python:FILE.py[:Class] or auto (default)")
    build.add_argument("--fields", default=None,
                       help="comma list (default per source: 21cmfast density,xhi; loreli density,xhi,tb)")
    build.add_argument("--targets", default=None, help="fields stored as targets (default xhi,tb)")
    build.add_argument("--z-grid", default=None,
                       help="native (default), uniform:ZMIN,ZMAX,N, comoving:ZMIN,ZMAX[,N] or FILE[.npy|.h5:key]")
    build.add_argument("--transverse", type=int, default=None, help="resample the transverse plane to N cells")
    build.add_argument("--splits", default=None,
                       help="'test' (all) or e.g. train=0.8,val=0.1,test=0.1 (seeded shuffle)")
    build.add_argument("--split-file", default=None,
                       help="JSON {split: [ids]}, {id: split} or run_metadata.json (<split>_cone_ids)")
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--limit", type=int, default=None)
    build.add_argument("--ids", default=None, help="comma list of sample ids to include")
    build.add_argument("--shard", type=int, default=0)
    build.add_argument("--num-shards", type=int, default=1)
    build.add_argument("--merge", action="store_true", help="join the --num-shards shard files into --out")
    build.add_argument("--remove-shards", action="store_true", help="with --merge: delete the shards")
    build.add_argument("--workers", type=int, default=1)
    build.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    build.add_argument("--compress", choices=["gzip", "lzf"], default=None)
    build.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                       help="source option, e.g. --option l_perp=300 --option pattern='*.h5'")
    build.add_argument("--quiet", action="store_true")

    describe = sub.add_parser("describe", help="summarise a dataset")
    describe.add_argument("--data", required=True)
    describe.add_argument("--target", default="xhi")
    describe.add_argument("--limit", type=int, default=3)
    describe.add_argument("--box-mpc", type=float, default=None)

    list_metrics = sub.add_parser("list-metrics", help="list registered metrics")
    list_metrics.add_argument("--json", action="store_true")

    sub.add_parser("selftest", help="run analytic self-checks")

    demo = sub.add_parser("demo", help="write synthetic data and score a toy model")
    demo.add_argument("--out", type=Path, default=Path("demo"))
    demo.add_argument("--n-samples", type=int, default=4)
    demo.add_argument("--shape", default="64,64,64")
    demo.add_argument("--quiet", action="store_true")

    plot = sub.add_parser("plot", help="figures from result directories (needs matplotlib)")
    plot.add_argument("figures", nargs="+", metavar="FIGURE",
                      help="figure or group names, 'all', or 'list' (prints the registry)")
    plot.add_argument("--run", action="append", default=[], type=Path,
                      help="result directory or a tree of them (repeatable)")
    plot.add_argument("--out", type=Path, default=None)
    plot.add_argument("--store", action="append", default=[], type=Path,
                      help="prediction store or .h5 cache (repeatable; default: 'predictions' "
                           "directories and *.cache.h5 next to the results)")
    plot.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                      help="figure option (repeatable), e.g. --opt vmax=0.5")
    plot.add_argument("--models", default=None, help="comma list and order of models (default: all)")
    plot.add_argument("--style", type=Path, default=None,
                      help='JSON {model: {"label": ..., "color": ..., "ls": ...}}')
    plot.add_argument("--ids", default=None, help="comma list of sample ids for gallery/deltas")
    plot.add_argument("--title", default="", help="data set named in figure titles")
    plot.add_argument("--vmax", type=float, default=None, help="cyl: colour saturation (same as --opt vmax=)")
    plot.add_argument("--dead", type=float, default=None, help="cyl: white band (same as --opt dead=)")
    plot.add_argument("--quiet", action="store_true")

    validate = sub.add_parser("validate", help="check eval21-run/1 files")
    validate.add_argument("files", nargs="+", type=Path)
    validate.add_argument("--strict", action="store_true",
                          help="also check that every dataset appears in /manifest")

    args = parser.parse_args(argv)
    if args.command == "plot":
        return _plot(args)
    if args.command == "run":
        return _run(args)
    if args.command == "build":
        return _build(args)
    if args.command == "describe":
        return _describe(args)
    if args.command == "list-metrics":
        return _list(args)
    if args.command == "selftest":
        return selftest.run()
    if args.command == "demo":
        return _demo(args)
    if args.command == "validate":
        return _validate(args)
    parser.error(f"unknown command {args.command}")
    return 2


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover (python < 3.11)
        raise SystemExit("--config requires Python 3.11+ (tomllib)") from None
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _run(args) -> int:
    config = _load_config(args.config)
    run_section = config.get("run", {})

    def pick(cli_value, key, default=None):
        if cli_value is not None:
            return cli_value
        return run_section.get(key, default)

    data_spec = pick(args.data, "data")
    if not data_spec:
        raise SystemExit("no data: pass --data or [run].data in the config")
    out = Path(pick(args.out, "out", "results"))
    target = pick(args.target, "target", "xhi")
    density = pick(args.density, "density", "density")
    split = pick(args.split, "split", None)
    limit = pick(args.limit, "limit", None)
    box_override = pick(args.box_mpc, "box_mpc", None)
    fno_root = pick(args.fno_root, "fno_root", None)
    device = pick(args.device, "device", None)
    models = dict(config.get("models", {}))
    for item in args.model:
        if "=" not in item:
            raise SystemExit(f"--model expects NAME=SPEC, got {item!r}")
        name, spec = item.split("=", 1)
        models[name] = spec
    if not models:
        raise SystemExit("no models: pass --model NAME=SPEC or [models]")
    if (not getattr(args, "legacy_dir", False) and out.suffix.lower() in {".h5", ".hdf5"}
            and len(models) > 1):
        raise SystemExit("a single .h5 --out holds one model; pass a directory or one --model")

    settings_values = dict(config.get("settings", {}))
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        settings_values[key] = _parse_value(value)
    for key in ("draws", "mode", "score_draw"):
        value = getattr(args, key, None)
        if value is not None:
            settings_values[key] = value
    # --target/--density pick the data field, the adapter output and what the metrics score:
    # one value for all three (an explicit --set target=... wins)
    for key, value in (("target", target), ("density", density)):
        if key in settings_values and settings_values[key] != value and getattr(args, key, None) is not None:
            raise SystemExit(f"--{key} {value} conflicts with --set {key}={settings_values[key]}")
        settings_values.setdefault(key, value)
    target, density = settings_values["target"], settings_values["density"]
    settings = Settings.from_dict(settings_values)

    metrics_section = config.get("metrics", {})
    if args.metrics:
        names = [part.strip() for part in args.metrics.split(",") if part.strip()]
    elif metrics_section.get("enabled"):
        names = list(metrics_section["enabled"])
    else:
        names = ["all"]
    options = config.get("metric_options", {})

    z_range = pick(getattr(args, "z_range", None), "z_range", None)
    if isinstance(z_range, str):
        z_range = tuple(float(v) for v in z_range.split(","))
    extra = {} if z_range is None else {"z_range": z_range}
    dataset = open_data(
        data_spec, target=target, density=density, split=None,
        limit=None, box_mpc=box_override, **extra)
    selection = _resolve_selection(dataset, args, run_section, split)
    if selection is not None:
        # An explicit row selection already encodes the split; a second split
        # filter would either be a no-op or silently drop rows.
        dataset.indices = selection
        dataset.split = None
    else:
        dataset.split = split
    dataset.limit = limit
    try:
        info = dataset.describe()
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    _preflight_cache_contracts(models, info)
    expect_shape = _resolve_expect_shape(args, info)
    if not args.quiet:
        print(f"data   {info}")
        if selection is None and split is None and "undeclared" in info.get(
                "splits", {}):
            print("warning: this file does not declare splits; every row is "
                  "being evaluated. Use --indices-json <run_metadata.json> "
                  "to reuse a checkpoint's test split.")
        print(f"metrics {', '.join(registry.available()) if names == ['all'] else ', '.join(names)}")

    legacy = bool(getattr(args, "legacy_dir", False))
    overwrite = bool(getattr(args, "overwrite", False))
    save_path = getattr(args, "save_predictions", None)
    cache_path = getattr(args, "cache", None)
    if save_path is not None and cache_path is not None and Path(save_path) != Path(cache_path):
        raise SystemExit("use --cache or --save-predictions, not both")
    cache_target = cache_path or save_path
    save_ids = [i.strip() for i in args.save_ids.split(",")] if args.save_ids else None
    save_n = None if args.save_n == 0 else args.save_n

    store = None
    cache = None
    if cache_target is not None and Path(cache_target).suffix.lower() in {".h5", ".hdf5"}:
        from .io.cache import CacheWriter
        cache_path = Path(cache_target)
        cache = CacheWriter(cache_path, ids=save_ids, n=save_n, overwrite=overwrite)
        store = cache
    elif cache_path is not None:
        from .io.cache import CacheWriter
        cache_path = Path(cache_path) / f"{target}.cache.h5"
        cache = CacheWriter(cache_path, ids=save_ids, n=save_n, overwrite=overwrite)
        store = cache
    elif save_path is not None:
        from .store import PredictionStore
        store = PredictionStore(save_path, ids=save_ids, n=save_n)

    h5_out = (not legacy) and out.suffix.lower() in {".h5", ".hdf5"}
    if h5_out and len(models) > 1:
        raise SystemExit("a single .h5 --out holds one model; pass a directory or one --model")
    if not legacy and not h5_out:
        out.mkdir(parents=True, exist_ok=True)

    from .io.runfile import default_path, write_run

    results = []
    try:
        for name, spec in models.items():
            if not args.quiet:
                print(f"\n[{name}] {spec}")
            adapter = adapters.build(spec, fno_root=fno_root, device=device,
                                     target=target)
            if expect_shape is not None and hasattr(adapter, "expect_shape"):
                adapter.expect_shape = expect_shape
            result = evaluate(
                adapter, dataset, settings, names, options,
                progress=None if args.quiet else _progress(name),
                expect_shape=expect_shape, store=store, model_name=name,
            )
            result.name = name
            if not args.quiet:
                _print_summary(result)
            if legacy:
                results.append(result)
                continue
            path = out if h5_out else default_path(out, name, target)
            try:
                write_run(path, result, info, settings.to_dict(), environment(),
                          command=" ".join(sys.argv), overwrite=overwrite)
            except FileExistsError as exc:
                raise SystemExit(str(exc)) from None
            if cache is not None:
                cache.flush()
            if not args.quiet:
                print(f"wrote {path}")
    finally:
        if cache is not None:
            cache.close()

    if legacy:
        payload = report.write(out, results, info, settings.to_dict(),
                               environment(), title=args.title)
        if not args.quiet:
            print(f"\nwrote {out}/results.json, summary.csv, report.md"
                  + (", curves.npz" if (out / "curves.npz").exists() else ""))
            print(f"schema {payload['schema']}")
    return 0


def _plot(args) -> int:
    from . import visualization
    from .store import PredictionStore

    if args.figures == ["list"]:
        figures = visualization.load_all()
        for group in sorted({f.group for f in figures.values()}):
            print(f"[{group}]")
            for f in figures.values():
                if f.group == group:
                    print(f"  {f.name:28s} {'(stored cubes) ' if 'store' in f.needs else ''}{f.help}")
        return 0
    if not args.run or args.out is None:
        raise SystemExit("eval21 plot: --run and --out are required")
    style = json.loads(args.style.read_text()) if args.style else None
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    ids = [i.strip() for i in args.ids.split(",")] if args.ids else None
    catalog = visualization.Catalog(visualization.find_runs(args.run), models=models, style=style, quiet=args.quiet)
    options = {}
    for item in args.opt:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--opt expects KEY=VALUE, got {item!r}")
        options[key] = _parse_value(value)
    for key in ("vmax", "dead"):
        if getattr(args, key) is not None:
            options[key] = getattr(args, key)
    from .io.cache import StoreGroup, open_cache

    stores = []
    candidates = [Path(p) for p in (args.store or _store_candidates(args.run))]
    for path in candidates:
        if path.is_dir() and (path / "index.json").exists():
            stores.append(PredictionStore.open(path))
        elif path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}:
            stores.append(open_cache(path))
    store = None
    if len(stores) == 1:
        store = stores[0]
    elif stores:
        store = StoreGroup(stores)
    if args.store and store is None:
        raise SystemExit(f"no prediction store or cache at {', '.join(str(p) for p in args.store)}")
    if not args.quiet:
        print(f"models {catalog.models}; targets {catalog.targets}; groups "
              + ", ".join(f"{t}: {len(catalog.groups[t])}" for t in catalog.targets)
              + (f"; store {store.dir} ({len(store.samples)} samples)" if store else "; no prediction store"))
    ctx = visualization.PlotContext(catalog=catalog, out=args.out, store=store, ids=ids, title=args.title,
                          options=options, quiet=args.quiet)
    visualization.run(args.figures, ctx)
    return 0


def _build(args) -> int:
    from . import build as builder

    if args.merge:
        builder.merge(args.out, args.num_shards, quiet=args.quiet, remove=args.remove_shards)
        return 0
    if args.folder is None:
        raise SystemExit("eval21 build: give the simulation folder")
    options = {}
    for item in args.option:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--option expects KEY=VALUE, got {item!r}")
        options[key] = _parse_value(value)
    split = lambda text: [v.strip() for v in text.split(",") if v.strip()] if text else None  # noqa: E731
    builder.build(args.folder, args.out, fmt=args.format, fields=split(args.fields),
                  targets=split(args.targets), z_grid=args.z_grid, transverse=args.transverse,
                  splits=args.splits, split_file=args.split_file, seed=args.seed, limit=args.limit,
                  ids=split(args.ids), shard=args.shard, num_shards=args.num_shards,
                  workers=args.workers, dtype=args.dtype, compress=args.compress,
                  source_options=options, quiet=args.quiet, command=" ".join(sys.argv))
    return 0


def _store_candidates(runs):
    out, seen = [], set()
    for run in runs:
        run = Path(run)
        bases = [run, run.parent, run.parent.parent]
        candidates = [base / "predictions" for base in bases]
        candidates.extend(sorted(p.parent for p in run.rglob("predictions/index.json")))
        if run.is_dir():
            candidates.extend(sorted(run.glob("*.cache.h5")))
            candidates.extend(sorted(run.rglob("*.cache.h5")))
        elif run.is_file():
            candidates.extend(sorted(run.parent.glob("*.cache.h5")))
        for path in candidates:
            if path not in seen:
                seen.add(path)
                out.append(path)
    return out


def _describe(args) -> int:
    dataset = open_data(args.data, target=args.target, limit=args.limit,
                        box_mpc=args.box_mpc)
    print(json.dumps(dataset.describe(), indent=2))
    return 0


def _list(args) -> int:
    items = {
        name: (cls.__doc__ or "").strip().split("\n")[0]
        for name, cls in sorted(registry.available().items())
    }
    if args.json:
        print(json.dumps(items, indent=2))
    else:
        for name, description in items.items():
            print(f"{name:<12} {description}")
    return 0


def _demo(args) -> int:
    from .data.synthetic import write_demo

    shape = tuple(int(part) for part in args.shape.split(","))
    manifest = write_demo(args.out / "data", args.n_samples, shape)
    print(f"wrote {manifest}")
    return _run(argparse.Namespace(
        config=None, data=str(manifest), model=["toy=synthetic:1.5"],
        metrics="all", out=args.out / "results", title="21cmEVAL demo",
        split=None, limit=None, target=None, density=None, fno_root=None,
        device=None, set=[], quiet=args.quiet, indices=None,
        indices_json=None, box_mpc=None, expect_shape=None, expect_nz=None,
        legacy_dir=False, overwrite=True, cache=None, save_predictions=None,
        save_n=8, save_ids=None, draws=None, mode=None, score_draw=None,
        z_range=None,
    ))


def _validate(args) -> int:
    from .io.runfile import validate_run

    bad = 0
    for path in args.files:
        try:
            problems = validate_run(path, strict=args.strict)
        except Exception as exc:  # noqa: BLE001 - report the path, keep checking the rest
            print(f"{path}: {exc}")
            bad += 1
            continue
        if problems:
            print(f"{path}: {len(problems)} problem(s)")
            for problem in problems:
                print(f"  - {problem}")
            bad += 1
        else:
            print(f"{path}: ok")
    return 1 if bad else 0


def _resolve_expect_shape(args, info: dict):
    explicit = getattr(args, "expect_shape", None)
    expect_nz = getattr(args, "expect_nz", None)
    if explicit and expect_nz is not None:
        raise SystemExit("use --expect-shape or --expect-nz, not both")
    if explicit:
        parts = [part.strip() for part in str(explicit).split(",")]
        if len(parts) != 3:
            raise SystemExit("--expect-shape expects NX,NY,NZ")
        try:
            return tuple(int(part) for part in parts)
        except ValueError:
            raise SystemExit("--expect-shape expects three integers") from None
    if expect_nz is not None:
        shape = info.get("shape")
        if not shape or len(shape) != 3:
            raise SystemExit("--expect-nz needs a 3-D dataset shape")
        return (int(shape[0]), int(shape[1]), int(expect_nz))
    return None


def _preflight_cache_contracts(models: dict, info: dict) -> None:
    """Enforce a checkpoint's recorded cache identity before loading a model."""
    from .adapters.fno21cm import validate_cache_contract

    for spec in models.values():
        adapter, _, argument = spec.partition(":")
        if adapter not in {"fno21cm", "fno"}:
            continue
        from .adapters.fno21cm import metadata_directory

        checkpoint = Path(argument).expanduser()
        metadata_path = metadata_directory(checkpoint) / "run_metadata.json"
        if not metadata_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text())
            validate_cache_contract(metadata, info, checkpoint)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None


def _resolve_selection(dataset, args, run_section: dict, split: str | None):
    """Row indices from an explicit list or from a run's recorded split.

    The reference resolves splits by cone id, not row position, because merged
    and reordered caches make positions stale. When the metadata records
    ``<split>_cone_ids`` and the dataset knows its ids, map through those and
    fail if any recorded cone is absent.
    """
    text = args.indices or run_section.get("indices")
    path = args.indices_json
    if text and path:
        raise SystemExit("use --indices or --indices-json, not both")
    if path:
        payload = json.loads(Path(path).read_text())
        block = payload.get("split", payload)
        name = split or "test"
        cone_key, index_key = f"{name}_cone_ids", f"{name}_indices"
        if cone_key in block:
            mapper = getattr(dataset, "row_indices_for_ids", None)
            if mapper is None:
                if index_key not in block:
                    raise SystemExit(
                        f"{path} records {cone_key} but this dataset has no "
                        f"ids to map them through, and no {index_key}")
                return [int(value) for value in block[index_key]]
            try:
                return mapper(block[cone_key])
            except KeyError as exc:
                raise SystemExit(str(exc)) from None
        if index_key in block:
            return [int(value) for value in block[index_key]]
        raise SystemExit(f"{path} has no split.{cone_key} or split.{index_key}")
    if text is None:
        return None
    if isinstance(text, list):
        return [int(value) for value in text]
    return _parse_indices(str(text))


def _parse_indices(text: str) -> list[int]:
    indices = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = part.split("-", 1)
            indices.extend(range(int(start), int(stop) + 1))
        else:
            indices.append(int(part))
    return indices


def _progress(name: str):
    def report_progress(index, total, sample_id):
        if index % 10 == 0 or index == total:
            suffix = f"/{total}" if total else ""
            print(f"\r  {name}: {index}{suffix} samples", end="", flush=True)
            if index == total:
                print()

    return report_progress


def _print_summary(result):
    flat = report.flatten(result.metrics)
    keys = ["voxel.rmse", "voxel.r2", "spectrum.active.coherence_ref",
            "history.tau.error_sigma_med", "efficiency.ms_per_sample"]
    parts = [f"{key}={flat[key]:.4g}" for key in keys if key in flat]
    if parts:
        print()
        print("  " + "  ".join(parts))
    for name, reason in result.skipped.items():
        print(f"  skipped {name}: {reason}")


def _parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "," in text:
        parts = [part.strip() for part in text.split(",")]
        try:
            return tuple(float(part) for part in parts)
        except ValueError:
            return tuple(parts)
    return text


if __name__ == "__main__":
    raise SystemExit(main())
