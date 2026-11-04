"""Command line interface.

    eval21 run --data cubes/ --model toy=synthetic:1.5 --out results/
    eval21 run --config eval.toml
    eval21 describe --data cubes/
    eval21 list-metrics
    eval21 selftest
    eval21 demo --out demo/
    eval21 plot all --run results/ --out figs/
    eval21 build 21cmfast /data/21cmfast --out cubes/fast --los nearest --z-grid 5.001,24.97,512
    eval21 build loreli /data/loreli_lc128 --out cubes/loreli --los comoving --dry-run
"""
from __future__ import annotations

import argparse
import json
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
    run.add_argument("--quiet", action="store_true")

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
    plot.add_argument("kind", choices=["cyl", "gallery", "deltas", "summary", "all"])
    plot.add_argument("--run", action="append", required=True, type=Path,
                      help="result directory or a tree of them (repeatable)")
    plot.add_argument("--out", type=Path, required=True)
    plot.add_argument("--models", default=None, help="comma list and order of models (default: all)")
    plot.add_argument("--style", type=Path, default=None,
                      help='JSON {model: {"label": ..., "color": ..., "ls": ...}}')
    plot.add_argument("--ids", default=None, help="comma list of sample ids for gallery/deltas")
    plot.add_argument("--title", default="", help="data set named in figure titles")
    plot.add_argument("--vmax", type=float, default=0.4, help="cyl: colour saturation of P_pred/P_true - 1")
    plot.add_argument("--dead", type=float, default=None, help="cyl: white band (default: the run's cyl_dead_band)")
    plot.add_argument("--quiet", action="store_true")

    _add_build_parser(sub)

    args = parser.parse_args(argv)
    if args.command == "build":
        return _build(args)
    if args.command == "plot":
        return _plot(args)
    if args.command == "run":
        return _run(args)
    if args.command == "describe":
        return _describe(args)
    if args.command == "list-metrics":
        return _list(args)
    if args.command == "selftest":
        return selftest.run()
    if args.command == "demo":
        return _demo(args)
    parser.error(f"unknown command {args.command}")
    return 2


def _add_build_parser(sub) -> None:
    build = sub.add_parser(
        "build", help="raw simulation lightcones -> eval21 npz cubes + manifest(s)",
        description="Build an eval21 dataset (one npz per lightcone + manifest.json per geometry "
                    "group) from raw 21cmFAST HDF5, LoReLi II .dat or generic per-file npz/h5 "
                    "lightcones. Resumable: complete outputs with the same build signature are kept.")
    build.add_argument("format", choices=["21cmfast", "loreli", "generic"])
    build.add_argument("source", help="folder with the raw lightcone files")
    build.add_argument("--out", type=Path, required=True, help="output directory")
    g = build.add_argument_group("fields")
    g.add_argument("--fields", default=None,
                   help="comma list of eval21 names (xhi,tb,density,los_velocity,...) or name=source_key")
    g.add_argument("--targets", default=None,
                   help="comma list; the first is manifest.json's target, each other one gets "
                        "manifest_<name>.json (default: xhi,tb where available)")
    g.add_argument("--rsd", action="store_true", help="21cmfast: use the *_with_rsds fields")
    g.add_argument("--density-convention", default=None,
                   help="source density: delta | one_plus_delta | rho (21cmfast default delta, "
                        "generic default one_plus_delta)")
    g = build.add_argument_group("line of sight")
    g.add_argument("--los", default="native", choices=["native", "nearest", "linear", "comoving"])
    g.add_argument("--z-grid", default=None,
                   help="z0,z1,n or a file (.npy, .npz z_grid, .json, text) for --los nearest|linear")
    g.add_argument("--z-range", default=None, help="z0,z1: restrict the LOS window")
    g.add_argument("--los-cell", type=float, default=None,
                   help="--los comoving spacing [Mpc] (default: the transverse cell)")
    g.add_argument("--allow-repeat", action="store_true",
                   help="--los nearest: allow a native plane to be used twice")
    g = build.add_argument_group("geometry")
    g.add_argument("--crop", type=int, default=None, help="centred transverse crop to N x N cells")
    g.add_argument("--crop-offset", default=None, help="X,Y corner of the crop instead of centred")
    g.add_argument("--group", default=None,
                   help="none (one manifest, cones cropped to the shared window), auto (cluster by "
                        "valid window; loreli default) or a groups.json {name: {cones, z}}")
    g.add_argument("--group-tol", type=float, default=0.15,
                   help="--group auto: max spread in first valid redshift within a group")
    g.add_argument("--min-planes", type=int, default=8)
    g = build.add_argument_group("selection and splits")
    g.add_argument("--ids", default=None, help="e.g. 0-99,250 or 10001-10010 (simulation numbers)")
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--only-split", default=None, help="build only cones of this split")
    g.add_argument("--split", default=None, help="one split name for every cone")
    g.add_argument("--split-file", type=Path, default=None, help="JSON {train: [ids], val: [...], test: [...]}")
    g.add_argument("--split-from", type=Path, default=None,
                   help="fno-21cm run_metadata.json (split.<name>_cone_ids)")
    g.add_argument("--split-fractions", default=None, help="train,val,test fractions (seeded hash per id)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--id-format", default="cone{id}", help="sample id / file name, default cone{id}")
    g = build.add_argument_group("output")
    g.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                   help="storage dtype (float16 falls back to float32 per field when unsafe)")
    g.add_argument("--compress", action="store_true", help="np.savez_compressed")
    g.add_argument("--workers", type=int, default=1)
    g.add_argument("--nonfinite", default="skip", choices=["skip", "error"],
                   help="cone with non-finite output values: skip it (listed in the manifest) or stop")
    g.add_argument("--overwrite", action="store_true", help="rebuild existing outputs")
    g.add_argument("--dry-run", action="store_true",
                   help="print files, fields, shapes, groups and size; writes only the scan cache")
    g.add_argument("--quiet", action="store_true")
    g = build.add_argument_group("loreli")
    g.add_argument("--loreli-kind", default="five", choices=["five", "dtb"],
                   help="five-field files (xhi, tb, density, ...) or dT_b-only files")
    g.add_argument("--l-perp", type=float, default=None, help="transverse size [Mpc] (default 294)")
    g.add_argument("--density-mean", default=None, choices=["slice", "cone"])
    g.add_argument("--loreli-params", type=Path, default=None,
                   help="loreli2_parameters.json: parameter values/status instead of the grid rule")
    g = build.add_argument_group("generic / file pattern")
    g.add_argument("--pattern", default=None, help="glob of the lightcone files")
    g.add_argument("--z-key", default=None)
    g.add_argument("--z-native", default=None, help="z0,z1: native planes uniform in z")
    g.add_argument("--distance-key", default=None)
    g.add_argument("--params-key", default=None)
    g.add_argument("--param-names", default=None, help="comma list for bare parameter arrays")
    g.add_argument("--box-mpc", default=None, help="number, dataset key or root attribute")
    g.add_argument("--omega-m", default=None, help="number, key or attribute")
    g.add_argument("--omega-m-param", default=None, help="parameter name holding Omega_m")
    g.add_argument("--hubble-h", type=float, default=None)
    g.add_argument("--los-axis", type=int, default=None)
    g.add_argument("--id-regex", default=None, help="regex with one group: id from the file stem")


def _pair(text, kind=float):
    if text is None:
        return None
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 2:
        raise SystemExit(f"expected two comma-separated values, got {text!r}")
    return tuple(kind(part) for part in parts)


def _build(args) -> int:
    import sys

    from .build import BuildError, BuildOptions, build, make_source

    split = lambda text: [part.strip() for part in text.split(",") if part.strip()] if text else None
    fractions = None
    if args.split_fractions:
        fractions = tuple(float(v) for v in args.split_fractions.split(","))
    source_options = {
        "rsd": args.rsd, "density_convention": args.density_convention,
        "loreli_kind": args.loreli_kind, "l_perp": args.l_perp, "density_mean": args.density_mean,
        "loreli_params": str(args.loreli_params) if args.loreli_params else None,
        "pattern": args.pattern, "z_key": args.z_key, "z_native": args.z_native,
        "distance_key": args.distance_key, "params_key": args.params_key,
        "param_names": split(args.param_names), "box_mpc": args.box_mpc, "omega_m": args.omega_m,
        "omega_m_param": args.omega_m_param, "hubble_h": args.hubble_h, "los_axis": args.los_axis,
        "id_regex": args.id_regex,
    }
    try:
        options = BuildOptions(
            out=args.out, fields=split(args.fields), targets=split(args.targets), los=args.los,
            z_grid=args.z_grid, z_range=_pair(args.z_range), los_cell=args.los_cell,
            allow_repeat=args.allow_repeat, crop=args.crop, crop_offset=_pair(args.crop_offset, int),
            dtype=args.dtype, compress=args.compress, workers=max(1, args.workers), limit=args.limit,
            ids=args.ids, only_split=args.only_split, split=args.split, split_file=args.split_file,
            split_from=args.split_from, split_fractions=fractions, seed=args.seed, group=args.group,
            group_tol=args.group_tol, min_planes=args.min_planes, id_format=args.id_format,
            nonfinite=args.nonfinite, overwrite=args.overwrite, dry_run=args.dry_run,
            quiet=args.quiet, argv=list(sys.argv),
            source_options={k: v for k, v in source_options.items() if v not in (None, False)})
        source = make_source(args.format, args.source, options)
        build(source, options, log=lambda *a: print(*a, flush=True))
    except (BuildError, ValueError, FileNotFoundError, KeyError) as exc:
        import os

        if os.environ.get("EVAL21_DEBUG"):
            raise
        raise SystemExit(f"eval21 build: {exc}") from None
    return 0


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

    settings_values = dict(config.get("settings", {}))
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        settings_values[key] = _parse_value(value)
    settings = Settings.from_dict(settings_values)

    metrics_section = config.get("metrics", {})
    if args.metrics:
        names = [part.strip() for part in args.metrics.split(",") if part.strip()]
    elif metrics_section.get("enabled"):
        names = list(metrics_section["enabled"])
    else:
        names = ["all"]
    options = config.get("metric_options", {})

    dataset = open_data(
        data_spec, target=target, density=density, split=None,
        limit=None, box_mpc=box_override)
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

    results = []
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
            expect_shape=expect_shape,
        )
        result.name = name
        results.append(result)
        if not args.quiet:
            _print_summary(result)

    payload = report.write(out, results, info, settings.to_dict(),
                           environment(), title=args.title)
    if not args.quiet:
        print(f"\nwrote {out}/results.json, summary.csv, report.md"
              + (", curves.npz" if (out / "curves.npz").exists() else ""))
        print(f"schema {payload['schema']}")
    return 0


def _plot(args) -> int:
    from . import viz

    style = json.loads(args.style.read_text()) if args.style else None
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    ids = [i.strip() for i in args.ids.split(",")] if args.ids else None
    catalog = viz.Catalog(viz.find_runs(args.run), models=models, style=style, quiet=args.quiet)
    if not args.quiet:
        print(f"models {catalog.models}; targets {catalog.targets}; groups "
              + ", ".join(f"{t}: {len(catalog.groups[t])}" for t in catalog.targets))
    if args.kind in ("cyl", "all"):
        viz.plot_cylindrical(catalog, args.out, vmax=args.vmax, dead=args.dead,
                             title=args.title or None, quiet=args.quiet)
    if args.kind in ("gallery", "all"):
        viz.plot_gallery(catalog, args.out / "gallery", ids=ids, quiet=args.quiet)
    if args.kind in ("deltas", "all"):
        viz.plot_deltas(catalog, args.out / "deltas", ids=ids, quiet=args.quiet)
    if args.kind in ("summary", "all"):
        viz.plot_summary(catalog, args.out, title=args.title, quiet=args.quiet)
    return 0


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
        config=None, data=str(manifest), model=[f"toy=synthetic:1.5"],
        metrics="all", out=args.out / "results", title="21cmEVAL demo",
        split=None, limit=None, target=None, density=None, fno_root=None,
        device=None, set=[], quiet=args.quiet, indices=None,
        indices_json=None, box_mpc=None, expect_shape=None, expect_nz=None,
    ))


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
