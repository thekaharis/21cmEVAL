"""Command line interface.

    cm21eval run --data cubes/ --model toy=synthetic:1.5 --out results/
    cm21eval run --config eval.toml
    cm21eval describe --data cubes/
    cm21eval list-metrics
    cm21eval selftest
    cm21eval demo --out demo/
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
        prog="cm21eval",
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

    list_metrics = sub.add_parser("list-metrics", help="list registered metrics")
    list_metrics.add_argument("--json", action="store_true")

    sub.add_parser("selftest", help="run analytic self-checks")

    demo = sub.add_parser("demo", help="write synthetic data and score a toy model")
    demo.add_argument("--out", type=Path, default=Path("demo"))
    demo.add_argument("--n-samples", type=int, default=4)
    demo.add_argument("--shape", default="64,64,64")
    demo.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)
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
    fno_root = pick(args.fno_root, "fno_root", None)
    device = pick(args.device, "device", None)
    indices = _resolve_indices(args, run_section, split)

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
        data_spec, target=target, density=density, split=split,
        limit=limit, box_mpc=settings.box_mpc, indices=indices)
    info = dataset.describe()
    if not args.quiet:
        print(f"data   {info}")
        print(f"metrics {', '.join(registry.available()) if names == ['all'] else ', '.join(names)}")

    results = []
    for name, spec in models.items():
        if not args.quiet:
            print(f"\n[{name}] {spec}")
        adapter = adapters.build(spec, fno_root=fno_root, device=device)
        result = evaluate(
            adapter, dataset, settings, names, options,
            progress=None if args.quiet else _progress(name),
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


def _describe(args) -> int:
    dataset = open_data(args.data, target=args.target, limit=args.limit)
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
        indices_json=None,
    ))


def _resolve_indices(args, run_section: dict, split: str | None):
    text = args.indices or run_section.get("indices")
    path = args.indices_json
    if text and path:
        raise SystemExit("use --indices or --indices-json, not both")
    if path:
        payload = json.loads(Path(path).read_text())
        block = payload.get("split", payload)
        key = f"{split or 'test'}_indices"
        if key not in block:
            raise SystemExit(f"{path} has no split.{key}")
        return [int(value) for value in block[key]]
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
