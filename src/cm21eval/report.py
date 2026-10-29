"""Report writers: JSON (complete), CSV (flat scalars), Markdown (readable)."""
from __future__ import annotations

import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = "cm21eval/1"


def jsonable(value):
    """Recursively convert numpy types and non-finite floats for strict JSON."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return jsonable(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def flatten(value, prefix: str = "") -> dict:
    """Dotted scalar leaves; arrays and lists are skipped."""
    out = {}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (bool, int, float, str)) or value is None:
        if value is not None and not isinstance(value, str):
            if not isinstance(value, bool) and not np.isfinite(value):
                return out
        out[prefix] = value
    return out


def write(directory, models, dataset_info: dict, settings: dict,
          environment: dict | None = None, title: str = "21cmEVAL report") -> dict:
    """Write ``results.json``, ``summary.csv``, ``report.md`` and ``curves.npz``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    environment = environment or {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    payload = {
        "schema": SCHEMA,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": environment,
        "dataset": dataset_info,
        "settings": settings,
        "models": {
            model.name: {
                "info": jsonable(model.info),
                "n_samples": model.n_samples,
                "seconds": model.seconds,
                "skipped": model.skipped,
                "metrics": jsonable(model.metrics),
            }
            for model in models
        },
    }
    (directory / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n")

    rows = []
    keys: list[str] = []
    for model in models:
        row = {"model": model.name}
        row.update({f"info.{k}": v for k, v in flatten(model.info).items()})
        row.update(flatten(model.metrics))
        for key in row:
            if key not in keys:
                keys.append(key)
        rows.append(row)
    with open(directory / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model"] + [k for k in keys if k != "model"])
        writer.writeheader()
        writer.writerows(rows)

    arrays = {}
    for model in models:
        for key, value in model.arrays.items():
            arrays[f"{model.name}__{key}"] = value
    if arrays:
        np.savez(directory / "curves.npz", **arrays)
    elif (directory / "curves.npz").exists():
        (directory / "curves.npz").unlink()

    stage_rows = []
    for model in models:
        for metric_name, result in model.metrics.items():
            if not isinstance(result, dict):
                continue
            for list_key in ("stages", "chunks"):
                for index, entry in enumerate(result.get(list_key) or []):
                    if not isinstance(entry, dict):
                        continue
                    label = entry.get("label", entry.get("z", index))
                    for key, value in flatten(entry).items():
                        if key in {"label", "k_targets", "delta2_k_targets",
                                   "kperp_centers", "kpar_centers"}:
                            continue
                        stage_rows.append({
                            "model": model.name, "metric": metric_name,
                            "group": list_key, "stage": label,
                            "key": key, "value": value,
                        })
    if stage_rows:
        keys = list(stage_rows[0])
        with open(directory / "stages.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(stage_rows)
    elif (directory / "stages.csv").exists():
        (directory / "stages.csv").unlink()

    (directory / "report.md").write_text(
        render_markdown(models, dataset_info, settings, title))
    return payload


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
HEADLINE = (
    ("RMSE", ("voxel", "rmse"), "{:.4f}"),
    ("R^2", ("voxel", "r2"), "{:.3f}"),
    ("r(k)", ("spectrum", "active", "coherence_ref"), "{:.3f}"),
    ("k_r90", ("spectrum", "active", "k_r90"), "{:.3f}"),
    ("tau err/sigma", ("history", "tau", "error_sigma_med"), "{:.2f}"),
    ("z50 err/sigma", ("history", "history", "z_xhi50", "error_sigma_med"), "{:.2f}"),
    ("front width excess", ("front", "front_width_excess_mpc"), "{:.2f}"),
    ("bubble W1", ("bubble", "active", "wasserstein_med_mpc"), "{:.2f}"),
    ("partial excess", ("observable", "active", "partial_fraction_excess"), "{:+.4f}"),
    ("ms/sample", ("efficiency", "ms_per_sample"), "{:.1f}"),
    ("params", ("efficiency", "parameters"), "{:,.0f}"),
)


def render_markdown(models, dataset_info: dict, settings: dict,
                    title: str = "21cmEVAL report") -> str:
    lines = [f"# {title}", ""]
    box = dataset_info.get("box_mpc")
    box = settings.get("box_mpc") if box is None else box
    evaluated = models[0].n_samples if models else dataset_info.get("n_samples", "?")
    total = dataset_info.get("n_total")
    scope = (f"{evaluated} of {total} samples" if total
             else f"{evaluated} samples")
    lines.append(
        f"Created {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
        f"{scope} · shape {dataset_info.get('shape')} · box {box} Mpc")
    location = dataset_info.get("path") or dataset_info.get("root")
    if location:
        z_range = dataset_info.get("z_range")
        detail = f" · z {z_range[0]:g}–{z_range[1]:g}" if z_range else ""
        lines.append(f"Data: `{location}`{detail}")
    lines.append("")

    columns = [(label, path, fmt) for label, path, fmt in HEADLINE
               if any(_get(model.metrics, path) is not None for model in models)]
    header = "| model | " + " | ".join(label for label, _, _ in columns) + " |"
    rule = "|" + "---|" * (len(columns) + 1)
    lines += [header, rule]
    for model in models:
        cells = []
        for _, path, fmt in columns:
            value = _get(model.metrics, path)
            cells.append("—" if value is None else fmt.format(value))
        lines.append(f"| {model.name} | " + " | ".join(cells) + " |")
    lines.append("")
    if models and models[0].skipped:
        lines.append("Skipped: " + "; ".join(
            f"`{name}` ({reason})" for name, reason in models[0].skipped.items()))
        lines.append("")

    for model in models:
        lines += [f"## {model.name}", ""]
        info = flatten(model.info)
        if info:
            keys = {"framework", "device", "model_kind", "checkpoint",
                    "spatial_shape", "in_channels", "target"}
            lines.append(" · ".join(
                f"{key}: {value}" for key, value in info.items()
                if key in keys and value is not None))
            lines.append("")
        for metric_name, result in model.metrics.items():
            if not result:
                continue
            lines += [f"### {metric_name}", ""]
            stages = result.get("stages") if isinstance(result, dict) else None
            if stages:
                lines += _stage_table(stages)
            else:
                scalars = flatten(result)
                if scalars:
                    lines += ["| key | value |", "|---|---|"]
                    for key, value in scalars.items():
                        lines.append(f"| {key} | {_format(value)} |")
                    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _stage_table(stages) -> list[str]:
    if not stages:
        return []
    columns = [key for key in stages[0]
               if key not in {"label", "k_targets", "delta2_k_targets"}]
    lines = ["| stage | " + " | ".join(columns) + " |",
             "|---|" + "---|" * len(columns)]
    for stage in stages:
        cells = [_format(stage.get(key)) for key in columns]
        lines.append(f"| {stage.get('label', '')} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _format(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "—"
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
            return f"{value:.3e}"
        return f"{value:.4g}"
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) > 5:
            try:
                array = np.asarray(value, dtype=float)
            except (TypeError, ValueError):
                return f"[{len(value)} values]"
            finite = array[np.isfinite(array)]
            if finite.size == 0:
                return "—"
            return (f"med {_format(float(np.median(finite)))} "
                    f"[{_format(float(finite.min()))}, "
                    f"{_format(float(finite.max()))}]")
        return "[" + ", ".join(_format(v) for v in value) + "]"
    return str(value)


def _get(mapping: dict, path: tuple):
    value = mapping
    for key in path:
        if not isinstance(value, dict):
            return None
        if key in value:
            value = value[key]
            continue
        matches = [k for k in value if k.startswith(key)]
        if len(matches) != 1:
            return None
        value = value[matches[0]]
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    return None
