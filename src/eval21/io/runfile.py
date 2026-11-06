"""The eval21 run file: one HDF5 file per ``(model, target, dataset)`` evaluation.

Schema ``eval21-run/1``. The layout is documented in ``docs/design/run-file.md``; the short
version::

    /                      attrs: schema, payload_schema, created, producer, model, target
    /manifest              JSON index of every dataset (dtype, dims, units)
    /meta/{environment,versions,command,settings}
    /dataset/{info,z_grid}
    /samples/{ids,splits,omega_m,params/{names,values}}
    /model/{info,meta}
    /metrics/<name>/{result,meta,arrays/<key>}

``open_run`` reads both a run file and a legacy run directory (``results.json`` + ``curves.npz``),
so callers do not branch on the storage format.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

SCHEMA = "eval21-run/1"
PAYLOAD_SCHEMA = "eval21/1"
# arrays larger than this get gzip + shuffle; small ones are cheap enough uncompressed
COMPRESS_ABOVE = 1 << 20


# ------------------------------------------------------------------------------------------- #
# helpers
# ------------------------------------------------------------------------------------------- #
def _text(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return str(value)


def _json_text(value) -> str:
    from ..report import jsonable

    return json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False)


def _write_json(group, name: str, value, specs: list | None = None, path: str = "") -> None:
    text = _json_text(value)
    if name in group:
        del group[name]
    group.create_dataset(name, data=np.array(text, dtype=_string_dtype()))
    if specs is not None:
        specs.append({"path": f"{path}/{name}".replace("//", "/"), "kind": "json",
                      "dtype": "utf-8", "dims": [], "units": ""})


def _read_json(group, name: str):
    if name not in group:
        raise KeyError(f"missing /{name} in {group.name}")
    return json.loads(_text(group[name][()]))


def _string_dtype():
    import h5py

    return h5py.string_dtype(encoding="utf-8")


def _write_strings(group, name: str, values, specs: list | None = None, path: str = "") -> None:
    array = np.array([str(v) for v in values], dtype=object)
    dataset = group.create_dataset(name, data=array, dtype=_string_dtype())
    if specs is not None:
        specs.append({"path": f"{path}/{name}".replace("//", "/"), "kind": "strings",
                      "dtype": f"str[{len(values)}]", "dims": ["item"], "units": ""})
    return dataset


def _write_array(group, name: str, value, specs: list, path: str, dims=None, units="") -> None:
    array = np.asarray(value)
    kwargs = {}
    if array.dtype.kind in {"U", "O", "S"}:
        flat = [str(v) for v in array.ravel()]
        dataset = group.create_dataset(name, data=np.array(flat, dtype=object).reshape(array.shape),
                                       dtype=_string_dtype())
        dtype = "utf-8"
    else:
        if array.nbytes > COMPRESS_ABOVE:
            kwargs = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
        dataset = group.create_dataset(name, data=array, **kwargs)
        dtype = str(array.dtype)
    dataset.attrs["dims"] = ",".join(dims or [])
    dataset.attrs["units"] = units or ""
    specs.append({"path": f"{path}/{name}".replace("//", "/"), "kind": "array",
                  "dtype": dtype, "shape": list(array.shape),
                  "dims": list(dims or []), "units": units or ""})


# ------------------------------------------------------------------------------------------- #
# array metadata
# ------------------------------------------------------------------------------------------- #
_RULES = {
    # spectra / cylindrical
    "spectrum_k": (["k"], "Mpc^-1"),
    "spectrum_highk_logmse": (["sample", "stage"], ""),
    "radial_k": (["k"], "Mpc^-1"),
    "radial_power_ratio_med": (["k"], ""),
    "radial_coherence_med": (["k"], ""),
    "eft_k": (["k"], "Mpc^-1"),
    "anisotropy_kperp": (["kperp"], "Mpc^-1"),
    "anisotropy_kperp_edges": (["kperp_edge"], "Mpc^-1"),
    "anisotropy_cyl_ratio": (["sample", "chunk", "kpar", "kperp"], ""),
    "anisotropy_cyl_r": (["sample", "chunk", "kpar", "kperp"], ""),
    "anisotropy_cyl_z": (["sample", "chunk"], "redshift"),
    "anisotropy_cyl_kpar": (["sample", "chunk", "kpar"], "Mpc^-1"),
    "cyl_ratio": (["sample", "chunk", "kpar", "kperp"], ""),
    "cyl_coherence": (["sample", "chunk", "kpar", "kperp"], ""),
    "cyl_ratio_median": (["chunk", "kpar", "kperp"], ""),
    "cyl_coherence_median": (["chunk", "kpar", "kperp"], ""),
    "cyl_kperp_edges": (["kperp_edge"], "Mpc^-1"),
    "cyl_kpar_edges": (["kpar_edge"], "Mpc^-1"),
    "cyl_kperp_max": ([], "Mpc^-1"),
    "cyl_chunk_z": (["sample", "chunk"], "redshift"),
    "cyl_chunk_xbar": (["sample", "chunk"], ""),
    "cyl_kpar_max": (["sample", "chunk"], "Mpc^-1"),
    "cyl_have": (["sample", "chunk"], ""),
    # bubbles
    "bubble_radius_mpc": (["radius"], "Mpc"),
    "bubble_radius_edges_mpc": (["radius_edge"], "Mpc"),
    "bubble_radius_centers_mpc": (["radius"], "Mpc"),
    # fronts / distance profiles
    "front_d_mpc": (["distance"], "Mpc"),
    "front_mean_truth": (["distance"], ""),
    "front_mean_pred": (["distance"], ""),
    "front_rmse": (["distance"], ""),
    "front_gradient_rms": (["distance"], ""),
    "front_counts": (["distance"], "count"),
    "front_width_truth": (["sample"], "Mpc"),
    "front_width_pred": (["sample"], "Mpc"),
    "front3d_d_mpc": (["distance"], "Mpc"),
    "front3d_mean_truth": (["distance"], ""),
    "front3d_mean_pred": (["distance"], ""),
    "front3d_rmse": (["distance"], ""),
    "front3d_gradient_rms": (["distance"], ""),
    "front3d_counts": (["distance"], "count"),
    # observable / dtb histograms
    "observable_xhi_edges": (["xhi_edge"], ""),
    "observable_dtb_edges": (["dtb_edge"], "mK"),
    "observable_xbar_edges": (["xbar_edge"], ""),
    "dtb_edges": (["dtb_edge"], "mK"),
    "dtb_xbar_edges": (["xbar_edge"], ""),
    # gallery
    "gallery_ids": (["sample"], ""),
    "gallery_stages": (["stage"], ""),
    "gallery_z": (["sample", "stage"], "redshift"),
    "gallery_xbar": (["sample", "stage"], ""),
    "gallery_planes": (["sample", "stage"], "index"),
    # parity / ensemble / voxel / survey
    "parity_count": (["true_bin"], "count"),
    "parity_sum_pred": (["true_bin"], ""),
    "parity_sum_pred2": (["true_bin"], ""),
    "parity_sum_truth": (["true_bin"], ""),
    "parity_sum_err2": (["true_bin"], ""),
    "parity_h2d": (["true_bin", "pred_bin"], "count"),
    "ensemble_rank_histogram": (["rank"], "count"),
    "voxel_sums": (["term"], ""),
    "efficiency_seconds": (["sample"], "s"),
    "survey_partial_bands": (["band", "edge"], ""),
}


def array_metadata(metric: str, key: str, shape: tuple, n_samples: int) -> tuple[list, str]:
    """Best-effort ``(dims, units)`` for one exported array.

    Explicit rules cover the arrays whose first axis is not a sample (k, radius, stage, bin,
    edges). Everything else falls back to ``sample`` when the first axis matches ``n_samples``,
    then to generic ``d<N>`` names. Unknown arrays never claim a meaning they may not have.
    """
    if key in _RULES:
        dims, units = _RULES[key]
        return list(dims), units
    if key.startswith("gallery_") and key.endswith(("_truth", "_pred")) and len(shape) == 4:
        return ["sample", "stage", "x", "y"], ""
    if key.startswith("observable_counts_"):
        if len(shape) == 2:
            return ["xhi_bin", "stage"] if "dtb" not in key else ["dtb_bin", "stage"], "count"
        return (["dtb_bin"] if "dtb" in key else ["xhi_bin"]), "count"
    if key.startswith("observable_") and "_hist_" in key:
        return (["dtb_bin"] if "dtb" in key else ["xhi_bin"]), "count"
    if key.startswith("bubble_mass_"):
        return (["sample", "radius"] if len(shape) == 2 else [f"d{i}" for i in range(len(shape))]), ""
    if key.startswith("bubble_stage_"):
        return (["sample", "stage"] if len(shape) == 2 else ["sample"]), ""
    if key.endswith("_ids") or key in {"gallery_ids"}:
        return ["sample"], ""
    if key == "history_z":
        return ["stage"], "redshift"
    if key == "history_omega_m":
        return ["sample"], ""
    if key.startswith("history_") and len(shape) == 1 and shape[0] == n_samples:
        return ["sample"], ""
    if key.startswith("dtb_") and key.endswith("_edges"):
        return (["dtb_edge"] if "dtb" in key else ["xbar_edge"]), ("mK" if "dtb" in key else "")
    if key.startswith("survey_"):
        return [f"d{i}" for i in range(len(shape))], ""
    if key.startswith("parity_") or key.startswith("ensemble_"):
        return [f"d{i}" for i in range(len(shape))], ""
    if len(shape) == 1 and shape[0] == n_samples:
        return ["sample"], ""
    if shape and shape[0] == n_samples:
        return ["sample"] + [f"d{i}" for i in range(1, len(shape))], ""
    return [f"d{i}" for i in range(len(shape))], ""


# ------------------------------------------------------------------------------------------- #
# writer
# ------------------------------------------------------------------------------------------- #
def _safe_model_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name)) or "model"


def default_path(out: Path, model: str, target: str) -> Path:
    return Path(out) / f"{_safe_model_name(model)}.{_safe_model_name(target)}.h5"


def write_run(path, result, dataset_info: dict, settings: dict, environment: dict | None = None,
              command: str = "", overwrite: bool = False) -> Path:
    """Write one :class:`~eval21.runner.ModelResult` as an ``eval21-run/1`` file."""
    import h5py

    from .. import __version__
    from ..runner import environment as default_environment

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    specs: list[dict] = []
    try:
        with h5py.File(tmp, "w", libver="earliest") as h5:
            h5.attrs["schema"] = SCHEMA
            h5.attrs["payload_schema"] = PAYLOAD_SCHEMA
            from datetime import datetime, timezone
            h5.attrs["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            h5.attrs["producer"] = f"eval21 {__version__}"
            h5.attrs["eval21_version"] = __version__
            h5.attrs["model"] = result.name
            h5.attrs["target"] = str(settings.get("target", "xhi"))

            meta = h5.create_group("meta")
            _write_json(meta, "environment", environment or default_environment(), specs, "/meta")
            _write_json(meta, "settings", settings, specs, "/meta")
            _write_json(meta, "versions", {
                "eval21": __version__, "numpy": np.__version__,
                "python": __import__("sys").version.split()[0],
                "framework": result.info.get("framework"),
                "framework_version": result.info.get("framework_version"),
            }, specs, "/meta")
            meta.create_dataset("command", data=np.array(str(command), dtype=_string_dtype()))
            specs.append({"path": "/meta/command", "kind": "string", "dtype": "utf-8",
                          "dims": [], "units": ""})

            dataset = h5.create_group("dataset")
            _write_json(dataset, "info", dataset_info, specs, "/dataset")
            z_grid = getattr(result, "z_grid", None)
            if z_grid is None and getattr(result, "samples", None):
                z_grid = result.samples[0].get("z_grid")
            if z_grid is not None:
                _write_array(dataset, "z_grid", np.asarray(z_grid, np.float64), specs, "/dataset",
                             ["z"], "redshift")

            samples = getattr(result, "samples", []) or []
            group = h5.create_group("samples")
            _write_strings(group, "ids", [s["id"] for s in samples], specs, "/samples")
            _write_strings(group, "splits", [s.get("split") or "" for s in samples], specs, "/samples")
            _write_array(group, "omega_m", np.array([np.nan if s.get("omega_m") is None else float(s["omega_m"])
                                                     for s in samples], float), specs, "/samples",
                         ["sample"], "")
            names, values = _params_matrix(samples)
            params = h5.create_group("samples/params")
            _write_strings(params, "names", names, specs, "/samples/params")
            _write_array(params, "values", values, specs, "/samples/params", ["sample", "parameter"], "")

            model = h5.create_group("model")
            _write_json(model, "info", result.info, specs, "/model")
            _write_json(model, "meta", {
                "n_samples": int(result.n_samples), "seconds": float(result.seconds),
                "skipped": result.skipped, "metric_meta": result.metric_meta,
            }, specs, "/model")

            metrics = h5.create_group("metrics")
            for name, value in result.metrics.items():
                mg = metrics.create_group(str(name))
                _write_json(mg, "result", value, specs, f"/metrics/{name}")
                _write_json(mg, "meta", (result.metric_meta or {}).get(name, {}), specs,
                            f"/metrics/{name}")
                arrays = {k.split("__", 1)[1]: v for k, v in result.arrays.items()
                          if k.startswith(f"{name}__")}
                if arrays:
                    ag = mg.create_group("arrays")
                    for key, array in arrays.items():
                        dims, units = array_metadata(name, key, np.shape(array), int(result.n_samples))
                        _write_array(ag, key, array, specs, f"/metrics/{name}/arrays", dims, units)

            manifest = {
                "schema": SCHEMA, "payload_schema": PAYLOAD_SCHEMA,
                "model": result.name, "target": str(settings.get("target", "xhi")),
                "n_samples": int(result.n_samples), "n_metrics": len(result.metrics),
                "metric_groups": list(result.metrics), "datasets": specs,
            }
            h5.create_dataset("manifest", data=np.array(_json_text(manifest), dtype=_string_dtype()))
            h5.flush()
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return path


def _params_matrix(samples: list[dict]) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    for sample in samples:
        for key in (sample.get("params") or {}):
            if key not in names:
                names.append(key)
    values = np.full((len(samples), len(names)), np.nan, float)
    for i, sample in enumerate(samples):
        for key, value in (sample.get("params") or {}).items():
            if value is not None and np.isfinite(float(value)):
                values[i, names.index(key)] = float(value)
    return names, values


# ------------------------------------------------------------------------------------------- #
# reader
# ------------------------------------------------------------------------------------------- #
class RunFile:
    """Reader for one ``eval21-run/1`` file."""

    def __init__(self, path):
        import h5py

        self.path = Path(path)
        self.h5 = h5py.File(self.path, "r")
        schema = _text(self.h5.attrs.get("schema", ""))
        major = schema.rsplit("/", 1)[-1].split(".", 1)[0] if "/" in schema else ""
        if not schema.startswith("eval21-run/") or major != "1":
            self.h5.close()
            raise ValueError(f"{self.path} is not a supported eval21 run file (schema {schema!r})")
        self.schema = schema

    # -- identity ---------------------------------------------------------------
    @property
    def model(self) -> str:
        return _text(self.h5.attrs.get("model", ""))

    @property
    def target(self) -> str:
        return _text(self.h5.attrs.get("target", ""))

    @property
    def dataset_info(self) -> dict:
        return _read_json(self.h5["dataset"], "info")

    @property
    def settings(self) -> dict:
        return _read_json(self.h5["meta"], "settings")

    @property
    def info(self) -> dict:
        return _read_json(self.h5["model"], "info")

    @property
    def meta(self) -> dict:
        return _read_json(self.h5["model"], "meta")

    @property
    def n_samples(self) -> int:
        return int(self.meta.get("n_samples", 0))

    @property
    def sample_ids(self) -> list[str]:
        if "samples" not in self.h5 or "ids" not in self.h5["samples"]:
            return []
        return [_text(v) for v in self.h5["samples/ids"][:]]

    def metric_names(self) -> list[str]:
        if "metrics" not in self.h5:
            return []
        return list(self.h5["metrics"])

    def result(self, metric: str) -> dict:
        return _read_json(self.h5[f"metrics/{metric}"], "result")

    def metric_meta(self, metric: str) -> dict:
        return _read_json(self.h5[f"metrics/{metric}"], "meta")

    def arrays(self, metric: str) -> dict:
        node = f"metrics/{metric}/arrays"
        if node not in self.h5:
            return {}
        return {key: self.h5[f"{node}/{key}"][()] for key in self.h5[node]}

    def array(self, metric: str, key: str) -> np.ndarray:
        return self.h5[f"metrics/{metric}/arrays/{key}"][()]

    def curves(self) -> dict:
        """``{model: {array key: array}}`` — the same shape the legacy reader returns."""
        out = {}
        for metric in self.metric_names():
            out.update(self.arrays(metric))
        return {self.model: out}

    def as_results(self) -> dict:
        meta = self.meta
        return {
            "schema": _text(self.h5.attrs.get("payload_schema", PAYLOAD_SCHEMA)),
            "created": _text(self.h5.attrs.get("created", "")),
            "environment": _read_json(self.h5["meta"], "environment"),
            "dataset": self.dataset_info,
            "settings": self.settings,
            "models": {
                self.model: {
                    "info": {**self.info, "draws": self.info.get("draws", 1)},
                    "n_samples": meta.get("n_samples", 0),
                    "seconds": meta.get("seconds", 0.0),
                    "skipped": meta.get("skipped", {}),
                    "metrics": {name: self.result(name) for name in self.metric_names()},
                }
            },
        }

    def validate(self, strict: bool = False) -> list[str]:
        problems = []
        for name in ("manifest", "meta", "dataset", "samples", "model", "metrics"):
            if name not in self.h5:
                problems.append(f"missing /{name}")
        if problems:
            return problems
        ids = self.sample_ids
        if len(ids) != self.n_samples:
            problems.append(f"/samples/ids has {len(ids)} rows, /model/meta says {self.n_samples}")
        for metric in self.metric_names():
            node = f"metrics/{metric}/arrays"
            if node not in self.h5:
                continue
            for key in self.h5[node]:
                ds = self.h5[f"{node}/{key}"]
                if "dims" not in ds.attrs or "units" not in ds.attrs:
                    problems.append(f"/{node}/{key} lacks dims/units")
                dims = str(ds.attrs.get("dims", ""))
                if not dims and ds.shape:
                    problems.append(f"/{node}/{key} has an empty dims attribute")
        if strict:
            try:
                manifest = json.loads(_text(self.h5["manifest"][()]))
            except Exception as exc:  # noqa: BLE001
                problems.append(f"manifest is not valid JSON: {exc}")
            else:
                listed = {d["path"] for d in manifest.get("datasets", [])}
                actual = set()
                def walk(name, obj):
                    import h5py

                    if name and name != "manifest" and isinstance(obj, h5py.Dataset):
                        actual.add("/" + name)
                self.h5.visititems(walk)
                for path in sorted(actual - listed):
                    problems.append(f"dataset {path} is not in the manifest")
        return problems

    def close(self) -> None:
        self.h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class LegacyRun:
    """Adapter over a legacy run directory (``results.json`` + ``curves.npz``)."""

    def __init__(self, path):
        self.path = Path(path)
        self.results = json.loads((self.path / "results.json").read_text())
        self.schema = self.results.get("schema", "")
        self._curves = None

    @property
    def model(self):
        models = list(self.results.get("models", {}))
        return models[0] if len(models) == 1 else ""

    @property
    def target(self):
        return self.results.get("settings", {}).get("target", "xhi")

    @property
    def dataset_info(self):
        return self.results.get("dataset", {})

    @property
    def settings(self):
        return self.results.get("settings", {})

    @property
    def info(self):
        models = self.results.get("models", {})
        return (models.get(self.model) or {}).get("info", {})

    @property
    def n_samples(self):
        models = self.results.get("models", {})
        return int((models.get(self.model) or {}).get("n_samples", 0))

    @property
    def sample_ids(self):
        return []

    def metric_names(self):
        models = self.results.get("models", {})
        return list((models.get(self.model) or {}).get("metrics", {}))

    def result(self, metric):
        return self.results["models"][self.model]["metrics"][metric]

    def metric_meta(self, metric):
        return {}

    def arrays(self, metric):
        return {k: v for k, v in self.curves().get(self.model, {}).items() if k.startswith(f"{metric}_")}

    def array(self, metric, key):
        return self.curves()[self.model][key]

    def curves(self):
        if self._curves is None:
            self._curves = {}
            file = self.path / "curves.npz"
            if file.exists():
                with np.load(file, allow_pickle=False) as data:
                    for key in data.files:
                        model, _, rest = key.partition("__")
                        _, _, short = rest.partition("__")
                        self._curves.setdefault(model, {})[short] = data[key]
        return self._curves

    def as_results(self):
        return self.results

    def validate(self, strict=False):
        return []

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_run(path):
    """Open an ``eval21-run/1`` file or a legacy run directory with one interface."""
    path = Path(path)
    if path.is_dir():
        if not (path / "results.json").exists():
            raise FileNotFoundError(f"{path} is neither a run file nor a run directory")
        return LegacyRun(path)
    if not path.exists():
        raise FileNotFoundError(f"no run at {path}")
    return RunFile(path)


def validate_run(path, strict: bool = False) -> list[str]:
    run = open_run(path)
    try:
        return run.validate(strict=strict)
    finally:
        run.close()
