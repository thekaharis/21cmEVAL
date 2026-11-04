"""Language-agnostic adapter: shell out to any inference program.

The command is run once per sample with ``{input}`` and ``{output}`` replaced
by temporary ``.npz`` paths::

    eval21 run --data cubes.npz \
        --model julia="command:julia infer.jl {input} {output}"

The input file holds every sample array under its field name plus a ``meta``
JSON blob (``meta`` key is a 0-d string array so ``np.load`` stays simple).
The program must write an ``.npz`` with one array per predicted target name.

This is the universal escape hatch: the only contract is "read npz, write
npz", which every scientific language supports. It is slower than an
in-process adapter because the process restarts per sample; for large sweeps
use a Python or Torch adapter instead.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..spec import Sample
from .base import BaseAdapter


class CommandModelAdapter(BaseAdapter):
    framework = "command"

    def __init__(self, command: str, timeout: float | None = None,
                 include_targets: bool = False):
        if "{input}" not in command or "{output}" not in command:
            raise ValueError(
                "command must contain {input} and {output} placeholders, got "
                f"{command!r}")
        self.command = command
        self.timeout = timeout
        # Off by default: writing the target into the input file lets a model
        # return it unchanged and score RMSE=0. A model that genuinely needs an
        # auxiliary target field must ask for it explicitly.
        self.include_targets = bool(include_targets)

    def predict(self, sample: Sample) -> dict:
        with tempfile.TemporaryDirectory(prefix="eval21-") as workdir:
            work = Path(workdir)
            input_path = work / "sample.npz"
            output_path = work / "prediction.npz"
            payload = dict(sample.inputs)
            if self.include_targets:
                payload.update(sample.targets)
            payload["meta"] = np.array(json.dumps({
                "id": sample.id,
                "box_mpc": sample.meta.box_mpc,
                "omega_m": sample.meta.omega_m,
                "z_grid": None if sample.meta.z_grid is None
                else np.asarray(sample.meta.z_grid).tolist(),
                "params": sample.meta.params,
            }))
            np.savez(input_path, **payload)
            command = self.command.format(input=input_path, output=output_path)
            result = subprocess.run(
                shlex.split(command), capture_output=True, text=True,
                timeout=self.timeout,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"command failed ({result.returncode}): {command}\n"
                    f"stderr:\n{result.stderr[-4000:]}")
            if not output_path.exists():
                raise FileNotFoundError(
                    f"command wrote no {output_path}\nstdout:\n{result.stdout[-2000:]}")
            with np.load(output_path) as data:
                return {key: data[key] for key in data.files}
