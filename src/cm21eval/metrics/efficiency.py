"""Cost of a forward pass.

Accuracy per parameter and per millisecond is how surrogate models are
compared in practice: a 0.002 RMSE win that costs 40x the inference time is
usually not a win. Timings are wall-clock over timed forward passes after a
warmup, reported as median with 16-84 percentiles, plus parameter counts and
peak accelerator memory when the adapter can provide it.

Cost is measured on the same samples the accuracy metrics see, one sample at a
time, so the number is "milliseconds per lightcone" -- the unit a pipeline
consumes.
"""
from __future__ import annotations

import numpy as np

from ..registry import register
from .base import Metric


@register("efficiency")
class EfficiencyMetrics(Metric):
    needs = ()

    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        self.seconds: list[float] = []
        self.slices: list[int] = []
        self.peak: float | None = None
        self.model_info: dict = {}

    def attach(self, model_info):
        self.model_info = model_info
        value = model_info.get("peak_memory_bytes")
        if value:
            self.peak = float(value) / 2 ** 20

    def update(self, sample, prediction, aux=None, runtime=None):
        if not runtime or "seconds" not in runtime:
            return
        self.seconds.append(float(runtime["seconds"]))
        self.slices.append(int(prediction[self.settings.target].shape[-1]))
        peak = runtime.get("peak_bytes")
        if peak:
            self.peak = max(self.peak or 0.0, float(peak) / 2 ** 20)

    def result(self):
        if not self.seconds:
            return {}
        seconds = np.asarray(self.seconds)
        median = float(np.median(seconds))
        return {
            "parameters": self.model_info.get("parameters"),
            "parameters_complex": self.model_info.get("complex_parameters"),
            "ms_per_sample": median * 1e3,
            "ms_per_sample_p16": float(np.percentile(seconds, 16)) * 1e3,
            "ms_per_sample_p84": float(np.percentile(seconds, 84)) * 1e3,
            "samples_per_s": 1.0 / median if median > 0 else None,
            "slices_per_s": float(np.sum(self.slices) / np.sum(seconds))
            if np.sum(seconds) > 0 else None,
            "peak_memory_mib": self.peak,
            "n_timed": int(seconds.size),
            "device": self.model_info.get("device"),
        }

    def arrays(self):
        return {"efficiency_seconds": np.asarray(self.seconds)} if self.seconds else {}
