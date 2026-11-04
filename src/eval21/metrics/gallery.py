"""Stored transverse slices for the gallery and delta-slice figures.

Not a score: for a few samples (the first ``gallery_samples``, or the ids in ``gallery_ids``) it
keeps the truth and prediction slices at the ionization stages ``gallery_stages`` (the slices
whose true slice-mean x_HI is nearest each stage; evenly spaced slices when the sample carries no
true x_HI), for every field in ``gallery_fields`` that both the sample (target or input) and the
prediction provide. ``eval21 plot gallery`` and ``eval21 plot deltas`` draw them, merging the
fields of several runs (e.g. an x_HI pass and a T_b pass) by sample id.
"""
from __future__ import annotations

import numpy as np

from ..registry import register
from .base import Metric
from .cylindrical import true_xhi


@register("gallery")
class GalleryMetric(Metric):
    pointwise = False
    def __init__(self, settings, options=None):
        super().__init__(settings, options)
        self.levels = tuple(float(s) for s in self.settings.gallery_stages)
        self.entries: list[dict] = []

    def _wanted(self, sample) -> bool:
        ids = tuple(str(i) for i in self.settings.gallery_ids)
        if ids:
            return str(sample.id) in ids
        return len(self.entries) < self.settings.gallery_samples

    def update(self, sample, prediction, aux=None, runtime=None):
        if not self._wanted(sample):
            return
        xhi = true_xhi(sample, self.settings.target, self.settings.stage_field)
        nz = sample.shape[2]
        if xhi is not None:
            xbar = xhi.mean(axis=(0, 1))
            planes = [int(np.argmin(np.abs(xbar - s))) for s in self.levels]
            source = "true x_HI"
        else:
            xbar = np.full(nz, np.nan)
            planes = [int(round(v)) for v in np.linspace(0, nz - 1, len(self.levels) + 2)[1:-1]]
            source = "evenly spaced (no true x_HI)"
        z = self.z_grid(sample)
        entry = {"id": str(sample.id), "planes": np.array(planes),
                 "z": np.array([np.nan if z is None else z[k] for k in planes], float),
                 "xbar": np.array([xbar[k] for k in planes], float), "source": source, "fields": {}}
        for field in self.settings.gallery_fields:
            truth = sample.targets.get(field, sample.inputs.get(field))
            pred = prediction.get(field)
            if truth is None or pred is None or np.ndim(truth) != 3 or np.shape(pred) != np.shape(truth):
                continue
            take = lambda a: np.moveaxis(np.asarray(a, np.float32)[..., planes], -1, 0)    # (stages, H, W)
            entry["fields"][field] = (take(truth), take(pred))
        if entry["fields"]:
            self.entries.append(entry)

    def result(self):
        return {"samples": [e["id"] for e in self.entries], "levels": list(self.levels),
                "fields": sorted({f for e in self.entries for f in e["fields"]}),
                "plane_source": sorted({e["source"] for e in self.entries})}

    def arrays(self):
        if not self.entries:
            return {}
        out = {"gallery_ids": np.array([e["id"] for e in self.entries]),
               "gallery_stages": np.array(self.levels, float),
               "gallery_z": np.stack([e["z"] for e in self.entries]),
               "gallery_xbar": np.stack([e["xbar"] for e in self.entries]),
               "gallery_planes": np.stack([e["planes"] for e in self.entries])}
        shape = next(iter(self.entries[0]["fields"].values()))[0].shape
        for field in sorted({f for e in self.entries for f in e["fields"]}):
            for k, part in enumerate(("truth", "pred")):
                out[f"gallery_{field}_{part}"] = np.stack([
                    e["fields"][field][k] if field in e["fields"] else np.full(shape, np.nan, np.float32)
                    for e in self.entries])                       # (samples, stages, H, W)
        return out
