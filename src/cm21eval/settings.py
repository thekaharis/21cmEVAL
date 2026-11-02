"""Suite-wide settings shared by every metric.

Metric-specific knobs (bin counts, ray counts, ...) also live here so one
TOML/JSON file fully describes a run and two reports are comparable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace


@dataclass(frozen=True)
class Settings:
    """Physical and numerical configuration of the metric suite."""

    # fields
    target: str = "xhi"
    density: str = "density"

    # geometry / cosmology
    box_mpc: float = 200.0
    omega_m: float = 0.30966
    omega_b: float = 0.04897
    hubble_h: float = 0.6766

    # stages
    threshold: float = 0.5
    stage_edges: tuple[float, ...] = (0.02, 0.2, 0.4, 0.6, 0.8, 0.98)
    active_range: tuple[float, float] = (0.05, 0.95)
    min_stage_slices: int = 2

    # spectra
    n_k_bins: int = 15
    k_targets: tuple[float, ...] = (0.1, 0.2, 0.5)
    eft_k_max: float = 0.25
    coherence_level: float = 0.9
    highk_k_min: float = 0.2          # cycles per pixel, as in losses.HighKPowerRatio
    highk_bins: int = 12

    # voxel / edge
    edge_width_fraction: float = 0.10
    active_variance_fraction: float = 0.05

    # bubble-size (mean free path) sampling
    rays_per_slice: int = 256
    slices_per_stage: int = 6
    bubble_bins: int = 24
    max_distance_mpc: float | None = None
    seed: int = 0

    # boundary-band front diagnostic
    front_bands_mpc: tuple[float, ...] = (2.0, 5.0)
    front_d_max_mpc: float = 30.0
    front_bin_mpc: float = 1.5

    # observable PDFs
    xhi_range: tuple[float, float] = (-0.2, 1.2)
    xhi_bins: int = 70
    partial_range: tuple[float, float] = (0.1, 0.9)
    dtb_range_mk: tuple[float, float] = (-5.0, 100.0)
    dtb_bins: int = 105

    # cylindrical P(k_perp, k_par) on LOS chunks (``cylindrical``)
    cyl_chunk_cells: int = 140                  # LOS cells per chunk, each one transverse cell long
    cyl_stages: tuple[float, ...] = (0.9, 0.7, 0.5, 0.3, 0.1)   # true slice-mean x_HI per chunk
    cyl_z_centers: tuple[float, ...] = ()       # fixed-redshift chunks instead of stages
    cyl_dead_band: float = 0.1                  # |P_pred/P_true - 1| counted as "within"
    cyl_native_cell_mpc: float | None = None    # transverse cell of a coarser native truth
    cyl_native_los_dz: float | None = None      # native LOS planes uniform in z with this spacing
    cyl_native_los_mpc: float | None = None     # ... or uniform in comoving distance

    # stored slices for ``cm21eval plot gallery`` / ``deltas`` (``gallery``)
    gallery_samples: int = 4                    # first N samples (ignored when gallery_ids is set)
    gallery_ids: tuple[str, ...] = ()
    gallery_stages: tuple[float, ...] = (0.8, 0.6, 0.4, 0.2)
    gallery_fields: tuple[str, ...] = ("xhi", "tb")

    # efficiency
    warmup: int = 3
    iters: int = 10
    benchmark_samples: int = 4

    def __post_init__(self):
        if self.box_mpc <= 0:
            raise ValueError("box_mpc must be positive")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must lie strictly between 0 and 1")
        edges = tuple(self.stage_edges)
        if len(edges) < 2 or any(b <= a for a, b in zip(edges[:-1], edges[1:])):
            raise ValueError("stage_edges must be strictly increasing")
        lo, hi = self.active_range
        if not lo < hi:
            raise ValueError("active_range must be increasing")
        if self.n_k_bins < 2:
            raise ValueError("n_k_bins must be at least 2")
        if not self.k_targets or any(t <= 0 for t in self.k_targets):
            raise ValueError("k_targets must be a non-empty list of positives")
        if self.eft_k_max <= 0:
            raise ValueError("eft_k_max must be positive")
        if self.front_bin_mpc <= 0 or self.front_d_max_mpc <= self.front_bin_mpc:
            raise ValueError("front_d_max_mpc must exceed front_bin_mpc > 0")
        if self.bubble_bins < 2:
            raise ValueError("bubble_bins must be at least 2")
        if self.rays_per_slice < 0 or self.slices_per_stage < 1:
            raise ValueError("rays_per_slice >= 0 and slices_per_stage >= 1")
        if self.min_stage_slices < 1:
            raise ValueError("min_stage_slices must be at least 1")
        if not 0 <= self.highk_k_min < 0.5:
            raise ValueError("highk_k_min must lie in [0, 0.5) cycles/pixel")
        if self.highk_bins < 1:
            raise ValueError("highk_bins must be positive")
        if not 0 < self.edge_width_fraction < 0.5:
            raise ValueError("edge_width_fraction must lie in (0, 0.5)")
        if self.xhi_range[0] >= self.xhi_range[1]:
            raise ValueError("xhi_range must be increasing")
        if self.warmup < 0 or self.iters < 1:
            raise ValueError("warmup >= 0 and iters >= 1")
        if self.cyl_chunk_cells < 4:
            raise ValueError("cyl_chunk_cells must be at least 4")
        if not (self.cyl_stages or self.cyl_z_centers):
            raise ValueError("set cyl_stages or cyl_z_centers")
        if not 0 < self.cyl_dead_band < 1:
            raise ValueError("cyl_dead_band must lie in (0, 1)")
        for name in ("cyl_native_cell_mpc", "cyl_native_los_dz", "cyl_native_los_mpc"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.cyl_native_los_dz is not None and self.cyl_native_los_mpc is not None:
            raise ValueError("set at most one of cyl_native_los_dz and cyl_native_los_mpc")
        if self.gallery_samples < 0 or not self.gallery_stages:
            raise ValueError("gallery_samples >= 0 and at least one gallery stage")

    def updated(self, **values) -> "Settings":
        """Return a copy with the given fields replaced (unknown keys fail)."""
        known = {f.name for f in fields(self)}
        unknown = set(values) - known
        if unknown:
            raise KeyError(f"unknown settings: {sorted(unknown)}")
        return replace(self, **values)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "Settings":
        return cls().updated(**{k: _coerce(k, v) for k, v in values.items()})


_TUPLES = {
    "stage_edges", "active_range", "k_targets", "front_bands_mpc",
    "xhi_range", "partial_range", "dtb_range_mk",
    "cyl_stages", "cyl_z_centers", "gallery_ids", "gallery_stages", "gallery_fields",
}


def _coerce(name: str, value):
    if name in _TUPLES and isinstance(value, list):
        return tuple(value)
    return value
