"""Cosmology and field statistics shared by every metric.

Torch-free by design: the suite must run on a laptop against saved cubes.
Every function takes plain arrays in, returns plain arrays out, so a metric
module stays a thin accumulation layer on top of these primitives.

Conventions
-----------
* Cubes are ``(Nx, Ny, Nz)``. ``x``/``y`` are periodic transverse axes,
  ``z`` is the lightcone (line-of-sight) axis and is *not* periodic.
* Transverse spectra are 2-D spectra of individual slices, in Mpc^2
  (``P_2D``), because the LOS axis mixes geometry with evolution.
* Stages are bins of the **truth** slice mean x_HI, so truth and prediction
  are always compared on the same slices.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# Constants (21cmFAST Planck-2018 defaults; Omega_m is sampled per cone)
# --------------------------------------------------------------------------- #
HUBBLE_H = 0.6766
OMEGA_B = 0.04897
OMEGA_M = 0.30966
Y_HE = 0.245
Z_HE_DOUBLE = 3.0

# Planck 2018 TT,TE,EE+lowE+lensing: tau = 0.0544 +- 0.0073,
# instantaneous-reionization midpoint z_re = 7.67 +- 0.73.
PLANCK_TAU = 0.0544
PLANCK_TAU_SIGMA = 0.0073
PLANCK_ZRE_SIGMA = 0.73

SIGMA_T_M2 = 6.6524587321e-29
C_M_S = 2.99792458e8
C_KM_S = 299792.458
M_PROTON_KG = 1.67262192369e-27
G_SI = 6.67430e-11
MPC_M = 3.0856775814913673e22

# numpy < 2.0 only has trapz
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------- #
# Reionization history
# --------------------------------------------------------------------------- #
def hubble_rate_si(z, omega_m: float = OMEGA_M, h: float = HUBBLE_H):
    """H(z) in 1/s for flat LCDM; radiation neglected (<1% at z <= 25)."""
    h0 = h * 100.0e3 / MPC_M
    z = np.asarray(z, dtype=np.float64)
    return h0 * np.sqrt(omega_m * (1.0 + z) ** 3 + (1.0 - omega_m))


def thomson_optical_depth(
    z_grid,
    xhi_mean,
    omega_m: float = OMEGA_M,
    omega_b: float = OMEGA_B,
    h: float = HUBBLE_H,
    y_he: float = Y_HE,
    z_he_double: float = Z_HE_DOUBLE,
    n_low: int = 2000,
) -> float:
    """CMB Thomson optical depth of a global reionization history.

    Helium is singly ionized alongside hydrogen and doubly ionized below
    ``z_he_double``. The history is fully ionized below the grid and fully
    neutral above it; the mean ionized fraction is clipped to [0, 1] so an
    emulator overshoot cannot produce unphysical electron densities.
    """
    z_grid = np.asarray(z_grid, dtype=np.float64)
    xhi_mean = np.asarray(xhi_mean, dtype=np.float64)
    order = np.argsort(z_grid)
    z_grid, xhi_mean = z_grid[order], xhi_mean[order]
    ionized = 1.0 - np.clip(xhi_mean, 0.0, 1.0)

    f_he = y_he / (4.0 * (1.0 - y_he))
    rho_crit = 3.0 * (h * 100.0e3 / MPC_M) ** 2 / (8.0 * np.pi * G_SI)
    n_h0 = rho_crit * omega_b * (1.0 - y_he) / M_PROTON_KG

    z_low = np.linspace(0.0, z_grid[0], n_low, endpoint=False)
    z = np.concatenate([z_low, z_grid])
    x_hii = np.concatenate([np.ones_like(z_low), ionized])
    x_e = x_hii * (1.0 + f_he) + np.where(z < z_he_double, f_he, 0.0)
    integrand = (
        C_M_S * SIGMA_T_M2 * n_h0 * (1.0 + z) ** 2 * x_e
        / hubble_rate_si(z, omega_m, h)
    )
    return float(_trapezoid(integrand, z))


def history_crossing(z_grid, xhi_mean, level: float) -> float:
    """Redshift where the global x_HI first reaches ``level`` from below.

    Scans from low to high redshift, i.e. returns the latest-time crossing,
    linearly interpolated. NaN when the level is never reached, or is already
    exceeded at the lowest redshift on the grid (crossing not resolved).
    """
    z_grid = np.asarray(z_grid, dtype=np.float64)
    xhi_mean = np.asarray(xhi_mean, dtype=np.float64)
    order = np.argsort(z_grid)
    z, x = z_grid[order], xhi_mean[order]
    above = np.flatnonzero(x >= level)
    if above.size == 0 or above[0] == 0:
        return float("nan")
    i = int(above[0])
    fraction = (level - x[i - 1]) / (x[i] - x[i - 1])
    return float(z[i - 1] + fraction * (z[i] - z[i - 1]))


def history_features(z_grid, xhi_mean) -> dict:
    """Crossing redshifts for x_HI = 0.25 / 0.5 / 0.75 and the duration."""
    z25 = history_crossing(z_grid, xhi_mean, 0.25)
    z50 = history_crossing(z_grid, xhi_mean, 0.50)
    z75 = history_crossing(z_grid, xhi_mean, 0.75)
    return {"z_xhi25": z25, "z_xhi50": z50, "z_xhi75": z75,
            "delta_z": z75 - z25}


def brightness_temperature_mk(
    xhi,
    delta,
    z_grid,
    omega_m: float = OMEGA_M,
    omega_b: float = OMEGA_B,
    h: float = HUBBLE_H,
):
    """21-cm brightness temperature in mK, saturated spin temperature, no RSD.

    ``dT_b = 27 x_HI (1+delta) (Omega_b h^2/0.023)
    sqrt(0.15/(Omega_m h^2) (1+z)/10)`` mK (Furlanetto, Oh & Briggs 2006).
    """
    z_grid = np.asarray(z_grid, dtype=np.float64)
    t0 = (
        27.0 * (omega_b * h ** 2 / 0.023)
        * np.sqrt(0.15 / (omega_m * h ** 2) * (1.0 + z_grid) / 10.0)
    )
    return t0[None, None, :] * np.asarray(xhi) * (1.0 + np.asarray(delta))


# --------------------------------------------------------------------------- #
# Slices, stages, one-point statistics
# --------------------------------------------------------------------------- #
def slice_mean(field: np.ndarray) -> np.ndarray:
    return np.asarray(field, dtype=np.float64).mean(axis=(0, 1))


def stage_labels(stage_edges, active_range) -> list[str]:
    labels = [
        f"xbar_HI {lo:.2f}-{hi:.2f}"
        for lo, hi in zip(stage_edges[:-1], stage_edges[1:])
    ]
    labels.append(f"active {active_range[0]:.2f}-{active_range[1]:.2f}")
    return labels


def stage_masks(xbar, stage_edges, active_range, min_slices: int = 1):
    """Boolean LOS masks per stage plus the trailing "active" pseudo-stage."""
    xbar = np.asarray(xbar, dtype=np.float64)
    index = np.digitize(xbar, stage_edges) - 1
    masks = []
    for s in range(len(stage_edges) - 1):
        mask = index == s
        masks.append(mask if mask.sum() >= min_slices else np.zeros_like(mask))
    masks.append((xbar >= active_range[0]) & (xbar <= active_range[1]))
    return masks


def slice_moments(field) -> dict:
    """Mean, variance and skewness of every transverse slice."""
    field = np.asarray(field, dtype=np.float64)
    mean = field.mean(axis=(0, 1))
    centered = field - mean
    variance = (centered ** 2).mean(axis=(0, 1))
    third = (centered ** 3).mean(axis=(0, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        skewness = np.where(variance > 0, third / variance ** 1.5, np.nan)
    return {"mean": mean, "variance": variance, "skewness": skewness}


def fit_even_polynomial(k, y, variance, k_max: float, n_terms: int = 2) -> dict:
    """Weighted least squares ``y(k) = sum_j c_j k^(2j)`` for ``k <= k_max``.

    Returns coefficients, 1-sigma errors from the WLS covariance, reduced
    chi^2 and the number of bins used. All-NaN when too few usable bins.
    """
    k = np.asarray(k, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    use = (k <= k_max) & np.isfinite(y) & np.isfinite(variance) & (variance > 0)
    empty = {
        "coef": np.full(n_terms, np.nan),
        "sigma": np.full(n_terms, np.nan),
        "chi2_red": float("nan"),
        "n_bins": int(use.sum()),
    }
    if use.sum() < n_terms + 1:
        return empty
    design = k[use, None] ** (2 * np.arange(n_terms))[None, :]
    weight = 1.0 / variance[use]
    normal = design.T @ (design * weight[:, None])
    try:
        covariance = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return empty
    coef = covariance @ (design.T @ (weight * y[use]))
    residual = y[use] - design @ coef
    dof = int(use.sum()) - n_terms
    return {
        "coef": coef,
        "sigma": np.sqrt(np.diag(covariance)),
        "chi2_red": float(np.sum(weight * residual ** 2) / dof),
        "n_bins": int(use.sum()),
    }


# --------------------------------------------------------------------------- #
# Distributions
# --------------------------------------------------------------------------- #
def histogram_w1(counts_a, counts_b, edges) -> float:
    """Wasserstein-1 distance between two histograms on shared ``edges``."""
    counts_a = np.asarray(counts_a, dtype=np.float64)
    counts_b = np.asarray(counts_b, dtype=np.float64)
    total_a, total_b = counts_a.sum(), counts_b.sum()
    if total_a <= 0 or total_b <= 0:
        return float("nan")
    cdf_gap = np.cumsum(counts_a / total_a - counts_b / total_b)
    return float(np.sum(np.abs(cdf_gap) * np.diff(edges)))


def wasserstein_1d(a, b) -> float:
    """Exact 1-D Wasserstein-1 distance between two sample sets."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    grid.sort()

    def cdf(values):
        return np.searchsorted(values, grid, side="right") / values.size

    return float(np.sum(np.abs(cdf(a) - cdf(b))[:-1] * np.diff(grid)))


def jensen_shannon(p, q) -> float:
    """Jensen-Shannon divergence (base e) between two discrete distributions."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)

    def kl(a, b):
        valid = a > 0
        return float(np.sum(a[valid] * np.log(a[valid] / b[valid])))

    return 0.5 * (kl(p, midpoint) + kl(q, midpoint))


# --------------------------------------------------------------------------- #
# Fourier: transverse radial binning
# --------------------------------------------------------------------------- #
class KBinner:
    """Radial k-binning of a 2-D transverse FFT grid, in physical 1/Mpc.

    Precomputes a ``(n_valid_modes, n_bins)`` indicator matrix so binned sums
    are one matmul per field. The k=0 mode and corner modes beyond the axis
    Nyquist are excluded.
    """

    def __init__(self, nx: int, ny: int, box_mpc: float, n_bins: int = 15):
        kx = 2 * np.pi * np.fft.fftfreq(nx, d=box_mpc / nx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, d=box_mpc / ny)
        k = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2).ravel()

        k_fund = 2 * np.pi / box_mpc
        k_ny = np.pi * min(nx, ny) / box_mpc
        edges = np.geomspace(k_fund, k_ny, n_bins + 1)
        edges[0] *= 1 - 1e-9
        edges[-1] *= 1 + 1e-9

        idx = np.digitize(k, edges) - 1
        self.valid = (idx >= 0) & (idx < n_bins) & (k > 0)
        self.matrix = np.zeros((int(self.valid.sum()), n_bins))
        self.matrix[np.arange(self.matrix.shape[0]), idx[self.valid]] = 1.0
        self.counts = self.matrix.sum(axis=0)
        self.edges = edges
        self.centers = np.sqrt(edges[:-1] * edges[1:])
        # unnormalized |FFT|^2 -> P(k) in Mpc^2:  P = |F|^2 L^2 / N^4
        self.power_norm = box_mpc ** 2 / float(nx * ny) ** 2

    def _binned(self, fields: np.ndarray, power: bool) -> np.ndarray:
        """(n_fields, Nx, Ny, Nz) -> (Nz, n_bins) sums, optionally cross."""
        f = np.fft.fftn(fields, axes=(1, 2))
        values = np.abs(f) ** 2 if power else np.real(f[0] * np.conj(f[1]))[None]
        flat = values.reshape(values.shape[0], -1, fields.shape[-1])
        return (flat[:, self.valid] * self.power_norm).sum(axis=0).T @ self.matrix

    def slice_sums(self, field: np.ndarray) -> np.ndarray:
        """Per-slice binned sums of |FFT|^2, shape (Nz, n_bins), P in Mpc^2."""
        return self._binned(np.asarray(field, dtype=np.float64)[None], True)

    def cross_sums(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Per-slice binned sums of Re(A conj(B)), same layout."""
        return self._binned(
            np.stack([np.asarray(a, np.float64), np.asarray(b, np.float64)]),
            False,
        )


def delta2(k, power):
    """Dimensionless 2-D power ``Delta^2(k) = k^2 P(k) / (2 pi)``."""
    return np.asarray(k, dtype=np.float64) ** 2 * np.asarray(power) / (2 * np.pi)


def safe_ratio(pt, pp, cross):
    """Power ratio ``pp/pt`` and coherence ``cross/sqrt(pt pp)``."""
    pt = np.asarray(pt, dtype=np.float64)
    pp = np.asarray(pp, dtype=np.float64)
    cross = np.asarray(cross, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(pt > 0, pp / pt, np.nan)
        denominator = np.sqrt(pt * pp)
        r = np.where(denominator > 0, cross / denominator, np.nan)
    return ratio, r


def k_crossing(k, curve, level: float) -> float:
    """Smallest k where ``curve`` first drops below ``level`` (log-interp)."""
    k = np.asarray(k, dtype=np.float64)
    curve = np.asarray(curve, dtype=np.float64)
    valid = np.isfinite(k) & np.isfinite(curve) & (k > 0)
    k, curve = k[valid], curve[valid]
    if k.size < 2:
        return float("nan")
    below = np.flatnonzero(curve < level)
    if below.size == 0:
        return float("nan")
    i = int(below[0])
    if i == 0:
        return float(k[0])
    k0, k1, c0, c1 = k[i - 1], k[i], curve[i - 1], curve[i]
    if c1 == c0:
        return float(k1)
    frac = (level - c0) / (c1 - c0)
    return float(np.exp(np.log(k0) + frac * (np.log(k1) - np.log(k0))))


# --------------------------------------------------------------------------- #
# Morphology: mean free path through the ionized mask
# --------------------------------------------------------------------------- #
def trace_ray_to_neutral_2d(
    ionized_mask: np.ndarray,
    start_xy: tuple[float, float],
    direction_xy: tuple[float, float],
    cell_size_mpc: tuple[float, float],
    max_distance_mpc: float,
) -> float:
    """Trace one periodic transverse ray to the first neutral cell.

    The ray starts inside an ionized cell. ``direction_xy`` is interpreted in
    physical coordinates and normalized internally. A digital differential
    analyzer advances cell boundary to cell boundary, so the result does not
    depend on a ray-marching step size.

    Returns NaN when no neutral cell is reached before ``max_distance_mpc``:
    a right-censored ray, as for a percolating or fully ionized slice.
    """
    mask = np.asarray(ionized_mask, dtype=bool)
    if mask.ndim != 2 or not all(size > 0 for size in mask.shape):
        raise ValueError("ionized_mask must be a non-empty 2-D array")
    sx, sy = (float(value) for value in start_xy)
    ux, uy = (float(value) for value in direction_xy)
    dx, dy = (float(value) for value in cell_size_mpc)
    max_distance = float(max_distance_mpc)
    if not np.all(np.isfinite([sx, sy, ux, uy, dx, dy, max_distance])):
        raise ValueError("ray inputs and distances must be finite")
    if dx <= 0 or dy <= 0 or max_distance <= 0:
        raise ValueError("cell sizes and max distance must be positive")
    norm = float(np.hypot(ux, uy))
    if norm == 0:
        raise ValueError("direction_xy must be non-zero")
    ux, uy = ux / norm, uy / norm

    nx, ny = mask.shape
    sx %= nx
    sy %= ny
    ix, iy = int(np.floor(sx)), int(np.floor(sy))
    if not mask[ix, iy]:
        raise ValueError("ray must start inside an ionized cell")

    vx, vy = ux / dx, uy / dy          # grid-cell units per Mpc

    def axis_state(position, index, velocity):
        if velocity > 0:
            return 1, (index + 1.0 - position) / velocity, 1.0 / velocity
        if velocity < 0:
            speed = -velocity
            return -1, (position - index) / speed, 1.0 / speed
        return 0, np.inf, np.inf

    step_x, next_x, delta_x = axis_state(sx, ix, vx)
    step_y, next_y, delta_y = axis_state(sy, iy, vy)

    while True:
        distance = min(next_x, next_y)
        if distance > max_distance:
            return float("nan")
        tolerance = 1e-12 * max(1.0, distance)
        if next_x <= distance + tolerance:
            ix = (ix + step_x) % nx
            next_x += delta_x
        if next_y <= distance + tolerance:
            iy = (iy + step_y) % ny
            next_y += delta_y
        if not mask[ix, iy]:
            return float(distance)


def mean_free_path_samples_2d(
    ionized_mask: np.ndarray,
    n_rays: int,
    cell_size_mpc: tuple[float, float],
    max_distance_mpc: float,
    seed=None,
) -> dict:
    """Sample a transverse mean-free-path bubble-size distribution.

    Starting cells are uniform over ionized area, sub-cell starts uniform and
    directions isotropic, so the distance distribution is ionized-area
    weighted. Boundaries are periodic. Only boundary hits are returned in
    ``distances_mpc``; censored rays are counted separately.
    """
    mask = np.asarray(ionized_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("ionized_mask must be 2-D")
    n_rays = int(n_rays)
    if n_rays < 0:
        raise ValueError("n_rays must be non-negative")
    cells = np.argwhere(mask)
    if n_rays == 0 or cells.size == 0:
        return {
            "distances_mpc": np.empty(0, dtype=np.float64),
            "n_requested": n_rays,
            "n_started": 0,
            "n_hit": 0,
            "n_censored": 0,
            "ionized_fraction": float(mask.mean()),
        }

    rng = np.random.default_rng(seed)
    chosen = cells[rng.integers(0, len(cells), size=n_rays)]
    offsets = rng.random((n_rays, 2))
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_rays)
    distances = np.empty(n_rays, dtype=np.float64)
    for index in range(n_rays):
        start = chosen[index].astype(np.float64) + offsets[index]
        direction = (np.cos(angles[index]), np.sin(angles[index]))
        distances[index] = trace_ray_to_neutral_2d(
            mask, (float(start[0]), float(start[1])), direction,
            cell_size_mpc, max_distance_mpc,
        )
    hit = np.isfinite(distances)
    return {
        "distances_mpc": distances[hit],
        "n_requested": n_rays,
        "n_started": n_rays,
        "n_hit": int(hit.sum()),
        "n_censored": int((~hit).sum()),
        "ionized_fraction": float(mask.mean()),
    }


def neutral_region_summary(field, threshold: float = 0.5) -> dict:
    """Connected-component summary of the neutral (``field >= threshold``) mask."""
    from scipy import ndimage

    labels, count = ndimage.label(np.asarray(field) >= threshold)
    volumes = np.bincount(labels.ravel())[1:].astype(np.float64)
    if volumes.size == 0:
        return {"count": 0, "mean_effective_radius_cells": 0.0,
                "volume_weighted_radius_cells": 0.0,
                "largest_fraction": 0.0}
    radii = (3.0 * volumes / (4.0 * np.pi)) ** (1.0 / 3.0)
    return {
        "count": int(count),
        "mean_effective_radius_cells": float(radii.mean()),
        "volume_weighted_radius_cells": float(np.average(radii, weights=volumes)),
        "largest_fraction": float(volumes.max() / field.size),
    }


def signed_distance_2d(mask, sampling) -> np.ndarray:
    """Signed distance to the mask boundary: positive inside, in Mpc.

    NaN everywhere when the slice has no boundary (all-True or all-False).
    """
    from scipy.ndimage import distance_transform_edt

    mask = np.asarray(mask, dtype=bool)
    if mask.all() or (~mask).all():
        return np.full(mask.shape, np.nan, dtype=np.float64)
    return (distance_transform_edt(mask, sampling=sampling)
            - distance_transform_edt(~mask, sampling=sampling))


def front_width(d_mpc, mean_field, lo: float = 0.1, hi: float = 0.9) -> float:
    """10-90 transition width (Mpc) of a mean x_HI-vs-signed-distance curve."""
    m = np.isfinite(mean_field)
    d, f = np.asarray(d_mpc)[m], np.asarray(mean_field)[m]
    order = np.argsort(d)
    d, f = d[order], f[order]
    if f.size == 0 or f.min() > lo or f.max() < hi:
        return float("nan")

    def crossing(level):
        idx = np.where(np.diff(np.sign(f - level)) != 0)[0]
        if len(idx) == 0:
            return float("nan")
        i = idx[0]
        f0, f1, d0, d1 = f[i], f[i + 1], d[i], d[i + 1]
        return d0 + (level - f0) * (d1 - d0) / (f1 - f0) if f1 != f0 else d0

    return float(crossing(hi) - crossing(lo))


def los_cell_size_mpc(z: float, dz_cell: float, omega_m: float = OMEGA_M,
                      h: float = HUBBLE_H) -> float:
    """Comoving size of one LOS cell at redshift z: dD = c dz / H(z)."""
    e_z = np.sqrt(omega_m * (1.0 + z) ** 3 + (1.0 - omega_m))
    return C_KM_S / (h * 100.0 * e_z) * dz_cell
