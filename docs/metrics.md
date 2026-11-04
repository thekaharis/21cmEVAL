# Metrics

A metric is one evaluation family. It receives the same `(sample, prediction)` pair as every
other enabled metric, updates a streaming accumulator, and returns a nested dict of JSON-safe
numbers at the end. Metrics never re-read the data and never keep a cube after `update`
returns.

```console
$ eval21 list-metrics
anisotropy   Cylindrical (k_perp, k_par) power ratio and coherence.
bubble       Bubble sizes from transverse mean-free paths.
cylindrical  Cylindrical power P(k_perp, k_par) on lightcone chunks: ratio, coherence, in-band share.
…
```

## The metric contract

```python
from eval21.registry import register
from eval21.metrics.base import Metric

@register("mystat")
class MyStat(Metric):
    needs = ("density",)        # required input fields (optional)
    targets = ("xhi",)          # restrict to these targets (None = any)
    pointwise = False           # True / "partial" / False (see below)

    def update(self, sample, prediction, aux=None, runtime=None): ...
    def result(self): ...       # nested dict, JSON-safe
    def arrays(self): ...       # optional arrays for curves.npz
```

| attribute | meaning |
|---|---|
| `needs` | input fields the metric requires; missing ones skip the metric with a reason |
| `targets` | target fields the metric is defined for; a T_b pass skips x_HI statistics |
| `pointwise` | `True`: voxel-by-voxel, skipped in `--mode generative`; `"partial"`: some keys are, stripped and listed in `_generative_dropped`; `False`: distributional only |
| `pointwise_keys` | key substrings to strip from a `"partial"` metric in generative mode |
| `wants_draws` | the metric reads every draw from `aux["draws"]` |
| `requires_rerun` | the runner re-runs the adapter on the x↔y-transposed sample and passes it in `aux["xy"]` |

Helpers on `Metric`: `self.target(sample)`, `self.prediction(pred)`, `self.stage_xbar(sample)`,
`self.stages(truth, sample)`, `self.require(sample, field)`, `self.box_mpc(sample)`,
`self.omega_m(sample)`, `self.z_grid(sample)`.

`result()` may return NumPy scalars and arrays; `report.jsonable` converts them and turns
non-finite floats into `null`.

## Stage-resolved reporting

Most metrics report per reionization stage. A stage is a bin of the **true** x_HI transverse
slice mean (`stage_field`, default `xhi`), not of the target: staging on truth pairs truth and
prediction and lets an x_HI pass and a T_b pass of the same cones use identical slices.

| setting | default | meaning |
|---|---|---|
| `stage_edges` | `(0.02, 0.2, 0.4, 0.6, 0.8, 0.98)` | five stage bins |
| `active_range` | `(0.05, 0.95)` | the trailing `active` pseudo-stage |
| `min_stage_slices` | `2` | stages thinner than this are reported but not fitted |

`stage_labels()` produces `xbar_HI 0.02-0.20` … and `active 0.05-0.95`; `report.md` and the
figures use these labels. Per-sample values are reduced to a median with 16–84 percentiles
(`median_band`) unless the metric's definition says otherwise.

Two conventions are worth stating because they are easy to misread:

- **Medians, not pooled voxels.** Stage tables quote the median over samples/slices, so a
  handful of pathological cones cannot dominate.
- **`null` means undefined.** An empty stage, an all-NaN fit, or a metric that never ran is
  `null` in JSON and `—` in Markdown, never `0`.

## Settings

All knobs live in one frozen dataclass and are serialised into `results.json["settings"]`.
Override with `--set key=value` or a `[settings]` TOML section.

### Run and geometry

| key | default | meaning |
|---|---|---|
| `target` | `"xhi"` | target field scored |
| `density` | `"density"` | density field name |
| `box_mpc` | `200.0` | transverse box, Mpc (fallback when a sample has none) |
| `omega_m` | `0.30966` | Ω_m (fallback) |
| `omega_b` | `0.04897` | Ω_b, for dT_b and τ_e |
| `hubble_h` | `0.6766` | h |
| `threshold` | `0.5` | ionized/neutral split (`x_HI < threshold` is ionized) |

### Stages and spectra

| key | default | meaning |
|---|---|---|
| `stage_field` | `"xhi"` | field whose slice mean defines stages |
| `stage_edges` | `(0.02, 0.2, 0.4, 0.6, 0.8, 0.98)` | stage bins |
| `active_range` | `(0.05, 0.95)` | active pseudo-stage |
| `min_stage_slices` | `2` | minimum slices per stage for fits |
| `n_k_bins` | `15` | log k bins for 2-D spectra |
| `k_targets` | `(0.1, 0.2, 0.5)` | k values quoted in headlines, 1/Mpc |
| `eft_k_max` | `0.25` | maximum k for the EFT fits, 1/Mpc |
| `coherence_level` | `0.9` | threshold for `k_r90` |
| `highk_k_min` | `0.2` | high-k band start, cycles/pixel |
| `highk_bins` | `12` | high-k band count |

### Voxel and edge

| key | default | meaning |
|---|---|---|
| `edge_width_fraction` | `0.10` | transverse edge width as a fraction of Nx |
| `active_variance_fraction` | `0.05` | slice std threshold for the active window (fraction of peak) |

### Bubble sizes

| key | default | meaning |
|---|---|---|
| `rays_per_slice` | `256` | MFP rays per slice (0 disables sampling) |
| `slices_per_stage` | `6` | slices sampled per stage |
| `bubble_bins` | `24` | histogram bins |
| `max_distance_mpc` | `None` | censoring distance; default the transverse box size |
| `seed` | `0` | ray RNG seed |

### Fronts

| key | default | meaning |
|---|---|---|
| `front_bands_mpc` | `(2.0, 5.0)` | band half-widths for band L2/H1 |
| `front_d_max_mpc` | `30.0` | signed-distance window |
| `front_bin_mpc` | `1.5` | distance bin width |

### Observable PDFs

| key | default | meaning |
|---|---|---|
| `xhi_range` | `(-0.2, 1.2)` | x_HI histogram range (clipped values) |
| `xhi_bins` | `70` | x_HI histogram bins |
| `partial_range` | `(0.1, 0.9)` | "partially ionized" definition |
| `dtb_range_mk` | `(-5.0, 100.0)` | dT_b histogram range, mK |
| `dtb_bins` | `105` | dT_b histogram bins |

### Cylindrical chunks

| key | default | meaning |
|---|---|---|
| `cyl_chunk_cells` | `140` | LOS cells per chunk (140 = 200 Mpc on the 21cmFAST grid) |
| `cyl_stages` | `(0.9, 0.7, 0.5, 0.3, 0.1)` | chunk centres by true slice-mean x_HI |
| `cyl_z_centers` | `()` | fixed-z chunk centres instead of stages |
| `cyl_dead_band` | `0.1` | \|P_pred/P_true − 1\| counted as "within" |
| `cyl_native_cell_mpc` | `None` | native transverse cell of an interpolated truth |
| `cyl_native_los_dz` | `None` | native LOS spacing uniform in z |
| `cyl_native_los_mpc` | `None` | native LOS spacing uniform in comoving distance |

### Gallery

| key | default | meaning |
|---|---|---|
| `gallery_samples` | `4` | first N samples stored (ignored when `gallery_ids` is set) |
| `gallery_ids` | `()` | explicit sample ids |
| `gallery_stages` | `(0.8, 0.6, 0.4, 0.2)` | target true slice-mean x_HI |
| `gallery_fields` | `("xhi", "tb")` | fields to store when the sample provides them |

### Ensembles, efficiency

| key | default | meaning |
|---|---|---|
| `draws` | `1` | predictions per sample |
| `mode` | `"pointwise"` | or `"generative"` |
| `score_draw` | `"first"` | `"first"` or `"mean"` |
| `ensemble_voxels` | `200000` | voxels sampled per sample for CRPS/rank statistics |
| `ensemble_n_los` | `16` | LOS positions for the x̄(z) PITs |
| `ensemble_k` | `(0.1, 0.3, 1.0)` | 1/Mpc, power PITs per stage |
| `store_draws` | `4` | draws kept by `--save-predictions` |
| `warmup` | `3` | benchmark warmup iterations |
| `iters` | `10` | benchmark timed iterations |
| `benchmark_samples` | `4` | samples passed to `adapter.benchmark()` |

`Settings.__post_init__` validates ranges: positive box, increasing `stage_edges` and
`active_range`, `0 < threshold < 1`, `0 < cyl_dead_band < 1`, at least one of `cyl_stages` /
`cyl_z_centers`, at most one native LOS spec, and so on.

## Metric reference

### `voxel`

Voxel-level accuracy over every evaluated voxel of the split. Accumulation is streaming, so
memory does not grow with the number of samples.

| key | meaning |
|---|---|
| `rmse`, `mse`, `mae`, `bias` | RMSE, MSE, mean absolute error, mean (pred − truth) |
| `r2` | `1 − Σ(p−t)² / Σ(t−mean t)²`, constant-mean reference |
| `pearson_r` | clip(cov(p,t) / σ_p σ_t, −1, 1) |
| `truth_mean`, `truth_std` | of the evaluated truth voxels |
| `r2_per_sample_mean`, `r2_per_sample_p16`, `r2_per_sample_p84` | distribution of per-sample R² |
| `gradient_rmse` | RMSE of periodic forward-difference gradients in the transverse plane |
| `mean_xhi_mae` | MAE of the per-cone global history x̄(z) |
| `high_k_power_ratio` | summed \|FFT\|² above `highk_k_min` (cycles/pixel), pred/truth |
| `high_k_cross_correlation` | cross/√(P_p P_t) over the same band |
| `n_voxels`, `n_samples` | counts |

R² is a *relative* score: x_HI is bimodal (mass at 0 and 1), so the constant-mean predictor is
poor and R² flatters every model. Quote RMSE and the conditional diagnostics alongside it.

### `edge`

Where in the lightcone the error lives. Whole-volume RMSE is dominated by trivially correct
saturated regions.

| key | meaning |
|---|---|
| `transverse_edge_rmse` | RMSE in the outer `edge_width_fraction` of the transverse plane |
| `transverse_interior_rmse` | RMSE inside |
| `active_window_rmse` | RMSE over slices whose transverse std exceeds `active_variance_fraction` of its peak |
| `active_window_fraction` | share of voxels in that window |
| `active_z_min`, `active_z_max` | median redshift bounds of the active window |
| `active_z_min_p16`, `active_z_max_p84` | 16–84 % band of those bounds |
| `global_history_rmse` | RMSE of x̄(z) |

### `spectrum`

Scale-resolved power. For every LOS slice the 2-D transverse power spectrum is binned radially
in physical 1/Mpc and aggregated by stage (truth slice mean).

```text
ratio(k)   = P_pred(k) / P_truth(k)
r(k)       = P_cross(k) / sqrt(P_truth P_pred)      coherence
Delta^2(k) = k^2 P(k) / 2 pi                        dimensionless
k_r90      = smallest k where r(k) < coherence_level
```

Top-level: `k_centers`, `stage_labels`, `stages`, `active`, `n_samples`.

| stage key | meaning |
|---|---|
| `label`, `n_slices` | stage label and slice count |
| `power_ratio_med`, `_p16`, `_p84` | median power ratio and band |
| `coherence_med`, `_p16`, `_p84` | median coherence and band |
| `delta2_truth_med`, `delta2_pred_med` | dimensionless power, Mpc² units folded in |
| `k_r90` | coherence scale, 1/Mpc |
| `highk_logmse` | stage-resolved `HighKPowerRatio` kernel (log-MSE above `highk_k_min`) |

`active` repeats the active stage and adds `ratio_k<k>` / `coherence_k<k>` at each
`k_targets` value, plus `k_ref` (0.2 1/Mpc), `ratio_ref`, `coherence_ref`, `k_r90`,
`highk_logmse`.

The LOS axis is deliberately *not* Fourier transformed: it mixes geometry with cosmic
evolution, so a 2-D transverse spectrum is the well-defined statistic. `k` therefore
corresponds to 3-D wavenumbers of roughly 1–2 k_perp.

### `radial`

Isotropic 3-D power ratio and coherence — a cross-check, not the headline. The LOS axis is
non-periodic, so the transform mixes geometry with evolution; a Hanning window suppresses the
edge discontinuity and radial binning uses physical cells with the mean LOS spacing.

`k_centers`, `power_ratio_med/p16/p84`, `coherence_med/p16/p84`,
`headline {k_ref, ratio_ref, coherence_ref, k_r90}`, `note`, `n_samples`.

### `anisotropy`

The lighter cylindrical variant: fixed-z chunks of 32 cells, median maps only, and a
low/high-k_par headline.

| key | meaning |
|---|---|
| `chunks[]` | per chunk: `z`, `kperp_centers`, `kpar_centers`, `ratio_low_kpar`, `ratio_high_kpar`, `coherence_low_kpar`, `coherence_high_kpar`, `anisotropy` (= ratio_high / ratio_low) |
| `headline` | `anisotropy_median`, `coherence_low_kpar`, `coherence_high_kpar` |
| `skipped_samples`, `n_samples` | counts |

The k_par = 0 plane is dropped, so both axes are strictly positive wavenumbers; ratios need no
physical normalisation because it cancels cell by cell.

### `cylindrical`

The full 2-D `(k_perp, k_par)` power spectrum of the 21-cm literature on lightcone chunks — the
LOS counterpart of `spectrum`. Structure along the line of sight, slice-to-slice jitter of
slice-wise models and LOS smoothing all show up at k_par.

Per chunk of `cyl_chunk_cells` LOS cells of one transverse cell each: the per-slice mean is
removed, a Hann window is applied along the LOS, then a 3-D FFT. `|F_t|²`, `|F_p|²` and
`Re(F_t F_p*)` are binned in k_perp (`n_k_bins` log bins) and k_par (one row per LOS frequency,
DC dropped). Per sample and chunk:

```text
ratio(k_perp, k_par) = P_pred / P_truth
r(k_perp, k_par)     = P_cross / sqrt(P_truth P_pred)
```

| key | meaning |
|---|---|
| `mode` | `"stage"` or `"z"` |
| `chunk_cells`, `chunk_mpc`, `dead_band` | geometry of the chunks |
| `stages[]` | per column: `label`, `value` (stage x̄_HI or z), `n_samples`, `z_median`, `within_dead_band`, `median_abs_error`, `median_coherence`, `resolved_cells` |
| `within_dead_band_mean` | mean share over columns |
| `kperp_max` | largest usable k_perp, 1/Mpc |

Chunk placement follows `cyl_stages` by default: the chunk is centred on the slice whose true
x_HI is nearest the stage; a sample whose history comes no closer than 0.05 is skipped for that
stage, and a chunk that would leave the cone is shifted and dropped if its centre then sits
more than 0.1 from the stage. With `cyl_z_centers` chunks sit at fixed redshifts instead.

LOS geometry follows the sample's `z_grid` (comoving distance, flat ΛCDM with the suite's
cosmology). A grid already uniform at the transverse cell size (within 0.2 %) is sliced
directly; any other grid is linearly resampled inside the chunk, and k_par above
`π / (coarsest spacing)` is marked unresolved. Unresolved cells are excluded from the in-band
share and hatched in the figures. `cyl_native_cell_mpc`, `cyl_native_los_dz` /
`cyl_native_los_mpc` declare that the *truth* was interpolated from a coarser native grid
(LoReLi II: 2.3 Mpc cells, planes uniform in z with dz = 10/1535) and add those limits.

`eval21 plot cyl` draws these maps; the per-sample maps are exported to `curves.npz` as
`<model>__cylindrical__cyl_*`.

### `bubble`

Ionized-bubble sizes from transverse mean-free paths. For each stage a handful of slices is
sampled (`slices_per_stage`, `rays_per_slice`); starting points are uniform over ionized area
and directions isotropic, so the distance to the first neutral cell is an ionized-area-weighted
MFP sample. Periodic transverse boundaries make a percolating slice return censored rays rather
than a fake edge.

Top-level: `radius_edges_mpc`, `radius_centers_mpc`, `stage_labels`, `stages`, `active`,
`n_samples`.

| stage key | meaning |
|---|---|
| `n_slices` | slices contributing |
| `wasserstein_med_mpc`, `_p16`, `_p84` | restricted W1 distance, censored rays folded in at `max_distance_mpc` |
| `js_divergence_med` | Jensen–Shannon divergence (base e) |
| `mean_size_truth_mpc`, `mean_size_pred_mpc` | mean bubble size |
| `relative_bias_med`, `_p16`, `_p84` | pred/truth − 1 of the mean size |
| `censored_truth`, `censored_pred` | share of rays that hit no neutral cell |
| `underflow_truth`, `underflow_pred` | share below the first radius bin |
| `ionized_truth`, `ionized_pred` | ionized area fraction |
| `mean_hit_truth_mpc`, `mean_hit_pred_mpc` | mean ray length without censoring |

`active` contains `wasserstein_med_mpc`, `js_divergence_med`, `relative_bias_med`.

A model that blurs fronts systematically over-produces small bubbles and under-produces large
ones; W1 catches that where voxel RMSE does not.

### `morphology`

A 3-D connected-component census of `x_HI < threshold`: ionized bubbles embedded in the neutral
IGM. Radii are effective radii of equal volume, in cells; multiply by the cell size to quote
Mpc. IoU is reported at the fixed threshold, with the predicted mask restricted to the true
ionized cell count (`matched`), and at ionized-fraction targets `f0.1` … `f0.9`.

```text
truth / pred: {count, mean_effective_radius_cells, volume_weighted_radius_cells, largest_fraction}
count_ratio, mean_radius_ratio, largest_fraction_ratio
iou: {threshold, matched, f0.1, f0.25, f0.5, f0.75, f0.9}
```

`count_ratio` and the radius ratios separate "wrong topology" from "wrong ionized fraction".

### `history`

Global x_HI(z) reduced to the numbers observers quote.

| key | meaning |
|---|---|
| `tau.truth_med`, `tau.pred_med` | τ_e median over samples |
| `tau.error_med` | median (pred − truth) |
| `tau.error_sigma_med` | median \|Δτ\| / σ_Planck, σ = 0.0073 |
| `tau.tau_truth_pull_from_planck` | (τ_truth − 0.0544) / 0.0073 |
| `history.z_xhi25/50/75.{truth_med, pred_med, error_med, error_sigma_med}` | crossing redshifts, σ_z = 0.73 |
| `history.delta_z.{truth_med, pred_med, error_med}` | z(0.75) − z(0.25), no σ |
| `history_rmse_med` | median RMSE of the x̄(z) curve |
| `n_samples` | count |

τ_e integrates a global history with helium single ionization and He II → He III at z = 3;
histories are clipped to [0, 1] so emulator overshoot cannot produce unphysical electron
densities. A model can win on voxel RMSE and still shift the history — that is the point of
this metric.

### `observable`

The 21-cm observable, not just x_HI. The prediction is re-expressed in brightness temperature
and compared through one-point statistics and power.

dT_b uses the saturated-spin-temperature, no-RSD limit

```text
dT_b = 27 x_HI (1+δ) (Ω_b h²/0.023) sqrt(0.15/(Ω_m h²) · (1+z)/10)  mK
```

with the *same* density for truth and prediction, so differences isolate the x_HI error. It is
not the 21cmFAST `brightness_temp` field (no T_S, no velocities).

Top-level: `stage_labels`, `stages`, `dxbar`, `active`, `skipped_samples`, `n_samples`.

| stage key | meaning |
|---|---|
| `w1_xhi` | W1 distance between x_HI PDFs |
| `w1_dtb_mk` | W1 between dT_b PDFs, mK |
| `partial_fraction_truth/pred/excess` | share in `partial_range` (0.1 < x_HI < 0.9) and pred − truth |
| `dtb_variance_ratio`, `dtb_skewness_delta` | second and third moments vs the global neutral fraction |
| `power_bias_significance_dtb/xhi` | median \|(P_pred − P_truth)/(P_truth √(2/N_modes))\| |
| `delta2_dtb_ratio[]`, `delta2_xhi_ratio[]` | Δ² ratios at each `k_targets` value |

`dxbar` holds the global-history bias `x̄_pred − x̄_truth`: `med`, `p16`, `p84`, `abs_max`,
`n_z`. `active` repeats the active stage's W1s, partial excess, variance ratio and
significances.

Power-bias significance is the emulator bias in units of the Gaussian sample variance of a
single slice; `1/significance²` is how many independent slices an observer must average before
the bias equals their statistical error.

### `dtb`

The model's own brightness temperature, scored like fno-21cm's `--dtb-from-npz`
(`viz/physical_evaluation.py`, `physical_dtb`). Where `observable` recomputes dT_b from x_HI in
the saturated limit, this metric takes the `tb` field the model actually predicts (mK) and
compares it with the simulated T_b. It runs only on a `--target tb` pass: on an x_HI run it
returns a note and increments `skipped_samples`.

Stages are the true x_HI slice means, so the sample must carry the true x_HI as an input (an
`xhi` field alongside the `tb` target, as in LoReLi's T_b manifests).

| key | meaning |
|---|---|
| `stage_labels`, `stages`, `active` | stage-resolved block; `active` is the last stage |
| stage `label`, `n_slices` | stage label and slice count |
| stage `w1_dtb_mk` | Wasserstein-1 distance between the pooled dT_b PDFs, mK |
| stage `delta2_dtb_ratio[]` | Δ²₂₁(pred)/Δ²₂₁(truth) at each `k_targets` value |
| stage `power_bias_significance_dtb[]` | (P_pred − P_truth)/σ_single-slice per k target |
| stage `k_targets` | the k values of those two lists, 1/Mpc |
| `dtb_range_mk` | histogram range actually used |
| `skipped_samples`, `n_samples` | counts |

Histogram range and bins are metric options: `[metric_options.dtb] dtb_range_mk = [-250, 100]`,
`dtb_bins = 350` (the default range is wider than `observable`'s, to cover the absorption
trough). The per-sample PDFs, per-slice moments and binned Δ² curves are exported as `dtb_*`
arrays. There is no `eval21 plot` figure for this metric yet; use `curves.npz`.

### `front3d`

### `eft`

Effective-field-theory bias diagnostics. With `u = x_HI − <x_HI>` per slice and the input
density `δ`:

```text
T(k)     = P_u,δ(k) / P_δ(k)                     transfer
P_ε(k)   = P_uu(k) − P_u,δ(k)² / P_δ(k)          stochasticity
r_u,δ    = P_u,δ / sqrt(P_uu P_δ)                correlation
```

Per stage, `T` is fitted as `b₁ + b_k2 k²` and `P_ε` as `P_ε,0 + P_ε,2 k²` for
`k ≤ eft_k_max`. Truth and prediction coefficients are compared cone by cone and the
difference is scaled by the truth fit's error.

Top-level: `stage_labels`, `stages`, `active`, `n_samples`.

| stage key | meaning |
|---|---|
| `b1_truth`, `b1_pred`, `b1_delta_sigma` | linear bias and (pred − truth)/σ_truth |
| `bk2_truth`, `bk2_pred` | k² correction |
| `eps0_truth`, `eps0_pred`, `eps0_ratio` | stochasticity amplitude and ratio |
| `transfer_ratio[]` | T_pred/T_truth at each `k_targets` value |
| `r_ud_truth`, `r_ud_pred` | correlation at the middle k target |

Normalisation follows fno-21cm exactly: `n_modes = N_modes · n_slices`,
`n_independent = n_modes/2`, `P_ε` is a per-mode power, non-positive `P_ε` bins are dropped,
and the fit variances are `P_ε/(n_independent P_δδ)` for the transfer function and
`2 P_ε²/n_independent` for the stochastic term.

A conditional-mean (MSE-trained) emulator usually reproduces T(k) but under-predicts P_ε: it
produces the mean field given the density and drops the part of x_HI the density does not
determine. That failure mode is invisible to RMSE. These are 2-D spectra of thin slices, so
read coefficient *differences*, not absolute errors, and treat the fit errors as approximate
(slices are not independent).

### `front`

Error versus distance to the ionization front. Every voxel is binned by its signed distance to
the nearest **true** front (positive into the neutral side); distances use physical transverse
sampling in Mpc, so anisotropic cells are handled correctly. Slices with no front are skipped.

| key | meaning |
|---|---|
| `d_centers_mpc`, `mean_truth`, `mean_pred` | signed-distance bins and mean profiles |
| `rmse_profile`, `gradient_rms_profile` | RMSE and gradient RMS per distance bin |
| `front_width_truth_mpc`, `front_width_pred_mpc` | 10–90 transition width of the pooled profile |
| `front_width_excess_mpc` | pred − truth |
| `front_width_per_sample_truth_mpc`, `_pred_mpc` | median of per-sample widths |
| `front_width_status` | `ok` or `unresolved: …` |
| `saturated_rmse` | RMSE where the truth is already 0 or 1 |
| `pred_where_truth_neutral_mean`, `pred_where_truth_ionized_mean` | conditional means |
| `bands` | `rmse_within_<b>mpc`, `h1_within_<b>mpc`, `errfrac_within_<b>mpc` and their `_mean` per sample, for `b` in `front_bands_mpc` |
| `n_samples`, `skipped_samples` | counts |

A diffuse front appears here as a larger width, in `spectrum` as suppressed high-k power and
low coherence, and in `bubble` as an excess of small bubbles; the three diagnostics should
agree.

### `front3d`

The 3-D counterpart of `front` (fno-21cm `boundary_band_diagnostic --mode 3d`). The signed
distance to the true `x_HI = threshold` surface is a 3-D Euclidean distance transform with
sampling `(dx, dy, los_cell_mpc)`; unlike `front`, the error gradient includes the LOS
derivative. A single mean LOS cell size is used, so distances are approximate along the
lightcone — the original's caveat. `los_cell_mpc` is a metric option (default 13 Mpc):
`[metric_options.front3d] los_cell_mpc = 13.0`.

| key | meaning |
|---|---|
| `los_cell_mpc` | LOS sampling used by the transform |
| `front_width_truth_mpc`, `front_width_pred_mpc` | 10–90 transition width of the pooled profile |
| `bands` | `L2_band<b>_mean`, `H1_band<b>_mean`, `errfrac_band<b>_mean` for each `front_bands_mpc` |
| `n_samples_with_fronts`, `n_samples` | samples that contained a front |

The metric is voxel-wise and x_HI-only: a `--mode generative` run skips it, and a T_b pass
skips it with the standard target reason. It exports the same profile and band arrays as
`front` under the `front3d_*` prefix.

### `parity`

Calibration: which part of the x_HI range the model gets wrong? Voxels are binned by their
**true** value (50 bins on [0, 1], values clipped in so x_HI = 1.0 is counted); each bin reports
the conditional distribution of the prediction.

| key | meaning |
|---|---|
| `bins[].center`, `count` | bin centre and occupancy |
| `bins[].truth_mean`, `pred_mean` | bin means |
| `bins[].pred_std`, `pred_p16`, `pred_p50`, `pred_p84` | spread, from a 50 × 400 count histogram |
| `bins[].bias`, `rmse` | pred_mean − truth_mean, RMSE within the bin |
| `headline.conditional_bias_max` | largest \|bias\| among bins with ≥ max(5, 0.1 % of voxels) |
| `headline.conditional_bias_max_count`, `conditional_bias_min_count` | its occupancy and the threshold |
| `headline.partial_rmse`, `partial_voxels` | RMSE over `partial_range` only |
| `n_voxels` | total |

Systematic departure of the conditional mean from the identity is the hedging-toward-the-mean
signature of an MSE-trained emulator in the transition range. Bin occupancy is reported with
it because x_HI is strongly bimodal and sparse mid-range bins would otherwise look as
trustworthy as saturated ones.

### `symmetry`

Transverse isotropy. The plane has no preferred axis, so a model whose output changes when the
input is transposed is fitting grid artefacts. The runner re-runs the adapter on
`Sample.transposed_xy()` and this metric reports the difference between the back-transposed
output and the original prediction.

`xy_rms` (median RMS difference), `xy_rms_p84`, `xy_relative` (RMS / prediction std),
`n_samples`. It needs no target, only a re-runnable adapter; at double the inference cost.

### `efficiency`

Cost of a forward pass, measured on the same samples the accuracy metrics see.

| key | meaning |
|---|---|
| `parameters`, `parameters_complex` | from the adapter's `describe()` |
| `ms_per_sample`, `ms_per_sample_p16`, `ms_per_sample_p84` | wall-clock per lightcone |
| `samples_per_s`, `slices_per_s` | throughput |
| `timing` | `forward_only` (dedicated benchmark) or `per_call` (timings from the loop) |
| `peak_memory_mib` | peak accelerator memory when the adapter reports it |
| `n_timed`, `device` | count and device |

If the adapter implements `benchmark(samples, warmup, iters)`, its pre-transferred, synchronised
timings replace the per-call numbers; `peak_memory_bytes` from the adapter becomes
`peak_memory_mib`. Accuracy per parameter and per millisecond is how surrogate models are
compared in practice: a 0.002 RMSE win that costs 40× the inference time is usually not a win.

### `ensemble`

Ensemble metrics for stochastic and generative models (`--draws N`). The runner passes every
draw in `aux["draws"]`, shape `(D, Nx, Ny, Nz)`. Without draws the metric returns a `note`
explaining how to enable it, or that all draws are identical (a deterministic model has no
ensemble to calibrate).

**Voxel calibration** (meaningful for a probabilistic surrogate conditioned on the same fields
as the truth):

| key | meaning |
|---|---|
| `crps_med` | fair CRPS per voxel, median over voxels |
| `spread_skill_med` | √((D+1)/D) · mean ensemble std / RMSE of the ensemble mean; 1 = calibrated |
| `rank_chi2` | χ² flatness of the rank histogram |
| `coverage50`, `coverage90` | rank-based central coverage |
| `coverage50_nominal`, `coverage90_nominal` | exact nominal values m/(D+1) |
| `rank_histogram` | counts over the D+1 ranks (ties broken at random, needed for x_HI = 0/1) |

`voxel_stages[]` repeats `crps` and `spread_skill` per stage.

**Summary-statistic calibration** (meaningful for both kinds): per sample the same statistics
are computed for the truth and for each draw — `xbar@z` at `ensemble_n_los` LOS positions,
`history` crossings and duration (x_HI targets), and `power@stage` Δ²(k) at `ensemble_k` — and
the truth is located in the draw distribution. Reported as randomised PIT
`(rank + U)/(D+1)`, `z = (truth − mean)/std`, KS distance, coverage, and mean z².

```text
statistics.<family>: {n_stats, pit_ks_med, coverage68, coverage68_nominal, coverage95,
                      coverage95_nominal, mean_z2_med, pit_hist}
per_statistic.<name>: {n, pit_ks, mean_z2, pit_hist, coverage68, coverage68_nominal,
                       coverage95, coverage95_nominal}
families: xbar, history, power
```

A calibrated model gives uniform PITs (KS ≈ 1/√n), coverage at nominal and mean z² ≈ 1. A PIT
piled at 0 and 1 means draws that are too narrow; a hump at 0.5 means too broad.

**Cosmic variance:** when several samples share identical parameters (seed repeats), the spread
of their truths is the simulator's own variance; `cosmic_variance` compares it with the draw
spread at those parameters.

```text
cosmic_variance: {n_groups, draw_over_truth_std_med: {family: ratio}}
```

A ratio ≈ 1 is right; ≪ 1 means mode collapse. Without repeats the entry says so.

### `gallery`

Not a score: the truth and prediction slices that `eval21 plot gallery` and `eval21 plot
deltas` draw. For the first `gallery_samples` samples (or the ids in `gallery_ids`) it keeps
the slices whose true slice-mean x_HI is nearest each `gallery_stages` level, for every field
in `gallery_fields` that both the sample and the prediction provide.

```text
samples, levels, fields, plane_source   # metadata; the slices go to curves.npz
```

Slices are stored as `gallery_<field>_truth` / `gallery_<field>_pred`,
shape `(samples, stages, H, W)`. A sample with no true x_HI (e.g. a T_b target with no x_HI
input) uses evenly spaced slices and records that in `plane_source`.

### `survey`

Not a score: an ionization-front survey of the truth and of the prediction (fno-21cm
`legacy/xhi2d/survey_truth_sharpness.py`). Every transverse plane is binned by its **own**
slice-mean x_HI (50 bins), and the following are accumulated over every LOS plane of every
sample:

- partially ionized fractions at three thresholds (`1e-3`, `0.01`, `0.1` from either phase);
- transverse and LOS total variation per voxel (periodic forward differences transversally,
  interior differences along z), and the 99.9th percentile of the transverse gradient;
- the band (`0.1 < x_HI < 0.9`) pixel fraction and a transition width in band pixels per unit
  of transverse total variation;
- a 200-bin histogram of voxel values.

`result()` reports the global fractions and plane count; the binned curves and histograms are
exported as `survey_*` arrays (`survey_partial_bands`, `survey_truth_*`, `survey_pred_*`).

```text
truth / pred: {frac_ionized, frac_neutral, frac_partial, n_planes}
n_samples
```

x_HI-only; a T_b pass skips it. It answers "how sharp is the field the model was trained on",
not "how good is the model"; the prediction half exists as a reference.

## Choosing a metric set

```bash
# Fast triage: voxel, history, parity — ≈55 s/cone on 16 CPU cores
--metrics voxel,history,parity

# Standard comparison: scales, bubbles, history, fronts
--metrics voxel,spectrum,bubble,history,front

# Stochastic / generative models
--draws 16 --mode generative --metrics spectrum,bubble,observable,history,ensemble

# A T_b pass: score the model's own brightness temperature
--target tb --metrics voxel,spectrum,dtb
```

`--metrics all` runs every registered metric whose needs are met. Metrics that were not run are
skipped and listed, never silently zeroed; figures that depend on them are skipped with that
reason.

## Metric options

`[metric_options.<name>]` is handed to that metric's constructor as keyword arguments
(`registry.build` → `create(name, settings, options.get(name, {}))`). Most built-in families
read `Settings`, so change their knobs with `--set` or `[settings]`. The exceptions are
`dtb` (`dtb_range_mk`, `dtb_bins`) and `front3d` (`los_cell_mpc`), which are per-metric by
nature:

```toml
[metric_options.dtb]
dtb_range_mk = [-250, 100]
dtb_bins = 350

[metric_options.front3d]
los_cell_mpc = 13.0
```

Custom metrics can use the same mechanism for any per-family configuration.

The distributed `examples/fno21cm_eval.toml` shows `n_k_bins` under
`[metric_options.spectrum]`; the effective value for the built-in `spectrum` metric is the
`Settings` field, so use `--set n_k_bins=20` (or `[settings] n_k_bins = 20`) to change it.
