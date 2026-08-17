# Methods: cumulative 2-D horizontal strain workflow

This document is the technical companion to Section 4 of the manuscript. Every fixed
numerical choice is recorded here so that a reader can audit the implementation against
the paper without reading the notebooks.

---

## 0. Inputs and common grid

| Input | Detail |
|---|---|
| Ascending stack | Sentinel-1A/B Track 64, frame `064A_05410_131313`, heading −10.146887°, mean incidence 39.6181° |
| Descending stack | Sentinel-1A/B Track 71, frame `071D_05377_131313`, heading −169.84503°, mean incidence 33.7677° |
| Processing | LiCSAR / LiCSBAS SBAS; GACOS tropospheric correction |
| Epochs | 80 common acquisitions, 27 May 2017 – 25 November 2019 |
| Reference epoch | 27 May 2017 |
| GNSS | 24 continuous stations, Nevada Geodetic Laboratory daily solutions |
| Rupture traces | California Geological Survey 2019 Ridgecrest mapping; two-segment finite source geometry (Paxton Ranch, Salt Wells Valley) |
| Output grid | UTM Zone 11N, EPSG:32611, 1-km spacing, footprint intersection of the two tracks |

The last pre-event acquisition pair is 4 July 2019; the descending pass was recorded at
13:51:41 UTC, **3 h 42 min** before the M<sub>w</sub> 6.4 foreshock origin time
(17:33:49 UTC).

> The 1-km grid spacing describes **output sampling only**. Effective spatial resolution is
> set by the interpolation and strain-support length scales below.

### Resampling and referencing

LOS rasters and pixel-specific E/N/U look-vector components are resampled by mask-aware
bilinear interpolation,

```
q̂(x*) = Σⱼ wⱼ mⱼ qⱼ / Σⱼ wⱼ mⱼ ,      s(x*) = Σⱼ wⱼ mⱼ
```

where `wⱼ` are the four bilinear weights, `qⱼ` the **source pixel value**, and
`mⱼ ∈ {0,1}` the validity mask. A target is retained only when `s(x*) ≥ 0.999`, so no
output cell is ever blended with a masked source pixel. LOS unit vectors are re-normalised
after resampling.

Track-71 pixels additionally require average coherence ≥ 0.30, residual RMS ≤ 5 mm,
unwrapping-gap count ≤ 2, and phase-closure error count ≤ 10. **This asymmetric screening is
the origin of the near-fault Track-71 gaps** that Step 4 reconstructs.

Cumulative LOS is referenced, before resampling, to the median displacement within 1.5 km
of GNSS station **P463**, so the datum does not depend on output-grid resolution.

---

## 1. GNSS vertical field

### 1a. Acquisition-time estimation

Cumulative station displacement is `ΔUₛ(t) = Uₛ(t) − Uₛ(t₀)`. Within uninterrupted daily
segments, positions are linearly interpolated between bracketing solutions with propagated
variance

```
σ²_U = (1−α)² σ²_L + α² σ²_R ,      α = (τₖ(t) − t_L)/(t_R − t_L)
```

For the two 4 July 2019 acquisitions — both preceding the foreshock — ordinary
interpolation would span the co-seismic discontinuity, so endpoints are predicted from a
30-day weighted pre-event trend solved by weighted least squares.

> Acquisition-time GNSS values are therefore **temporally estimated quantities**, not
> independent observations.

### 1b. Spatial interpolation

Seven local interpolation families were compared under pre-event calibration and
independent temporal and station holdouts. The selected model is an adaptive local
Gaussian process with a squared-exponential kernel

```
K(h) = σ_f² · exp[ −½ (h/ℓ)² ] ,      ℓ = 15 km
```

zero nugget, station uncertainties on the covariance diagonal, and adaptive local support
chosen from candidate radii 35–120 km subject to: ≥ 5 contributing stations, occupancy of
≥ 3 of 8 azimuthal sectors, and the target lying inside the local station convex hull.
**No map-wide plane is fitted.** The support topology is constructed once, before the event
period, and reused at every epoch.

Independent pre-event holdout: RMSE **4.52 mm** (T64) and **3.13 mm** (T71); 90 % interval
coverage **0.81** and **0.92**.

---

## 2. Vertical-to-LOS correction

The interpolated vertical field is projected with the native LiCSAR vertical look
coefficient and subtracted from the referenced LOS:

```
h_k(x,t) = s_k · d_k(x,t) − R[ l_{U,k}(x) · Û_k(x,t) ] ,    s₆₄ = s₇₁ = +1
```

where `R` applies the same temporal and spatial reference as the InSAR field. A sign audit
against GNSS selected `s₆₄ = s₇₁ = +1`; **neither track is negated**. Uplift and subsidence
are carried by the sign of `U`, so no conditional addition rule is required.

---

## 3. Two-track E–N inversion

```
[ h_a ]   [ l_{E,a}  l_{N,a} ] [ E ]
[ h_d ] = [ l_{E,d}  l_{N,d} ] [ N ]
```

solved pixelwise in closed form. Cells are rejected where `|Δ| ≤ 1e−6` or the geometry
condition number exceeds **8**, reflecting the weak north constraint that follows from two
near-polar, right-looking geometries.

Propagated covariance carries fixed LOS uncertainties (**24.34 mm** ascending, **16.28 mm**
descending) and the vertical-correction uncertainty, the latter treated as **fully
correlated between tracks** because both derive from the same GNSS network. Output is
pointwise `σ_E`, `σ_N`, `Cov(E,N)` at every epoch.

---

## 4. Near-fault Track-71 completion

Within 18 km of mapped rupture the domain contains **3,494** 1-km cells:

| Category | Cells |
|---|---:|
| Both tracks observed, no reconstruction | 2,979 |
| Track 64 observed, Track 71 reconstructed by cokriging | 502 |
| Paired-neighbour spatial completion (M0) | 13 |
| **Total** | **3,494** |

Reconstruction uses fixed local, fault-side-aware, Track-64-conditioned latent E–N
universal cokriging with Matérn-3/2 correlation

```
ρ(h) = (1 + √3·h/ℓ) · exp(−√3·h/ℓ) ,      ℓ = 8 km
```

24-km donor support, at most 48 paired samples, an affine E/N drift, and a **finite
fault-side connectivity rule**: a donor whose segment to the target crosses a mapped finite
fault segment is excluded. Valid Track-71 values are never overwritten.

The completed field covers all 3,494 near-fault cells, **including the 305 cells within
1 km of the mapped rupture**.

### Validation — read this before using the near-fault products

| Evaluation interval | Buffered spatial CV MAE |
|---|---:|
| Calibration (≤ 29 May 2019) | ≈ 2.6 mm |
| 22 June 2019 | 3.2 mm |
| 4 July 2019 | 7.1 mm |
| 16 July 2019 (event control) | ≈ 28 mm |

Track-64 conditioning gives a **−0.1 %** gain relative to the paired-track-only baseline —
a negligible degradation, not an improvement. Reconstruction error grows roughly tenfold
into the window of greatest scientific interest. All near-fault products are therefore
**retrospective, model-assisted sensitivity maps**, not validated recovery of the missing
Track-71 observations.

---

## 5. Cumulative 2-D strain

A fixed joint east–north local affine generalised-least-squares operator is built once and
reused at every epoch:

```
E(x) = E₀ + Eₓ·Δx + E_y·Δy
N(x) = N₀ + Nₓ·Δx + N_y·Δy
```

| Parameter | Value |
|---|---|
| Displacement-sample lattice | 2 km |
| Strain-output grid | 1 km |
| Support radius | **8 km** |
| Gaussian weighting bandwidth | 4 km |
| Minimum samples per target | 16 |
| Off-fault sample exclusion | > 10 km from rupture |
| Off-fault target exclusion | > 18 km from rupture (= 10 + 8) |

Kernel weights are `exp[−½ (d/bandwidth)²]`. Parameter uncertainty is `Cov(β̂) = H⁻¹`.

Components:

```
ε_EE = ∂E/∂x
ε_NN = ∂N/∂y
γ_EN = ∂E/∂y + ∂N/∂x
δ    = ε_EE + ε_NN
ω    = ½ (∂N/∂x − ∂E/∂y)
```

Strain is reported in **microstrain (µstrain)**, vertical-axis rotation in **microradians
(µrad)**. These are direct spatial derivatives of cumulative displacement — not strain
rates, and not temporally accumulated interval strain.

Median propagated pointwise 1σ: ε_EE 1.21, ε_NN 6.60, γ_EN 6.94, δ 6.72 µstrain; ω 3.45 µrad.

---

## 6. Change detection

Innovations are exact temporal increments of cumulative strain,

```
Δ_τ ε_c(x,t) = ε_c^cum(x,t) − ε_c^cum(x, t−τ) ,     τ ∈ {12, 24} days
r_{τ,c} = Δ_τ ε_c / τ ,    σ_{r,τ,c} = √2 · σ_c / τ
```

| Period | Definition |
|---|---|
| Calibration | windows ending on or before 29 May 2019 |
| Pre-event surveillance | 29 May – 4 July 2019 |
| Event control | exactly 4 – 16 July 2019 |

A robust baseline centre and MAD scale from the calibration period give a standardised
innovation `z_c(x,t)`, retaining variance in excess of the formally propagated uncertainty.
Two tests run on a **4-km inference lattice** across all five descriptors.

**Maximum signed spatial-cluster mass.** Cells with `|z_c| ≥ 1.96` and consistent sign are
grouped by 8-neighbour connectivity; clusters of ≥ 4 cells are retained and scored

```
M_g = Σ_{x∈g} [ s·z_c(x,t) − 1.96 ]
T(t) = max_{c,s,g} M_g
```

The maximum jointly accounts for the search across space, strain component, sign, and
candidate cluster — this is the family-wise error control.

**Page CUSUM.** On the 90th percentile of `|z_c|` pooled across components and inference
cells per 12-day window, standardised against pre-event median and robust scale to give
`e_t`:

```
C_t = max[0, C_{t−1} + e_t − k] ,   C₀ = 0 ,   k = 0.5
```

**Empirical null.** Both statistics are compared to consecutive pre-event baseline blocks
of matching duration:

```
p = (1 + N(T_null ≥ T_obs)) / (1 + n_null)
```

### Results

| Test | Period | Observed | *p* | Significant (α=0.05) |
|---|---|---:|---:|---|
| Max signed cluster mass | Pre-event surveillance | 0.000 | 1.000 | No |
| Max signed cluster mass | 4–16 July event control | 5.804 | 0.0377 | Yes |
| Page CUSUM (90th-pct \|z\|) | Pre-event surveillance | 0.075 | 0.481 | No |
| Page CUSUM (90th-pct \|z\|) | 4–16 July event control | 3.759 | 0.0185 | Yes |

Only the **12-day** series is formally tested. The 24-day series peaks higher (≈9.8) at the
same date but a 24-day window spanning the event extends into the post-earthquake period
and is incompatible with the exact 4–16 July control design.

> **Honest caveat.** A pre-event cluster-mass excursion around October 2018 (≈9.8, 12-day
> series) exceeds the reported event-control value of 5.804. It is one of the baseline
> blocks contributing to the empirical count. The detection margin is modest and rests on a
> small empirical null.

---

## 7. Dilatation lobe partition

To avoid sign cancellation in a whole-domain signed median, two masks are fixed once on the
**29 May 2019** reference epoch from the 20th and 80th percentiles of near-fault dilatation
(699 cells each, 20 % of the 3,494-cell domain) and held fixed at every subsequent epoch,
so change reflects the displacement field rather than a moving spatial definition.

| Date | Upper lobe (P80) | Lower lobe (P20) |
|---|---:|---:|
| 29 May 2019 | +13.51 µstrain | −15.27 µstrain |
| 22 June 2019 | +24.59 µstrain | −25.24 µstrain |
| 4 July 2019 | +34.04 µstrain | −27.92 µstrain |

These are **near-fault reconstruction-domain values** — see the Step 4 validation caveat.

---

## 8. Finite-aperture cross-fault displacement jump

Evaluated as a point-sampled discontinuity at a fixed **3-km** perpendicular offset either
side of each segment centreline (6-km total aperture), at 14 points along the central 76 %
of each segment (`f ∈ [0.12, 0.88]`, uniform spacing), bilinearly interpolated on the
filled 1-km grid:

```
x_i^(±) = c_i ± 3.0 · n̂ ,     n̂ = [â_y, −â_x]
Δu_∥ = median_i [ ΔE_i·â_x + ΔN_i·â_y ]
Δu_⊥ = median_i [ ΔE_i·n̂_x + ΔN_i·n̂_y ]
```

with the interquartile range across the 14 points as the uncertainty envelope.

---

## Interpretation boundary

The north component is less strongly constrained than east because both Sentinel-1 tracks
have near-polar viewing geometries and their nominal same-date acquisitions differ by
roughly 12 hours. This is independent of the vertical correction and would persist under an
error-free vertical constraint; resolving north directly requires an additional, suitably
oriented viewing geometry.

Cumulative fields are descriptive. Statistical significance is established only on
independently calibrated temporal innovations. Temporal coincidence alone is not
interpreted as earthquake preparation or predictive behaviour.
