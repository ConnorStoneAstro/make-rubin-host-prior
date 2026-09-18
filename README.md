# make-rubin-host-prior

A fully convolutional, energy-based diffusion prior over static scenes in the
vicinity of a host galaxy, trained on Rubin DP1 image patches. Intended as the
prior term in a forward model that extracts a point-source transient light curve
from LSST visit images.

The score is the exact gradient of a scalar energy, so it is a conservative
field — a genuine score, not a network that approximates one.

## Install

```bash
python -m pip install -e ".[dev]"
```

JAX, equinox, optax, numpy and h5py are all that the model and training side
need. The extraction side additionally needs the LSST Science Pipelines
(r29.2.0), which are not pip-installable — on NERSC they come from the stack
environment. `rubin_host_prior.rubin.extract` imports the stack lazily, so
everything else works on a laptop.

## Quick start, no cluster needed

```bash
python scripts/smoke_test.py --steps 400
```

Generates synthetic DP1-like shards, estimates the per-band offsets, builds the
loader, trains briefly, checkpoints, reloads and samples. Run it after any
change to the model or the transform; it exercises everything except the Butler,
so porting to NERSC only has to debug that part.

```bash
python -m pytest            # ~2 min, no cluster, no LSST stack
```

## The pipeline

### 1. Extract patches (on NERSC, inside the stack)

```bash
python scripts/extract_dp1_patches.py --out data/ecdfs --bands r i --n-hosts 500 -v
```

Selects extended objects from the per-tract `object` table, finds covering
`visit_image`s, cuts a **jittered** stamp near each host, runs the artefact gate,
and writes sharded HDF5 plus a manifest.

Read `data/ecdfs/summary.json` before trusting the output. It carries
`rejection_counts`, and those statistics are the only way to see whether the
selection function is biased — if bright, dense galaxy centres are being
rejected, the training set is skewed against exactly the regime this project
exists to model.

It also carries `correlation_length_native_flux_px`, accumulated over every
accepted patch as they are written (streaming, so it costs no memory). Treat that
as provenance only: at native resolution the small lags are dominated by the PSF,
and the log transform changes the correlation structure. The number to act on is
the pooled, log-space one from step 2.

### 2. Derive the config from the data

```bash
python scripts/prepare_config.py --shards data/ecdfs/shards --out config.json
```

The per-band offsets and the σ range are not free hyperparameters; they follow
from the noise level and dynamic range. This measures them and prints two checks
worth reading:

- `sky_scatter` should come out near `1 / k_sigma` (0.2 by default). If not, the
  offsets are wrong and the bands are not on a common footing.
- `clipped_fraction` should be ~1e-5 or below. Larger means `k_sigma` is too
  small, or the shards still contain over-subtraction the gate missed.

It also reports the **correlation length** of the pooled log-space patches and
compares it to the model's `2R` crop — this is the authoritative measurement, and
the one that decides how much context your analysis needs. Note that ~60% of the
pixel variance sits in the zero-lag noise delta; the estimator renormalises at
lag 1 to exclude it, because a naive 1/e crossing on the raw profile returns
ξ ≈ 1 regardless of galaxy size (measured: wrong by 4×).

### 3. Train

```bash
python scripts/train.py --shards data/ecdfs/shards --config config.json --out runs/ecdfs
```

### 4. Sample

```bash
python scripts/sample.py --checkpoint runs/ecdfs/final --n 16 --out samples
```

## Design decisions, and why

### An energy model, not a score network

`ConvEnergyNet` returns a scalar; `score(model, x, sigma)` is `-∇ₓE`. Because it
is an exact gradient, the implied log-density is path-independent and the
Jacobian of the score is symmetric (both are tested). A freely parameterised
score network satisfies neither exactly.

The cost: every training step differentiates through a gradient, so a step is
roughly 2–3× a conventional score network of the same size.

Two details that follow from the score, not the energy, being the trained
quantity:

- **The head is initialised small, not zero.** Zero-initialising a final layer
  is standard practice, but here `score = -W_head · ∂h/∂x`, so a zero head makes
  both the score *and* `∂score/∂θ` vanish for every upstream parameter — at step
  0 only the head would get a gradient. Same argument applies to the FiLM
  projections. Both are small-but-nonzero, and a test asserts every parameter
  receives a gradient at step 0.
- **The head has no bias.** A constant added to the energy is invisible to the
  score: pure gauge.

### Valid convolutions everywhere, and the `2R` crop

No padding, so no border artefacts and no dependence on patch size — the model
runs on any scene above `4R + 1` pixels, where `R = n_layers` for 3×3 kernels.

But the energy is a *sum* over the final feature map, and that does not weight
input pixels equally. Energy cell `p` sees input pixels `[p, p+2R]`, so

```
∂E/∂xᵢ = Σ over p ∈ [max(0, i−2R), min(i, E−1)]
```

contains all `2R+1` terms only for `2R ≤ i ≤ H−2R−1`. Outside that window the
sum runs over a *subset* of kernel offsets, so a border pixel's score is a
different linear functional of the weights than an interior pixel's — not merely
smaller (kernel weights have either sign), but systematically different, and the
network has no path to the missing terms.

**Two questions follow, and they have different answers.** Conflating them is
easy and wrong.

**1. Which pixels give clean training signal?** Exactly the ones more than `2R`
from the edge. For a field whose precision is banded at `2R` — which is what
this energy represents in the bulk — a window's marginal precision is
`K_ww − K_wo K_oo⁻¹ K_ow`, and `K_wo` is nonzero only within `2R` of the edge. So
the contamination has support *exactly* there and is identically zero beyond,
not merely small. Measured in an exactly solvable Gaussian analogue:

| distance from edge | 0 | 2 | 4 | **8 = 2R** | 16 |
|---|---|---|---|---|---|
| relative error in the training signal | 88% | 67% | 45% | **10⁻¹⁵** | 10⁻¹⁵ |

`2R` is therefore not a safety margin — it is the precise boundary. The loss
crop is **derived from `n_layers` and `kernel_size`, never configured**, and
`train()` prints the arithmetic at startup so that changing either announces
what it did:

```
valid-convolution geometry (derived from the architecture, not configured)
  8 x 3x3 valid convolutions  ->  receptive radius R = 1*8 = 8
  loss crop = 2R = 16 px from every side   (beyond 2R the training signal is exactly unbiased)
  smallest usable patch = 4R + 1 = 33 px
    patch   64x64   -> energy map 48x48, loss on interior 32x32 (25% of pixels)
    patch   96x96   -> energy map 80x80, loss on interior 64x64 (44% of pixels)
  at inference: a trustworthy region of N px needs a canvas of N + 32 px
```

**2. How much context does a scene need before its middle is trustworthy?**
This is *not* set by `R`. It is set by the **correlation length** ξ of the
images — how far apart two pixels must be before they stop being related, which
here means roughly how big the galaxies are. The edge's influence spreads as far
as pixels remain correlated. Measured in the same Gaussian analogue, in 2D:

| margin / ξ | error in the prior's width, at the centre |
|---|---|
| 0.5 | 45% |
| 0.8 | 13% |
| 1.1 | 4% |
| 1.3 | 1.2% |

So you want a margin of roughly **1.5–3 × ξ**. On synthetic patches ξ ≈ 6 pooled
pixels, giving `2R/ξ ≈ 2.5` at `L = 8` — comfortable. **Measure it on real DP1
data** (see below); if ξ comes back above ~15 pooled pixels, `2R = 16` is only
~1 × ξ and you need more layers or a larger analysis margin.

There is nothing ceremonious about the "padded canvas". It amounts to: *the edge
of a generated scene is unreliable, so make the scene bigger than the part you
care about.* `sample_interior` is a convenience that adds `2R` per side and
returns the middle; you can equally generate whatever size you like and ignore
the rim. For the eventual light-curve fit the same rule applies in plain terms:
model a scene extending at least 2ξ beyond the region whose photometry you
report.

For the record, a **full-field** loss (no crop) was measured against this. It has
an accuracy floor around 3% error in the prior's width that barely improves with
patch size or data, it slightly contaminates the learned bulk statistics, and it
conflicts with variable patch sizes (6× worse at the larger sizes). It is better
than crop-plus-margin only when the margin is under ~1.5 × ξ, i.e. when the
crop-trained model is being used outside the regime it was built for.

### Variable patch sizes

`PatchConfig.out_sizes` adds extra training sizes, cycled round-robin across
batches (a batch must be shape-homogeneous, so one jit compilation per distinct
size). This does three things:

- **Recovers the signal the crop discards.** The fraction of a patch that reaches
  the loss is `((H−4R)/H)²`. Since the score is computed on every pixel and the
  crop throws the border away, that fraction is also the per-step compute
  efficiency. **Bigger patches need bigger extracted stamps**, and this is what
  should drive `--native-size`:

  | `native_size` | max training size | interior at that size | KB/patch | GB per 50k | translation room |
  |---|---|---|---|---|---|
  | 224 | 74 | 25% (at 64px) | 331 | 15.8 | 32 px |
  | 288 | 96 | 44% | 519 | 24.7 | **0 px** |
  | 384 | 128 | 56% | 789 | 37.6 | **0 px** |
  | **416** (default) | 138 | **56%** (at 128px) | 931 | 44.4 | 32 px |

  The default targets **128 px training patches**, where 56% of each patch clears
  the crop against 25% at 64 px. `native_size` is 416 rather than the 384 that
  128 px strictly needs, because 384 is exactly 3×128 and would leave no room to
  translate the crop — silently disabling an augmentation that is otherwise free
  and exact. `Config.check_sizes()` warns if you land in that state.

  Larger stamps are more efficient in both directions: per stored byte the
  training signal roughly doubles from 224 to 416, because the fixed `2R` border
  tax is amortised over more area. The costs are extraction time (butler reads
  scale with area) and slightly more rejections near detector edges.

  `Config.usable_size_range()` reports the bounds — `(33, 138)` for the defaults —
  and `Config.check_sizes()` warns about sizes below the model's minimum or that
  spend most of themselves on the margin. `prepare_config.py` and `train()` print
  both.

  A reasonable mix is `--out-sizes 64 96 128`. Step cost scales with
  `batch × H²`, so a 128 px patch costs 4× a 64 px one; `train()` prints the
  score-pixel count per step, and halving the batch is the first thing to try if
  device memory is tight.

- **Costs nothing in accuracy.** Every size estimates the same size-independent
  bulk potential, so mixing them is free — verified as bit-identical to
  single-size training in the Gaussian analogue. This is only true *because* the
  loss is cropped; under a full-field loss the sizes make contradictory demands
  on shared weights.
- **Augments the data**, since each size takes a different random sub-crop.

Translation room is capped at the reference size's room (`max_translate_native`).
Without that cap a small crop would roam the whole stamp and land mostly on blank
sky far from the host, silently changing the data distribution with patch size.

The pooled cache serves any size at or below the one it was built at, by
sub-cropping: a sub-crop of a pooled, transformed image is exactly the pooled
transform of the corresponding native sub-region, because pooling is local and
the transform is pointwise.

### σ conditioning: FiLM only, no spatial normalisation

`log σ` → frozen random Fourier features → MLP → per-layer `(scale, shift)`
applied uniformly across each feature map.

**There is deliberately no batch/group/layer normalisation anywhere.** Any layer
that aggregates over pixels would make the output depend on the patch size,
quietly destroying the "runs on any scene" property. FiLM is spatially constant,
so it does not.

Activations are restricted to C¹ functions (`silu` by default). `relu` is absent
from the registry by design: a piecewise-linear activation gives a
piecewise-constant score with discontinuities.

The Fourier basis is frozen and derived from `fourier_seed` in the config, not
from the model's init key. It lives in a static field so AdamW's weight decay
cannot shrink it away — but static fields are pytree *metadata*, so deriving it
from the init key would make two models structurally incompatible, breaking
`tree_map` (hence the EMA) and silently corrupting checkpoint reload. There is a
regression test.

### VE SDE, no preconditioning

`x_σ = x + σ·ε` with a geometric schedule and log-uniform σ sampling. The learned
score *is* `∇ₓ log p_σ(x)` on the data's own scale — nothing needs unscaling
before the prior meets a likelihood, and the `σ → 0` limit is the prior you want.

One reparameterisation is on by default: `E = Ẽ / σ` (`sigma_scaling`). It leaves
the loss and the exactness of the score untouched, but makes `σ·score = -∇Ẽ`
free of any schedule-wide trend, so one set of weights does not have to span
decades of score magnitude. Set `sigma_scaling="none"` for the unmodified energy.

### The log-space transform

```
forward:   x = log1p( max(f / b_band, r) ) / c   =   log( 1 + max(f/b_band, r) ) / c
inverse:   f = b_band · expm1( c · x )           =   b_band · ( e^{c·x} − 1 )
```

with `r = floor_ratio = −0.9` and `c = log_scale = 1`. `b_band = k_sigma ×
σ_pooled` is a per-band offset in nJy.

**The `+1` inside `log1p` is the boost**, and it is what carries negative pixels
through the logarithm. DP1 images are background-subtracted, so roughly half of
all sky pixels are negative; dividing by `b_band` rescales them but leaves them
negative. Adding `b_band` is what makes the argument positive. In flux units the
transform is `log((f + b)/b)`.

Three properties follow:

- **No hard floor in the ordinary regime, and no point mass.**
  `log(max(f, floor))` would pile 40–50% of every patch onto one value;
  `log1p(f/b)` is smooth and strictly monotonic through zero instead.
- **Band-agnostic.** Dividing by `b_band` puts every band's sky level at `x ≈ 0`
  with scatter `≈ 1/k_sigma`, so u-band and y-band patches land in the same
  place and one prior covers all six. There is no band label anywhere in the
  model. `PatchDataset.stats()["sky_scatter"]` should come out near `1/k_sigma`;
  if it does not, the offsets are wrong.
- **Linear where it matters, logarithmic where it must be.** Near the noise floor
  `x ≈ f/(b·c)`, a pure rescaling, so additive Gaussian pixel noise stays
  additive and Gaussian. In the bright regime it is logarithmic, which tames the
  ~10⁴ dynamic range of a galaxy core.

#### `k_sigma` is a modelling choice, not a numerical guard

`inverse` is `b·expm1(c·x)` and `expm1 → −1`, so the model can represent flux in
**`(−b_band, +∞)` and nothing below**. `k_sigma` is therefore a hard bound on how
negative a pixel the prior can express. The clip at `floor_ratio` bites slightly
earlier, at `−0.9·k_sigma` σ.

This binds harder than it looks, because **pooling does not treat noise and
smooth offsets alike**: 3×3 averaging divides the *noise* by 3, while a smooth
background offset does not average down at all. An over-subtracted region `D`
sigma deep natively is `3D` sigma deep in the pooled data the model sees.

Fraction of pixels driven onto the floor, by halo depth and `k_sigma`:

| dark halo depth | pooled depth | `k=5` | `k=10` | `k=15` |
|---|---|---|---|---|
| 0 (clean sky) | 0σ | 3×10⁻⁶ | ~0 | ~0 |
| 1.0σ native | 3σ | **6.7×10⁻²** | 1×10⁻⁹ | ~0 |
| 2.0σ native | 6σ | **0.93** | 1×10⁻³ | ~0 |
| 3.0σ native | 9σ | **1.0** | 0.50 | 3×10⁻⁶ |

Measured end to end on synthetic shards carrying a 1.5σ halo: `k_sigma=5` clips
8.4% of pixels and inflates `sky_scatter` from 0.2 to 0.65; `k_sigma=10` clips
none. **The default is therefore `k_sigma = 10`**, and raising it costs almost
nothing — the dynamic range of the representation barely changes, since that is
set by the physical S/N rather than by `k`.

`PatchDataset.flux_headroom()` measures what your data actually needs — the
distribution of the most-negative pooled pixel in units of pooled sky noise, and
the implied minimum `k_sigma`. `prepare_config.py` prints it and warns if
`k_sigma` is too small. **Run it before training**, especially if you are keeping
background-subtraction artefacts rather than gating them out.

A forward model in log space needs no Jacobian: generate the model scene in `x`,
map to flux with `transform.inverse`, and compare to the data.
`transform.jacobian` exists if you ever want a density in flux units.

### Pooling in flux, then log — not the other way round

Pooling averages **fluxes**; the log is applied after. Averaging log-fluxes would
compute a geometric mean and bias every structured patch low.

`block_mean` by an integer factor is exact: independent pixel noise is divided by
exactly `pool_factor` and stays uncorrelated between output pixels. `area_resample`
(for scale jitter) reduces to `block_mean` exactly at integer factors — there is a
test — but at non-integer factors output pixels share input pixels and become
slightly correlated. **That is a real change to the noise properties, so
`scale_jitter` defaults to 0.**

### Augmentation that leaves the noise alone

- **Dihedral (D4, 8 elements)** — exact re-indexings. The multiset of pixel values
  is unchanged, so the noise distribution and its pixel-to-pixel independence are
  untouched. On by default.
- **Translation** — integer *native*-pixel shifts of the crop. One native pixel is
  1/3 of an output pixel, so this is sub-output-pixel positional augmentation with
  **no interpolation at all**. On by default. This is why `native_size` (416)
  exceeds `nominal_crop` (384).
- **Scale jitter** — interpolates, so off by default. `AugmentConfig.scale_jitter`
  turns it on.

No added noise, no added blur, and none is offered.

### The artefact gate

Three DP1 facts drive it:

1. **Half the mask planes are never set in `visit_image`** (`STREAK`, `NO_DATA`,
   `UNMASKEDNAN`, `VIGNETTED`, `SENSOR_EDGE`, `CLIPPED`, `REJECTED`,
   `DETECTED_NEGATIVE`, `INEXACT_PSF`). A gate built on them passes everything, so
   pixel finiteness and variance positivity are tested directly, and satellite
   trails come from the matching `difference_image` mask (`StreakCache`) rather
   than from the visit image's own STREAK plane.
2. **Several artefacts have no mask plane at all** — stray light, ghosts, amp
   jumps, fringing, tree rings, crosshatch, and the dangerous ones here, *dark
   edge* and *dark halo*: background **over-subtraction**. Those put a smooth
   negative bowl into exactly the low-surface-brightness regime this project
   cares about, and a prior trained on them learns that galaxies sit in negative
   bowls.
3. **Tolerances for a generative model differ from tolerances for photometry.**
   The published DP1 table says "retain" for `CR` and `INTRP` because an
   interpolated pixel barely perturbs a flux. Here the model is learning a
   distribution over pixel values, and an interpolated pixel is a smooth
   synthetic patch teaching structure that is not in the sky. `CR` and `INTRP`
   are tighter than the documentation suggests; every fraction is recorded in the
   manifest so they can be loosened later without re-reading pixels.

`background_floor` locates the sky floor as a low percentile per block of a
coarse grid, corrected for the percentile's own Gaussian offset. Two choices
matter:

- **A percentile, not a fitted surface.** A quadratic fit to a patch containing a
  bright galaxy absorbs the galaxy and then extrapolates strongly negative
  towards the corners, reporting a bowl that is not there. A low percentile is
  blind to positive sources by construction.
- **One-sided.** A *depressed* floor is over-subtraction; a *raised* one is
  starlight. Gating on the magnitude would discard the brightest hosts.

**By default it records but does not reject** (`max_depression=None`). The data
is taken as-is, background-subtraction artefacts included, and the prior is
allowed to learn them — the right call when the artefacts are a property of the
current processing that a later release will improve, since you retrain rather
than filter. Pass `--max-depression 0.3` to reject instead. Note the interaction:
keeping depressed regions is exactly what forces a larger `k_sigma`, because
those pixels have to remain representable.

Tested against bright galaxies, faint galaxies, edge galaxies, galaxies filling
the whole stamp (all accepted) and 6σ bowls, 1σ bowls, gradients and uniform
offsets (all rejected). Blank-sky noise floor stays inside ±0.25, against a
threshold of 0.3.

`DETECTED` is never a rejection reason — gating on it would throw away every
patch containing a galaxy — and `gate` raises if you try.

### Storage

Shards hold **native-resolution stamps in physical units** (nJy), plus variance,
mask, the mask plane dictionary, PSF stamp and moments, `x0y0`, WCS-derived
position, MJD and neighbour summary. Pooling, the log transform and the offsets
all happen in the loader, so any of them can change without re-extracting.

`x0y0` and the mask plane dictionary are mandatory, not optional: LSST boxes have
non-zero origins, bit assignments are not guaranteed stable across releases, and
without both a saved stamp cannot be mapped back to the sky or interpreted.

`PatchDataset.build_pooled_cache()` writes a derived pooled-and-transformed array
keyed by a hash of the transform config, for fast iteration. It only supports
dihedral augmentation (the crop is baked in), so the default loader path reads
native stamps.

## Porting to NERSC: what to verify

The model, loss, loader and gate are all tested locally. The Butler layer is not,
and these specific items are flagged `WARN` in `rubin/extract.py`:

- [ ] **Repo alias.** `"dp1"` is the RSP label. Check `Butler.get_known_repos()`
      or `$DAF_BUTLER_REPOSITORY_INDEX`; pass a path if it differs. Never open the
      shared mirror writeable.
- [ ] **Component names.** `visit_image.wcs` is the verified idiom;
      `visit_image.bbox` follows the same pattern but is unconfirmed. The
      containment pre-check degrades to "unknown" rather than failing, and the
      stamp shape is checked after the read regardless, so a clipped stamp is
      rejected rather than padded.
- [ ] **`_mjd` accessor chain.** `stamp.visitInfo.date.toAstropy().mjd` is
      unverified; falls back to the long form, then to NaN.
- [ ] **`detect_isPrimary`.** Standard in LSST object tables but absent from the
      DP1 tutorials. Requested, with a fall back to deduplicating on `objectId`.
      Without one of the two, tract-overlap regions are silently oversampled.
- [ ] **`difference_image.mask` component read** for the streak verdict. Falls
      back to NaN (no rejection) where it fails or no difference image exists.
- [ ] Confirm `BUNIT`, but do not trust it — the header once reported `'adu'` for
      nJy pixels (DM-51270). Pixels are nJy either way.

## Open questions and known gaps

- **Visit images vs coadds.** `--dataset-type` accepts both. Visit images are
  closest to the data you will eventually analyse and, being unwarped, have
  near-independent pixel noise — which matters, since `estimate_band_offsets`
  assumes pooling divides the noise by `pool_factor`. On a coadd, warping
  correlates neighbouring pixels and pooling reduces noise by less, so that
  estimate would be optimistic. The cosmic-ray question is empirical: extract a
  few thousand of each and compare `rejection_counts`.
- **Visit-level cuts not implemented.** Seeing, zeropoint, sky background and
  PSF-star count per (visit, detector) would remove whole swathes before any
  pixel is read. The DP1 column names for `visit_summary` were not verifiable
  from the material available, and guessing them would be worse than omitting
  them. Build it as a parquet table once and join it in.
- **Satellite trails rely on Rubin's own detection.** A Radon/Hough pass per
  (visit, detector) would catch the faint trails that escaped masking — which are
  precisely the ones that would teach the model to generate straight lines.
- **Correlation length on real data is the open empirical question.** Everything
  about how much context the model needs follows from it, and the synthetic
  generator only crudely imitates the real host size distribution.
- **`sigma_max` should dominate the data's own scale** or the `t = 1` marginal is
  not really Gaussian. `suggest_sigma_range` sets it from the 99th percentile of
  per-patch range; check it against `stats()` on the real data.
- **Multi-band.** Patches are single-band, 1-channel, all bands pooled into one
  dataset, no band label. `in_channels` is configurable, so a registered
  multi-band stack from coadds is a loader change, not a rewrite.
- **The Langevin corrector has a known `+O(snr⁴)` variance bias** (+1.4% at
  `snr=0.16`, <0.2% at 0.05). `pflow_sample` has none and is the default.

## Layout

```
src/rubin_host_prior/
  geometry.py        valid-conv shape arithmetic; read before choosing a patch size
  config.py          every dataclass that must travel with a checkpoint
  nn/                layers.py (FiLM, Fourier, ConvBlock), energy.py (net + score)
  diffusion/         sde.py (VE), loss.py (DSM + interior crop), sampler.py
  training/          trainer.py, ema.py, checkpoint.py
  data/              transform.py, pooling.py, augment.py, shards.py, dataset.py,
                     diagnostics.py (correlation length), synthetic.py (DP1-like
                     fake data for offline testing)
  rubin/             quality.py (stack-free gate), extract.py (lazy LSST imports)
scripts/             extract_dp1_patches.py, prepare_config.py, train.py,
                     sample.py, smoke_test.py
tests/               ~100 tests, no cluster and no LSST stack required
```
