# make-rubin-host-prior

A fully convolutional, energy-based diffusion prior over static scenes in the
vicinity of a host galaxy, trained on Rubin DP2 `deep_coadd` image patches. Intended as the
prior term in a forward model that extracts a point-source transient light curve
from LSST visit images. The prior itself trains on DP2 `deep_coadd` patches.

The score is the exact gradient of a scalar energy, so it is a conservative
field — a genuine score, not a network that approximates one.

## Install

```bash
python -m pip install -e ".[dev]"
```

JAX, equinox, optax, numpy and h5py are all that the model and training side
need. The extraction side additionally needs the LSST Science Pipelines
(r30.0.11), which are not pip-installable — on NERSC they come from the stack
environment. `rubin_host_prior.rubin.extract` imports the stack lazily, so
everything else works on a laptop.

## Quick start, no cluster needed

```bash
python scripts/smoke_test.py --steps 400
```

Generates synthetic DP2-like shards, estimates the softening scales, builds the
loader, trains briefly, checkpoints, reloads and samples. Run it after any
change to the model or the transform; it exercises everything except the Butler,
so porting to NERSC only has to debug that part.

```bash
python -m pytest            # ~2 min, no cluster, no LSST stack
```

## The pipeline

### 1. Extract patches (on NERSC, inside the stack)

```bash
python scripts/extract_dp2_patches.py --out data/ecdfs --bands r i --n-hosts 2000 -v
```

Selects extended objects from the per-tract `object` table, finds covering
`deep_coadd` patches, cuts a **jittered** stamp near each host, runs the artefact
gate, and writes sharded HDF5 plus a manifest. A host yields at most one patch
per band, so `--n-hosts` sets the training-set size fairly directly.

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

### 2. Look at the diagnostic figures

Extraction writes them automatically to `<out>/diagnostics` (`--no-plots` to
skip). Once a config exists, regenerate with the loader figures too:

```bash
python scripts/diagnose.py --shards data/ecdfs/shards --config config.json
```

In rough order of how often they catch something:

| figure | what to look for |
|---|---|
| `rejections.png` | **the important one.** Rejection reasons, and accepted vs rejected sky noise. If the rejected patches are systematically brighter or denser, the gate is discarding exactly the regime this project models, and the tolerances need loosening. |
| `transform.png` | native flux → pooled → log space for a few patches, plus the pixel-value histogram. The sky peak must sit on `x = log(log 2) ≈ −0.37` inside the predicted scatter band, with sources clear of it. If it doesn't, the softening scales are wrong. |
| `training_batch.png` | exactly what the network receives: pooled, log-space, augmented, at the training size(s), on a shared colour scale so the spread between patches is visible. |
| `cutouts.png` | raw stamps as `asinh(flux / sky noise)` — a stretch in σ units with a pinned low end, so bands of very different depth are directly comparable. |
| `hosts.png` | the selected population: size, distortion, magnitude, blendedness, band counts, sky noise, PSF size, nearest galaxy/star, neighbour counts, and the extraction jitter. |

The host size/magnitude/distortion panels come from `hosts.parquet`, which
extraction writes alongside the shards — those quantities are known only at
selection time and are not carried in the shard metadata.

Note the distortion convention: `|e| = (Ixx−Iyy, 2Ixy)/(Ixx+Iyy)`, which is
`(1−q²)/(1+q²)`, roughly twice the `(1−q)/(1+q)` shear convention at modest
ellipticity.

### 3. Derive the config from the data

```bash
python scripts/prepare_config.py --shards data/ecdfs/shards --out config.json
```

The per-band softening scales and the σ range are not free hyperparameters;
they follow from the noise level and dynamic range. This measures them — from
pooled patches, not the variance planes, since coadd noise is correlated — and
prints two checks worth reading:

- `sky_scatter` should match `expected_sky_scatter(softening_sigma)`
  (= `0.721 / softening_sigma`). If not, the per-band softening scales are wrong
  and the bands are not on a common footing.
- the sky pedestal (`0.693 × softening_sigma`, in σ) should stay under 1, and the
  flux above which the exponential model map is accurate should sit below
  anything you care about photometrically.

It also reports the **correlation length** of the pooled log-space patches and
compares it to the model's `2R` crop — this is the authoritative measurement, and
the one that decides how much context your analysis needs. Note that ~60% of the
pixel variance sits in the zero-lag noise delta; the estimator renormalises at
lag 1 to exclude it, because a naive 1/e crossing on the raw profile returns
ξ ≈ 1 regardless of galaxy size (measured: wrong by 4×).

### 4. Train

```bash
python scripts/train.py --shards data/ecdfs/shards --config config.json --out runs/ecdfs
```

### 5. Sample

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
pixels, giving `2R/ξ ≈ 2.5` at `L = 8` — comfortable. **Measure it on real DP2
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
forward (data):   x = log( softplus(f / s_band) ) / c       softplus(u) = log(1 + e^u)
model map:        f = s_band · exp(c · x)                   strictly positive
```

`s_band = softening_sigma × σ_pooled` is a per-band *softening* scale in nJy.

**These two are deliberately not inverses, and that is the point.** A source
cannot emit negative flux, so the prior's reachable domain in flux space must be
strictly positive — hence the plain exponential, which maps all of ℝ to `(0, ∞)`.
Measured flux, by contrast, *is* negative wherever noise dips below the
subtracted sky, and the right thing to do with those pixels is carry them
smoothly toward zero rather than represent them faithfully.

`s·exp(c·forward(f))` equals `softplus_s(f)` exactly, so the entire discrepancy
between the data and what the model can express is the softening and nothing
else. Three consequences:

- **Bright flux passes through untouched.** `softplus(u) → u` exponentially
  fast: within 1.6% at `f = 3s`, 0.1% at `5s`, exact in double precision by
  `10s`. Anything detected is represented far inside its own photometric error.
- **Negative flux vanishes smoothly.** `softplus(u) → e^u`, so `x → f/s`: the
  negative tail is *linear* in flux, which keeps Gaussian pixel noise Gaussian
  instead of compressing it. (A `√(f²+4s²)` style softening fails here — it goes
  as `−log|f|` and distorts the noise.)
- **There is no floor anywhere.** `softplus` is strictly positive on all of ℝ, so
  no clipping, no point mass, no NaN, and **no bound on how negative an input
  pixel may be**. Background over-subtraction of any depth is representable.

The cost is a pedestal: the model's sky sits at `softplus(0)·s = 0.693·s` rather
than zero. That is why `softening_sigma` defaults to **1.0** — it keeps the
pedestal at 0.69σ, below the noise it replaces, while making the exponential map
accurate to 0.1% above 5σ. Raising it buys a tighter, less skewed noise
distribution in `x` at the price of a pedestal climbing above the noise and a
proportionately higher flux threshold for accuracy.

An **ELU-style softening was considered and rejected**: `ELU(u) + 1` leaves a
permanent `+s` offset on positive flux — still 3.3% high at `f = 30s` — whereas
softplus converges to the identity exponentially.

`inverse_exact` undoes `forward` exactly (to ~10⁻¹⁵), including negative flux,
for round-trip checks. `inverse` is what a forward model calls.

A forward model in log space needs no Jacobian: generate the scene in `x`, map to
flux with `inverse`, compare to the data. `jacobian` is simply `c·f` if you ever
want a density in flux units.

Check `stats()["sky_scatter"]` against `expected_sky_scatter(softening_sigma)`
(= `0.721 / (softening_sigma · c)`); a large disagreement means the per-band
softening scales are wrong, which would put the bands on different footings and
break the single band-agnostic prior. `prepare_config.py` does this for you.

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

DP2 is not DP1 with more data, and the differences all land here.

**Plane names changed and bit numbers are dynamic.** `SAT` → `SATURATED`,
`CR` → `COSMIC_RAY`, `INTRP` → `INTERPOLATED`, `EDGE` → `DETECTION_EDGE`, and the
bit assignments are not stable across releases. A DP1-era gate does not error on
DP2 — it matches nothing and passes every stamp. Nothing here hard-codes a bit:
extraction repacks each mask into a `uint32` using a mapping derived from the
coadd's own `mask.schema`, stores that mapping with the shard, and `gate`
**raises** if none of the planes it is asked to gate on appear in the mapping.
That last check is the difference between a loud failure and a silently useless
training set.

**No-data is carried by the variance, not by a plane.** DP2 variance holds `inf`
where there were no contributing exposures, including the cores of saturated
stars. The fraction is measured, with a tolerance, and zero tolerance at the
centre where the transient goes. Rejecting merely because non-finite variance is
*present* would discard every stamp containing a bright neighbour.

**`SATURATED` gets a fraction, not zero.** DP1 excluded any saturation outright.
On a DP2 coadd the saturated core of a bright neighbour lands in a great many
stamps, and a scene with a bright neighbour is exactly the regime this project
models — so a small fraction away from the centre is kept, and the manifest
records how much.

**`INEXACT_PSF` and `REJECTED` are not quality cuts.** They cover a large
fraction of the DP2 coadd, so gating on them keeps almost nothing. They are
recorded as per-stamp covariates (`frac_inexact_psf`, `frac_rejected`,
`frac_no_data`) so a cut can still be made from the manifest without re-reading
pixels — and so `hosts.png` can show you whether the PSF your forward model
relies on is approximate over most of the training set. `DETECTED` is
informational; rejecting on it would reject every patch containing a galaxy.

**Neither release masks satellite trails in what you train on.** DP2 coadds have
no `STREAK` plane at all. Trails and unmasked electronics artefacts need your own
detection step; nothing here will catch them.

Tolerances for `COSMIC_RAY` and `INTERPOLATED` are tighter than the documentation
recommends, deliberately: that guidance is written for *measurement*, where an
interpolated pixel barely perturbs a flux, whereas here the model is learning a
distribution over pixel values and an interpolated pixel is smooth synthetic fill
teaching structure that is not in the sky.

**There is no background check.** DP2 coadds are over-subtracted around extended
galaxies and the background is recoverable via `apply_background`. The images are
taken **as delivered**, without restoration, and every shard records
`background_restored=0` so a set made the other way is distinguishable.

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

The model, loss, loader and gate are tested locally. The Butler and `lsst.images`
layer is not. **The DP2 skill shipped only its overview — `references/dp2-facts.md`
and `references/dp2-images-api.md` were not available** — so the items below are
written from the overview's description of the semantics, not from verified
signatures. They are collected so each is a one-line fix.

- [ ] **`REPO` / `COLLECTION`** in `rubin/extract.py` — `"dp2"` and
      `"LSSTCam/DP2"` are guesses. Check `Butler.get_known_repos()`,
      `$DAF_BUTLER_REPOSITORY_INDEX`, `butler.collections.query("*")`.
- [ ] **`DP2_ATTRS`** — the `CellCoadd` attribute names for image, variance,
      mask, psf, wcs, schema and origin. Every access goes through `_attr`,
      which on failure reports what the object actually offers, so a wrong name
      produces a useful error rather than a crash.
- [ ] **`PIXEL_ORIGIN`.** DP2 has two conventions: `sky_projection` works in
      *tract* coordinates, `astropy_wcs` in *patch-local*. Mixing them misplaces
      a position by up to a patch — far enough to land on the wrong galaxy, close
      enough to look plausible. This module commits to patch-local and then
      round-trips **every** stamp centre back to the sky, rejecting anything more
      than `CENTRE_TOLERANCE_ARCSEC` from the position asked for. If the
      convention is wrong, the first patch fails loudly with `centre_mismatch`
      in the manifest instead of quietly producing a mis-centred training set.
      **Check the rejection counts for `centre_mismatch` on your first run.**
- [ ] **`Box.factory` is `[y, x]`** — numpy order, the opposite of DP1's
      `Box2I(x, y)`. `_stamp_box` assumes this.
- [ ] **`coadd[box]` returns a view**, so `_extract` calls `.copy()`. If the API
      instead returns a copy this is merely wasteful, not wrong.
- [ ] **PSF accessor.** `psf_bundle` tries several spellings and gives up
      cleanly; the moments are computed from the returned stamp locally, so they
      do not depend on an unverified accessor.
- [ ] **Object-table columns** assumed unchanged from DP1 (`coord_ra`,
      `shape_xx`, `refExtendedness`, `{b}_cModelFlux`, `detect_isPrimary`).
- [ ] **Field coordinates.** ECDFS is a standard deep-drilling field so DP2 very
      likely covers it, but the DP2 field list was not available.
- [ ] Pixels are nJy in both releases; do not apply a calibration step.

## Open questions and known gaps

- **Source injection on DP2 is uncharted.** `CoaddInjectTask` against a
  `CellCoadd` is untested, and DP1 is the only release with per-visit pixels and
  difference images — so an injection campaign or a diffim comparison still has
  to happen on DP1 even though the prior trains on DP2.
- **Coadd noise is correlated, and that is handled by measurement.** Warping
  onto the skymap grid makes neighbouring pixels share flux, so averaging `P²`
  pixels reduces the noise by *less* than `P`. `measure_pooled_sky_noise` reads
  the pooled patches directly rather than deriving the value from the variance
  plane — at a realistic 0.8 px correlation width the derived value is **50%
  low**, which would put the transform's turnover in the wrong place. Worth
  checking the measured softening scales against the variance planes on real
  data to see how large the real effect is.
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
  plots.py           diagnostic figures (matplotlib imported lazily)
  data/              transform.py, pooling.py, augment.py, shards.py, dataset.py,
                     diagnostics.py (correlation length), synthetic.py (DP2-like
                     fake data for offline testing)
  rubin/             quality.py (stack-free gate), extract.py (lazy LSST imports)
scripts/             extract_dp2_patches.py, diagnose.py, prepare_config.py,
                     train.py, sample.py, smoke_test.py
tests/               ~100 tests, no cluster and no LSST stack required
```
