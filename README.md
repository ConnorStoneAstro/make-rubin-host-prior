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

### Valid convolutions everywhere, and the `4R` geometry

No padding, so no border artefacts and no dependence on patch size — the model
runs on any scene above `2R + 1` pixels, where `R = n_layers` for 3×3 kernels.

But the energy is a *sum* over the final feature map, and that does not weight
input pixels equally. Energy cell `p` sees input pixels `[p, p+2R]`, so

```
∂E/∂xᵢ = Σ over p ∈ [max(0, i−2R), min(i, E−1)]
```

contains all `2R+1` terms only for `2R ≤ i ≤ H−2R−1`. Outside that window the
sum runs over a *subset* of kernel offsets, so a border pixel's score is a
different linear functional of the weights than an interior pixel's — not merely
smaller (kernel weights have either sign), but systematically different, and the
network has no path to the missing terms. Consequences:

- The denoising loss is cropped to the interior `H − 4R` window.
- **Sampling needs a padded canvas.** `sample_interior` generates `out_size + 4R`
  and returns the middle.
- **Inference on a region of interest of size `Rgn` needs `Rgn + 4R` pixels of
  input.** With `L = 8`, that is 64 pixels of context.

`geometry.describe` prints the arithmetic for a given configuration:

```
>>> from rubin_host_prior import geometry
>>> geometry.describe(64, 8)
'input 64x64 -> energy map 48x48 (R=8, margin=16); loss interior 32x32 (25% of pixels)'
```

25% of pixels is the price of 8 layers on a 64-pixel patch. `L = 6` gives 40×40
(39%). Choose deliberately.

One further consequence of the summed energy: **energies are extensive in scene
area**, so they are only comparable between scenes of equal size. That is what
makes the model a translation-invariant prior over scenes of any size.

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
x = log1p(f / b_band) / c          f = b_band · expm1(c · x)
```

`b_band` is a per-band soft offset in nJy, nominally `k_sigma ×` the *pooled* sky
noise of that band. Three properties follow:

- **No hard floor, no point mass.** DP1 `visit_image` pixels are
  background-subtracted, so roughly half of all sky pixels are negative.
  `log(max(f, floor))` would pile 40–50% of every patch onto one value;
  `log1p(f/b)` is smooth and strictly monotonic through zero.
- **Band-agnostic.** Dividing by `b_band` puts every band's sky level at `x ≈ 0`
  with scatter `≈ 1/k_sigma`, so u-band and y-band patches land in the same place
  and one prior covers all six. There is no band label anywhere in the model.
- **Linear where it matters, logarithmic where it must be.** Near the noise floor
  `x ≈ f/(b·c)`, a pure rescaling, so additive Gaussian pixel noise stays
  additive and Gaussian. In the bright regime it is logarithmic, which tames the
  ~10⁴ dynamic range of a galaxy core.

The clip at `floor_ratio = -0.9` is a numerical guard for artefacts, not a
modelling choice: at `k_sigma = 5` a legitimate sky pixel reaches it only at 9.5σ
(~4×10⁻⁶ of pixels in practice). `PatchDataset.stats()` reports how often it
fires.

A forward model in log space needs no Jacobian: generate the model scene in `x`,
map to flux with `transform.inverse`, and compare to the data. `transform.jacobian`
exists if you ever want a density in flux units.

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
  **no interpolation at all**. On by default. This is why `native_size` (224)
  exceeds `nominal_crop` (192).
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
                     synthetic.py (DP1-like fake data for offline testing)
  rubin/             quality.py (stack-free gate), extract.py (lazy LSST imports)
scripts/             extract_dp1_patches.py, prepare_config.py, train.py,
                     sample.py, smoke_test.py
tests/               ~100 tests, no cluster and no LSST stack required
```
