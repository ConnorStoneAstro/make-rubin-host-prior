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

The softening scale is **measured per band**, so a band the shards contain no
patches in gets no scale — inventing one would put its turnover wherever the
guess landed. The script prints the patch count per band and says so when one is
empty; a band missing entirely usually means the run hit its `--n-patches` target
before reaching it, or that no coadds exist for those tracts.

The transform is nonetheless **always indexed over the whole of `BANDS`**, with
NaN where a scale was never measured. `band_idx` in the shards is a global index
into `BANDS`, so the softening tuple has to be too: building it over the subset
that happened to have patches silently re-bases the indexing, and with
`('r','i','z','y')` measured, index 4 runs off the end while index 2 quietly
returns the i-band scale for an r-band patch. `PatchDataset.from_shards` checks
that every band actually present has a finite scale and names the ones that do
not, so the failure arrives where the band can be identified rather than as a
NaN in training.

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

    soften:   f_s = s_band * softplus(f / s_band)       softplus(u) = log(1+e^u)
    forward:  x   = log(f_s / s_band) / c
    model:    f   = s_band * exp(c * x)                 strictly positive

The softening is written as `s · softplus(f/s)` because that form has **one**
parameter, it is a flux, and above it the map is the identity: `softplus(u) → u`
exponentially fast, so bright pixels pass through untouched and the entire
adjustment is confined to the low and negative regime. `softplus` itself comes
from the library — `jax.nn.softplus`, or `numpy.logaddexp(0, u)` — rather than
being hand-rolled. The one branch that remains belongs to the *logarithm*, not
to softplus: below `u = -745` softplus underflows in float64 and its log is
`-inf`, where `log(softplus(u)) → u` is exact to 1e-9.

`s · exp(c · forward(f))` reproduces the softened flux exactly, so the entire
discrepancy between the data and what the model can express is the softening and
nothing else.

**The softening suppresses the sky, and that is the point.** With `s` at two
sigma of the pooled per-band noise, pixels within the noise are compressed
towards a pedestal at `softplus(0)·s = 1.39σ` while anything detected is
untouched. The prior is meant to describe what a galaxy looks like, not what
this realisation of the sky looked like — and a prior that reproduces noise
faithfully spends capacity on something the likelihood already models. This is a
deliberate reversal: the transform used to soften at 1σ specifically to keep the
noise distribution intact.

`softening_sigma` is that scale in units of the measured noise, and is the knob.
Lower preserves the noise more faithfully at the cost of a skewed, heavy-tailed
`x`; higher flattens the sky harder and pushes the flux at which the exponential
map becomes accurate proportionately up (0.1% above 5σ at `softening_sigma = 1`,
above 10σ at 2).

A source cannot emit negative flux, so the prior's reachable domain in flux space
must be strictly positive — hence the plain exponential, which maps all of ℝ to
(0, ∞). The data transform is therefore **not** exactly invertible, and should
not be: measured flux *is* negative wherever noise takes it below the subtracted
sky, and the right thing to do with those pixels is let them approach zero
smoothly. An ELU-style softening was considered and rejected: `ELU(u)+1` leaves a
permanent `+s` offset on positive flux (still 3.3% high at `f = 30s`), whereas
softplus converges to the identity exponentially.

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

**No-data is carried twice over.** DP2 has a `NO_DATA` plane *and* `inf` in the
variance where there were no contributing exposures, including the cores of
saturated stars. The plane is excluded outright. The variance is treated more
gently — its fraction is measured, with a tolerance and zero tolerance at the
centre — because rejecting merely because non-finite variance is *present* would
discard every stamp containing a bright neighbour.

**`SATURATED` gets a fraction, not zero — a deliberate deviation.** Rubin's own
guidance (tutorial 202.5) is to exclude `NO_DATA` and `SATURATED`, which for
*pixels* in a measurement is right. For whole training stamps it is not: on a
coadd the saturated core of a bright neighbour lands in a great many stamps, and
a scene with a bright neighbour is exactly the regime this project models, so a
blanket cut would reproduce the selection bias the rejection statistics exist to
expose. A small fraction is kept away from the centre, zero at the centre where
the transient goes, and the fraction is recorded either way. Set
`FRAC_TOL["SATURATED"] = 0.0` to follow the guidance literally. `NO_DATA` *is*
excluded outright, as are `DETECTION_EDGE` pixels.

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

### `n_hosts` is not the size of the training set

A host yields at most one cutout per band, and the gate rejects a share of those,
so `--n-hosts 128` produces nowhere near 128 cutouts — `n_accepted` in the
summary is the number that actually got written, and the first thing to read
beside it is `rejection_counts`, which says where the rest went. A stamp can fail several
gates at once, so those counts sum to more than `n_rejected`; `first_rejection_counts`
gives the first reason only, which is what the figure and the summary used to
disagree about. `diagnostic_percentiles` gives the distribution of every gated
quantity over every attempt, which is what a threshold should be chosen from.

Hosts are drawn from **the whole DP2 footprint** by default, not from a disc
around a field centre. With a 3″ size cut that is the difference between a
workable sample and almost nothing: big galaxies are rare per square degree, so
the way to get more of them is more sky, not more draws from the same 0.3°.
`--radius-deg` still restricts to a field if you want one.

**The host cuts are sent to TAP** (`--host-source tap`, the default). They are a
selection, and a selection is what a query service is for: the footprint is ~10⁹
rows and the survivors ~10⁴, so filtering where the catalogue already lives is
the difference between moving the survivors and moving the catalogue. One ADQL
query replaces reading every row of ~1000 object tables.

Note this is the opposite conclusion to the *cutout* service, and for the
symmetric reason. TAP is asked for a selection whose result is tiny; the cutout
service would be asked to ship pixels that are already on local disk. The right
question is never "remote or local" but "is the answer smaller than the input".

Only the numeric cuts go into the WHERE clause. The boolean Sersic failure flags
are fetched and applied locally, because how a boolean compares in ADQL is
backend-specific and a wrong guess silently returns nothing; the point-source
cross-check and the cross-tract dedupe are not expressible there either. There is
no `ORDER BY` — the tutorial is explicit that sorting burdens a shared service,
and the stratified draw happens locally regardless. The job is submitted async
and deleted afterwards, including on failure.

**You do not need to be on the RSP.** TAP is an IVOA standard and the Rubin
endpoint is an ordinary TAP service behind a bearer token, so `pyvo` talks to it
from anywhere — `lsst.rsp` exists only on the RSP itself, where it wraps exactly
this. The endpoint comes from Rubin's own public discovery document
(`https://data.lsst.cloud/repertoire/discovery`, no auth needed), which for DP2
gives `https://data.lsst.cloud/api/tap`; `--tap-url` overrides it.

The token is read from `ACCESS_TOKEN`, `NUBLADO_TOKEN` or `RSP_TOKEN`, then from
`/etc/nublado/secrets/token`, `~/.rsp-token`, `~/.rsp_token` — the same
precedence `lsst.rsp` uses, so a notebook and a login node behave alike. Make one
at data.lsst.cloud under *Security tokens* with the `read:tap` scope. It is
deliberately **not** a command-line argument: that would put it in shell history
and in every process listing on a shared machine. It is attached only to requests
whose URL is under the TAP endpoint, because a session-wide header follows
redirects and one redirect off-host hands the token to whoever answered.

If TAP answers **401**, run `python scripts/check_tap.py`. It separates the four
things that all present as one 401: no token, a token Gafaelfawr does not
recognise (expired, revoked, or not an RSP token — `ACCESS_TOKEN` is a generic
name that other software sets too, which is why the source it came from is
logged), a valid token without the `read:tap` scope, and a query the service
refuses. The token value is never printed, only its type prefix. `tap_client`
runs the scope check itself before submitting anything, so the failure arrives
with a reason attached rather than as a 401 from inside a job.

TAP needs network, which a batch node may not have. That is what `--host-cache`
is for: query once where there is a network, cache, and extraction then runs with
no service at all. `--limit-hosts` puts a `TOP N` on the query.

**`--host-source butler`** is the offline route: scan the object tables through
the repo instead, one tract at a time, applying the cuts before anything is
concatenated so what is held is the host list and not the footprint. It reads far
more but needs nothing beyond the repo. It is not a silent fallback — ask for it.
`--limit-tracts` bounds a test run.

Sampling deliberately does *not* happen during either scan: a stratified draw has
to see the whole pool or it stratifies within tracts instead of across them.

TAP results are unmasked before use. A VOTable NULL comes back masked, and
`np.asarray` on a masked column hands back the raw buffer with no hint that part
of it is not data — for a float that is usually NaN and harmless, but for an
integer like `patch` it is whatever was in memory, which would file a host under
a patch it is nowhere near. Nulls in `patch`/`tract` are logged and set to -1 so
they match nothing; rows with no sky position are dropped with a count.

A position that will not project is not fatal either. It means "not in this
patch" — either the row had no coordinates or the patch does not cover that piece
of sky — so it is a rejection, not a crash. If **`PATCH_CHECK_AFTER` patches in a
row hold none** of the hosts the catalogue assigned to them, the run says so and
prints where the hosts actually projected against the patch's pixel span, because
that pattern means the `Object.patch` column and the `deep_coadd` dataId `patch`
are not the same numbering rather than a run of unlucky edges.
`n_patches_with_hosts` / `n_patches_without_hosts` in the summary is the same
thing after the fact.

Per-tract queries are constrained with `data_id=`, not with a `where` string.
The expression language bit once and silently: in `where="tract = :tract"` the
bind key **shadows the dimension of the same name**, so it resolved as
`tract = tract` — true for every row. The query returned the whole repo,
truncated at the default 20000, and since patch indices repeat across tracts the
client-side patch filter let refs from anywhere through. Hosts were matched
against same-numbered patches in other tracts and projected ~200 000 pixels away.
Refs are now filtered on tract client-side as well, and a ref from the wrong
tract is reported rather than dropped quietly.

The sweep is then **tract-major**, which is what makes the footprint affordable.
The object table is per tract, so the neighbour index is built once per tract and
thrown away; and within a tract only the patches that actually hold a host are
asked for, since under a cut this selective a patch holds one or two hosts and
sweeping every patch that overlaps a field loads a great many that hold none.
Candidates for a patch are the hosts the catalogue assigned to it — testing every
host against every patch is quadratic and unaffordable once the list spans the
sky.

**Only the stamp's pixels are read.** `butler.get(ref, parameters={'bbox': box})`
returns a `CellCoadd` of just that region without loading the patch (DP2 tutorial
104.5). A patch is ~4100 px square and a stamp is 416, so that is about two
orders of magnitude less I/O — and with a 3″ cut the hosts are thin enough that
there is rarely a second stamp in a patch to amortise a whole read against.

Everything else about a patch comes from **component reads**, which move no
pixels either: `sky_projection`, `bbox`, `psf` and `provenance`.

`grid` and `bounds` are *not* among them, which cost a run its whole optimisation
once: `CellCoadd.grid` and `CellCoadd.bounds` are Python properties reading
through to `self._psf.bounds`, not stored components, so asking the butler for
them fails and forces a whole-patch read for something the `psf` component
already carries. They are taken from the PSF object instead — `psf.bounds` is the
`CellGridBounds`, and `psf.bounds.grid` the `CellGrid`.

If `provenance` turns out not to be served either, per-cell visit counts go away
and depth boundaries fall to the measured `variance_step`, with a warning —
rather than silently paying ~100× the I/O for a covariate. `--min-visits` is the
exception: it is asked for explicitly, so it loads whole patches to honour it.

Both paths fall back. If components are refused the patch is read whole and its
attributes used — and the run says **which** component forced that, once, because
silently taking ~100× the pixel I/O is exactly the kind of thing that should be
visible. If a bbox read is refused, or comes back missing a plane — checked on
the spot, since an absent `variance` would otherwise surface as an
`AttributeError` hours in — the run switches to whole patches and says so.

**A stamp is tested against the cell grid, not the image.** A patch at the edge
of coverage has cells that were never built: its image `bbox` is the full patch
while `bounds.bbox` covers only the populated part, and slicing outside that
raises rather than returning empty pixels. `bounds` is the right predicate and
excludes individually missing cells too. Because the corner test cannot see a
hole in the *middle* of a stamp, the cells the stamp covers are checked against
`bounds.missing` separately. And whatever else goes wrong cutting one stamp is
recorded as `cut_failed` and the run continues — one stamp is one stamp.

`--n-patches` is the number to ask for when you want a training set of a given
size. Extraction then works towards it: draw a batch of hosts, sweep every coadd
patch, and if it is still short draw another batch — sized from the yield it has
actually observed, since that depends on the field, the band set and how tight
the gate is, none of which are known in advance — excluding hosts already tried.
It stops when the target is met, the catalogue runs out, or `--max-rounds` is
reached, and says which. `--max-patches` is the older hard stop and never tops
up.

Two consequences worth knowing. The patch list is **shuffled**, because the sweep
stops the moment the target is reached and the butler returns patches ordered by
band: left in order, a run that stopped early would be entirely g-band and
entirely one corner of the field. And topping up is not free — whatever the gate
rejects, it rejects preferentially, so a set filled over several rounds is drawn
deeper into the catalogue than one filled by the first round. Read
`rejection_counts` before deciding that is acceptable.

### One file describes the whole run

```bash
$EDITOR extraction.yaml
python scripts/extract_dp2_patches.py --config extraction.yaml
```

`extraction.yaml` is **checked into the repository**, not generated. It holds
everything a run does — where to look, which catalogue to ask, what a host is,
what a usable stamp is, how much to write. The script has no defaults of its
own, so a run is reproducible from a file you can read, diff and check in, and a
flag and a config key cannot disagree.

The cuts used to be spread across defaults on `select_hosts`, defaults on
`host_adql`, module constants in `rubin.quality` and command-line flags that
sometimes overrode one and not the other. A key the file does not recognise — or
a whole mistyped section — is an error rather than a silently ignored line, since
a cut that looks applied and is not is the worst of the three outcomes. YAML
rather than JSON so the reasoning can sit next to the numbers.

Extraction prints `describe()` before it runs: the faint limit, the surface
brightness limit, and **which of the two binds at each size**. Size and
brightness are not independent — `mu_e = m + 2.5·log10(2π·a·b)` — so a magnitude
limit and a surface-brightness limit can quietly exclude each other over exactly
the range you care about, and the answer to "why did this find nothing" is
usually in those four lines.

### Visibility is magnitude; extent is size

`max_mag` is the primary host cut. Surface brightness decides whether a *fit* is
real, but it is a poor proxy for "I can see it": a tight `max_mu_e` selects
**concentrated** light, which is the opposite of what a prior over galaxy
structure wants — it favours exactly the compact objects that look like point
sources. So `max_mu_e` is left loose, as a bound on runaway fits rather than a
selector, and total flux does the work.

Extent is `min_reff_arcsec` (intrinsic, from the multiband Sersic fit) together
with `min_deconvolved_px`, which is the same claim made against the image:
`T² = ((ixx+iyy) − (ixxPSF+iyyPSF))/2`, exactly zero for a point source at any
seeing.

### Host selection on DP2

The DP2 Object table differs from DP1 in ways that break code silently rather
than loudly, so `select_hosts` is built to fail loudly instead:

- **There is no band-independent `shape_xx`.** Second moments are per band
  (`{band}_ixx`, `{band}_iyy`, `{band}_ixy`, in pixel²). `host_trace_radius_px`
  raises if they are absent rather than returning NaN, because every caller uses
  it to avoid a sample dominated by the smallest, faintest galaxies.
- **TAP's `dp2.Object` and the butler's `object` parquet are not the same table.**
TAP serves derived columns the pipeline never wrote — `{band}_cModelMag` among
them — and asking the butler for one fails the whole read with a formatter
error. The SDM schema describes the TAP view. So the host selection (TAP) and
the neighbour index (butler) get different column lists, each chosen for what
its source has and its caller needs, and a butler read that fails on a column
says which of the two tables it is talking to.

**All six bands carry photometry and shapes.** `u` through `y` all have
  `_cModelFlux`, `_ixx` and `_sersicFlux`. An earlier version of this file said
  only `ugri` did; that came from reading the *rendered* schema page in excerpts,
  which is long enough to truncate mid-table and give a confidently wrong answer.
  Check column questions against the schema YAML in `lsst/sdm_schemas`
  (`python/lsst/sdm/schemas/drp_base.yaml`), which is small enough to grep and
  carries the units as `ivoa:unit`.
- **There are no `detect_*` columns at all**, so `detect_isPrimary` is not
  available for dropping duplicates — and tracts and patches overlap, so a
  source in an overlap region appears twice, across two tracts under two
  *different* `objectId`s. `dedupe_hosts` therefore collapses near coincidences
  on the sky (0.5″) as well as repeated ids.
- DP2 also offers continuous `{band}_sizeExtendedness` and
  `{band}_model_extendedness`, either a better primary cut than the hard 0/1
  `refExtendedness` if the sample turns out to need one, and
  the band-independent `sersic_*` block — `reff_major`/`reff_minor` in arcsec,
  `index`, `theta`, `rho`, and per-band `{band}_sersicFlux` for all six bands —
  which is what the size cut and the diagnostic plots use.

**Surface brightness is the cut that decides whether a host is a galaxy.** Not
size. Nothing else in the selection requires the object to be *visible*: with a
3″ half-light radius the old 360 nJy floor admitted objects at
μ_e = 29.4 mag/arcsec², some 2.4 mag/arcsec² fainter than one sigma of sky per
square arcsecond in r. At that signal-to-noise the multiband Sersic fit is
degenerate along (n, R_e, flux) and walks off to a large radius around an
invisible envelope while the real light stays in a few pixels. Those rows pass
every size cut and arrive as point-like blobs — which is exactly what the
cutouts figure was showing.

So `--max-mu-e` (default **24.5**) is the primary cut, written server-side as
`flux >= K · reff_major · reff_minor` — multiplication only, since `LOG10` and
`POWER` are not guaranteed across ADQL dialects and a clause the service
silently declines to apply is worse than one it refuses.

The sample is therefore **surface-brightness limited, not size limited**. The
size bounds are wide — 0.7″ to 12″, a factor of 17 — because that is the range
asked for, "a couple of arcsec across up to very large". A galaxy of ordinary
brightness at R_e = 3″ covers about 30 pooled pixels of visible isophote, so 3″
was never too small; it was the wrong knob.

**The point-source cut is referenced to the PSF.** `{band}_ixx` and friends are
HSM adaptive moments on the *PSF-convolved* coadd, so a star's trace radius is
whatever the seeing was — 2.0 px at median DP2 seeing, against a `min_trace_px`
of 1.75, which therefore rejected nothing. The cut is now on
`T² = ((ixx+iyy) − (ixxPSF+iyyPSF))/2`, which a point source makes exactly zero.

**Two things worth knowing about the trace radius**, since it is what the old
hosts figure plotted: adaptive moments are flux-weighted toward the core and
run 1.4× smaller than R_e for an exponential and 4–9× smaller for a de
Vaucouleurs. A sample correctly cut at R_e ≥ 3″ plots at 0.7–2.2″ in trace
radius. Seeing sizes below the cut in that panel is what a *working* cut looks
like in the wrong units.

**The bright end is a saturation flag, not a flux ceiling.** The old 3e6 nJy
ceiling (r = 15.2) sits 1.5–3 mag *below* where cores actually saturate, so it
was blocking precisely the nearly-saturating galaxies wanted. The ceiling is now
nominal and `{band}_pixelFlags_saturatedCenter` does the work, which is literally
"this core is not saturated". `interpolatedCenter` goes with it: an interpolated
core is synthetic structure exactly where the transient goes.

**Stratification bins are fixed, not data-driven.** Edges taken from the
sample's own min and max hand whole bins to whatever tail exists — so with
runaway fits in the pool, the stratification written to rescue rare large
galaxies was preferentially rescuing rare bad fits instead.

`min_trace_px` stays as a second, non-parametric floor from the adaptive moments,
a cross-check against a runaway fit.

**The flux ceiling had to move with it.** Size and flux are not independent: a
galaxy with a 3″ half-light radius and an ordinary effective surface brightness
of 22 mag/arcsec² has r ≈ 17.6, nine times brighter than the 36000 nJy ceiling
this used to carry. That ceiling was set for a 1″ population and would have
annihilated the size cut. It is now 3e6 nJy (r ≈ 15.2); saturated cores are the
gate's job rather than this one's. `select_hosts` also notices when a size cut
and a flux range are nearly disjoint and says so, rather than silently returning
nothing.

**Size stratification draws from equal-width bins in log half-light radius, not
quantiles.**
This matters and was wrong until recently: quantile bins hold equal numbers by
construction, so drawing equally from each is *exactly* a uniform sample and
stratifies nothing. With equal-width bins the sample carries ~9× more
well-resolved hosts than a uniform draw, falling to 1× as the request approaches
the whole population — you cannot over-sample galaxies that are not there.

### Cell structure, and the depth step it causes

DP2 coadds are cell-based — a 22×22 grid of **150-pixel** cells, each coadded
from its own set of input visits (typically 11–33). Depth and PSF are therefore
piecewise constant, with genuine discontinuities at cell edges. A 416-pixel
native stamp spans roughly **3×3 cells**, so essentially every training patch
straddles them, and staying inside one cell is not an option: 150 native pixels
is 50 pooled, which with `L = 8` leaves an 18×18 loss interior.

Where the input set actually changes, the noise level steps across a straight
cell edge and the stamp comes out visibly patchworked — most obviously in y,
which has the fewest visits and so the largest fractional step. **No mask plane
flags this.** It is not a defect: the pixels are all real, they are just not
equally deep, and Rubin has nothing to say about it. What records it is the
*variance plane*, which is where depth lives.

So `variance_step` measures it directly — the ratio between the highest and the
lowest block-wise variance floor across the stamp, 1.0 for a uniform stamp and
the depth ratio of the two cells for a boundary. Two things keep a galaxy from
being mistaken for a step, which matters because source Poisson variance is
one-sided and the biggest, best hosts would be rejected first:

- Pixels the *image* shows to be source are dropped before any floor is taken.
  In sky-limited data a source only inflates the variance once its flux
  approaches the sky per pixel, which is a detection at S/N of order √(sky
  counts) — far above the 3σ threshold used — so this removes every pixel where
  the source could matter, with a wide margin.
- What is taken within a block is the 10th percentile, not a mean or median. At
  the 25th, a bright 3″ galaxy reads as a step of 1.9; at the 10th it reads 1.16.
  A real step is measured identically at any percentile — the ratio of two like
  quantiles is unbiased — so the low one is free.

With both, a σ = 30 px galaxy at 200× the sky reads 1.01 while a true 1.3 step
still reads 1.31. The default threshold is **1.5 in variance**, which is 1.22 in
noise σ; the ratio goes into the manifest whether the stamp passes or not, so it
can be retuned from `variance_step` without re-reading pixels. `--max-variance-step
inf` keeps everything.

**And the coadd states the step exactly, before any pixel is read.**
`CellCoadd.provenance.contributions` is a table of `{visit, detector, cell}` —
which observation went into which cell. DP2 exposures share an integration time,
so the number of distinct visits in a cell *is* its depth, and the ratio across
the cells a stamp covers is the step. `cell_visit_counts` reads it once per
patch; the ratio is gated as `cell_depth_ratio` and `n_visits_min` /
`n_visits_max` go into the manifest.

`deep_coadd_input_summary` is not an alternative: Rubin documents it as
patch-level and states outright that it does not record which visit-detector
images contributed to each cell.

The two measures are kept because neither subsumes the other. Coaddition is
inverse-variance weighted, so cells with equal visit counts still differ by
whatever the seeing and the sky did — the counts cannot see that and the variance
can; and provenance can be absent or spelled differently, in which case the
variance is all there is. Two independent conventions meet here — the grid's
`(i, j)` and the provenance table's cell columns — and nothing guarantees they
agree on which one is x, so the first stamp checks that its cells appear in the
table at all and, if they do not, says so and falls back rather than rejecting
every stamp for the most confusing possible reason.

`n_cells_spanned` is still recorded as the footprint, and `visits_per_cell` in
the summary gives the depth of the **whole run** — every cell of every patch
swept. That replaces a log line that reported the first patch only: it read like
a property of the run, so it moved whenever anything perturbed the RNG stream
that decides which tract is visited first. Removing the per-host jitter did
exactly that, and the reported depth went from 1–9 visits to 1–1 with no change
to the data at all. The per-patch line is still printed, but it names its patch.

**Absolute depth is a separate question from depth *variation*.** Early DP2
outside the deep fields runs 1–3 visits per cell, which is a different sky from a
deep coadd — and at 1–3 visits a single-visit difference between neighbouring
cells is a `cell_depth_ratio` of 2 or 3, so the 1.5 default rejects nearly
everything there. `n_visits_min` is recorded for every stamp and `--min-visits`
gates on it, off by default: whether a shallow coadd belongs in this prior is a
judgement about the prior, not about the pixels.

### Only the image is stored

The prior is a distribution over pixels. It never sees a variance plane, a mask
or a PSF, so carrying them tripled the shards to no purpose. They are still
*read* during extraction — they are what the quality gate is made of, and a bbox
read returns them anyway — and then dropped. What survives is a handful of
scalars per stamp (`sky_noise`, `variance_step`, `frac_no_data`, the visit
counts), which is what a later cut from the manifest needs.

This is also why the cutout service stays the wrong tool even though it returns
images only: the pixels are already on local disk, and the planes the gate needs
come free with the read that fetches them.

### Old output directories are refused, not silently read

Shards carry a `schema` attribute. Schema 1 stored variance, mask and PSF arrays
beside a different metadata set, and it would still *open* here — missing
metadata columns fill with -1 — which is precisely the danger: a stale set
trains or plots without complaint. `ShardSet.open` refuses it, and refuses a
shard missing any metadata column too. A `hosts.parquet` written before the
Sersic switch is recognised by its per-band cModel radii and says so.

### Nothing falls back

A read either answers or ends the run. Every quiet degradation this code had
turned out to be a bug wearing a disguise: `grid` and `bounds` were assumed to be
butler components, were not, and silently forced a whole-patch read on every
stamp at ~100× the I/O for several runs before a warning caught it. A component
that does not answer, a mask schema that changes mid-run, an object table that is
missing for a tract hosts were selected from — all now raise, naming what to fix.

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

The model, loss, loader and gate are tested locally; the Butler and `lsst.images`
layer is not. These are now written against the verified DP2 reference rather
than guessed, so the list is short:

- [ ] **`Butler("dp2", collections="dp2")`** — both are literally `"dp2"`.
      Off-platform, confirm the alias with `Butler.get_known_repos()` or
      `$DAF_BUTLER_REPOSITORY_INDEX`.
- [x] **Butler API generation.** `DataCoordinate` stopped being a `Mapping` in
      daf_butler v27, so `dict(data_id)` now falls through to sequence iteration
      and raises `KeyError: 0`. `_data_id_dict` goes through `.mapping` /
      `.required` instead, and still handles the old form.
- [ ] **`deep_coadd.bbox` and `.sky_projection` as component reads.** Both are
      documented, but if either is refused the loop falls back to loading the
      whole patch (slower, identical result) and warns once, rather than
      rejecting every ref in the field.
- [ ] **`grid.index_of(x=, y=)` returning `.i`/`.j`.** Used only for the
      `n_cells_spanned` covariate, and degrades to `-1` if the attribute names
      differ.
- [ ] **Field choice.** ECDFS (tract 5063) is the best-characterised DP2 field
      and the one the tutorials use; check dp2.lsst.io before choosing on cadence
      grounds.
- [ ] **`min_trace_px = 1.75`** in `select_hosts` is a DP1-era ComCam PSF size.
      It is now only a cross-check behind the 1″ half-light cut, so it matters
      less, but check it against the DP2 PSF before leaning on it.
- [ ] **`provenance.contributions` column names.** The API documents the table
      as `{visit, detector, cell}` without pinning the spellings, and `CellIJ`
      cannot survive into an astropy column as one object, so
      `CONTRIB_CELL_COLUMNS` tries `cell_i/cell_j`, `cell_x/cell_y`, `i/j`,
      `x/y` and logs the real names if none match. One run says which it is.
- [ ] **`--max-variance-step 1.5`** was chosen from what a depth step looks like,
      not from DP2 statistics. Look at the `variance step` panel in `hosts.png`
      and the rejection counts after a real run: if it is rejecting a large
      fraction, the cells in this field differ more than assumed and the
      threshold, not the data, is what should move.
- [ ] **`centre_mismatch` count in the manifest should be zero.** Every stamp
      centre is projected back to the sky and compared against the position
      asked for. DP2 has two pixel-origin conventions — `sky_projection` is
      tract, `astropy_wcs` is patch-local — and mixing them is an error of up to
      a full patch (~4000 px), far enough to land in the wrong galaxy and close
      enough to look plausible. This module works in tract coordinates
      throughout, which is what `Box.factory` and `bbox.contains` expect, and the
      round-trip makes any residual geometry error loud rather than silent.

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
