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
python scripts/extract_dp2_patches.py --config extraction.yaml
```

One ADQL query against `dp2.Object` returns the host list; the list is shuffled
and **walked host by host**, and each host is turned into up to one stamp per
band, centred on it, read as a bbox out of its own patch. The artefact gate runs
on each, and what survives is written as sharded HDF5 plus a manifest.

Everything the run does comes from `extraction.yaml`, which is checked in. The
script has no defaults of its own, so there is no way for a command-line flag and
a config key to disagree; `--repo` and `--collection` name the butler, and that
is all.

Read `<out>/summary.json` before trusting the output. It carries
`rejection_counts`, and those statistics are the only way to see whether the
selection function is biased — if bright, dense galaxy centres are being
rejected, the training set is skewed against exactly the regime this project
exists to model.

It also carries `correlation_length_native_flux_px`, accumulated over every
accepted patch as they are written (streaming, so it costs no memory). Treat that
as provenance only: at native resolution the small lags are dominated by the PSF,
and the log transform changes the correlation structure. The number to act on is
the pooled, log-space one from step 2.

### Reading `summary.json`

A **host** is a catalogue object. A **stamp** is one cutout of one host in one
band, so a single host yields up to `len(bands)` of them — which is why
`stamps_attempted` is several times `hosts_selected`, and why subtracting
rejections from it does not give back the hosts you asked for. The `counts`
block keeps the two words apart:

| key | means |
|---|---|
| `host_candidates_in_catalogue` | rows the ADQL returned, after the flags and the dedupe |
| `hosts_tried` | how far down the shuffled list the walk got |
| `hosts_with_no_coadd_for_their_patch` | of those, ones the repo has no `deep_coadd` for |
| `hosts_too_near_the_edge_of_coverage` | of those, ones whose stamp fitted inside no built cell grid |
| `stamps_attempted` | rows in the manifest |
| `stamps_accepted` / `stamps_rejected` | written / gated away |
| `stamps_requested` | `run.n_stamps` from the config |
| `shards_written` | HDF5 files under `<out>/shards` |

`rejection_counts` counts **every** reason a stamp failed, so it sums to more
than `stamps_rejected`. `seconds` says where the wall time went, and
`visits_per_cell` how deep the field was. The same thing is printed in prose at
the end of a run.

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
| `transform.png` | native flux → pooled → log space for a few patches, plus the pixel-value histogram. The sky must pile up on the red line at `x = log(s·log 2)` — one line, since one scale serves every band — inside the predicted scatter band, with sources clear of it. If it doesn't, the softening scale is wrong. |
| `training_batch.png` | exactly what the network receives: pooled, log-space, augmented, at the training size(s), on a shared colour scale so the spread between patches is visible. |
| `cutouts.png` | raw stamps as `asinh(flux / sky noise)` — a stretch in σ units with a pinned low end, so bands of very different depth are directly comparable. |
| `hosts.png` | the selected population: size, distortion, magnitude, surface brightness, Sérsic index, blendedness, band counts, sky noise and depth. |

The host panels come from `hosts.parquet`, which extraction writes alongside the
shards — those quantities are known only at selection time and are not carried in
the shard metadata.

Size and shape both come from the multiband Sérsic fit, the same fit the size cut
is made on, rather than from per-band adaptive moments: a second, PSF-convolved
answer to the same question is a thing to have to reconcile. Note the distortion
convention: `|e| = (1−q²)/(1+q²)` with `q = b/a`, roughly twice the `(1−q)/(1+q)`
shear convention at modest ellipticity.

### 3. Derive the config from the data

```bash
python scripts/prepare_config.py
```

Both paths default: `--shards` to `<ExtractionConfig.out>/shards` and `--out` to
`config.json`. Pass them when they differ.

**`config.py` is the one source of truth for defaults.** Every flag on
`prepare_config.py`, `train.py` and `sample.py` that names a config field
defaults to `None` and is applied only when it is actually given; the numbers
themselves live in the dataclasses in `src/rubin_host_prior/config.py`, and
`sample.py` reads them out of the checkpoint the model was trained with.

This is enforced by a test (`test_no_script_decides_a_config_value_for_itself`)
because it has already gone wrong twice. A flag with a number of its own
overrides the config on *every* run, passed or not: `prepare_config.py` reset
`softening_sigma` to the 1.0 of the old preserve-the-noise design, and reset
`out_size` to 64 — so the 128 px patch geometry `PatchConfig` documents was
never the geometry anything trained on. Three kinds of flag legitimately carry
values, and none of them is a config field: paths (`--shards`, `--out`,
`--config`, `--resume`), machine properties (`--devices`, `--max-in-memory-gb`),
and per-invocation choices (`--seed`, `--sampler`, `--weights`, `--n-stats`).

**Measured fields are `None`, not a plausible number.** `transform.softening`,
`sde.sigma_min`, `sde.sigma_max` and `sde.data_mean` follow from the data, so
`config.py` does not pretend to choose them — it says `None` and
`prepare_config.py` fills them in. `LogFluxTransform.from_config` and
`VESDE.from_config` both refuse a config still carrying `None` and name the
script to run.

`None` is doing real work beyond documentation: it distinguishes *not measured
yet* from *measured, and it came out at 10.0*. That is what lets the script
measure only what is missing and leave a value you set yourself alone — setting
the field is the only way to override the heuristic, and there is deliberately
no flag for it. When a `--base-config` already carries measured values they are
kept and listed on stdout, because a number carried over was not measured
against *these* shards, and after a change to `out_size` the statistics it came
from no longer exist.

While those fields carried defaults (`0.01`, `10.0`, `0.0`) none of them could
ever take effect: `prepare_config.py` overwrote all three unconditionally, so
the file claimed a say it did not have.

The softening scale and the σ range are not free hyperparameters;
they follow from the noise level and dynamic range. This measures them — from
pooled patches, not the variance planes, since coadd noise is correlated — and
prints two checks worth reading:

- `sky_scatter` should match `expected_sky_scatter(softening_sigma)`
  (= `0.721 / softening_sigma`). If not, the softening scale is wrong. It is a
  *typical* width: one scale serves every band, so a deeper or shallower band
  scatters proportionally less or more about the shared sky level.
- the flux above which the exponential model map is accurate should sit below
  anything you care about photometrically. The sky pedestal
  (`0.693 × softening_sigma`, in σ) is *meant* to sit above 1σ — 1.39σ at the
  default `softening_sigma = 2` — because suppressing the sky is the point of
  the softening. An earlier version of this line said to keep it under one
  sigma, from when the aim was to preserve the noise distribution.

It also reports the **correlation length** of the pooled log-space patches and
compares it to the model's `2R` crop — this is the authoritative measurement, and
the one that decides how much context your analysis needs. Note that ~60% of the
pixel variance sits in the zero-lag noise delta; the estimator renormalises at
lag 1 to exclude it, because a naive 1/e crossing on the raw profile returns
ξ ≈ 1 regardless of galaxy size (measured: wrong by 4×).

The softening scale is **one number for every band**, measured from the pooled
sky noise across all the patches. It used to be a dict per band, which brought
with it a whole apparatus that is now gone: a band the shards had no patches in
had no scale, so the tuple had to be indexed over the whole of `BANDS` with NaN
in the gaps (building it over the subset that happened to have patches re-bases
the indexing and hands an r-band patch the i-band scale), and
`PatchDataset.from_shards` had to check that every band present had a finite one.
None of that exists now — there is a scale or there is not, and
`LogFluxTransform.from_config` refuses a config without one.

The script still prints the patch count per band, because a band missing
entirely usually means the extraction hit its `n_stamps` target before reaching
it, or that no coadds exist for those tracts. It just no longer changes the
transform.

### 4. Train

```bash
python scripts/train.py --shards data/ecdfs/shards --config config.json --out runs/ecdfs
```

Writes into `--out`:

| path | what |
|---|---|
| `log.jsonl` | one line per log step, eval and checkpoint |
| `checkpoints/step-XXXXXXXX/` | the weights at each of `n_checkpoints` points |
| `samples/step-XXXXXXXX.png` | a square grid drawn from the EMA model there |
| `latest/` | the newest checkpoint, **with** the optimiser state |
| `final/` | the end of the run |

**Checkpoints are a count, not an interval.** `train.n_checkpoints` (default 10)
spreads them evenly over `steps`, with the last landing exactly on the final
step; an interval has to be recomputed by hand every time `steps` changes, and
getting it wrong means either one checkpoint or thousands. Asking for more
checkpoints than there are steps gives one per step rather than duplicates.

Each one keeps the weights and the EMA but **not** the optimiser state, which is
two more copies of the parameters and is only ever wanted for the most recent
checkpoint — that lives in `latest/`, overwritten each time. Ten checkpoints
therefore cost the weights ten times and the optimiser once.

**Each checkpoint draws `n_samples` (default 64) from the EMA model** and writes
them as an 8×8 grid. The EMA rather than the live weights, because that is what
inference uses. They are in the same log space and the same style as
`training_batch.png`, so put the two side by side — that comparison is the point,
and it only works because nothing is stretched differently between them. The
colour scale is shared across the whole grid: per-panel scaling would make every
sample look equally structured, including the ones that are noise.

**What a checkpoint costs.** It is `2 × sample_steps` batched backward passes —
Heun takes two score evaluations per step, each over the whole batch on a canvas
`4R` larger than the sample, and a score evaluation *is* a backward pass because
the score is a gradient. The count is printed before training starts.

On a NERSC GPU, 64 samples at the default 128 steps is **7 s**, plus a ~10 s
one-off compile of the sampler. Ten checkpoints cost about a minute — nothing
beside the training they punctuate.

On a CPU the same thing is *hours*: a 1M-parameter 8-layer model on a 72 px
canvas measured 4.7 s per sample per score evaluation, so 64×128 extrapolates to
around a day. Use `--n-samples 0` there rather than trimming steps.

That ratio is about **11000×**, not the 200× this file briefly claimed while
`sample_steps` was set to 32 on the strength of a CPU measurement. The error was
scaling across batch size and hardware at once: a GPU absorbs a batch of 64
almost for free, where a CPU pays linearly for every sample in it. A measured
number on the machine you will actually use beats an extrapolated one, and there
is no reason to economise on a minute.

Sampling is also the one step in training that can exhaust device memory on its
own, so a failure is logged with its reason and training continues — hours of
training must not be lost to a diagnostic. The log line for each checkpoint
carries `sample_mean`, `sample_std` and `sample_nonfinite`; a diverged sampler
produces `inf` rather than an error, and a grid of those looks like a blank
figure, so the count is what tells you.

`--n-samples 0` skips sampling; `--n-checkpoints 0` skips checkpoints entirely.

### Using the whole node

Training is data-parallel across **every GPU JAX can see**, which is what a
scheduler that allocates whole nodes wants. Nothing has to be passed for that;
`--devices 1` opts out.

The parameters are replicated on every device and `--batch-size` is the
**global** batch, split evenly across them. Four GPUs therefore make the *same
run* go faster rather than changing it: same global batch, same learning rate,
same loss curve. Raise `--batch-size` deliberately if a larger batch is what you
want — and then revisit the learning rate, which four devices on their own never
force you to do.

The gradients are all-reduced **every step**, not averaged periodically. The
model is 1.01 M parameters = 4.1 MB, so an all-reduce moves about 6 MB and takes
tens of microseconds over NVLink, against a step that is a
gradient-of-a-gradient: under 1%. Local SGD — run independently, average every
*k* steps — exists to hide a slow link *between nodes*. Inside one node it would
trade exact gradients for nothing, and cost a second optimiser state per replica,
a rule for what happens to Adam's moments and to the EMA at each average, and a
checkpoint format that has to decide which replica it is.

Because the weights are replicated rather than sharded, **a checkpoint says
nothing about how many devices wrote it**. A chunk trained on four GPUs resumes
on one and vice versa, which matters when the queue gives you what it has. The
device count is a command-line argument and deliberately *not* a config field,
for the same reason: it is a property of the allocation, not of the run. It is
recorded in the `log.jsonl` header as `n_devices`, where it is provenance rather
than something anything reads back.

`--batch-size` must be a multiple of the device count; the error says which
number to use. The evaluation batch and the checkpoint samples are left
replicated — they run every few thousand steps, so the redundant work is
invisible, and that keeps `--eval-size` and `--n-samples` free of a divisibility
rule of their own.

**Launch one process that can see all four GPUs**, not one process per GPU.
`srun --ntasks=1` with the whole node's GPUs bound to it is what this wants;
`--ntasks=4` starts four independent single-GPU trainings that will overwrite
each other's checkpoints, and nothing in the output will say so — each one is a
valid-looking run. `n_devices` in the `log.jsonl` header is where to check.

This is single-node. Spreading across nodes needs `jax.distributed.initialize`
and a larger mesh, but not different trainer code — try plain sharding across
nodes before reaching for anything cleverer.

### Running it in chunks

A scheduler that will not give you a long job means the run has to be a series
of short ones. The same command line works for every chunk:

```bash
python scripts/train.py --shards data/ecdfs/shards --config config.json \
    --out runs/ecdfs --resume
```

Bare `--resume` takes `<out>/latest` if it exists and starts fresh if it does
not, so the first submission and every later one are identical.

**`train.steps` is the length of the whole run, not of one chunk.** It is what
the cosine schedule decays over and what the checkpoint spacing is computed
from; setting it to the chunk length would restart the schedule every time. A
chunk runs from wherever the last one stopped until that total, or until a
signal.

**Stop the job before the scheduler kills it.** On `SIGUSR1` — what
`sbatch --signal=B:USR1@300` sends ahead of the wall clock — the loop finishes
the step it is on, writes `latest` with the optimiser state, logs a `stopped`
event, and returns. It deliberately does **not** write `final`: the run is not
finished, and the absence of `final` is what tells you there is more to do. No
samples are drawn on the way out, since the point is to be gone inside the grace
period. The handler only sets a flag — saving a checkpoint from inside a signal
handler would run JAX and filesystem work at an arbitrary point in a step — and
the previous disposition is restored when `train()` returns.

**Four things have to come back**, and the optimiser state is the one that gets
forgotten:

| | why |
|---|---|
| weights | the obvious one |
| EMA copy | a separate set of parameters, and the one inference uses |
| **optimiser state** | Adam's moments, **and** the schedule's step count — optax drives warmup and cosine decay from a counter inside it, so a fresh one re-runs the warmup from zero learning rate on a half-trained model |
| step number | so the EMA warmup continues instead of treating a half-trained model as new |

That is why only `latest/` can be resumed from: the numbered checkpoints carry
no optimiser state by design. The RNG key is folded with the start step so a
resumed chunk does not replay noise draws it has already used, and `train.py`
seeds the batch stream from the start step for the same reason — an iterator
cannot be fast-forwarded, so the same seed would replay the same batches.

### 5. Sample

```bash
python scripts/sample.py --checkpoint runs/ecdfs/final --out samples
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
runs on any scene above `4R + 1` pixels, where `R = r · Σ dilations` and
`r = (k−1)/2`. An undilated 3×3 stack has `R = n_layers`; see **Reach** below
for why that is not the only option.

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

### Reach: dilated branches, summed

Reach is `r · Σ dilations`, not `r · n_layers`. A 3×3 conv with dilation `d`
reads the same nine taps spread over `d` times the span — identical parameters,
identical arithmetic, `d` times the distance — so a doubling series buys
geometric growth for linear depth: `1, 2, 4, 8, 16, 1` reaches `R = 32` in six
layers where 32 undilated layers would be needed.

`EnergyConfig.channels` and `.dilations` hold **one tuple per branch**. Each
branch is its own stack ending in a 1×1 head; their energy maps are centre-
cropped to a common size and summed. A sum of energies is an energy, so the
score stays an exact gradient however many branches there are.

**The default is both branches** — `FINE_*` summed with `COARSE_*`, `R = 32`,
`2R = 64`. That is not cosmetic: `prepare_config.py` starts from these defaults,
so a config written without `--base-config` describes whatever they say. While
the default was one branch, every freshly generated config quietly had `R = 8`
and a 16 px loss crop, with the long-range branch absent and nothing in the file
to show it had ever been there. One branch is still a fine configuration — pass
a single tuple for each — it is just not one to arrive at by accident.

**Dilations are never inferred.** Setting `channels` without `dilations` is an
error, not a shorthand for all-ones. A branch's reach is `r · Σ dilations`, and
the loss crop, the minimum patch size and how much context the loader carries
all follow from it; it is not a number to leave unwritten.

**Why one stack and not two branches.** It was two: a wide undilated stack of
`R = 8` summed with a narrow dilated one of `R = 32`, on the reasoning that a
long-range path need not be wide. That model trained, and every measurement of
it came out right — the long-range branch carried 25–60% of the score, and the
training crops plainly held galaxies wider than its reach — but its samples
still contained no structure above ~16 px.

What measurement could not rule out is that **a sum lets the branches compete**.
`E = E_fine + E_coarse` gives the optimiser two ways to explain the same
residual and nothing in the objective to say which should win; the branch with
10× the parameters and 16× the arithmetic can answer for both, and a coarse
branch that is *used* is not the same as one that is *needed*. A single stack at
full width removes the choice: there is no cheap path to hide in.

The dilations alternate the doubling series with ones — `1, 1, 2, 1, 4, 1, 8,
1, 16, 1`, `R = 36` — so every long-range layer is followed by a local one that
integrates what it gathered.

It costs about 1.8× the previous step: 98 GMAC against 54, from 1.6× the
arithmetic per pixel on a patch 1.13× larger.

**Why dilation rather than pooling.** Pooling reaches as far for fewer FLOPs,
but it downsamples, and a stack with total stride `j` is invariant only to
shifts that are multiples of `j`. The resulting bias is locked to the *pixel
lattice* rather than to the image — and the fine branch cannot cancel it, because
a stride-1 valid CNN is exactly translation-equivariant and so can only produce
content-locked structure. Worse, a sampling run evaluates the score ~128 times
on the same canvas at the same phase, so a grid-locked bias integrates coherently
where a random one would average away. `test_dilation_keeps_the_score_translation_equivariant`
is what pins this down.

A doubling series leaves no holes: after `n` layers the reach is `2ⁿ − 1`, and
the next dilation `2ⁿ` is within `2R + 1` of what is already covered. Gridding
comes from *repeating* a large dilation with no small ones beneath it.

### Residual skips

`EnergyConfig.residual` is **on by default**: a centre-cropped skip around every
layer after the first. Two things about it are deliberate.

**A skip that cannot be made is an error, not a silent omission.** It needs
equal channel counts, so `residual=True` requires uniform widths within a
branch and says so; a tapering stack raises rather than quietly training without
the skips it was asked for. The one exception is structural and stated: the
first layer maps `in_channels` to the branch width, so there is nothing to add
to its output.

**The sum is divided by √2.** Two independent unit variances add to two, so an
unscaled skip doubles the variance per block — over nine blocks the stack
arrives at the head with ~180× the signal a plain stack would, the deliberately
small head no longer keeps the initial energy landscape flat, and training
starts at a **loss of 1.78**: worse than predicting no score at all. Measured,
with the alternatives:

| | energy-map std | initial loss |
|---|---|---|
| no residual | 5.7e-4 | 0.996 |
| residual, unscaled | 1.0e-1 | **1.775** |
| residual, ÷√2 | 3.6e-3 | 1.007 |
| residual, `head_init_scale` 1e-4 | 1.0e-3 | 0.996 |

Shrinking the head works too, but by a factor that depends on the depth and
would have to be retuned for every architecture. The √2 does not.

**What it costs.** `R = 32` means a **64 px crop per side**. Under the old
contract — patch in, `patch − 4R` trained on — a 512 px native stamp pooled to
170 would have left a 42 px interior, 6% of the arithmetic. That is what the
context border exists to avoid.

### The context border

**`out_size` is the size the loss is computed on, not the size the net is fed.**
The loader carries `2R` pixels of context on every side and the net is fed
`out_size + 4R`, so `crop_interior` lands on exactly the nominal crop:

```
fed 256 = [ 64 context ][ 128 nominal crop ][ 64 context ]
loss on         ^-- these 128 --^
```

The border is taken **from the rest of the stamp wherever there is any**, and
only the shortfall is reflect-padded. For `native_size` 512, `nominal_crop` 384
and a centred crop that is 21 real pooled pixels and 43 reflected per side;
translation moves real context from one side to the other rather than creating
more, so a crop pushed to one edge has a wholly synthetic border there and a
wholly real one opposite. `Config.real_context()` is the number, and
`prepare_config.py` and `train()` both print the split.

**Padding cannot manufacture clean signal.** The loss pixels whose receptive
field holds no reflected pixel number ~43 per side — exactly what a bare crop of
the whole stamp would have given. What the border buys is training on the *rest*
of the nominal crop as well, at the cost of a seam somewhere in those pixels'
receptive fields. That is a deliberate trade, taken because the alternative is
6% efficiency:

| | loss region | never sees the seam | clean px |
|---|---|---|---|
| context 64 | 128 | 43 (11%) | 1,820 |
| no context, whole stamp as one patch | 42 | 42 (100%) | 1,764 |

The defect it introduces is mild for a reason worth stating: **mirrored sky is
valid sky.** The augmentation already teaches dihedral invariance, so reflected
pixels are drawn from the same distribution as real ones. The only thing
reflection creates that nature does not is the *correlation across the seam* — a
locally mirror-symmetric neighbourhood. That is far milder than the truncated
energy sum the `2R` crop exists to avoid, which no amount of training can fix.
`reflect` and not `symmetric`, so the edge pixel is not duplicated.

The loss is unweighted: once padded, every pixel in the nominal crop counts the
same. If symmetric artefacts ever show up in the checkpoint sample grids, the
knobs are `nominal_crop` (smaller crop, more real context) and the coarse
branch's reach (shorter dilation series, smaller `2R`).

Two things deliberately do *not* see the border: `correlation_length` and
`stats` pass `context=0`, because a reflected border would put a mirror
correlation straight into ξ and double-count pixels in the sky scatter that sets
the σ range.

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

    soften:   f_s = s * softplus(f / s)                 softplus(u) = log(1+e^u)
    forward:  x   = log(f_s)                            → log(f) for f ≫ s
    model:    f   = exp(x)                              strictly positive

**`x` is log flux in nJy, absolutely.** `softplus(u) → u`, so `s·softplus(f/s) → f`
and the forward map converges to plain `log(f)`: a 4×10⁵ nJy pixel lands at
12.899 in *every* band, to machine precision. The model map is then `exp(x)` with
no band in it at all — `LogFluxTransform.inverse` takes no `band_index` — so a
forward model composing this prior with a likelihood in nJy has no per-band
offset to undo.

**There is one softening scale, `s`, for every band.** It was briefly per band,
and that bought nothing once `forward` became `log(f_s)`: a per-band scale only
moved each band's sky to its own `log(s_band·log 2)`, spreading the levels over
1.27 in `x`, while the thing the prior is about — the flux of the scene — was
already band-independent. With one scale the sky lands at `log(s·log 2)`
*everywhere*, so the transform gives an absolute flux scale **and** a common sky
level.

What differs between bands is the *width* of the sky about that level, and that
is the honest difference: a deep band scatters less than a shallow one, and
flattening that out was all the per-band scale was doing.
`expected_sky_scatter(softening_sigma)` is therefore a *typical* width — a band
deeper or shallower than the noise the scale was measured from scatters
proportionally less or more.

Earlier still, the transform divided by `s_band` inside the logarithm. That also
gave a common sky level, but by making `x` a *relative* quantity: the same flux
meant a different `x` in each band, and the absolute scale of the signal was what
got given away. Scene flux is what this prior exists to describe.

One thing follows that is handled rather than assumed away: **`SDEConfig.data_mean`.**
VE only adds noise, so the `t = 1` marginal keeps the data's mean, and
`prior_sample` must start from `N(data_mean, σ_max²)`. This was implicitly zero
while the sky sat at −0.37; at +3.0 a prior sample centred on zero starts half a
`σ_max` away from the distribution the score was trained on. `prepare_config.py`
measures it.

The softening is written as `s · softplus(f/s)` because that form has **one**
parameter, it is a flux, and above it the map is the identity: `softplus(u) → u`
exponentially fast, so bright pixels pass through untouched and the entire
adjustment is confined to the low and negative regime. `softplus` itself comes
from the library — `jax.nn.softplus`, or `numpy.logaddexp(0, u)` — rather than
being hand-rolled. The one branch that remains belongs to the *logarithm*, not
to softplus: below `u = -745` softplus underflows in float64 and its log is
`-inf`, where `log(softplus(u)) → u` is exact to 1e-9.

`exp(forward(f))` reproduces the softened flux exactly, so the entire
discrepancy between the data and what the model can express is the softening and
nothing else.

`soften(f, s)` is the definition, and `LogFluxTransform.soften` applies it with a
band's own scale. `forward` does not call it — it goes through `log_softplus`,
for the underflow reason above — so a test pins `forward == log(soften(f))` and
`inverse == soften` to keep the implementation and the definition one thing.
`softening_sigma` likewise has a single home in `TransformConfig`;
`estimate_softening` takes it as a required argument rather than carrying a
default of its own, and `prepare_config.py`'s `--softening-sigma` overrides the
config only when it is actually passed.

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
`patches.max_plane_fraction.SATURATED: 0.0` in `extraction.yaml` to follow the
guidance literally. `DETECTION_EDGE` pixels *are* excluded outright; `NO_DATA` is
measured as an inf-variance fraction instead — see below.

### Two questions the catalogue cannot answer

Every host cut runs on a **catalogue quantity** — magnitude, surface brightness,
Sérsic radius. That makes them statements about what the pipeline measured, and
when the pipeline is wrong they are no help at all. Two failure modes get through
all of them, and both are caught in the gate, on the pixels:

**Nothing at the centre.** A Sérsic fit that found nothing still has a magnitude
and a radius, so it passes every limit and arrives as a stamp of empty sky.
`centre_sigma` is the median of the central 16 px in units of the sky noise —
about 0 for empty sky, tens for a real host — and `min_centre_sigma` (2.0) gates
it. 16 px is 3.2″, comfortably inside a host selected at `reff ≥ 1.5″` and small
enough that an empty centre cannot hide behind the galaxy's outskirts.

**A crowd of stars instead of a galaxy.** Both are bright and both fill the
`DETECTED` plane, so no mask fraction separates them. The count of distinct
sources does: `n_peaks` is a smoothed local-maximum count, and `max_peaks` (200)
gates it. A host is one peak plus a few companions; a field of nothing but stars
runs to hundreds.

Getting that count right took two goes. The naive version — every 8-connected
local maximum above 5σ — counts the noise riding on a bright galaxy, so it
measures *how much of the stamp is bright* rather than how many sources are in
it: a single smooth galaxy scored 102. Requiring each peak to be the largest
within a PSF-sized radius fixes it (1 for that galaxy, 185 for a field of 200
stars), and the 3 px smoothing is what lets the radius stay near the PSF instead
of growing to suppress noise.

Both are **recorded for every stamp** whether they gate or not, so the thresholds
can be retuned from `diagnostic_percentiles` in the summary without re-reading
pixels; `None` disables either. Neither costs a query — they run on the stamp
that has already been read.

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

### `run.n_stamps` counts cutouts, not hosts

A host yields at most one cutout per band, and the gate rejects a share of those,
so the two units differ by nearly a factor of six. `n_stamps` is the one that
sizes the training set: the walk continues down the shuffled catalogue until it
has that many accepted cutouts, or until the catalogue runs out. The check
happens between hosts, so a run can overshoot by up to one host's worth of bands
rather than leaving its last host with an arbitrary subset of them.

Topping up is not free of consequences. Whatever the gate rejects, it rejects
preferentially, so a target that takes most of the catalogue to fill is drawn
from a different population than one filled in the first hundred hosts.
`rejection_counts` says where the rest went; a stamp can fail several gates at
once, so those counts sum to more than `stamps_rejected`.
`diagnostic_percentiles` gives the distribution of every gated quantity over
every attempt, which is what a threshold should be chosen from.

Hosts are drawn from **the whole DP2 footprint** by default, not from a disc
around a field centre. With a selective size cut that is the difference between a
workable sample and almost nothing: big galaxies are rare per square degree, so
the way to get more of them is more sky, not more draws from the same 0.3°.
`sky.radius_deg` still restricts to a field if you want one.

**The host cuts are sent to TAP**, and that is the only way the host list is
built. They are a
selection, and a selection is what a query service is for: the footprint is ~10⁹
rows and the survivors ~10⁴, so filtering where the catalogue already lives is
the difference between moving the survivors and moving the catalogue. One ADQL
query replaces reading every row of ~1000 object tables.

Note this is the opposite conclusion to the *cutout* service, and for the
symmetric reason. TAP is asked for a selection whose result is tiny; the cutout
service would be asked to ship pixels that are already on local disk. The right
question is never "remote or local" but "is the answer smaller than the input".

The WHERE clause is where the host cuts **are** applied. They are not applied
again after the rows arrive: re-running a cut the service already made is a
second answer to the same question, and two answers drift. Three things are left
for the client, because a query cannot do them — the boolean Sérsic failure flags
and the saturated/interpolated core flags, since how a boolean compares in ADQL
is backend-specific and a wrong guess silently returns nothing; and the
cross-tract dedupe, which needs the whole pool at once. There is no `ORDER BY` —
the tutorial is explicit that sorting burdens a shared service, and the draw
happens locally regardless. The job is submitted async and deleted afterwards,
including on failure.

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

TAP needs network, which a batch node may not have. That is what
`catalogue.cache` is for: query once where there is a network, and extraction
then runs from the parquet with no service at all. `catalogue.limit_hosts` puts a
`TOP N` on the query.

A cache has the cuts **baked into it**, so the ADQL that produced one is written
beside it as `<cache>.sql` and compared against what the current config asks for.
Reusing a cache after editing `extraction.yaml` is a cut that looks applied and
is not — the same failure the config file exists to prevent — so a mismatch stops
the run and prints both queries. Delete the two files to re-query.

There used to be a second route, scanning the per-tract `object` tables through
the butler, for a node with no network at all. It is gone. Two ways to build the
same list meant two column lists, two places the cuts were applied, and a
question about which one a given `hosts.parquet` came from; the cache covers the
offline case at a fraction of the reading.

TAP results are unmasked before use. A VOTable NULL comes back masked, and
`np.asarray` on a masked column hands back the raw buffer with no hint that part
of it is not data — for a float that is usually NaN and harmless, but for an
integer like `patch` it is whatever was in memory, which would file a host under
a patch it is nowhere near. Nulls in `patch`/`tract` are logged and set to -1 so
they match nothing; rows with no sky position are dropped with a count.

A host whose stamp does not fit inside the built cells of its patch is a
rejection, not a crash: it is too near the edge of coverage, which is a genuine
data condition on early DP2 and part of the selection function. But if **none of
the first `PATCH_CHECK_AFTER` hosts** produces a stamp at all, the run stops and
says so, because that pattern means the `Object.patch` column and the
`deep_coadd` dataId `patch` are not the same numbering — or `native_size` is too
large to fit inside a patch anywhere — rather than a run of unlucky edges.
Walking the whole catalogue to discover that and then writing an empty set is the
outcome worth avoiding.

Coadd queries are constrained with `data_id=`, not with a `where` string. The
expression language bit once and silently: in `where="tract = :tract"` the bind
key **shadows the dimension of the same name**, so it resolved as
`tract = tract` — true for every row. The query returned the whole repo,
truncated at the default 20000, and since patch indices repeat across tracts the
client-side patch filter let refs from anywhere through. Hosts were matched
against same-numbered patches in other tracts and projected ~200 000 pixels away.
A ref whose data id is not the one asked for now ends the run.

### The walk is host-major

One host, one position, one cutout per band. The loop used to be organised by
patch — group the hosts by the patch the catalogue assigned them, sweep tract by
tract, slice every host that fell inside each patch from one read — because that
amortises a whole-patch read across the hosts in it. Once the pixels come from a
bbox read there is nothing to amortise, and with a selective size cut a patch
holds one or two hosts anyway.

What that structure cost was everything built to support it: the tract sweep, the
`by_patch` grouping, the accepted-`(host, band)` set guarding against tract
overlap, the round-and-top-up machinery, and the patch-yield diagnostics. A host
is now visited once, by position, so it **cannot** be extracted twice and none of
that bookkeeping has anything to guard.

**Only the stamp's pixels are read.** `butler.get(ref, parameters={'bbox': box})`
returns a `CellCoadd` of just that region without loading the patch (DP2 tutorial
104.5). A patch is ~4100 px square and a stamp is 512, so that is about two
orders of magnitude less I/O.

Everything else about a patch comes from **component reads**, which move no
pixels either: `sky_projection` and `provenance`.

Those three started as **76% of the runtime** — 303 s of 398 s on a real run,
against 80 s for the pixels themselves. They are per-(tract, patch, band)
quantities that the old patch-major sweep amortised for free and a host-major
walk would otherwise pay per host, so two things claw it back:

- the shuffled host list is **ordered** by patch — ordered, not grouped; the loop
  is still one host at a time, and patches keep the shuffled order of the first
  host drawn into each, so *where* the set is drawn from is unchanged;
- a **one-patch component cache** then catches every repeat, because that
  ordering guarantees the repeats are consecutive. It is keyed on the patch and
  holds all of its bands: a key including the band would be cleared on every
  band of a single host and never hit.

The same cache holds the **ref query**, which is one `query_datasets` per host
asking an identical question for every host in a patch. Leaving it out of the
first version of the cache was worth 22% of a run on its own.

The reads are timed separately (`read: wcs`, `read: provenance`) rather than as
one total, because "component reads: 303 s" does not say which of them to
attack. That measurement is what killed a confident guess: the PSF looked like
the obvious culprit — `bounds` is a bounding box and a set of missing cells, and
getting it meant deserialising a 22×22 grid of per-cell PSFs — but split out it
was 51.7 s against 45.0 s for the WCS and 25.3 s for provenance. No single read
dominated. After the cache the profile was flat: pixels 108 s, ref queries 68 s,
psf 52 s, wcs 45 s, provenance 25 s.

### The read is the fit test

There was a pre-check before the pixel read: take the coadd's cell grid, and ask
whether both stamp corners lay inside the cells that were actually built and
whether any cell in the *middle* of the stamp was missing. Both conditions make
the bbox read raise anyway — so the pre-check bought nothing but an earlier
answer, at the cost of the PSF read that produced the grid.

It is gone. `butler.get(ref, parameters={'bbox': box})` is wrapped in a `try`,
and a failure is a rejection with reason `off_the_grid`. The cell indices the
depth check needs then come off the **stamp**, which is a `CellCoadd` of just
that region and already in hand — they must be the patch's own `(i, j)` and not
indices relative to the sub-region, which is exactly what the existing
provenance cross-check asserts on the first stamp of a run.

**What that cost.** The pre-check could tell "this stamp is off the grid" from
"this read failed"; a `try` cannot. A burst of `OSError` would otherwise be
counted as a selection effect and quietly bias the training set. So every
distinct exception type is logged once at WARNING with its message, and
`stamp_read_failures` in the summary maps each type to the first message seen —
empty on a healthy run, and a type other than the geometry error there means the
repo is broken rather than the sky being ragged.

Worth recording for anyone who reaches for the cell grid again: `CellCoadd.grid`
and `CellCoadd.bounds` are **not** butler components. They are Python properties
reading through to `self._psf.bounds`, so asking the butler for them fails and
falls back to a whole-patch read — which is how this code spent several runs at
~100× the pixel I/O without anyone noticing. The grid now comes off the stamp,
which is a `CellCoadd` and carries it directly.

Nothing falls back. A component the repo will not serve ends the run, naming the
role and the component it was asked for. The alternative — read the whole patch
and carry on — is ~100× the pixel I/O for an identical result, and it ran that
way undetected for several runs before a warning caught it.

### Why not the cutout service

DP2 tutorial 103.6 describes exactly the shape this walk has: name a position,
get back a cutout with the metadata a `deep_coadd` carries. It is the *SODA*
service, though — a remote HTTP endpoint at `data.lsst.cloud`, reached through
`lsst.rsp.RSPDiscovery` and `get_pyvo_auth`, neither of which exists off the
platform. Using it from a batch node would mean reimplementing both, then paying
an ObsCore lookup, a datalink resolution (15-minute expiry) and a SODA request
for every host in every band, with every pixel crossing the WAN.

`butler.get(ref, parameters={'bbox': box})` is the same request against pixels
that are already on local disk, and it returns the same `CellCoadd`, provenance
and cell grid included. The rule that settles it is not "remote or local" but
*is the answer smaller than the input*: TAP is asked for a selection out of 10⁹
rows and hands back 10⁴, so it belongs where the catalogue is; the cutout service
would be asked to ship bytes that are already here.

**A stamp is tested against the cell grid, not the image** — by the read itself.
A patch at the edge of coverage has cells that were never built: its image `bbox`
is the full patch while the cell grid covers only the populated part, and slicing
outside that raises rather than returning empty pixels. So does a hole in the
middle. A host that hits either is rejected as `off_the_grid` and counted: being
too near the edge of coverage is a real property of early DP2 and part of the
selection function.

The host list is **shuffled** before the walk, because the walk stops the moment
`n_stamps` is reached — in catalogue order a run that stopped early would be
drawn entirely from one corner of the footprint.

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
sometimes overrode one and not the other. They are now stated once. `gate` has no
defaults for any of its tolerances — every one is a required argument, filled by
`PatchCuts.gate_kwargs()` — so there is no second answer for the config to drift
away from, and a test asserts that the two signatures match. A key the file does
not recognise, or a whole mistyped section, is an error rather than a silently
ignored line, since a cut that looks applied and is not is the worst of the three
outcomes. YAML rather than JSON so the reasoning can sit next to the numbers.

Extraction prints `describe()` before it runs: the faint limit, the surface
brightness limit, and **which of the two binds at each size**. Size and
brightness are not independent — `mu_e = m + 2.5·log10(2π·a·b)` — so a magnitude
limit and a surface-brightness limit can quietly exclude each other over exactly
the range you care about, and the answer to "why did this find nothing" is
usually in those four lines.

### Visibility is magnitude; extent is one size measurement

`max_mag` is the primary host cut. Surface brightness decides whether a *fit* is
real, but it is a poor proxy for "I can see it": a tight `max_mu_e` selects
**concentrated** light, which is the opposite of what a prior over galaxy
structure wants — it favours exactly the compact objects that look like point
sources. So `max_mu_e` is left loose, as a bound on runaway fits rather than a
selector, and total flux does the work.

That distinction was learned the hard way. Nothing else in the selection requires
the object to be *visible*: with a 3″ half-light radius the old 360 nJy floor
admitted objects at μ_e = 29.4 mag/arcsec², some 2.4 mag/arcsec² fainter than one
sigma of sky per square arcsecond in r. At that signal-to-noise the multiband
Sérsic fit is degenerate along (n, R_e, flux) and walks off to a large radius
around an invisible envelope while the real light stays in a few pixels. Those
rows pass every size cut and arrive as point-like blobs — which is exactly what
the cutouts figure was showing. The surface-brightness bound is written
server-side as `flux >= K · reff_major · reff_minor`, multiplication only, since
`LOG10` and `POWER` are not guaranteed across ADQL dialects and a clause the
service silently declines to apply is worse than one it refuses.

**Extent is `min_reff_arcsec` and nothing else** — the half-light major axis of
the multiband Sérsic fit, in arcsec, measured before PSF convolution. There used
to be a second, non-parametric size cut alongside it, built from the per-band HSM
adaptive moments with the PSF removed in quadrature,
`T² = ((ixx+iyy) − (ixxPSF+iyyPSF))/2`. It was correct, and it is gone: two
measurements of the same quantity means carrying the question of which one to
believe, and five extra columns per band to answer it with. The Sérsic fit is one
morphology fit to all six bands at once and is the better of the two.

That cut is also the reason to say what a *wrong* size cut looks like. Adaptive
moments are flux-weighted toward the core and run 1.4× smaller than R_e for an
exponential and 4–9× smaller for a de Vaucouleurs, so a sample correctly cut at
R_e ≥ 3″ plotted at 0.7–2.2″ in the old trace-radius panel. Seeing sizes below
the cut is what a *working* cut looks like in the wrong units. And an absolute
threshold on the raw moments meant nothing at all: a star's trace radius is
whatever the seeing was — 2.0 px at median DP2 seeing, against a `min_trace_px`
of 1.75, which therefore rejected nothing.

**The bright end is a saturation flag, not a flux ceiling.** An early 3e6 nJy
ceiling (r = 15.2) sat 1.5–3 mag *below* where cores actually saturate, so it was
blocking precisely the nearly-saturating galaxies wanted. `min_mag` is now
nominal and `{band}_pixelFlags_saturatedCenter` does the work, which is literally
"this core is not saturated". `interpolatedCenter` goes with it: an interpolated
core is synthetic structure exactly where the transient goes.

### Host selection on DP2

The DP2 Object table differs from DP1 in ways that break code silently rather
than loudly:

- **There is no band-independent `shape_xx`.** Second moments are per band
  (`{band}_ixx`, `{band}_iyy`, `{band}_ixy`, in pixel²), and they are measured on
  the PSF-convolved coadd. None of them are read any more — see above — but a
  DP1-era column name would fail the whole query rather than one column.
- **All six bands carry photometry and shapes.** `u` through `y` all have
  `_cModelFlux`, `_ixx` and `_sersicFlux`. An earlier version of this file said
  only `ugri` did; that came from reading the *rendered* schema page in excerpts,
  which is long enough to truncate mid-table and give a confidently wrong answer.
  Check column questions against the schema YAML in `lsst/sdm_schemas`
  (`python/lsst/sdm/schemas/drp_base.yaml`), which is small enough to grep and
  carries the units as `ivoa:unit`.
- **There are no `detect_*` columns at all**, so `detect_isPrimary` is not
  available for dropping duplicates — and tracts overlap, so a source in an
  overlap region appears twice, under two *different* `objectId`s. `dedupe_hosts`
  therefore collapses near coincidences on the sky (0.5″) as well as repeated
  ids. This is no longer about patches: a host is visited once, by position, so
  it cannot be extracted twice by the walk. It is about the catalogue listing one
  galaxy under two rows, which the walk would take at face value.
- **TAP's `dp2.Object` view and the butler's `object` parquet are not the same
  table.** TAP serves derived columns the pipeline never wrote —
  `{band}_cModelMag` among them — and asking the butler for one fails the whole
  read with a formatter error. Only the TAP view is read now, so there is one
  column list rather than two that had to be kept from drifting.
- DP2 also offers continuous `{band}_sizeExtendedness` and
  `{band}_model_extendedness`, either a better primary cut than the hard 0/1
  `refExtendedness` if the sample turns out to need one, and
  the band-independent `sersic_*` block — `reff_major`/`reff_minor` in arcsec,
  `index`, `theta`, `rho`, and per-band `{band}_sersicFlux` for all six bands.

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
can be retuned from `variance_step` without re-reading pixels.
`max_variance_step: null` keeps everything.

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
can. Two independent conventions meet here — the grid's `(i, j)` and the
provenance table's cell columns — and nothing guarantees they agree on which one
is x, so the first stamp checks that its cells appear in the table at all and, if
they do not, ends the run naming both spellings. Transposed, every lookup misses
and every stamp reads as zero-visit, which is the most confusing possible reason
to reject a whole field.

`n_cells_spanned` is recorded as the footprint, and `visits_per_cell` in the
summary gives the depth of the **whole run** — every cell of every patch read.
That replaces a log line that reported the first patch only: it read like a
property of the run, so it moved whenever anything perturbed the RNG stream that
decides which host is visited first. Removing the per-host jitter did exactly
that, and the reported depth went from 1–9 visits to 1–1 with no change to the
data at all.

**Absolute depth is a separate question from depth *variation*.** Early DP2
outside the deep fields runs 1–3 visits per cell, which is a different sky from a
deep coadd — and at 1–3 visits a single-visit difference between neighbouring
cells is a `cell_depth_ratio` of 2 or 3, so the 1.5 default rejects nearly
everything there. `n_visits_min` is recorded for every stamp and `min_visits`
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
that does not answer, a mask schema that changes mid-run, a WCS that does not
round-trip, an empty or unreadable `provenance.contributions`, a cell grid that
will not answer `index_of`, a ref whose data id is not the one asked for — all
raise, naming what to fix.

The same rule applies to having two ways of doing one thing, which is the same
disguise in a different coat. There is one route to the host list (TAP), one size
measurement (`sersic_reff_major`), one place each cut is stated
(`extraction.yaml` — `gate` has no tolerance defaults of its own, so it cannot
disagree), and one way pixels are read (a bbox around the host). What used to be
"the other option" was in every case the one nobody ran, which is to say the one
nobody would notice had broken.

### Storage

Shards hold **native-resolution stamps in physical units** (nJy) and nothing
else in the way of arrays, plus a row of scalars per stamp: `x0`/`y0`, the host
position and id, the pixel scale, the sky noise, the depth the stamp was cut at
and the fractions the gate measured. Pooling, the log transform and the offsets
all happen in the loader, so any of them can change without re-extracting.

The prior is a distribution over pixels and never sees a variance plane, a mask
or a PSF, so carrying them tripled the storage to no purpose. They are still
*read* during extraction — they are what the gate is made of — and then dropped;
what survives of them is that row of scalars, which is what a later cut from the
manifest needs.

`x0`/`y0` and the mask plane dictionary are mandatory, not optional: LSST boxes
have non-zero origins, bit assignments are not guaranteed stable across releases,
and without both a saved stamp cannot be mapped back to the sky or interpreted.

**Every batch is built from the shards.** There was once a second loader path
serving a pre-pooled, pre-transformed cache keyed by a hash of the config. It was
faster, but it baked the crop in — so it lost the translation augmentation — and
it was one more derived file to fall out of step with the shards it came from.
The shards load fast enough; the cache is gone rather than switched off.

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
- [x] **`deep_coadd.sky_projection`, `.psf` and `.provenance` as component
      reads.** A component the repo will not serve ends the run. It used to fall
      back to loading the whole patch, which is ~100× the I/O for an identical
      result, and did so silently for several runs — note that `grid` and
      `bounds` are *not* components at all but properties reading through to
      `psf.bounds`, which is how that fallback got triggered on every stamp.
- [ ] **`grid.index_of(x=, y=)` returning `.i`/`.j`.** The cell grid is what the
      depth and missing-cell checks are made of, so it raises rather than
      degrading if the attribute names differ.
- [ ] **Field choice.** ECDFS (tract 5063) is the best-characterised DP2 field
      and the one the tutorials use; check dp2.lsst.io before choosing on cadence
      grounds.
- [ ] **`provenance.contributions` column names.** The API documents the table
      as `{visit, detector, cell}` without pinning the spellings, and `CellIJ`
      cannot survive into an astropy column as one object, so
      `CONTRIB_CELL_COLUMNS` tries `cell_i/cell_j`, `cell_x/cell_y`, `i/j`,
      `x/y` and logs the real names if none match. One run says which it is.
- [ ] **`max_variance_step`** was chosen from what a depth step looks like,
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
extraction.yaml      the whole description of an extraction run, checked in
src/rubin_host_prior/
  geometry.py        valid-conv shape arithmetic; read before choosing a patch size
  config.py          every dataclass that must travel with a checkpoint
  selection.py       extraction.yaml's dataclasses: the host and patch cuts
  nn/                layers.py (FiLM, Fourier, ConvBlock), energy.py (net + score)
  diffusion/         sde.py (VE), loss.py (DSM + interior crop), sampler.py
  training/          trainer.py, ema.py, checkpoint.py
  plots.py           diagnostic figures (matplotlib imported lazily)
  data/              transform.py, pooling.py, augment.py, shards.py, dataset.py,
                     diagnostics.py (correlation length), synthetic.py (DP2-like
                     fake data for offline testing)
  rubin/             quality.py (stack-free gate), extract.py (lazy LSST imports)
scripts/             extract_dp2_patches.py, diagnose.py, prepare_config.py,
                     check_tap.py, train.py, sample.py, smoke_test.py
tests/               no cluster and no LSST stack required; fakes.py is a butler
                     small enough to read, which is where the walk is tested
```
