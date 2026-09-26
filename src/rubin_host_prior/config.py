"""Configuration dataclasses.

Everything that changes the meaning of a trained model lives here so it can be
serialised next to the weights.  The rule: if you cannot reproduce the training
data transform from the checkpoint, the prior is unusable in a forward model.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BANDS = ("u", "g", "r", "i", "z", "y")


#: The texture branch: wide, undilated, R = 8 (17 px across).
FINE_CHANNELS: tuple[int, ...] = (32, 64, 96, 128, 128, 128, 128, 128)
FINE_DILATIONS: tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1, 1)

#: The long-range branch: narrow, dilated, R = 32 (65 px across) in six layers.
#: Summed with the fine one, it is what lets the model represent a galaxy larger
#: than the texture branch can see.  It costs about 6% of the fine branch's
#: arithmetic and 9% more parameters; what it really costs is the loss crop,
#: which goes from 16 px per side to 64 -- see ``data.pooling`` for how the
#: loader supplies that much context.
COARSE_CHANNELS: tuple[int, ...] = (32, 32, 32, 32, 32, 32)
COARSE_DILATIONS: tuple[int, ...] = (1, 2, 4, 8, 16, 1)


@dataclass(frozen=True)  # hashable: stored as a static field on the module
class EnergyConfig:
    """Fully convolutional energy, summed over one or more parallel branches.

    ``channels`` and ``dilations`` hold **one tuple per branch**.  Each branch is
    an independent stack of valid convolutions ending in a 1x1 head, and their
    energy maps are centre-cropped to a common size and added.  A sum of energies
    is an energy, so the score stays an exact gradient however many there are.

    **The default is both branches**, ``FINE_*`` and ``COARSE_*``, because that
    is the model this project trains.  A config written without one would
    describe a different architecture from the one that is going to be used,
    and since ``prepare_config.py`` starts from these defaults, the difference
    would be silent: the first sign of it is a training run with a 16 px loss
    crop instead of 64.  One branch is still a perfectly good configuration --
    pass a single tuple for each -- it is just not the one to get by accident.

    **Why more than one.**  Reach is ``r * sum(dilations)``, so a long-range
    branch needs few layers -- but every layer of a *single* stack would have to
    be as wide as the widest, and that width is there for texture, not for
    large-scale structure.  Two branches let the long-range path run narrow: the
    suggested ``COARSE_*`` pair reaches R = 32 for about 5% of the fine branch's
    arithmetic.

    **Why dilation and not pooling.**  Pooling reaches the same distance more
    cheaply, but it downsamples, and a stack with total stride ``j`` is invariant
    only to shifts that are multiples of ``j``.  The resulting score carries a
    bias locked to the pixel lattice rather than to the image, which the other
    branch cannot cancel -- it is translation-equivariant, so it can only
    produce content-locked structure -- and which accumulates coherently over the
    hundred-odd score evaluations of a sampling run.  Dilation never
    downsamples, so full pixel-level equivariance survives.

    A doubling series leaves no holes: after ``n`` layers the reach is
    ``2^n - 1``, and the next dilation ``2^n`` is within ``2R + 1`` of what is
    already covered.  Gridding comes from *repeating* a large dilation with no
    small ones beneath it, not from doubling.
    """

    in_channels: int = 1
    #: One tuple per branch.  ``((32, 64, ...),)`` would be one branch of eight.
    channels: tuple[tuple[int, ...], ...] = (FINE_CHANNELS, COARSE_CHANNELS)
    #: Per-layer dilation, same shape as ``channels``, and always stated: a
    #: branch's reach is ``r * sum(dilations)``, so leaving it to be inferred
    #: would mean the single most consequential number in the geometry -- the
    #: loss crop follows from it -- was one nobody wrote down.
    #: ``(1, 2, 4, 8, 16, 1)`` reaches R = 32 in six layers, the trailing 1
    #: mixing neighbouring long-range features back together.
    dilations: tuple[tuple[int, ...], ...] = (FINE_DILATIONS, COARSE_DILATIONS)
    kernel_size: int = 3
    activation: str = "silu"  # must be C^1; see nn.layers.ACTIVATIONS
    embed_dim: int = 128  # width of the log-sigma embedding MLP
    n_fourier: int = 32  # random Fourier features of log(sigma)
    fourier_scale: float = 1.0  # std of the random frequencies
    fourier_seed: int = 0  # fixes the frozen basis; see nn.layers
    head_init_scale: float = 0.01  # small, not zero -- see nn.energy
    film_init_scale: float = 0.01  # small, not zero -- see nn.layers.FiLM
    residual: bool = False  # center-cropped skip where channel counts match
    sigma_scaling: str = "inverse_sigma"  # "inverse_sigma" | "none"

    def __post_init__(self) -> None:
        flat = [c for c in self.channels if isinstance(c, int)]
        if flat:
            raise ValueError(
                "channels is now one tuple per branch, e.g. ((32, 64, 96),) for "
                "a single branch; this one is a flat list of widths. The config "
                "predates branches -- re-run scripts/prepare_config.py."
            )
        if not self.channels:
            raise ValueError("an energy needs at least one branch")
        if len(self.channels) != len(self.dilations):
            raise ValueError(
                f"{len(self.channels)} channel tuples but "
                f"{len(self.dilations)} dilation tuples; there must be one of "
                f"each per branch. If you set channels and left dilations "
                f"alone, state them too -- they are never inferred, because a "
                f"branch's reach is r * sum(dilations) and the loss crop, the "
                f"patch size and the loader's context all follow from it."
            )
        for i, (c, d) in enumerate(zip(self.channels, self.dilations)):
            if not c:
                raise ValueError(f"branch {i} has no layers")
            if len(c) != len(d):
                raise ValueError(
                    f"branch {i} has {len(c)} layers but {len(d)} dilations; "
                    f"every layer needs one"
                )
            if any(x < 1 for x in d):
                raise ValueError(f"branch {i} has a dilation below 1: {tuple(d)}")

    @property
    def n_branches(self) -> int:
        return len(self.channels)

    @property
    def n_layers(self) -> int:
        """Total across every branch -- for parameter counts and log lines, not
        for geometry, which depends on the dilations rather than the depth."""
        return sum(len(c) for c in self.channels)

    @property
    def receptive_radius(self) -> int:
        """``R``: the largest reach among the summed branches."""
        from . import geometry

        return geometry.receptive_radius(self.dilations, self.kernel_size)

    @property
    def loss_margin(self) -> int:
        return 2 * self.receptive_radius


@dataclass
class SDEConfig:
    """Variance-exploding SDE, geometric sigma schedule, no preconditioning.

    **All three are measured, not chosen**, which is why all three are None
    rather than carrying plausible-looking numbers.  They follow from the data's
    noise level and dynamic range, and ``prepare_config.py`` fills them in from
    the shards -- exactly as it does ``TransformConfig.softening``.

    None also makes "not yet measured" distinguishable from "measured and it
    came out at 10.0", which is what lets the script measure only what is
    missing and leave a value you set yourself alone.  While these carried
    defaults there was no such distinction: a number written here was silently
    overwritten on the next run, so the defaults could never take effect and the
    file claimed a say it did not have.
    """

    #: Below the sky scatter in x, where the score is dominated by noise the
    #: data already contains.
    sigma_min: float | None = None
    #: Must dominate the data's own spread, or the ``t = 1`` marginal is not
    #: really Gaussian and sampling starts from the wrong distribution.
    sigma_max: float | None = None
    #: Mean of ``x`` over the training set.  VE does not move the mean, so the
    #: ``t = 1`` marginal is centred here and ``prior_sample`` has to start from
    #: the same place.  This was implicitly zero while the transform put every
    #: band's sky at ``log(log 2) = -0.37``; under absolute log flux the sky sits
    #: near +3, and a prior sample centred on zero starts half a ``sigma_max``
    #: away from the distribution it is meant to be drawn from.
    data_mean: float | None = None


@dataclass
class TransformConfig:
    """Flux -> log-space transform.

        x = log(s_band * softplus(f / s_band))        f = exp(x)

    ``x`` is log flux in nJy, absolutely: ``softplus(u) -> u``, so the forward
    map converges to plain ``log(f)`` and the model map is ``exp(x)`` with no
    band in it.  A forward model composing this prior with a likelihood in nJy
    has no per-band offset to undo.

    The model map is a plain exponential, so the prior's reachable domain in
    flux space is strictly positive -- a source cannot emit negative flux.  The
    data transform is therefore deliberately not its exact inverse: measured
    flux goes negative wherever noise takes it below the subtracted sky, and
    those pixels are smoothly carried towards zero instead.

    ``s = softening_sigma * pooled sky noise`` sets where the softening turns
    over, and is the knob that decides how hard the sky is flattened.  At 2.0
    the pedestal sits at 1.39 sigma and pixels within the noise are compressed
    towards it: the prior describes the galaxy rather than this realisation of
    the sky, which is what the likelihood is for.

    **One scale, for every band.**  It used to be a dict per band, which bought
    the transform nothing once ``forward`` became ``log(f_s)`` -- see
    ``data.transform``.  With one scale the sky lands at ``log(s * log 2)``
    everywhere, and the bands differ only in how wide the sky is about it.
    """

    #: nJy.  Measured by ``prepare_config.py``; None until then.
    softening: float | None = None
    softening_sigma: float = 2.0  # s = softening_sigma * pooled sky noise


@dataclass
class PatchConfig:
    """Native cutout geometry and the pooling that produces a training image.

    ``native_size`` is deliberately larger than ``nominal_crop`` so that scale
    jitter can go in both directions and so that integer translations in native
    pixels (which are exact -- no interpolation) give sub-pooled-pixel jitter.

    The defaults target 128 px training patches, where 56% of each patch clears
    the ``2R`` loss crop (against 25% at 64 px).  ``native_size`` is 416 rather
    than the 384 that 128 px strictly needs: 384 would be exactly 3 x 128,
    leaving no room at all to translate the crop and silently disabling an
    augmentation that is otherwise free and exact.  32 native pixels of slack is
    +/- 5.3 pooled pixels, far more than the +/- 1 pooled pixel needed to cover
    every sub-pixel phase.
    """

    native_size: int = 512  # pixels cut from the coadd
    nominal_crop: int = 384  # native pixels feeding one training image
    out_size: int = 128  # nominal_crop / pool_factor; the reference size
    pool_factor: int = 3
    #: Extra training sizes.  Each batch is drawn at one size (a batch must be
    #: shape-homogeneous), cycling over ``training_sizes``.  Larger patches
    #: spend proportionally less of themselves on the cropped border, so mixing
    #: sizes both augments the data and recovers loss signal.  Every size shares
    #: the same weights and estimates the same size-independent bulk potential,
    #: so this costs nothing in accuracy -- which is only true because the loss
    #: is cropped; a full-field loss makes different sizes fight each other.
    out_sizes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.nominal_crop != self.out_size * self.pool_factor:
            raise ValueError(
                f"nominal_crop ({self.nominal_crop}) must equal out_size * "
                f"pool_factor ({self.out_size} * {self.pool_factor})"
            )
        if self.native_size < self.nominal_crop:
            raise ValueError("native_size must be >= nominal_crop")
        for s in self.out_sizes:
            if s * self.pool_factor > self.native_size:
                raise ValueError(
                    f"out_size {s} needs {s * self.pool_factor} native pixels "
                    f"but native_size is {self.native_size}"
                )

    @property
    def training_sizes(self) -> tuple[int, ...]:
        """Every size the loader will emit, ``out_size`` first."""
        return (self.out_size,) + tuple(
            s for s in sorted(set(self.out_sizes)) if s != self.out_size
        )

    @property
    def max_translate_native(self) -> int:
        """Native-pixel translation room, fixed at the *reference* size.

        Smaller training crops leave more room in the stamp, but letting them
        wander that far would change the data distribution with size -- small
        patches would mostly land on blank sky away from the host.  Capping the
        offset at the reference size's room keeps every size looking at the same
        neighbourhood.
        """
        return max(self.native_size - self.nominal_crop, 0)


@dataclass
class AugmentConfig:
    """Augmentations that do not touch the noise properties.

    ``dihedral`` and ``translate`` are exact re-indexings.  ``scale_jitter`` is
    *not*: it resamples, which correlates neighbouring pixels slightly.  It is
    off by default for that reason -- turn it on deliberately.
    """

    dihedral: bool = True  # the 8 rotations/reflections of the square
    translate: bool = True  # integer native-pixel shifts of the crop
    scale_jitter: float = 0.0  # +/- fractional deviation from pool_factor


@dataclass
class TrainConfig:
    batch_size: int = 128
    steps: int = 2_000_000
    learning_rate: float = 1e-4
    warmup_steps: int = 2_000
    cosine_decay: bool = False
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    log_every: int = 1000
    seed: int = 0
    #: Validation cadence and size.  Here rather than as a default on
    #: ``train.py`` for the reason everything else in this file is here: a
    #: default that lives in a script is a second source of truth, and two
    #: sources of truth drift.
    eval_every: int = 5000
    eval_size: int = 32

    #: Checkpoints spread evenly over the run, rather than an interval that has
    #: to be recomputed every time ``steps`` changes.  The last one lands on the
    #: final step.  0 disables them.
    n_checkpoints: int = 25
    #: Samples drawn from the EMA model at each checkpoint and written as a
    #: square grid, so the run's progress is visible as pictures rather than only
    #: as a loss curve.  0 disables sampling.  64 is an 8x8 grid.
    n_samples: int = 64
    #: Probability-flow ODE steps per sample.  This is the knob that decides
    #: what a checkpoint costs, and it is easy to get badly wrong: Heun takes
    #: two score evaluations per step, each over the whole batch on a canvas 4R
    #: larger than the sample, and a score evaluation is a backward pass because
    #: the score *is* a gradient.  So a checkpoint is ``2 * sample_steps``
    #: batched backward passes.
    #:
    #: Measured on a NERSC GPU: 64 samples at 128 steps is **7 s**, plus a ~10 s
    #: one-off compile.  Ten checkpoints therefore cost about a minute, which is
    #: nothing beside the training they punctuate, so this is not a number worth
    #: economising on.
    #:
    #: It was briefly set to 32 on the strength of a CPU measurement scaled to a
    #: GPU by a guessed factor of 200.  The real ratio for this workload is
    #: ~11000x: a GPU absorbs the batch almost for free where a CPU pays linearly
    #: for it, so scaling across both batch size and hardware at once was never
    #: going to hold.  On a CPU this is hours either way -- use ``n_samples = 0``
    #: there rather than trimming steps.
    sample_steps: int = 128

    def checkpoint_steps(self) -> list[int]:
        """The steps to checkpoint at: ``n_checkpoints`` of them, evenly spread,
        the last landing exactly on ``steps``."""
        if self.n_checkpoints <= 0 or self.steps <= 0:
            return []
        n = min(self.n_checkpoints, self.steps)
        return [round(self.steps * (i + 1) / n) for i in range(n)]


@dataclass
class Config:
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    sde: SDEConfig = field(default_factory=SDEConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    patch: PatchConfig = field(default_factory=PatchConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def real_context(self, out_size: int | None = None) -> float:
        """Pooled px of genuine sky either side of a centred nominal crop.

        The loss discards ``2R`` from every side, so the loader carries that
        much context along; this is how much of it the stamp can supply.  The
        rest is reflected, which is a deliberate trade -- see
        ``data.pooling.pool_to_training_grid``.  Note that a crop is not always
        centred: translation moves real context from one side to the other, it
        does not create more.
        """
        p = self.patch
        out_size = p.out_size if out_size is None else out_size
        return (p.native_size / p.pool_factor - out_size) / 2

    def usable_size_range(self) -> tuple[int, int]:
        """``(smallest, largest)`` training size this config can actually serve.

        The lower bound used to be architectural -- a patch had to exceed ``4R``
        or the loss had no interior.  That is gone: the loader carries ``2R`` of
        context on every side, so the loss lands on the whole nominal crop
        whatever its size.  What remains is the stamp: the nominal crop itself
        must be real pixels, so ``out_size * pool_factor`` cannot exceed
        ``native_size``.  How much of the *context* is real is a separate
        question -- ``real_context``.
        """
        return 1, self.patch.native_size // self.patch.pool_factor

    def check_sizes(self) -> list[str]:
        """Warnings about the configured training sizes; empty means fine."""
        _, hi = self.usable_size_range()
        margin = self.energy.loss_margin
        out = []
        # A size that does not fit the stamp cannot get here: PatchConfig
        # rejects it at construction, and dataclasses.replace re-runs that.
        for s in self.patch.training_sizes:
            if self.real_context(s) <= 0:
                out.append(
                    f"size {s} leaves no real context: the whole {margin} px "
                    f"border on every side would be reflection, so nothing "
                    f"outside the nominal crop is new information. Extract "
                    f"larger stamps, or reduce out_size below {hi}."
                )
            frac = (s / (s + 2 * margin)) ** 2
            if frac < 0.10:
                out.append(
                    f"size {s} is fed {s + 2 * margin} px to train on {s} -- "
                    f"only {100 * frac:.0f}% of the arithmetic reaches the "
                    f"loss. Use a larger out_size, or less reach."
                )
        if self.augment.translate and self.patch.max_translate_native == 0:
            out.append(
                f"translation augmentation has no room: native_size "
                f"({self.patch.native_size}) equals out_size * pool_factor. "
                f"Enlarge native_size or reduce out_size -- translation is exact "
                f"and free, so losing it silently is a waste."
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        sections = {
            "energy": (EnergyConfig, _tuples(d.get("energy", {}), ("channels", "dilations"), 2)),
            "sde": (SDEConfig, d.get("sde", {})),
            "transform": (TransformConfig, d.get("transform", {})),
            "patch": (PatchConfig, _tuples(d.get("patch", {}), ("out_sizes",))),
            "augment": (AugmentConfig, d.get("augment", {})),
            "train": (TrainConfig, d.get("train", {})),
        }
        return cls(**{name: _section(name, kind, raw) for name, (kind, raw) in sections.items()})

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _section(name: str, kind, raw: dict[str, Any]):
    """Build one config section, refusing a key it does not have.

    A config is written by ``prepare_config.py`` and then sits on disk across
    code changes, so it outlives them.  Without this a renamed field surfaces as
    ``TypeError: __init__() got an unexpected keyword argument``, which is true
    and says nothing about what to do.  The answer is the same whatever the key
    was, so that is what it says.
    """
    known = {f.name for f in dataclasses.fields(kind)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            f"config section {name!r} has no field(s) {unknown}; it has "
            f"{sorted(known)}. This config predates the current code; re-run "
            f"scripts/prepare_config.py to write a new one."
        )
    return kind(**raw)


def _tuples(d: dict[str, Any], keys: tuple[str, ...], depth: int = 1) -> dict[str, Any]:
    """JSON round-trips tuples as lists; put them back, ``depth`` levels deep.

    ``EnergyConfig.channels`` is a tuple of tuples -- one per branch -- and it is
    a *static* pytree field, so it has to come back hashable or two models built
    from the same file compare unequal and ``tree_map`` across them fails.
    """
    out = dict(d)
    for k in keys:
        if k in out and out[k] is not None:
            out[k] = tuple(tuple(v) for v in out[k]) if depth == 2 else tuple(out[k])
    return out
