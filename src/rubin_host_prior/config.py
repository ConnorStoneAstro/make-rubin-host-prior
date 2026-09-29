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


#: Eight layers at 64 channels, dilations up a doubling series and back down
#: again, reach ``sum = 30``.  The descending tail is the standard de-gridding
#: construction: a layer at dilation ``d`` samples a lattice of spacing ``d``,
#: and the smaller dilations after it mix the lattices back together.
#:
#: **R is chosen by the stamp, not by the architecture.**  A loss pixel's score
#: depends on ``2R`` around it, so training it on real sky needs
#: ``out_size + 4R`` pooled pixels of real sky, and a 512 native stamp at
#: ``pool_factor`` 3 has 170.  At ``out_size`` 32 that caps R at 34.  Reach
#: beyond that is not free and not neutral: it is bought with reflected sky, so
#: the large-scale part of the score gets trained on a mirror symmetry that
#: nature does not have -- which is the leading explanation for samples that
#: contain only small structure.  ``Config`` raises rather than pad, so a
#: config asking for more reach than the stamp can feed says so.
#:
#: Two earlier architectures produced no large-scale structure: a wide undilated
#: stack summed with a narrow dilated one (the long-range branch carried 25-60%
#: of the score, so it was not being ignored), and a single 128-wide stack at
#: R=36.  Both were fed a majority-reflected border, which neither diagnostic
#: was looking at.
DEFAULT_CHANNELS: tuple[int, ...] = (64,) * 8
DEFAULT_DILATIONS: tuple[int, ...] = (1, 2, 4, 8, 8, 4, 2, 1)

#: Said whenever `channels` and `dilations` disagree, because by far the most
#: likely reason is that only one of them was given.
_NOT_INFERRED = (
    "If you set channels and left dilations alone, state them too -- they are "
    "never inferred, because a branch's reach is r * sum(dilations) and the "
    "loss crop, the patch size and the loader's context all follow from it."
)


@dataclass(frozen=True)  # hashable: stored as a static field on the module
class EnergyConfig:
    """Fully convolutional energy, summed over one or more parallel branches.

    ``channels`` and ``dilations`` hold **one tuple per branch**.  Each branch is
    an independent stack of valid convolutions ending in a 1x1 head, and their
    energy maps are centre-cropped to a common size and added.  A sum of energies
    is an energy, so the score stays an exact gradient however many there are.

    **The default is one branch**, ``DEFAULT_*``: eight layers at 64 channels
    with a rise-and-fall dilation series, R = 30.  Several branches still work and are
    tested -- the sum of any number of energies is an energy -- but the default
    is one, because a sum lets the branches compete to explain the same residual
    and nothing in the objective decides which should win.

    ``prepare_config.py`` starts from these defaults, so whatever they say is
    what a freshly written config describes.  That is why the architecture lives
    here rather than in the script.

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
    channels: tuple[tuple[int, ...], ...] = (DEFAULT_CHANNELS,)
    #: Per-layer dilation, same shape as ``channels``, and always stated: a
    #: branch's reach is ``r * sum(dilations)``, so leaving it to be inferred
    #: would mean the single most consequential number in the geometry -- the
    #: loss crop follows from it -- was one nobody wrote down.
    #: ``(1, 2, 4, 8, 8, 4, 2, 1)`` reaches R = 30 in eight layers, the
    #: descending tail mixing the coarse lattices back together.
    dilations: tuple[tuple[int, ...], ...] = (DEFAULT_DILATIONS,)
    kernel_size: int = 3
    activation: str = "silu"  # must be C^1; see nn.layers.ACTIVATIONS
    embed_dim: int = 128  # width of the log-sigma embedding MLP
    n_fourier: int = 32  # random Fourier features of log(sigma)
    fourier_scale: float = 1.0  # std of the random frequencies
    fourier_seed: int = 0  # fixes the frozen basis; see nn.layers
    head_init_scale: float = 0.01  # small, not zero -- see nn.energy
    film_init_scale: float = 0.01  # small, not zero -- see nn.layers.FiLM
    #: Centre-cropped skip around every layer after the first.  Requires uniform
    #: widths within a branch, and says so rather than quietly dropping the
    #: skips it cannot make.  The first layer is the one structural exception:
    #: it changes the channel count from ``in_channels``, so there is nothing to
    #: add to its output.
    residual: bool = True
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
                f"each per branch. {_NOT_INFERRED}"
            )
        for i, (c, d) in enumerate(zip(self.channels, self.dilations)):
            if not c:
                raise ValueError(f"branch {i} has no layers")
            if len(c) != len(d):
                raise ValueError(
                    f"branch {i} has {len(c)} layers but {len(d)} dilations; "
                    f"every layer needs one. {_NOT_INFERRED}"
                )
            if any(x < 1 for x in d):
                raise ValueError(f"branch {i} has a dilation below 1: {tuple(d)}")
            if self.residual and len(set(c)) > 1:
                raise ValueError(
                    f"residual=True needs uniform widths within a branch so "
                    f"there is something to add the skip to, but branch {i} is "
                    f"{tuple(c)}. Use one width, or set residual=False -- the "
                    f"skips are not quietly dropped where they do not fit."
                )

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

    Three numbers, not four: ``nominal_crop`` is ``out_size * pool_factor`` and
    is a property rather than a field.  It used to be stored and checked against
    that product, which made it a second statement of something the other two
    already fix, and one that had to be edited in step with them by hand.

    ``native_size`` is deliberately larger than the crop.  The slack is what the
    loader translates within -- integer native-pixel shifts are exact, no
    interpolation, so they give sub-pooled-pixel positional augmentation for
    free -- and it is also the only real sky available for the context border
    the loss crop needs (see ``Config.real_context``).  At the defaults it is
    512 - 384 = 128 native px, 42.7 pooled, split between the two sides.
    """

    native_size: int = 512  # pixels cut from the coadd
    out_size: int = 32  # the reference training size, in pooled pixels
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
        if self.native_size < self.nominal_crop:
            raise ValueError(
                f"native_size ({self.native_size}) must be >= nominal_crop "
                f"({self.out_size} * {self.pool_factor} = {self.nominal_crop})"
            )
        for s in self.out_sizes:
            if s * self.pool_factor > self.native_size:
                raise ValueError(
                    f"out_size {s} needs {s * self.pool_factor} native pixels "
                    f"but native_size is {self.native_size}"
                )

    @property
    def nominal_crop(self) -> int:
        """Native pixels feeding one training image: ``out_size * pool_factor``."""
        return self.out_size * self.pool_factor

    @property
    def training_sizes(self) -> tuple[int, ...]:
        """Every size the loader will emit, ``out_size`` first."""
        return (self.out_size,) + tuple(
            s for s in sorted(set(self.out_sizes)) if s != self.out_size
        )


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
    steps: int = 200_000
    learning_rate: float = 1e-4
    warmup_steps: int = 2_000
    cosine_decay: bool = False
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema_decay: float = 0.998
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
    n_checkpoints: int = 50
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

    def fed_size(self, out_size: int | None = None) -> int:
        """What the loader hands the model: ``out_size + 2 * loss_margin``.

        The loss crops ``2R`` from every side, so this is the size that leaves
        exactly ``out_size`` behind.  One place rather than the four it was
        spelled out in -- the trainer's report, its sampling canvas, the config
        summary and ``check_sizes`` -- and the arithmetic itself still lives in
        ``geometry``, which is the module that exists for it.
        """
        from . import geometry

        out_size = self.patch.out_size if out_size is None else out_size
        return geometry.min_input_for_region(
            out_size, self.energy.dilations, self.energy.kernel_size
        )

    def fed_native(self, out_size: int | None = None) -> int:
        """Native pixels the loader must read to serve one training image.

        ``fed_size * pool_factor``: the loss region plus the ``2R`` of context
        on every side that the loss will crop away again.  **Every one of them
        has to be real sky** -- see ``data.pooling.pool_to_training_grid``, which
        raises rather than invent the shortfall.
        """
        return self.fed_size(out_size) * self.patch.pool_factor

    def max_translate_native(self) -> int:
        """Native px the crop may wander, keeping the fed window on real sky.

        This cannot live on ``PatchConfig``: the window includes ``2R`` of
        context per side and ``R`` is the energy's.  Taken at the *largest*
        training size, which serves both purposes at once -- it is the size with
        the least room, so no size can walk its context off the stamp, and every
        size then looks at the same neighbourhood rather than small crops roaming
        out onto blank sky.
        """
        largest = max(self.patch.training_sizes)
        return max((self.patch.native_size - self.fed_native(largest)) // 2, 0)

    def sample_size(self) -> int:
        """Pooled px of scene a checkpoint sample grid should show: ``4R``.

        **Not ``out_size``.**  That is the loss region, and the stamp caps it --
        at the defaults it is 32 px while the score reaches ``2R = 60``, so a
        sample the size of the loss region could not display the largest
        structure the model is even able to represent, and "no large-scale
        structure in the samples" would be a statement about the figure.  At
        inference the canvas costs nothing but compute, since nothing has to
        come off the sky, so it is set by the architecture instead: twice the
        reach, which shows a feature at the reach with room either side of it.
        """
        return max(self.patch.out_size, 2 * self.energy.loss_margin)

    def real_context(self, out_size: int | None = None) -> float:
        """Pooled px of genuine sky either side of a centred nominal crop.

        The loss discards ``2R`` from every side, so the loader carries that
        much context along, and all of it must be real: anything less is an
        error, not a reflected border.  So this is ``>= loss_margin`` for any
        config that can serve ``out_size`` at all, and the excess is the room
        translation wanders in.  Note that a crop is not always centred:
        translation moves real context from one side to the other, it does not
        create more.
        """
        p = self.patch
        out_size = p.out_size if out_size is None else out_size
        return (p.native_size / p.pool_factor - out_size) / 2

    def usable_size_range(self) -> tuple[int, int]:
        """``(smallest, largest)`` training size this config can actually serve.

        The loader carries ``2R`` of context on every side and none of it may be
        invented, so the whole fed window -- ``out_size + 4R`` pooled pixels --
        has to come out of the stamp.  That is the binding constraint, and it is
        much tighter than the nominal crop alone: at R = 30 a 512 px stamp at
        pool 3 serves 170 pooled px, of which 120 are context.
        """
        p = self.patch
        return 1, max(p.native_size // p.pool_factor - 4 * self.energy.receptive_radius, 0)

    def check_sizes(self) -> list[str]:
        """Warnings about the configured training sizes; empty means fine."""
        _, hi = self.usable_size_range()
        margin = self.energy.loss_margin
        out = []
        # PatchConfig rejects a *nominal crop* bigger than the stamp at
        # construction; what it cannot see is the context, which is four times
        # R again and is where the room actually goes.
        for s in self.patch.training_sizes:
            if self.fed_native(s) > self.patch.native_size:
                out.append(
                    f"size {s} needs {self.fed_native(s)} native px -- {s} for "
                    f"the loss and {margin} pooled px of context on every side "
                    f"-- but the stamp is {self.patch.native_size}. The loader "
                    f"will refuse it rather than reflect the shortfall. Reduce "
                    f"out_size to {hi} or below, cut R (now "
                    f"{self.energy.receptive_radius}), or extract larger stamps."
                )
            frac = (s / self.fed_size(s)) ** 2
            if frac < 0.10:
                out.append(
                    f"size {s} is fed {self.fed_size(s)} px to train on {s} -- "
                    f"only {100 * frac:.0f}% of the arithmetic reaches the "
                    f"loss. This is the price of an all-real context and it "
                    f"cannot be tuned away here: the stamp caps out_size at "
                    f"{hi}, so this config's ceiling is "
                    f"{100 * (hi / self.fed_size(hi)) ** 2:.0f}%. To do better, "
                    f"extract larger stamps or cut R."
                )
        if self.augment.translate and self.max_translate_native() == 0:
            out.append(
                f"translation augmentation has no room: the fed window "
                f"({self.fed_native()} native px) fills the stamp "
                f"({self.patch.native_size}). Enlarge native_size, or reduce "
                f"out_size or R -- translation is exact and free, so losing it "
                f"silently is a waste."
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
