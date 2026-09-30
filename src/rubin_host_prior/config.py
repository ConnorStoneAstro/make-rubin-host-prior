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

from . import geometry

BANDS = ("u", "g", "r", "i", "z", "y")


#: Ten layers at 32 channels, dilations up a doubling series to 32 and back
#: down, reach ``sum = 78``.  The ascent reaches, the descent de-grids: a layer
#: at dilation ``d`` samples a lattice of spacing ``d``, and the smaller
#: dilations after it mix the lattices back together.
#:
#: **R is no longer bounded by the stamp**, because the convolutions are
#: same-mode.  Under valid convolutions a loss pixel needed ``2R`` of real sky
#: on every side, which capped R at 34 for a 512 px stamp; R = 78 was measured
#: to leave a usable window of 16x16 on a 256 grid, i.e. nothing.  The cap is
#: gone and the cost moved: at 128 px, 52% of the mean receptive field is now
#: zero padding, and past ``2R = 128`` further reach buys only more of it.
#:
#: This is a demonstrator width.  32 channels is 191k parameters, half of the
#: 64-channel run before it and a ninth of the 128-channel one; raise it for a
#: production run, remembering that arithmetic goes as the square of the width.
#:
#: The sequence of failures behind this: a two-branch model (the long branch
#: carried 25-60% of the score, so it was not being ignored), then a single
#: 128-wide stack at R=36, both fed a majority-reflected border and both
#: producing nothing above ~16 px.  Removing the reflection, at R=30 on all-real
#: sky, finally gave faint elongated structure past 16 px -- the first positive
#: signal -- which is what this trades resolution of the border for reach.
DEFAULT_CHANNELS: tuple[int, ...] = (32,) * 10
DEFAULT_DILATIONS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 8, 4, 2, 1)

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

    **The default is one branch**, ``DEFAULT_*``: ten layers at 32 channels
    with a rise-and-fall dilation series, R = 78.  Several branches still work and are
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
    #: ``(1, 2, 4, 8, 16, 32, 8, 4, 2, 1)`` reaches R = 78 in ten layers, the
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
    #: Pixels discarded from every side before the loss.  **Configured, not
    #: derived.**  Under valid convolutions this was forced to ``2R``, because
    #: outside that window a pixel's score was a different linear functional of
    #: the weights and no amount of training could fix it.  Same-mode padding
    #: gives every pixel a score, so the margin became a choice -- and the right
    #: choice is 0: the model is size-locked, training and inference use the
    #: same grid and the same zero padding, so the border is a fixed part of the
    #: operator rather than an artefact. Cropping it would leave those pixels
    #: untrained and their samples undefined.  Raise it only to test whether the
    #: border is hurting the interior.
    loss_margin: int = 0

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
    free.  At the defaults it is 512 - 256 = 256 native px, +/-25.6 arcsec, and
    it now matters more than it used to: same-mode padding lets the network read
    its distance from the border, every stamp is centred on its host, and
    without translation the two together let a model score well by learning
    "bright blob in the middle" instead of anything about galaxies.  Moving the
    host around the frame is the defence.

    The context border is gone.  It existed because valid convolutions needed
    ``2R`` of real sky on every side of the loss region; same-mode convolutions
    give every pixel a score, so the loader feeds exactly ``out_size``.
    """

    native_size: int = 512  # pixels cut from the coadd
    out_size: int = 128  # the training grid, in pooled pixels
    pool_factor: int = 2
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
    def max_translate_native(self) -> int:
        """Native-pixel translation room, fixed at the *reference* size.

        Back on ``PatchConfig``, where it belongs: with no context border there
        is nothing outside the crop to keep on the stamp, so this depends on the
        patch alone and not on the energy's reach.  Smaller training crops leave
        more room, but letting them wander that far would change the data
        distribution with size -- small patches would mostly land on blank sky
        away from the host -- so the offset is capped at the reference size's
        room and every size looks at the same neighbourhood.
        """
        return max(self.native_size - self.nominal_crop, 0)

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
    #: Rungs on the eval's sigma ladder.  Decoupled from ``eval_size``, which it
    #: used to be equal to for the wrong reason: the eval paired example ``i``
    #: with ``sigma[i]``, so one number served as both the number of patches and
    #: the number of noise levels, and the curve it produced was a picture of
    #: which patch got which sigma.  Every patch now sees every rung.
    eval_sigmas: int = 32

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

    @property
    def input_offset(self) -> float:
        """The sky level in ``x``: ``log(softening * log 2)``.

        Subtracted from the scene before the first layer so the zeros the
        convolutions pad with sit at the sky rather than 5-10 sigma below it.
        Computed from the softening the config already stores rather than being
        a field of its own, so there is no second number to drift.  Zero when
        the softening has not been measured yet, which is the honest answer:
        there is no sky level until there is a transform.
        """
        from math import log

        s = self.transform.softening
        return log(s * log(2.0)) if s else 0.0

    def build_model(self, key):
        """The sanctioned way to construct the energy from a config.

        ``ConvEnergyNet`` needs ``input_offset``, which is the transform's sky
        level and therefore belongs to no single sub-config.  Building the model
        by hand with ``ConvEnergyNet(config.energy, key=...)`` silently gets an
        offset of 0, and since a checkpoint's skeleton is rebuilt from the saved
        config, the reloaded model would then compute different scores from the
        same weights.  Going through here is what makes those two agree.
        """
        from .nn.energy import ConvEnergyNet

        return ConvEnergyNet(self.energy, input_offset=self.input_offset, key=key)

    def usable_size_range(self) -> tuple[int, int]:
        """``(smallest, largest)`` training size this config can serve.

        Just the stamp now: the crop is ``out_size * pool_factor`` native pixels
        and it has to come out of ``native_size``.  Under valid convolutions
        this also had to hold ``4R`` of context, which capped a 512 px stamp at
        34 px of reach; same-mode padding removed that, which is the whole
        reason the reach could go to 78.
        """
        p = self.patch
        return 1, p.native_size // p.pool_factor

    def check_sizes(self) -> list[str]:
        """Warnings about the configured training sizes; empty means fine."""
        _, hi = self.usable_size_range()
        margin = self.energy.loss_margin
        R = self.energy.receptive_radius
        out = []
        # A size that does not fit the stamp cannot get here: PatchConfig
        # rejects it at construction, and dataclasses.replace re-runs that.
        for s in self.patch.training_sizes:
            if 2 * R >= s:
                out.append(
                    f"size {s} is smaller than the score's reach 2R = {2 * R}: "
                    f"every pixel already sees every other one, so reach past "
                    f"R = {s // 2} adds no sky, only padding -- "
                    f"{100 * geometry.padding_fraction(s, self.energy.dilations, self.energy.kernel_size):.0f}% "
                    f"of the mean receptive field is zeros. Deliberate if you "
                    f"are over-providing reach; otherwise cut R or train on a "
                    f"larger grid (this stamp serves {hi})."
                )
            if margin and 2 * margin >= s:
                out.append(
                    f"size {s} has no loss region left after cropping {margin} "
                    f"px from every side. Reduce energy.loss_margin."
                )
        if self.augment.translate and self.patch.max_translate_native == 0:
            out.append(
                f"translation augmentation has no room: native_size "
                f"({self.patch.native_size}) equals out_size * pool_factor. "
                f"Enlarge native_size or reduce out_size -- translation is exact "
                f"and free, and with zero-padded convolutions it is also what "
                f"stops the model learning the host's position instead of its "
                f"shape."
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
