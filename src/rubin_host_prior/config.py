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


@dataclass(frozen=True)  # hashable: stored as a static field on the module
class EnergyConfig:
    """Fully convolutional energy network."""

    in_channels: int = 1
    channels: tuple[int, ...] = (32, 64, 96, 128, 128, 128, 128, 128)
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

    @property
    def n_layers(self) -> int:
        return len(self.channels)


@dataclass
class SDEConfig:
    """Variance-exploding SDE, geometric sigma schedule, no preconditioning."""

    sigma_min: float = 0.01
    sigma_max: float = 10.0


@dataclass
class TransformConfig:
    """Flux -> log-space transform.

        x = log(softplus(f / s_band)) / c        f = s_band * exp(c * x)

    The model map is a plain exponential, so the prior's reachable domain in
    flux space is strictly positive -- a source cannot emit negative flux.  The
    data transform is therefore deliberately not its exact inverse: measured
    flux goes negative wherever noise takes it below the subtracted sky, and
    those pixels are smoothly carried towards zero instead.

    ``s_band = softening_sigma * pooled sky noise`` sets where the softening
    turns over.  See ``data.transform`` for the full rationale; in brief, 1.0
    keeps the sky pedestal (``0.693 * s``) below the noise it replaces while
    making ``inverse`` accurate to 0.1% above 5 sigma.
    """

    band_softening: dict[str, float] = field(default_factory=dict)  # nJy, per band
    log_scale: float = 1.0  # "c" above
    softening_sigma: float = 1.0  # s_band = softening_sigma * pooled sky noise


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

    native_size: int = 416  # pixels cut from the visit image
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
    batch_size: int = 32
    steps: int = 200_000
    learning_rate: float = 2e-4
    warmup_steps: int = 2_000
    cosine_decay: bool = False
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    log_every: int = 100
    ckpt_every: int = 5_000
    seed: int = 0


@dataclass
class Config:
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    sde: SDEConfig = field(default_factory=SDEConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    patch: PatchConfig = field(default_factory=PatchConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def usable_size_range(self) -> tuple[int, int]:
        """``(smallest, largest)`` training size this config can actually serve.

        The lower bound comes from the architecture (a patch must exceed ``4R``
        or the loss has no interior); the upper bound from the extracted stamp
        (``native_size // pool_factor``).  Neither dataclass knows both halves,
        which is why this lives on ``Config`` -- and why it is easy to configure a
        size that one half allows and the other does not.
        """
        from . import geometry

        lo = 4 * geometry.receptive_radius(
            self.energy.n_layers, self.energy.kernel_size
        ) + 1
        hi = self.patch.native_size // self.patch.pool_factor
        return lo, hi

    def check_sizes(self) -> list[str]:
        """Warnings about the configured training sizes; empty means fine."""
        from . import geometry

        lo, hi = self.usable_size_range()
        out = []
        if lo > hi:
            out.append(
                f"no usable patch size: the model needs > {lo - 1} px but the "
                f"{self.patch.native_size} px stamps only yield {hi} px. "
                f"Extract larger stamps or use fewer layers."
            )
        for s in self.patch.training_sizes:
            if s < lo:
                out.append(f"size {s} is below the minimum {lo} (model crops "
                           f"{2 * geometry.receptive_radius(self.energy.n_layers, self.energy.kernel_size)} px per side)")
            elif s > hi:
                out.append(f"size {s} needs {s * self.patch.pool_factor} native "
                           f"px but the stamps are {self.patch.native_size}")
            else:
                frac = (geometry.interior_size(s, self.energy.n_layers,
                                               self.energy.kernel_size) / s) ** 2
                if frac < 0.10:
                    out.append(f"size {s} spends {100 * (1 - frac):.0f}% of itself "
                               f"on the cropped margin -- little signal per step")
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
        return cls(
            energy=EnergyConfig(**_tuples(d.get("energy", {}), ("channels",))),
            sde=SDEConfig(**d.get("sde", {})),
            transform=TransformConfig(**d.get("transform", {})),
            patch=PatchConfig(**_tuples(d.get("patch", {}), ("out_sizes",))),
            augment=AugmentConfig(**d.get("augment", {})),
            train=TrainConfig(**d.get("train", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _tuples(d: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """JSON round-trips tuples as lists; put them back."""
    out = dict(d)
    for k in keys:
        if k in out and out[k] is not None:
            out[k] = tuple(out[k])
    return out
