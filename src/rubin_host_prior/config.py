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
    """Flux -> log-space pixel transform.

        x = log1p(f / b_band) / c          f = b_band * expm1(c * x)

    ``b_band`` is a per-band soft offset in nJy (nominally ``k_sigma`` times the
    pooled sky noise of that band).  Dividing by it before the log puts every
    band's sky level at x ~ 0 with scatter ~ 1 / k_sigma, which is what makes a
    single band-agnostic prior reasonable.  The transform is linear in the
    noise-dominated regime and logarithmic in the bright regime, so the huge
    dynamic range of galaxy cores is compressed without a hard floor and
    without a point mass.
    """

    band_offsets: dict[str, float] = field(default_factory=dict)  # nJy, per band
    log_scale: float = 1.0  # "c" above
    k_sigma: float = 5.0  # b_band = k_sigma * pooled sky noise (provenance only)
    floor_ratio: float = -0.9  # clip f / b_band at this; -0.9 -> x_min = log(0.1)/c


@dataclass
class PatchConfig:
    """Native cutout geometry and the pooling that produces a training image.

    ``native_size`` is deliberately larger than ``nominal_crop`` so that scale
    jitter can go in both directions and so that integer translations in native
    pixels (which are exact -- no interpolation) give sub-pooled-pixel jitter.
    """

    native_size: int = 224  # pixels cut from the visit image
    nominal_crop: int = 192  # native pixels feeding one training image
    out_size: int = 64  # nominal_crop / pool_factor
    pool_factor: int = 3

    def __post_init__(self) -> None:
        if self.nominal_crop != self.out_size * self.pool_factor:
            raise ValueError(
                f"nominal_crop ({self.nominal_crop}) must equal out_size * "
                f"pool_factor ({self.out_size} * {self.pool_factor})"
            )
        if self.native_size < self.nominal_crop:
            raise ValueError("native_size must be >= nominal_crop")


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
            patch=PatchConfig(**d.get("patch", {})),
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
