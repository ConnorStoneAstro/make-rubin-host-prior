"""The flux transform, the noise schedule, the sampler, and ``Prior`` itself.

Small and stable, so these are transcribed rather than generated (unlike
``_net.py``, which is the architecture and is copied byte for byte by
``scripts/export_standalone.py`` in the training repo).
``tests/test_standalone.py`` there loads a checkpoint through both paths and
compares the numbers, so a divergence here shows up as a failing test rather
than as a wrong prior.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ._net import NCSNpp, NCSNppEnergy
from ._support import NCSNppConfig

_LINEAR_BELOW = -20.0
_EXPM1_ABOVE = 30.0


def _xp(a):
    """numpy or jax.numpy, whichever matches the input, so one implementation
    serves a numpy pipeline and a jax forward model."""
    return jnp if type(a).__module__.startswith("jax") else np


def _softplus(u, xp):
    if xp is not np:
        return jax.nn.softplus(u)
    return np.logaddexp(0.0, u)


# -- the flux transform ----------------------------------------------------


@dataclass(frozen=True)
class LogFluxTransform:
    """``x = log(s * softplus(f / s))``, with ``s`` in nJy.

    One scale for every band: ``x`` is log flux in nJy absolutely, so a scene is
    a scene and nothing per-band enters the map.
    """

    softening: float

    def soften(self, flux, floor=0.0):
        """``floor + s * softplus((f - floor) / s)``: nJy in, nJy out.

        ``floor`` is where the map bends over -- the flux the sky is carried
        towards and never goes below -- and ``s`` is how sharp the bend is.
        The training loader passes a *per stamp* floor, a multiple of that
        cutout's own sky noise; ``floor = 0`` is the original one-parameter
        softening exactly.
        """
        xp = _xp(flux)
        return floor + self.softening * _softplus((flux - floor) / self.softening, xp)

    def forward(self, flux, floor=0.0):
        """nJy -> the model's space.  Finite for every real input, negative sky
        included.

        Written as ``log_softplus(f/s) + log(s)`` rather than by composing
        ``soften`` with a logarithm: below ``f/s = -745`` the softened flux
        underflows to zero in float64 and its log is ``-inf``, where this
        returns the exact limit instead.
        """
        xp = _xp(flux)
        s = self.softening
        u = (flux - floor) / s
        val = self.soften(flux, floor)
        limit = u + math.log(s)
        degenerate = (xp.asarray(floor) == 0) & (u < _LINEAR_BELOW)
        direct = (val > 0) & ~degenerate
        return xp.where(direct, xp.log(xp.where(direct, val, 1.0)), limit)

    def inverse(self, x):
        """The model's space -> nJy: ``exp(x)``.  **What a forward model calls.**

        Not the inverse of ``forward``: ``inverse(forward(f))`` is ``soften(f)``
        exactly, at every flux.  That gap is the whole discrepancy between the
        data and what the prior can express, and it is why a scene drawn from
        the prior is strictly positive.  Use ``inverse_exact`` to undo
        ``forward``.
        """
        return _xp(x).exp(x)

    def inverse_exact(self, x, floor=0.0):
        """The true inverse of ``forward``: ``s * log(expm1(exp(x) / s))``.

        For recovering a measurement, negative sky included.  Agrees with
        ``inverse`` to double precision wherever the flux is more than ~10 s.
        """
        xp = _xp(x)
        s = self.softening
        zero_floor = xp.asarray(floor) == 0
        v = xp.where(zero_floor, xp.exp(x - math.log(s)),
                     (xp.exp(xp.where(zero_floor, 0.0, x)) - floor) / s)
        cx = xp.where(zero_floor, x - math.log(s),
                      xp.log(xp.where(v > 0, v, 1e-300)))
        # Three regimes.  Large v: expm1 would overflow and log(expm1(v)) == v.
        # Very negative cx: v underflows, expm1(v) ~ v, so log(expm1(v)) ~ cx.
        # In between the direct expression is fine.
        big = v > _EXPM1_ABOVE
        small = cx < _LINEAR_BELOW
        mid = xp.where(big | small, 1.0, v)
        direct = xp.log(xp.expm1(mid))
        return floor + s * xp.where(big, v, xp.where(small, cx, direct))

    def jacobian(self, x):
        """``df/dx`` for the model map, which is simply ``f`` itself."""
        return self.inverse(x)

    def sky_level_at(self, floor=0.0):
        """Where zero flux lands in ``x``: ``log(floor + s * log 2)``."""
        return _xp(floor).log(floor + self.softening * math.log(2.0))

    @property
    def sky_level(self) -> float:
        """``sky_level_at(0)``: where the sky sits with no floor."""
        return math.log(self.softening * math.log(2.0))


# -- the noise schedule ----------------------------------------------------


@dataclass(frozen=True)
class VESDE:
    """Variance exploding, geometric in sigma: ``x_sigma = x + sigma * eps``."""

    sigma_min: float
    sigma_max: float
    data_mean: float = 0.0

    def sigma(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def t_of_sigma(self, sigma):
        return jnp.log(sigma / self.sigma_min) / math.log(
            self.sigma_max / self.sigma_min)

    def prior_sample(self, key, shape):
        """The ``t = 1`` marginal, ``N(data_mean, sigma_max^2)``.  VE only adds
        noise, so the marginal keeps the data's mean."""
        return self.data_mean + self.sigma_max * jax.random.normal(key, shape)


@eqx.filter_jit
def pflow_sample(model, key, shape, sde, n_steps: int = 256, heun: bool = True):
    """Deterministic probability-flow ODE, ``dx/dsigma = -sigma * score``.

    Heun's method by default: two score evaluations per step, and worth it --
    Euler needs several times more steps for the same accuracy.
    """
    sigmas = sde.sigma(jnp.linspace(1.0, 0.0, n_steps + 1))
    x = sde.prior_sample(key, shape)
    batch = shape[0]

    def step(x, i):
        s_cur, s_next = sigmas[i], sigmas[i + 1]
        d_cur = -s_cur * _batched_score(model, x, jnp.full((batch,), s_cur))
        x_next = x + (s_next - s_cur) * d_cur
        if heun:
            d_next = -s_next * _batched_score(
                model, x_next, jnp.full((batch,), s_next))
            x_next = x + (s_next - s_cur) * 0.5 * (d_cur + d_next)
        return x_next, None

    x, _ = jax.lax.scan(step, x, jnp.arange(n_steps))
    return x


def _batched_score(model, x, sigma):
    return jax.vmap(lambda a, s: model.score(a, s))(x, sigma)


@eqx.filter_jit
def _score(model, x, sigma):
    return _batched_score(model, x, sigma)


@eqx.filter_jit
def _energy(model, x, sigma, pixel_offset):
    return jax.vmap(lambda a, s: model(a, s, pixel_offset))(x, sigma)


# -- the prior -------------------------------------------------------------


@dataclass(frozen=True)
class Prior:
    """A trained prior, with the schedule and the flux transform it belongs to.

    **Size-locked.**  The convolutions pad with zeros, so the grid the model was
    trained on is part of the operator rather than a window onto it.  ``size``
    is that grid; a different one is a different prior, and nothing here will
    stop you -- the result simply means nothing.
    """

    model: object
    sde: VESDE
    transform: LogFluxTransform
    architecture: str
    size: int
    pool_factor: int
    step: int
    source: Path
    config: dict

    # -- what it is --------------------------------------------------------

    @property
    def is_energy(self) -> bool:
        """Whether ``energy`` is available -- i.e. whether the score is an exact
        gradient.  False for ``architecture="ncsnpp"``, which predicts the score
        directly and defines no potential."""
        return isinstance(self.model, NCSNppEnergy)

    @property
    def sigma_min(self) -> float:
        return self.sde.sigma_min

    @property
    def sigma_max(self) -> float:
        return self.sde.sigma_max

    @property
    def in_channels(self) -> int:
        return self.model.in_channels

    # -- the flux mapping --------------------------------------------------

    def to_x(self, flux):
        """nJy -> the model's space.  Takes no band."""
        return self.transform.forward(flux)

    def to_flux(self, x):
        """The model's space -> nJy: ``exp(x)``.  **What a forward model calls.**

        Not the inverse of ``to_x`` -- see ``LogFluxTransform.inverse``.  Use
        ``to_flux_exact`` to recover a measurement.
        """
        return self.transform.inverse(x)

    def to_flux_exact(self, x):
        """The true inverse of ``to_x``, negative sky included."""
        return self.transform.inverse_exact(x)

    # -- what a sampler or an HMC needs ------------------------------------

    def score(self, x, sigma: float | None = None):
        """``grad_x log p_sigma(x)``, same shape in as out.

        Accepts ``(H, W)``, ``(C, H, W)`` or ``(B, C, H, W)``.  ``sigma``
        defaults to ``sigma_min``, the sharpest distribution the model was
        trained on; outside ``[sigma_min, sigma_max]`` it extrapolates silently.
        """
        batched, shape = self._as_batch(x)
        return _score(self.model, batched,
                      self._sigmas(sigma, len(batched))).reshape(shape)

    def energy(self, x, sigma: float | None = None, pixel_offset: float = 0.0):
        """``E(x, sigma)``, a scalar per scene, with ``score == -grad_x E``.

        **What HMC needs that a score model cannot give**: the accept/reject
        step is on the Hamiltonian, so it wants the potential and not only its
        gradient.  Up to an additive constant, which is pure gauge, ``-energy``
        is the unnormalised log density.

        ``pixel_offset`` is subtracted from each pixel's contribution before
        the sum (a function of sigma alone leaves every ratio unchanged); see
        ``_net.scalar_energy`` for why a sampler wants that in float32.
        """
        if not self.is_energy:
            raise TypeError(
                f"{self.source} was trained with architecture "
                f"{self.architecture!r}, which predicts the score directly and "
                f"is not the gradient of anything -- there is no energy to "
                f"return. It needs a checkpoint trained with architecture "
                f"'ncsnpp_energy'."
            )
        batched, shape = self._as_batch(x)
        e = _energy(self.model, batched, self._sigmas(sigma, len(batched)),
                    jnp.asarray(pixel_offset))
        return e[0] if len(shape) < 4 else e

    def log_prob(self, x, sigma: float | None = None, pixel_offset: float = 0.0):
        """``-energy``: the unnormalised log density, up to a constant.

        Differences between scenes at the same sigma are meaningful; the
        absolute value is not -- the partition function is not available.
        ``pixel_offset`` (a per-pixel energy level, a function of sigma alone)
        is subtracted inside the sum, see ``energy``.
        """
        return -self.energy(x, sigma, pixel_offset)

    def sample(self, key, n: int = 1, n_steps: int = 256, heun: bool = True):
        """Draw ``n`` scenes on the prior's own grid, in the model's space."""
        shape = (n, self.in_channels, self.size, self.size)
        return pflow_sample(self.model, key, shape, self.sde, n_steps=n_steps,
                            heun=heun)

    # -- internals ---------------------------------------------------------

    def _as_batch(self, x):
        a = jnp.asarray(x)
        if a.ndim == 2:
            return a[None, None], a.shape
        if a.ndim == 3:
            return a[None], a.shape
        if a.ndim == 4:
            return a, a.shape
        raise ValueError(
            f"expected (H, W), (C, H, W) or (B, C, H, W), got shape {a.shape}")

    def _sigmas(self, sigma, n):
        return jnp.full((n,), self.sde.sigma_min if sigma is None else sigma)

    def summary(self) -> str:
        n = sum(leaf.size for leaf in jax.tree_util.tree_leaves(
            eqx.filter(self.model, eqx.is_inexact_array)))
        return "\n".join([
            f"{self.architecture} prior from {self.source} at step {self.step:,}",
            f"  {n:,} parameters on a {self.size}x{self.size} grid "
            f"({self.size * self.pool_factor} native px, pool {self.pool_factor})",
            f"  sigma in [{self.sigma_min:.4g}, {self.sigma_max:.4g}], "
            f"softening {self.transform.softening:.4g} nJy, sky at x = "
            f"{self.transform.sky_level:.3f}",
            f"  energy available: {self.is_energy}",
        ])


def _input_offset(tr_cfg: dict) -> float:
    """The typical sky level in ``x``: ``log(c + s * log 2)``.

    Subtracted from the scene before the first layer so the zeros the
    convolutions pad with sit at the sky.  ``c`` is the *typical* sky floor --
    the midpoint of the per-stamp draw times the pooled sky noise, which is
    recovered as ``softening / softening_sigma`` rather than stored again.

    **This must match ``Config.input_offset`` in the training package exactly.**
    It is a static field on the module, so it is not restored by
    deserialisation -- the skeleton's value is the one that gets used, and a
    mismatch computes different scores from the same weights with nothing to
    say so.  It silently did, once: the floor raised the sky from 1.24 to 3.10
    and this copy was still returning the unfloored number.
    ``tests/test_standalone.py`` compares the two loaders' scores for exactly
    that reason.
    """
    s = float(tr_cfg["softening"])
    if not s:
        return 0.0
    lo = float(tr_cfg.get("floor_sigma_lo") or 0.0)
    hi = float(tr_cfg.get("floor_sigma_hi") or 0.0)
    softening_sigma = float(tr_cfg.get("softening_sigma") or 0.0)
    c = 0.0
    if hi > 0 and softening_sigma > 0:
        c = 0.5 * (lo + hi) * (s / softening_sigma)
    return math.log(c + s * math.log(2.0))


def load_prior(directory, which: str = "ema") -> Prior:
    """Rebuild a trained prior from a checkpoint directory.

    ``which`` is ``"ema"`` (the default, and what inference should use) or
    ``"model"``, the live weights.  Point it at ``final/`` or at any
    ``checkpoints/step-*/``.

    The checkpoint is self-describing: ``config.json`` travels with the weights
    and carries the architecture, the sigma schedule, the softening scale and
    the pooling, so none of those is supplied here.  That is deliberate -- a
    prior whose transform had to be restated at the call site is one that can be
    used wrongly without anything saying so.
    """
    d = Path(directory)
    cfg = json.loads((d / "config.json").read_text())
    architecture = cfg.get("architecture", "energy")
    if architecture not in ("ncsnpp", "ncsnpp_energy"):
        raise ValueError(
            f"{d} was trained with architecture {architecture!r}, which this "
            f"loader does not carry -- it has the NCSN++ U-Net only, in the "
            f"two forms 'ncsnpp' and 'ncsnpp_energy'. Load it with the full "
            f"rubin_host_prior package instead."
        )

    sde_cfg, tr_cfg, patch = cfg["sde"], cfg["transform"], cfg["patch"]
    for name, section in (("sigma_min", sde_cfg), ("sigma_max", sde_cfg),
                          ("softening", tr_cfg)):
        if section.get(name) is None:
            raise ValueError(
                f"{d / 'config.json'} has no {name}; it was never measured, so "
                f"this checkpoint cannot describe its own schedule or transform."
            )

    net_cfg = NCSNppConfig.from_dict(cfg.get("ncsnpp", {}))
    net_cfg.check()
    transform = LogFluxTransform(softening=float(tr_cfg["softening"]))
    sde = VESDE(float(sde_cfg["sigma_min"]), float(sde_cfg["sigma_max"]),
                float(sde_cfg.get("data_mean") or 0.0))
    input_offset = _input_offset(tr_cfg)
    common = dict(sigma_min=sde.sigma_min, sigma_max=sde.sigma_max,
                  input_offset=input_offset, key=jax.random.key(0))
    if architecture == "ncsnpp":
        skeleton = NCSNpp(net_cfg, **common)
    else:
        if net_cfg.energy_form == "dae" and sde_cfg.get("data_std") is None:
            raise ValueError(
                f"{d} uses the 'dae' energy, whose prefactor is "
                f"(data_std^2 + sigma^2)/sigma^2, but its config has no "
                f"sde.data_std."
            )
        skeleton = NCSNppEnergy(
            net_cfg, data_std=float(sde_cfg.get("data_std") or 1.0), **common)

    name = {"ema": "ema.eqx", "model": "model.eqx"}[which]
    path = d / name
    if not path.exists():  # e.g. trained with the EMA disabled
        path = d / "model.eqx"
    model = eqx.tree_deserialise_leaves(path, skeleton)
    step = json.loads((d / "state.json").read_text())["step"]
    return Prior(
        model=model, sde=sde, transform=transform, architecture=architecture,
        size=int(patch["out_size"]), pool_factor=int(patch["pool_factor"]),
        step=int(step), source=d, config=cfg,
    )
