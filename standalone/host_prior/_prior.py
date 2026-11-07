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

from ._net import NCSNpp, NCSNppEncoderEnergy, NCSNppEnergy
from ._support import EnergyModel, NCSNppConfig

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


def _sigmoid(u, xp):
    if xp is not np:
        return jax.nn.sigmoid(u)
    return 0.5 * (np.tanh(0.5 * u) + 1.0)


# -- the flux transform ----------------------------------------------------


@dataclass(frozen=True)
class LogFluxTransform:
    """``x = log(s * softplus(f / s))``, with ``s`` in nJy.

    One scale for every band: ``x`` is log flux in nJy absolutely, so a scene is
    a scene and nothing per-band enters the map.
    """

    softening: float
    #: Hand ``floor - pedestal`` back above the knee so bright flux is exact.
    #: Read from the checkpoint; absent in one written before the term
    #: existed, where it is False and the map is what it always was.
    restore_bright_end: bool = False

    def soften(self, flux, floor=0.0, pedestal=None):
        """``pedestal + s * softplus((f - floor) / s)``: nJy in, nJy out.

        ``floor`` is where the map bends over, ``pedestal`` is what it bends
        over *to* -- the faintest flux the prior can represent -- and ``s`` is
        how sharp the bend is.  The training loader passes a *per stamp*
        floor, a multiple of that cutout's own sky noise, and one pedestal for
        the whole set.  ``pedestal = None`` means "the same as floor".

        **The bright end is not the identity** when they differ: the map tends
        to ``f - (floor - pedestal)``, about 13 nJy low at the trained scales.
        So ``inverse`` (``exp(x)``) is that much under the real flux however
        bright the pixel, and ``inverse_exact`` is the one that recovers a
        measurement.

        Unless ``restore_bright_end`` is set, which adds ``(floor - pedestal)
        * sigmoid((f - floor) / s)`` -- zero at the floor, the whole offset
        above it -- and makes the bright end exact again.  It costs a gain
        above one in a ring around the floor, and the closed-form inverse.
        """
        xp = _xp(flux)
        if pedestal is None:
            pedestal = floor
        u = (flux - floor) / self.softening
        out = pedestal + self.softening * _softplus(u, xp)
        if self.restore_bright_end:
            out = out + (floor - pedestal) * _sigmoid(u, xp)
        return out

    def forward(self, flux, floor=0.0, pedestal=None):
        """nJy -> the model's space.  Finite for every real input, negative sky
        included.

        Written as ``log_softplus(f/s) + log(s)`` rather than by composing
        ``soften`` with a logarithm: below ``f/s = -745`` the softened flux
        underflows to zero in float64 and its log is ``-inf``, where this
        returns the exact limit instead.
        """
        xp = _xp(flux)
        s = self.softening
        if pedestal is None:
            pedestal = floor
        u = (flux - floor) / s
        val = self.soften(flux, floor, pedestal)
        # ``span`` is what ``soften - pedestal`` is proportional to as
        # ``u -> -inf``; ``sigmoid`` decays like ``e^u`` just as ``softplus``
        # does, so the restoration term adds to the coefficient.
        span = s + (floor - pedestal) if self.restore_bright_end else s
        limit = u + xp.log(xp.asarray(span))
        # The pedestal, not the floor, decides this: it is what the output
        # cannot go below, so it is what stops the log underflowing.
        degenerate = (xp.asarray(pedestal) == 0) & (u < _LINEAR_BELOW)
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

    def inverse_exact(self, x, floor=0.0, pedestal=None):
        """The true inverse of ``forward``: ``s * log(expm1(exp(x) / s))``.

        For recovering a measurement, negative sky included.  Agrees with
        ``inverse`` to double precision wherever the flux is more than ~10 s.
        """
        xp = _xp(x)
        s = self.softening
        if pedestal is None:
            pedestal = floor
        if self.restore_bright_end:
            return self._inverse_restored(x, floor, pedestal)
        zero_floor = xp.asarray(pedestal) == 0
        v = xp.where(zero_floor, xp.exp(x - math.log(s)),
                     (xp.exp(xp.where(zero_floor, 0.0, x)) - pedestal) / s)
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

    def _inverse_restored(self, x, floor, pedestal):
        """``inverse_exact`` for the restored map, by Newton on the residual.

        Solves ``s * softplus(t) + lift * sigmoid(t) == exp(x) - pedestal``
        for ``t = (f - floor) / s``; there is no closed form.  Both starting
        points are exact asymptotes of the residual, so four steps already
        reach the forward map's own precision -- 20 is margin, not need.
        """
        xp = _xp(x)
        s = self.softening
        lift = floor - pedestal
        span = s + lift
        target = xp.exp(x) - pedestal
        cx = xp.log(xp.where(target > 0, target, 1e-300) / span)
        small = cx < _LINEAR_BELOW
        start = xp.where(target > span, (target - lift) / s, cx)
        t = xp.where(small, 0.0, start)
        for _ in range(20):
            z = _sigmoid(t, xp)
            phi = s * _softplus(t, xp) + lift * z - target
            dphi = s * z + lift * z * (1.0 - z)
            t = t - phi / xp.where(dphi > 1e-300, dphi, 1e-300)
        return floor + s * xp.where(small, cx, t)

    def jacobian(self, x):
        """``df/dx`` for the model map, which is simply ``f`` itself."""
        return self.inverse(x)

    def sky_level_at(self, floor=0.0, pedestal=None):
        """Where zero flux lands in ``x``: ``log(soften(0))``.

        Not ``log(floor + s * log 2)``, which is this only at ``floor = 0``
        where ``softplus(0) = log 2``.  That expression was the unfloored one
        with the floor added on and was 0.13 nats out at the trained scales.
        """
        xp = _xp(floor)
        return xp.log(self.soften(xp.asarray(floor) * 0.0, floor, pedestal))

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
    #: What the soft floor bends over to, nJy; 0 when the set had no floor.
    #: Carried rather than recomputed, so ``to_x`` and ``to_flux_exact`` use
    #: exactly the number ``load_prior`` read off the config.  Last because it
    #: is the only field with a default and a dataclass will not take one
    #: before the rest; ``load_prior`` passes every field by keyword anyway.
    pedestal: float = 0.0

    # -- what it is --------------------------------------------------------

    @property
    def is_energy(self) -> bool:
        """Whether ``energy`` is available -- i.e. whether the score is an exact
        gradient.  False for ``architecture="ncsnpp"``, which predicts the score
        directly and defines no potential.

        Against ``EnergyModel`` and not a list of classes, which is what the
        training package's ``inference.Prior`` does and what keeps the two in
        step.  It was ``isinstance(self.model, NCSNppEnergy)``, and
        ``NCSNppEncoderEnergy`` is a sibling rather than a subclass of that,
        so an encoder checkpoint produced a correct score and then refused to
        give up its energy -- which is the one thing HMC needs."""
        return isinstance(self.model, EnergyModel)

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
        return self.transform.forward(flux, pedestal=self.pedestal)

    def to_flux(self, x):
        """The model's space -> nJy: ``exp(x)``.  **What a forward model calls.**

        Not the inverse of ``to_x`` -- see ``LogFluxTransform.inverse``.  Use
        ``to_flux_exact`` to recover a measurement.
        """
        return self.transform.inverse(x)

    def to_flux_exact(self, x):
        """The true inverse of ``to_x``, negative sky included."""
        return self.transform.inverse_exact(x, pedestal=self.pedestal)

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
                f"'ncsnpp_energy' or 'ncsnpp_encoder'."
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


def _floor_k(tr_cfg: dict) -> float:
    """The sky floor in units of the pooled sky noise, 0 when there is none.

    ``floor_sigma`` is one number for the whole set.  It used to be a per-stamp
    draw from ``[floor_sigma_lo, floor_sigma_hi]``, and an older checkpoint
    carries that pair instead; the midpoint is what the training package used
    for the input offset then, so reading it that way is exact rather than a
    guess at what the run did.
    """
    if "floor_sigma" in tr_cfg:
        return float(tr_cfg["floor_sigma"] or 0.0)
    lo = float(tr_cfg.get("floor_sigma_lo") or 0.0)
    hi = float(tr_cfg.get("floor_sigma_hi") or 0.0)
    return 0.5 * (lo + hi)


def _pedestal_of(tr_cfg: dict) -> float:
    """What the soft floor bends over to, nJy, or 0 when there is no floor.

    **Absent means an older checkpoint**, written before the pedestal existed,
    whose floor bent over to the floor itself.  Guessing either way would be a
    silent change to what ``exp(x)`` means, so this refuses instead.
    """
    if _floor_k(tr_cfg) <= 0:
        return 0.0
    if "floor_pedestal" not in tr_cfg:
        raise ValueError(
            "this config has a sky floor but no transform.floor_pedestal, so "
            "it was written before the floor and the pedestal were separated. "
            "Its map bent over to the floor itself; add "
            '"floor_pedestal": null to say so explicitly, or the value the '
            "run used. Reading it either way without being told would change "
            "what exp(x) means by about 6 nJy, silently."
        )
    got = tr_cfg["floor_pedestal"]
    if got is None:
        # Explicitly the old behaviour: pedestal == the per-stamp floor.
        return None
    return float(got)


def _input_offset(tr_cfg: dict) -> float:
    """The typical sky level in ``x``: ``log(soften(0))`` at the typical floor.

    Subtracted from the scene before the first layer so the zeros the
    convolutions pad with sit at the sky.  ``c`` is the *typical* sky floor --
    ``floor_sigma`` times the pooled sky noise, which is recovered as
    ``softening / softening_sigma`` rather than stored again.

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
    k = _floor_k(tr_cfg)
    softening_sigma = float(tr_cfg.get("softening_sigma") or 0.0)
    c = 0.0
    if k > 0 and softening_sigma > 0:
        c = k * (s / softening_sigma)
    pedestal = _pedestal_of(tr_cfg)
    if pedestal is None:
        pedestal = c
    # log(soften(0)) = log(pedestal + s * softplus(-c/s)), plus the
    # restoration term when it is on.  It read log(c + s * log 2) until the
    # pedestal landed, which is this only at c = 0 and was 0.13 nats out at
    # the trained scales.
    u = -c / s
    sp = math.log1p(math.exp(-abs(u))) + max(u, 0.0)
    val = pedestal + s * sp
    if tr_cfg.get("restore_bright_end"):
        val = val + (c - pedestal) / (1.0 + math.exp(-u))
    return math.log(val)


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
    if architecture not in ("ncsnpp", "ncsnpp_energy", "ncsnpp_encoder"):
        raise ValueError(
            f"{d} was trained with architecture {architecture!r}, which this "
            f"loader does not carry -- it has the NCSN++ backbone only, as "
            f"'ncsnpp', 'ncsnpp_energy' and 'ncsnpp_encoder'. Load it with "
            f"the full rubin_host_prior package instead."
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
    transform = LogFluxTransform(
        softening=float(tr_cfg["softening"]),
        restore_bright_end=bool(tr_cfg.get("restore_bright_end")),
    )
    sde = VESDE(float(sde_cfg["sigma_min"]), float(sde_cfg["sigma_max"]),
                float(sde_cfg.get("data_mean") or 0.0))
    input_offset = _input_offset(tr_cfg)
    pedestal = _pedestal_of(tr_cfg)
    if pedestal is None:
        ss = float(tr_cfg.get("softening_sigma") or 0.0)
        k = _floor_k(tr_cfg)
        pedestal = k * (transform.softening / ss) if (k > 0 and ss) else 0.0
    common = dict(sigma_min=sde.sigma_min, sigma_max=sde.sigma_max,
                  input_offset=input_offset, key=jax.random.key(0))
    if architecture == "ncsnpp":
        skeleton = NCSNpp(net_cfg, **common)
    else:
        # Both energies need data_std for the 'dae' prefactor, and both
        # restore their sigma-offset bands from the checkpoint like any other
        # array -- they are module fields, not metadata.
        if net_cfg.energy_form == "dae" and sde_cfg.get("data_std") is None:
            raise ValueError(
                f"{d} uses the 'dae' energy, whose prefactor is "
                f"(data_std^2 + sigma^2)/sigma^2, but its config has no "
                f"sde.data_std."
            )
        cls = (NCSNppEncoderEnergy if architecture == "ncsnpp_encoder"
               else NCSNppEnergy)
        skeleton = cls(
            net_cfg, data_std=float(sde_cfg.get("data_std") or 1.0), **common)

    name = {"ema": "ema.eqx", "model": "model.eqx"}[which]
    path = d / name
    if not path.exists():  # e.g. trained with the EMA disabled
        path = d / "model.eqx"
    model = eqx.tree_deserialise_leaves(path, skeleton)
    step = json.loads((d / "state.json").read_text())["step"]
    return Prior(
        model=model, sde=sde, transform=transform, architecture=architecture,
        pedestal=float(pedestal),
        size=int(patch["out_size"]), pool_factor=int(patch["pool_factor"]),
        step=int(step), source=d, config=cfg,
    )
