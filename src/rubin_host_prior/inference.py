"""Load a trained prior and use it, without the rest of the package.

    from rubin_host_prior.inference import load_prior

    prior = load_prior("runs/ncsnpp-energy/final")
    x = prior.to_x(flux_njy)                   # nJy -> the model's space
    u = prior.energy(x)                        # scalar, for HMC's Hamiltonian
    g = prior.score(x)                         # -grad_x E, for its leapfrog

**You do need the package installed, and there is no way around that**: an
equinox checkpoint stores *leaves*, so rebuilding a model means rebuilding the
skeleton, which means the architecture code.  What you do not need is the
extras.  This path reaches the four core dependencies and nothing else --
``jax``, ``equinox``, ``numpy``, ``h5py`` on import, and ``optax`` once
``load_prior`` runs -- so no matplotlib, no astropy, no pyvo, and nothing from
the LSST stack, which stays behind the ``extract`` extra.  (``h5py`` and
``optax`` arrive through the ``data`` and ``training`` package ``__init__``s
rather than through anything here; both are plain wheels.)

A checkpoint is self-describing.  ``config.json`` travels next to the weights
and carries the architecture, the sigma schedule, the softening scale and the
pooling -- so a ``Prior`` knows its own grid, its own noise range and its own
flux mapping, and none of those has to be supplied again at the call site.
That is deliberate: a prior whose transform had to be restated somewhere else
is a prior that can be used wrongly.

Everything here is a thin wrapper over ``training.checkpoint``,
``diffusion`` and ``data.transform``.  Nothing is computed differently; what
this adds is one object that holds the four things that have to agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from .config import Config
from .data.transform import LogFluxTransform
from .diffusion.sampler import pflow_sample, reverse_sde_sample
from .diffusion.sde import VESDE
from .nn.energy import EnergyModel
from .nn.score import ScoreModel, batched_score, n_parameters


@eqx.filter_jit
def _score(model, x, sigma):
    return batched_score(model, x, sigma)


@eqx.filter_jit
def _energy(model, x, sigma):
    return jax.vmap(lambda a, s: model(a, s))(x, sigma)


@dataclass(frozen=True)
class Prior:
    """A trained prior, with the schedule and the flux transform it belongs to.

    **Size-locked.**  Both architectures pad with zeros, so the grid the model
    was trained on is part of the operator rather than a window onto it.
    ``size`` is that grid; a different one is a different prior, and the methods
    here will not stop you but the result means nothing.
    """

    model: ScoreModel
    config: Config
    sde: VESDE
    transform: LogFluxTransform
    step: int
    source: Path

    # -- what it is --------------------------------------------------------

    @property
    def size(self) -> int:
        """The pooled grid the prior is defined on."""
        return self.config.patch.out_size

    @property
    def pool_factor(self) -> int:
        """Native coadd pixels per model pixel."""
        return self.config.patch.pool_factor

    @property
    def architecture(self) -> str:
        return self.config.architecture

    @property
    def is_energy(self) -> bool:
        """Whether ``energy`` is available -- i.e. whether the score is an exact
        gradient.  False for ``architecture="ncsnpp"``, which predicts the score
        directly and defines no potential."""
        return isinstance(self.model, EnergyModel)

    @property
    def sigma_min(self) -> float:
        return self.sde.sigma_min

    @property
    def sigma_max(self) -> float:
        return self.sde.sigma_max

    # -- the flux mapping --------------------------------------------------

    def to_x(self, flux):
        """nJy -> the model's space, ``log(s * softplus(f / s))``.

        Takes no band: one softening scale serves every band, so a scene is a
        scene.  Defined and finite for any real input, negative sky included.
        """
        return self.transform.forward(flux)

    def to_flux(self, x):
        """The model's space -> nJy: ``exp(x)``.  **This is what a forward model
        calls.**

        It is the *model map*, not the inverse of ``to_x``: ``to_flux(to_x(f))``
        is ``soften(f) = s * softplus(f / s)`` exactly, at every flux.  That gap
        is the whole discrepancy between the data and what the prior can
        express, and it is deliberate -- it is also why a scene drawn from the
        prior can never contain negative flux.  Above ~10 softening scales the
        two agree to double precision; below that, ``soften`` is the shape the
        model actually has.

        Use ``to_flux_exact`` to undo ``to_x``, e.g. to recover measured flux
        including its negative sky pixels.
        """
        return self.transform.inverse(x)

    def to_flux_exact(self, x):
        """The true inverse of ``to_x``, negative sky included.

        For round trips and for recovering measurements.  Not what a forward
        model should use -- see ``to_flux``.
        """
        return self.transform.inverse_exact(x)

    # -- what a sampler or an HMC needs ------------------------------------

    def score(
        self, x: Float[Array, "..."], sigma: float | None = None
    ) -> Float[Array, "..."]:
        """``grad_x log p_sigma(x)``, same shape in as out.

        Accepts ``(H, W)``, ``(C, H, W)`` or ``(B, C, H, W)`` and gives back
        what it was given.  ``sigma`` defaults to ``sigma_min``, the sharpest
        distribution the model was trained on; the model is **not** trained
        outside ``[sigma_min, sigma_max]`` and will extrapolate silently.
        """
        batched, shape = self._as_batch(x)
        s = _score(self.model, batched, self._sigmas(sigma, len(batched)))
        return s.reshape(shape)

    def energy(
        self, x: Float[Array, "..."], sigma: float | None = None
    ) -> Float[Array, "..."]:
        """``E(x, sigma)``, a scalar per scene, with ``score == -grad_x E``.

        **This is what HMC needs that a score model cannot give**: the
        accept/reject step is on the Hamiltonian, so it wants the potential
        itself and not only its gradient.  Up to an additive constant that is
        pure gauge, ``-energy`` is the unnormalised log density.
        """
        if not self.is_energy:
            raise TypeError(
                f"{self.source} was trained with architecture "
                f"{self.architecture!r}, which predicts the score directly and "
                f"is not the gradient of anything -- there is no energy to "
                f"return. Train with architecture 'ncsnpp_energy' (or 'energy') "
                f"if you need a potential, e.g. for HMC's accept/reject step."
            )
        batched, shape = self._as_batch(x)
        e = _energy(self.model, batched, self._sigmas(sigma, len(batched)))
        return e[0] if len(shape) < 4 else e

    def log_prob(self, x, sigma: float | None = None):
        """``-energy``: the unnormalised log density, up to an additive constant.

        Not a normalised log probability and it does not pretend to be -- the
        partition function is not available.  Differences between scenes at the
        same sigma are meaningful; the absolute value is not.
        """
        return -self.energy(x, sigma)

    def sample(
        self, key: PRNGKeyArray, n: int = 1, sampler: str = "pflow", **kwargs
    ) -> Float[Array, "b c h w"]:
        """Draw ``n`` scenes on the prior's own grid, in the model's space."""
        shape = (n, self.model.in_channels, self.size, self.size)
        if sampler == "pflow":
            return pflow_sample(self.model, key, shape, self.sde, **kwargs)
        if sampler == "sde":
            return reverse_sde_sample(self.model, key, shape, self.sde, **kwargs)
        raise ValueError(f"unknown sampler {sampler!r}; use 'pflow' or 'sde'")

    # -- internals ---------------------------------------------------------

    def _as_batch(self, x):
        """``(B, C, H, W)`` plus the shape to put back, for 2, 3 or 4 axes."""
        a = jnp.asarray(x)
        if a.ndim == 2:
            return a[None, None], a.shape
        if a.ndim == 3:
            return a[None], a.shape
        if a.ndim == 4:
            return a, a.shape
        raise ValueError(
            f"expected (H, W), (C, H, W) or (B, C, H, W), got shape {a.shape}"
        )

    def _sigmas(self, sigma, n):
        return jnp.full((n,), self.sde.sigma_min if sigma is None else sigma)

    def summary(self) -> str:
        return "\n".join([
            f"{self.architecture} prior from {self.source} at step {self.step:,}",
            f"  {n_parameters(self.model):,} parameters on a {self.size}x{self.size} "
            f"grid ({self.size * self.pool_factor} native px, pool "
            f"{self.pool_factor})",
            f"  sigma in [{self.sigma_min:.4g}, {self.sigma_max:.4g}], "
            f"softening {self.transform.softening:.4g} nJy, sky at x = "
            f"{self.transform.sky_level:.3f}",
            f"  energy available: {self.is_energy}",
        ])


def load_prior(directory: str | Path, which: str = "ema") -> Prior:
    """Rebuild a trained prior from a checkpoint directory.

    ``which`` is ``"ema"`` (the default, and what inference should use -- it is
    what the training loop drew its samples from) or ``"model"``, the live
    weights.  Point this at ``final/``, or at any ``checkpoints/step-*/``.
    """
    from .training.checkpoint import load_checkpoint

    d = Path(directory)
    model, config, step = load_checkpoint(d, which=which)
    return Prior(
        model=model,
        config=config,
        sde=VESDE.from_config(config.sde),
        transform=LogFluxTransform.from_config(config.transform),
        step=int(step),
        source=d,
    )
