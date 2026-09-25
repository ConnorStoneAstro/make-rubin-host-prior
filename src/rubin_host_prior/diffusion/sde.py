"""Variance-exploding SDE with a geometric sigma schedule.

    forward:   x_sigma = x + sigma * eps,        eps ~ N(0, I)
    schedule:  sigma(t) = sigma_min * (sigma_max / sigma_min) ** t,   t in [0, 1]

No preconditioning of the network output.  The virtue of VE here is that the
learned score *is* ``grad_x log p_sigma(x)`` on the data's own scale: nothing has
to be unscaled before the prior is combined with a likelihood, and the
``sigma -> 0`` limit is the prior you actually want.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ..config import SDEConfig


@dataclass(frozen=True)
class VESDE:
    sigma_min: float = 0.01
    sigma_max: float = 10.0
    #: Mean of the training data in x.  VE does not move the mean, so the
    #: ``t = 1`` marginal is centred here and not on zero.
    data_mean: float = 0.0

    @classmethod
    def from_config(cls, config: SDEConfig) -> "VESDE":
        """Build from a config, refusing one whose range was never measured.

        All three are None until ``prepare_config.py`` measures them.  A default
        used here instead would be wrong in a way nothing reports: the wrong
        ``sigma_max`` trains the score on noise levels the data never reaches,
        and the wrong ``data_mean`` starts every sample half a ``sigma_max``
        from the distribution it is meant to come from.
        """
        missing = [n for n in ("sigma_min", "sigma_max", "data_mean")
                   if getattr(config, n) is None]
        if missing:
            raise ValueError(
                f"the config has no {', '.join(missing)}. Run "
                f"scripts/prepare_config.py on the shards; it measures the "
                f"sigma range and the data mean from them."
            )
        if not 0 < config.sigma_min < config.sigma_max:
            raise ValueError(
                f"need 0 < sigma_min < sigma_max, got sigma_min="
                f"{config.sigma_min} and sigma_max={config.sigma_max}"
            )
        return cls(float(config.sigma_min), float(config.sigma_max),
                   float(config.data_mean))

    @property
    def log_ratio(self) -> float:
        """``log(sigma_max / sigma_min)``, the schedule's rate constant.

        ``math.log``, not ``jnp.log``: under jit even a Python scalar passed to
        ``jnp`` is staged out into a tracer, and ``float()`` of a tracer raises.
        """
        return math.log(self.sigma_max / self.sigma_min)

    def sigma(self, t: Float[Array, "..."]) -> Float[Array, "..."]:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def t_of_sigma(self, sigma: Float[Array, "..."]) -> Float[Array, "..."]:
        return jnp.log(sigma / self.sigma_min) / self.log_ratio

    def g2(self, t: Float[Array, "..."]) -> Float[Array, "..."]:
        """``d(sigma^2)/dt``, the squared diffusion coefficient."""
        return 2.0 * self.sigma(t) ** 2 * self.log_ratio

    # -- training-time helpers -------------------------------------------

    def sample_sigma(
        self, key: PRNGKeyArray, shape: tuple[int, ...]
    ) -> Float[Array, "..."]:
        """Log-uniform over ``[sigma_min, sigma_max]``.

        Uniform in ``t`` under the geometric schedule, i.e. equal training
        weight per decade of noise -- which is what you want when the data
        spans decades of amplitude.
        """
        return self.sigma(jax.random.uniform(key, shape))

    def perturb(
        self,
        key: PRNGKeyArray,
        x: Float[Array, "b ..."],
        sigma: Float[Array, " b"],
    ) -> tuple[Float[Array, "b ..."], Float[Array, "b ..."]]:
        """Return ``(x + sigma * eps, eps)`` broadcasting sigma over pixel axes."""
        eps = jax.random.normal(key, x.shape, dtype=x.dtype)
        s = sigma.reshape(sigma.shape + (1,) * (x.ndim - sigma.ndim))
        return x + s * eps, eps

    # -- sampling-time helpers -------------------------------------------

    def prior_sample(
        self, key: PRNGKeyArray, shape: tuple[int, ...]
    ) -> Float[Array, "..."]:
        """Draw from the ``t = 1`` marginal: ``N(data_mean, sigma_max^2)``.

        VE only *adds* noise, so the marginal keeps the data's mean; the width is
        ``sqrt(sigma_max^2 + var(data))``, which is ``sigma_max`` to the extent
        that ``sigma_max`` dominates the data's own spread.  Check that:
        ``sigma_max`` should be at least the largest pixel-to-pixel spread in
        the training set.

        ``data_mean`` is not cosmetic.  It was implicitly zero while the
        transform put every band's sky at ``log(log 2) = -0.37``; under absolute
        log flux the sky sits near +3, and starting the reverse process from
        ``N(0, sigma_max^2)`` would be starting half a ``sigma_max`` away from
        the distribution the score was trained on.
        """
        return self.data_mean + self.sigma_max * jax.random.normal(key, shape)
