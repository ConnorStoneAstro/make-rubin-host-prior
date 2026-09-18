"""Exponential moving average of the weights.

Diffusion models are notoriously sensitive to this -- the EMA weights are
typically much better than the live weights, and for a prior that will be reused
many times it is the EMA copy you want to ship.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def ema_decay_at(step: int | Float[Array, ""], decay: float) -> Float[Array, ""]:
    """Warmed-up decay: ``min(decay, (1 + step) / (10 + step))``.

    Without the warmup the average is dominated by the random initialisation for
    the first ``1 / (1 - decay)`` steps, which at ``decay = 0.999`` is a thousand
    steps of a useless EMA copy.
    """
    step = jnp.asarray(step, dtype=jnp.float32)
    return jnp.minimum(decay, (1.0 + step) / (10.0 + step))


def ema_update(ema: eqx.Module, model: eqx.Module, decay: Float[Array, ""]):
    """``ema <- decay * ema + (1 - decay) * model`` over inexact array leaves."""

    def mix(e, m):
        if eqx.is_inexact_array(e):
            return decay * e + (1.0 - decay) * m
        return e

    return jax.tree_util.tree_map(mix, ema, model)
