"""Samplers for the reverse process.

A warning specific to this architecture: the model is **size-locked**.  Its
convolutions are same-mode and zero-padded, so it will happily run on any grid,
but the padding it learned to expect belongs to the grid it was trained on.
Sample at ``PatchConfig.out_size`` and nothing else; a different size is not a
different view of the same prior, it is a different operator.

This used to say the opposite -- generate a canvas larger than you need and keep
the middle -- which was right for valid convolutions, where the outer ``2R``
pixels had a truncated score.  Same-mode padding scores every pixel, so there is
nothing to throw away and nothing to enlarge.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from ..nn.energy import ConvEnergyNet, batched_score
from .sde import VESDE


def _sigma_schedule(sde: VESDE, n_steps: int) -> Float[Array, " n"]:
    """Geometric sigma grid from ``sigma_max`` down to ``sigma_min``."""
    return sde.sigma(jnp.linspace(1.0, 0.0, n_steps + 1))


@eqx.filter_jit
def pflow_sample(
    model: ConvEnergyNet,
    key: PRNGKeyArray,
    shape: tuple[int, int, int, int],
    sde: VESDE,
    n_steps: int = 256,
    heun: bool = True,
) -> Float[Array, "b c h w"]:
    """Deterministic probability-flow ODE, integrated in sigma.

    ``dx/dsigma = -sigma * score(x, sigma)``.  Heun's method (the EDM default)
    costs two score evaluations per step and is worth it -- Euler needs several
    times more steps for the same accuracy.
    """
    sigmas = _sigma_schedule(sde, n_steps)
    x = sde.prior_sample(key, shape)
    batch = shape[0]

    def step(x, i):
        s_cur, s_next = sigmas[i], sigmas[i + 1]
        d_cur = -s_cur * batched_score(model, x, jnp.full((batch,), s_cur))
        x_next = x + (s_next - s_cur) * d_cur
        if heun:
            d_next = -s_next * batched_score(model, x_next, jnp.full((batch,), s_next))
            x_next = x + (s_next - s_cur) * 0.5 * (d_cur + d_next)
        return x_next, None

    x, _ = jax.lax.scan(step, x, jnp.arange(n_steps))
    return x


@eqx.filter_jit
def reverse_sde_sample(
    model: ConvEnergyNet,
    key: PRNGKeyArray,
    shape: tuple[int, int, int, int],
    sde: VESDE,
    n_steps: int = 1000,
    n_corrector: int = 1,
    snr: float = 0.16,
) -> Float[Array, "b c h w"]:
    """Predictor-corrector sampling of the reverse VE SDE (Song et al. 2021).

    Predictor: Euler-Maruyama on ``dx = -g2(t) * score * dt + g(t) dw``.
    Corrector: ``n_corrector`` Langevin steps at fixed sigma, with the step size
    set from the signal-to-noise ratio ``snr`` so it adapts to the score's scale.
    Set ``n_corrector=0`` for plain Euler-Maruyama.

    The corrector carries a known positive bias in the sampled variance, from the
    Euler-Maruyama discretisation of the Langevin chain: the stationary variance
    grows by roughly ``4 * snr**4`` per corrector step.  Measured against an
    analytic Gaussian target it is +1.4% at ``snr=0.16``, +0.8% at 0.10 and under
    0.2% at 0.05.  Keep ``snr <= 0.1`` if the width of the distribution matters,
    or use ``pflow_sample``, which has no such bias.
    """
    ts = jnp.linspace(1.0, 0.0, n_steps + 1)
    dt = 1.0 / n_steps
    batch = shape[0]
    key, k0 = jax.random.split(key)
    x = sde.prior_sample(k0, shape)

    d = int(np.prod(shape[1:]))
    z_norm = jnp.sqrt(jnp.asarray(float(d)))  # E||z|| for d standard normals

    def corrector_step(carry, _):
        x, sig, key = carry
        key, k_noise = jax.random.split(key)
        s = batched_score(model, x, jnp.full((batch,), sig))
        # Song et al. eq. (C.1): step size from the ratio of noise to score norm.
        # The *expected* noise norm, not a sampled one: sharing a single draw
        # between the step size and the injected noise correlates them, and
        # E[||z||^2 z z^T] exceeds E[||z||^2] E[z z^T] by a factor 1 + 2/d.  That
        # inflates the stationary variance by ~8 snr^2 / d per step, which is
        # invisible on a large patch and badly wrong on a small one.
        s_norm = jnp.linalg.norm(s.reshape(batch, -1), axis=1)
        eps = 2.0 * (snr * z_norm / (s_norm + 1e-12)) ** 2
        eps = eps.reshape((batch,) + (1,) * (x.ndim - 1))
        z = jax.random.normal(k_noise, x.shape)
        x = x + eps * s + jnp.sqrt(2.0 * eps) * z
        return (x, sig, key), None

    def step(carry, i):
        x, key = carry
        t = ts[i]
        sig = sde.sigma(t)
        g2 = sde.g2(t)
        key, k_pred = jax.random.split(key)
        s = batched_score(model, x, jnp.full((batch,), sig))
        # Reverse-time Euler-Maruyama: dx = -g2 * score * dt + g * dw, dt < 0.
        x = x + g2 * s * dt + jnp.sqrt(g2 * dt) * jax.random.normal(k_pred, x.shape)
        if n_corrector:
            sig_next = sde.sigma(ts[i + 1])
            key, k_corr = jax.random.split(key)
            (x, _, _), _ = jax.lax.scan(
                corrector_step, (x, sig_next, k_corr), jnp.arange(n_corrector)
            )
        return (x, key), None

    (x, _), _ = jax.lax.scan(step, (x, key), jnp.arange(n_steps))
    return x


def sample_scene(
    model: ConvEnergyNet,
    key: PRNGKeyArray,
    out_size: int,
    sde: VESDE,
    n_samples: int = 1,
    sampler: str = "pflow",
    **kwargs,
) -> Float[Array, "b c h w"]:
    """Sample an ``out_size`` scene.  Pass the grid the model was trained on.

    No canvas and no crop: same-mode convolutions score every pixel, so the
    scene that comes out is the scene that was asked for.  The predecessor,
    ``sample_interior``, generated ``out_size + 4R`` and kept the middle, which
    is what valid convolutions needed and would now be actively wrong -- the
    padding is part of the operator, so a larger canvas puts it somewhere the
    model has never seen it.

    ``sde`` is required.  It used to default to ``VESDE()``, whose own field
    defaults are a schedule no trained model has: sampling would run, produce
    plausible-looking noise, and report nothing wrong.
    """
    shape = (n_samples, model.config.in_channels, out_size, out_size)
    if sampler == "pflow":
        return pflow_sample(model, key, shape, sde, **kwargs)
    if sampler == "sde":
        return reverse_sde_sample(model, key, shape, sde, **kwargs)
    raise ValueError(f"unknown sampler {sampler!r}; use 'pflow' or 'sde'")
