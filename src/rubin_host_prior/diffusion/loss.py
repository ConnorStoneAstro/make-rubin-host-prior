"""Denoising score matching, evaluated only where the score is fully supported.

The valid-convolution energy attenuates the score within ``2R`` pixels of the
edge (see ``geometry``).  Training on those pixels would force the network to
compensate for a structural deficit it cannot fix, corrupting the interior in
the process, so they are cropped out of the residual.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ..nn.energy import ConvEnergyNet, batched_score
from .sde import VESDE


def crop_interior(a: Float[Array, "... h w"], margin: int) -> Float[Array, "... h2 w2"]:
    """Drop ``margin`` pixels from every side of the trailing two axes."""
    if margin == 0:
        return a
    if a.shape[-1] <= 2 * margin or a.shape[-2] <= 2 * margin:
        raise ValueError(
            f"array of spatial shape {a.shape[-2:]} has no interior left after "
            f"cropping {margin} from each side; use a larger patch or fewer layers"
        )
    return a[..., margin:-margin, margin:-margin]


def dsm_loss(
    model: ConvEnergyNet,
    x: Float[Array, "b c h w"],
    key: PRNGKeyArray,
    sde: VESDE,
    margin: int | None = None,
) -> Float[Array, ""]:
    """``E || sigma * score(x + sigma*eps, sigma) + eps ||^2`` over the interior.

    The ``sigma^2`` weighting implied by writing the residual this way is the
    standard likelihood weighting: it makes the loss O(1) at every noise level,
    so the log-uniform sigma schedule really does spend equal effort per decade.
    """
    if margin is None:
        margin = model.loss_margin
    k_sigma, k_eps = jax.random.split(key)
    sigma = sde.sample_sigma(k_sigma, (x.shape[0],))
    x_noisy, eps = sde.perturb(k_eps, x, sigma)
    s = batched_score(model, x_noisy, sigma)
    residual = sigma[:, None, None, None] * s + eps
    return jnp.mean(crop_interior(residual, margin) ** 2)


def dsm_loss_by_sigma(
    model: ConvEnergyNet,
    x: Float[Array, "b c h w"],
    sigma: Float[Array, " b"],
    key: PRNGKeyArray,
    sde: VESDE,
    margin: int | None = None,
) -> Float[Array, " b"]:
    """Per-example loss at *prescribed* noise levels.

    For validation: a loss curve against sigma tells you which part of the
    schedule is underfit, which an aggregate number hides completely.

    Two things to know when reading that curve.  It should *fall* with sigma --
    for data of scale ``tau`` the optimal residual variance is
    ``tau^2 / (tau^2 + sigma^2)``, which tends to 1 as sigma falls (a tiny amount
    of added noise is unidentifiable) and to 0 as sigma grows.  And each point is
    a single example, so its Monte Carlo error is about
    ``sqrt(2 / n_interior_pixels)`` -- roughly 4% for a 32x32 interior.  Scatter
    of that size is noise, not structure.
    """
    if margin is None:
        margin = model.loss_margin
    x_noisy, eps = sde.perturb(key, x, sigma)
    s = batched_score(model, x_noisy, sigma)
    residual = sigma[:, None, None, None] * s + eps
    return jnp.mean(crop_interior(residual, margin) ** 2, axis=(1, 2, 3))


def mean_dsm_loss(
    model: ConvEnergyNet,
    batches,
    n_batches: int,
    key: PRNGKeyArray,
    sde: VESDE,
    margin: int | None = None,
) -> float:
    """Average ``dsm_loss`` over ``n_batches``, for comparing two checkpoints.

    A single batch's loss is far too noisy to compare models with: the sigma
    draw alone moves it by O(0.1), since the optimal loss runs from ~1 at small
    sigma to ~0 at large.  Average before drawing any conclusion.
    """
    total = 0.0
    for _ in range(n_batches):
        key, k = jax.random.split(key)
        total += float(dsm_loss(model, jnp.asarray(next(batches)), k, sde, margin))
    return total / n_batches
