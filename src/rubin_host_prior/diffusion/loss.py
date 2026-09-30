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
    sigmas: Float[Array, " n"],
    key: PRNGKeyArray,
    sde: VESDE,
    margin: int | None = None,
) -> Float[Array, "n b"]:
    """Loss of **every** patch at **every** sigma: shape ``(n_sigma, b)``.

    For validation: a loss curve against sigma tells you which part of the
    schedule is underfit, which an aggregate number hides entirely.

    **It used to give one patch per sigma**, pairing example ``i`` with
    ``sigma[i]``, and that made the curve unreadable. The scatter between
    neighbouring points was not the sigma dependence, it was the difference
    between a blank-sky patch and one with a bright galaxy in it -- swings of
    0.68 to 0.04 to 0.45 at adjacent sigmas, reproducible from one eval to the
    next only because the batch and the pairing were fixed. A docstring here
    claimed the scatter was Monte Carlo at the ~4% level and that anything that
    size was noise; the real scatter was ten times that and came from the
    patches, not the noise draw.

    Every patch now sees every sigma, with an independent noise draw per sigma,
    so averaging across the batch axis leaves the sigma dependence and nothing
    else. Cost is ``n_sigma`` batches instead of one, evaluated one at a time so
    the memory is a single batch; at 32 of each that is a few seconds on a
    schedule that runs every few thousand steps.
    """
    if margin is None:
        margin = model.loss_margin
    sigmas = jnp.atleast_1d(jnp.asarray(sigmas))
    batch = x.shape[0]

    def at(carry):
        sigma, k = carry
        full = jnp.full((batch,), sigma)
        x_noisy, eps = sde.perturb(k, x, full)
        s = batched_score(model, x_noisy, full)
        residual = full[:, None, None, None] * s + eps
        return jnp.mean(crop_interior(residual, margin) ** 2, axis=(1, 2, 3))

    keys = jax.random.split(key, len(sigmas))
    # `lax.map` and not `vmap`: vmapping would put n_sigma batches of a
    # gradient-of-a-gradient on the device at once, which is the one thing here
    # that can exhaust it.
    return jax.lax.map(at, (sigmas, keys))


def gaussian_loss_floor(
    x: Float[Array, "b c h w"], sigmas: Float[Array, " n"]
) -> Float[Array, " n"]:
    """The loss the best *Gaussian* model of this data would reach, per sigma.

    Averaged over Fourier modes, ``P_k / (P_k + sigma^2)`` with ``P_k`` the
    data's own per-mode power. It is what ``dsm_loss_by_sigma`` should be
    compared against, and it is the number every hand-wave about "is the model
    underfit here" has been standing in for.

    **An upper bound on what is achievable, not a lower one.** The true score of
    a non-Gaussian distribution carries more information than its covariance, so
    the real optimum sits at or below this. Which makes the comparison one-sided
    and useful: a model *above* this curve is definitely underfit at that sigma,
    while one below it is exploiting structure a Gaussian cannot.
    """
    v = jnp.asarray(x)
    if v.ndim == 4:
        v = v[:, 0]
    v = v - jnp.mean(v, axis=(-2, -1), keepdims=True)
    h = v.shape[-1]
    # Per-mode power, averaged over the batch.  Parseval: the mean over modes of
    # `power` is the per-pixel variance, which is what makes the ratio below the
    # fraction of variance that survives at this sigma.
    power = jnp.mean(jnp.abs(jnp.fft.fft2(v)) ** 2, axis=0) / h ** 2
    s2 = jnp.asarray(sigmas)[:, None] ** 2
    flat = power.reshape(1, -1)
    return jnp.mean(flat / (flat + s2), axis=1)


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
