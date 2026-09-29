"""Building blocks for the energy network.

Two constraints shape everything here:

1.  **No spatial aggregation anywhere except the final sum.**  That rules out
    batch/group/layer normalisation over pixels -- any such layer would make the
    output depend on the size of the patch, destroying the "works on any scene
    larger than the receptive field" property.  Conditioning is therefore
    FiLM: a spatially constant per-channel scale and shift.
2.  **C^1 activations.**  The score is a gradient of the network, so a
    piecewise-linear activation (ReLU) gives a piecewise-constant score with
    discontinuities.  Use a smooth one.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

#: Variance-preserving normalisation for a residual sum: two independent unit
#: variances add to two, so the sum is divided by this to keep it at one.
SQRT2 = jnp.sqrt(2.0)

# Smooth activations only.  relu/leaky_relu are deliberately absent: they are C^0,
# and d/dx of the resulting energy is discontinuous.
ACTIVATIONS = {
    "silu": jax.nn.silu,
    "gelu": jax.nn.gelu,
    "softplus": jax.nn.softplus,
    "tanh": jnp.tanh,
    "elu": jax.nn.elu,  # C^1 at 0 but not C^2
}


def get_activation(name: str):
    try:
        return ACTIVATIONS[name]
    except KeyError:
        raise ValueError(
            f"unknown or non-smooth activation {name!r}; "
            f"choose from {sorted(ACTIVATIONS)}"
        ) from None


class FourierFeatures(eqx.Module):
    """Random Fourier features of ``log(sigma)``, with a frozen basis.

    The frequencies are a *static* field, not an array leaf, so the optimiser
    cannot see them.  ``stop_gradient`` alone would not be enough: AdamW's
    decoupled weight decay shrinks every parameter it can reach regardless of
    gradient, so with ``weight_decay > 0`` the frequencies would drift towards
    zero and the sigma conditioning would quietly go flat.

    They are therefore derived from ``fourier_seed`` in the config rather than
    from the model's init key.  This matters more than it looks: a static field
    is part of the pytree *metadata*, so two models whose frequencies differ are
    structurally incompatible -- ``tree_map`` across them fails (which breaks the
    EMA), and ``tree_deserialise_leaves`` does not restore them at all, so a
    reloaded checkpoint would silently keep the skeleton's basis and compute
    different scores from the same weights.  Deriving them from the config,
    which travels with the checkpoint, makes every model built from one config
    share one basis.
    """

    freqs: tuple[float, ...] = eqx.field(static=True)

    def __init__(self, n_features: int, scale: float = 1.0, seed: int = 0):
        self.freqs = tuple(
            float(f)
            for f in scale * jax.random.normal(jax.random.key(seed), (n_features,))
        )

    def __call__(self, log_sigma: Float[Array, ""]) -> Float[Array, " 2n"]:
        theta = 2.0 * jnp.pi * jnp.asarray(self.freqs) * log_sigma
        return jnp.concatenate([jnp.sin(theta), jnp.cos(theta)])


class SigmaEmbedding(eqx.Module):
    """log(sigma) -> Fourier features -> 2-layer MLP -> embedding vector."""

    fourier: FourierFeatures
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear
    act: callable = eqx.field(static=True)

    def __init__(
        self,
        embed_dim: int,
        n_fourier: int,
        fourier_scale: float,
        activation: str,
        fourier_seed: int = 0,
        *,
        key: PRNGKeyArray,
    ):
        k1, k2 = jax.random.split(key)
        self.fourier = FourierFeatures(n_fourier, fourier_scale, fourier_seed)
        self.l1 = eqx.nn.Linear(2 * n_fourier, embed_dim, key=k1)
        self.l2 = eqx.nn.Linear(embed_dim, embed_dim, key=k2)
        self.act = get_activation(activation)

    def __call__(self, sigma: Float[Array, ""]) -> Float[Array, " d"]:
        h = self.fourier(jnp.log(sigma))
        return self.l2(self.act(self.l1(h)))


class FiLM(eqx.Module):
    """Per-channel (scale, shift) from the sigma embedding.

    Applied as ``h * (1 + s) + b`` so the block starts out as an unmodulated
    convolution.  The projection is initialised *small but not zero*: exactly
    zero would make ``dloss/demb = W_proj^T dloss/ds`` vanish, so the whole
    sigma-embedding MLP would receive no gradient at step 0.
    """

    proj: eqx.nn.Linear

    def __init__(
        self,
        embed_dim: int,
        channels: int,
        init_scale: float = 0.01,
        *,
        key: PRNGKeyArray,
    ):
        proj = eqx.nn.Linear(embed_dim, 2 * channels, key=key)
        w = (
            init_scale
            * jax.random.normal(key, proj.weight.shape)
            / jnp.sqrt(embed_dim)
        )
        proj = eqx.tree_at(lambda m: m.weight, proj, w)
        self.proj = eqx.tree_at(lambda m: m.bias, proj, jnp.zeros_like(proj.bias))

    def __call__(
        self, h: Float[Array, "c h w"], emb: Float[Array, " d"]
    ) -> Float[Array, "c h w"]:
        scale, shift = jnp.split(self.proj(emb), 2)
        return h * (1.0 + scale)[:, None, None] + shift[:, None, None]


class ConvBlock(eqx.Module):
    """same conv -> FiLM -> activation, with an optional skip.

    ``radius`` is the per-side reach, ``(k - 1) // 2 * dilation``.  It no longer
    shrinks anything: the convolution is same-mode, so the grid comes out the
    size it went in and the skip is a plain sum.  The radius is still what sets
    how far the block sees, so it is kept for the geometry report.

    **Zero padding.**  ``padding_mode`` is left at the equinox default, which
    pads with zeros rather than reflecting.  Reflection was tried on the *data*
    border and is the leading explanation for a run that learned no structure
    above ~16 px: a mirrored border is symmetric at every scale, and the large
    scales are almost all border.  Zeros invent no structure; what they invent
    is an edge, which a size-locked model can learn once and for all.  See
    ``ConvEnergyNet.input_offset`` for why the *input* is centred first.
    """

    conv: eqx.nn.Conv2d
    film: FiLM
    act: callable = eqx.field(static=True)
    radius: int = eqx.field(static=True)
    residual: bool = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        embed_dim: int,
        activation: str,
        residual: bool,
        film_init_scale: float = 0.01,
        dilation: int = 1,
        *,
        key: PRNGKeyArray,
    ):
        kc, kf = jax.random.split(key)
        # Dilation spreads the same k*k taps over d times the span: identical
        # parameters and identical arithmetic, reaching d times as far.  Under
        # same-mode padding it costs no shape either -- what it costs is a wider
        # band of pixels whose taps land in the padding, r*d per side.
        conv = eqx.nn.Conv2d(
            in_channels, out_channels, kernel_size, padding="SAME",
            dilation=dilation, key=kc
        )
        # He-style init for a smooth, roughly half-rectifying activation: the
        # equinox default (uniform 1/sqrt(fan_in)) loses variance layer over layer
        # and an 8-deep stack arrives at the head with almost no signal.
        fan_in = in_channels * kernel_size * kernel_size
        w = jax.random.normal(kc, conv.weight.shape) * jnp.sqrt(2.0 / fan_in)
        conv = eqx.tree_at(lambda m: m.weight, conv, w)
        conv = eqx.tree_at(lambda m: m.bias, conv, jnp.zeros_like(conv.bias))
        self.conv = conv
        self.film = FiLM(embed_dim, out_channels, film_init_scale, key=kf)
        self.act = get_activation(activation)
        self.radius = (kernel_size - 1) // 2 * dilation
        if residual and in_channels != out_channels:
            raise ValueError(
                f"a residual skip needs {in_channels} == {out_channels}: there "
                f"is nothing to add a {in_channels}-channel input to a "
                f"{out_channels}-channel output. This used to be silently "
                f"switched off, which meant a config asking for skips could get "
                f"none and never hear about it."
            )
        self.residual = residual

    def __call__(
        self, h: Float[Array, "c h w"], emb: Float[Array, " d"]
    ) -> Float[Array, "c2 h2 w2"]:
        out = self.act(self.film(self.conv(h), emb))
        if self.residual:
            # No crop: same-mode convolution leaves the skip the same shape as
            # the output.  Divided by sqrt(2), so the block preserves variance
            # instead of doubling it.  Without this a 10-layer stack arrives at
            # the head with ~180x the signal a plain stack would, the small head
            # no longer keeps the initial energy landscape flat, and training
            # starts at a loss of 1.78 -- worse than predicting no score at all.
            # Scaling the head down instead would work, but the factor would
            # depend on the depth and would have to be retuned for every
            # architecture; this does not.
            out = (out + h) / SQRT2
        return out
