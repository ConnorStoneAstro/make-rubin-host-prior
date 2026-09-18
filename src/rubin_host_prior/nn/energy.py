"""The energy model, and the score derived from it.

``ConvEnergyNet`` maps a scene and a noise level to a scalar.  The score used by
the diffusion machinery is ``-grad_x E(x, sigma)``, an exact gradient by
construction -- so it is a conservative vector field and a genuine score, not a
network that merely approximates one.  The practical consequences: the implied
log-density is path-independent, the Jacobian of the score is symmetric, and you
can evaluate relative log-probabilities of scenes directly.

Because every convolution is "valid", the network is translation-equivariant and
has no zero-padding border artefacts, and it accepts any input at least
``min_input_size`` pixels on a side.  See ``geometry`` for the arithmetic, and in
particular for why the loss is restricted to an interior window.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .. import geometry
from ..config import EnergyConfig
from .layers import ConvBlock, SigmaEmbedding


class ConvEnergyNet(eqx.Module):
    """Stack of valid convolutions summed into a scalar energy."""

    blocks: tuple[ConvBlock, ...]
    head: eqx.nn.Conv2d
    embed: SigmaEmbedding
    config: EnergyConfig = eqx.field(static=True)

    def __init__(self, config: EnergyConfig, *, key: PRNGKeyArray):
        if config.sigma_scaling not in ("inverse_sigma", "none"):
            raise ValueError(f"bad sigma_scaling {config.sigma_scaling!r}")
        self.config = config
        keys = jax.random.split(key, config.n_layers + 2)
        self.embed = SigmaEmbedding(
            config.embed_dim,
            config.n_fourier,
            config.fourier_scale,
            config.activation,
            config.fourier_seed,
            key=keys[0],
        )
        widths = (config.in_channels,) + tuple(config.channels)
        self.blocks = tuple(
            ConvBlock(
                widths[i],
                widths[i + 1],
                config.kernel_size,
                config.embed_dim,
                config.activation,
                config.residual,
                config.film_init_scale,
                key=keys[i + 1],
            )
            for i in range(config.n_layers)
        )
        # 1x1 projection to a single channel, then sum.  No bias: a constant
        # added to the energy is invisible to the score, so it is pure gauge.
        #
        # The head is initialised *small but not zero*.  This differs from the
        # usual zero-init of a final layer, and deliberately: here the trained
        # quantity is the score, -W_head . dh/dx, so a zero head makes the score
        # identically zero AND makes dscore/dtheta zero for every upstream
        # parameter.  At step 0 only the head would receive a gradient.  A small
        # head keeps the initial energy landscape nearly flat (loss ~ 1, as it
        # should be) while letting gradients reach the whole network immediately.
        head = eqx.nn.Conv2d(
            config.channels[-1], 1, 1, padding=0, use_bias=False, key=keys[-1]
        )
        fan_in = config.channels[-1]
        w_head = (
            config.head_init_scale
            * jax.random.normal(keys[-1], head.weight.shape)
            / jnp.sqrt(fan_in)
        )
        self.head = eqx.tree_at(lambda m: m.weight, head, w_head)

    # -- geometry ---------------------------------------------------------

    @property
    def n_layers(self) -> int:
        return len(self.blocks)

    @property
    def receptive_radius(self) -> int:
        return geometry.receptive_radius(self.n_layers, self.config.kernel_size)

    @property
    def loss_margin(self) -> int:
        return geometry.loss_margin(self.n_layers, self.config.kernel_size)

    @property
    def min_input_size(self) -> int:
        return geometry.min_input_size(self.n_layers, self.config.kernel_size)

    # -- forward ----------------------------------------------------------

    def energy_map(
        self, x: Float[Array, "c h w"], sigma: Float[Array, ""]
    ) -> Float[Array, "1 e e"]:
        """The per-cell energy density, before summation.  Useful for diagnostics."""
        # Without this check an undersized scene produces a zero-size energy map,
        # so the energy is sum([]) == 0 and the score is identically zero -- valid
        # arrays all the way down, and silently meaningless.
        h, w = x.shape[-2:]
        need = self.min_input_size
        if min(h, w) < need:
            raise ValueError(
                f"scene is {h}x{w} but a model with {self.n_layers} "
                f"{self.config.kernel_size}x{self.config.kernel_size} valid "
                f"convolutions needs at least {need} pixels per side"
            )
        emb = self.embed(jnp.asarray(sigma))
        h = x
        for block in self.blocks:
            h = block(h, emb)
        return self.head(h)

    def __call__(
        self, x: Float[Array, "c h w"], sigma: Float[Array, ""]
    ) -> Float[Array, ""]:
        sigma = jnp.asarray(sigma)
        e = jnp.sum(self.energy_map(x, sigma))
        if self.config.sigma_scaling == "inverse_sigma":
            # A reparameterisation, not a change to the loss or to the exactness
            # of the score: E = E~ / sigma makes grad(E~) ~ epsilon at every noise
            # level, so one set of weights does not have to span decades of score
            # magnitude.  Set sigma_scaling="none" for the unmodified energy.
            e = e / sigma
        return e


# -- score ----------------------------------------------------------------


def energy(
    model: ConvEnergyNet, x: Float[Array, "c h w"], sigma: Float[Array, ""]
) -> Float[Array, ""]:
    return model(x, sigma)


def score(
    model: ConvEnergyNet, x: Float[Array, "c h w"], sigma: Float[Array, ""]
) -> Float[Array, "c h w"]:
    """``-grad_x E``, i.e. an estimate of ``grad_x log p_sigma(x)``."""
    return -jax.grad(energy, argnums=1)(model, x, sigma)


def batched_energy(
    model: ConvEnergyNet, x: Float[Array, "b c h w"], sigma: Float[Array, " b"]
) -> Float[Array, " b"]:
    return jax.vmap(energy, in_axes=(None, 0, 0))(model, x, sigma)


def batched_score(
    model: ConvEnergyNet, x: Float[Array, "b c h w"], sigma: Float[Array, " b"]
) -> Float[Array, "b c h w"]:
    return jax.vmap(score, in_axes=(None, 0, 0))(model, x, sigma)


def n_parameters(model: eqx.Module) -> int:
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    return sum(leaf.size for leaf in leaves)
