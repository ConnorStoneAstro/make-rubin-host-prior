"""The energy model, and the score derived from it.

``ConvEnergyNet`` maps a scene and a noise level to a scalar.  The score used by
the diffusion machinery is ``-grad_x E(x, sigma)``, an exact gradient by
construction -- so it is a conservative vector field and a genuine score, not a
network that merely approximates one.  The practical consequences: the implied
log-density is path-independent, the Jacobian of the score is symmetric, and you
can evaluate relative log-probabilities of scenes directly.

Every convolution is **same-mode and zero-padded**, so the energy map is the
size of the scene and every pixel has a score.  The price is that the network is
no longer translation-equivariant -- it can read its distance from the border --
and is therefore **size-locked**: a model trained on a 128x128 grid is a prior
over 128x128 scenes and is not defined on any other size.  ``geometry`` has the
reach arithmetic and what fraction of it lands on padding.

This replaced valid-mode convolutions, which were equivariant and size-agnostic
but needed ``2R`` pixels of real context on every side of the loss region.  At
``R = 78`` that is 156 px per side, which no 512 px stamp can supply: measured,
the usable window on a 256 grid was 16x16.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .. import geometry
from ..config import EnergyConfig
from .layers import ConvBlock, SigmaEmbedding


class EnergyBranch(eqx.Module):
    """One stack of same-mode convolutions ending in a 1x1 head.

    Complete on its own, so its contribution is already a scalar field over the
    scene -- which is what makes summing several of them still an energy, and
    the score still an exact gradient.
    """

    blocks: tuple[ConvBlock, ...]
    head: eqx.nn.Conv2d
    radius: int = eqx.field(static=True)

    def __init__(
        self,
        config: EnergyConfig,
        channels: tuple[int, ...],
        dilations: tuple[int, ...],
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, len(channels) + 1)
        widths = (config.in_channels,) + tuple(channels)
        self.blocks = tuple(
            ConvBlock(
                widths[i],
                widths[i + 1],
                config.kernel_size,
                config.embed_dim,
                config.activation,
                # Never on the first layer: it maps in_channels to the branch
                # width, so its input and output have different channel counts
                # and there is nothing to add.  EnergyConfig states this rule
                # and rejects any *other* width change, so a skip is only absent
                # where it is structurally impossible.
                config.residual and i > 0,
                config.film_init_scale,
                dilations[i],
                key=keys[i],
            )
            for i in range(len(channels))
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
            channels[-1], 1, 1, padding=0, use_bias=False, key=keys[-1]
        )
        w_head = (
            config.head_init_scale
            * jax.random.normal(keys[-1], head.weight.shape)
            / jnp.sqrt(channels[-1])
        )
        self.head = eqx.tree_at(lambda m: m.weight, head, w_head)
        self.radius = geometry.branch_radius(dilations, config.kernel_size)

    def __call__(
        self, x: Float[Array, "c h w"], emb: Float[Array, " d"]
    ) -> Float[Array, "1 e e"]:
        h = x
        for block in self.blocks:
            h = block(h, emb)
        return self.head(h)


class ConvEnergyNet(eqx.Module):
    """Parallel stacks of same-mode convolutions, summed into a scalar energy.

    One branch is the ordinary case.  Several still work -- a sum of energies is
    an energy -- and now simply add cell for cell, since same-mode padding
    leaves every branch's map the size of the scene.  See ``EnergyConfig`` for
    why dilation rather than pooling.
    """

    branches: tuple[EnergyBranch, ...]
    embed: SigmaEmbedding
    config: EnergyConfig = eqx.field(static=True)
    #: Subtracted from the scene before the first layer, so that the zeros the
    #: convolutions pad with sit at the sky rather than far below it.
    #:
    #: ``x`` is absolute log flux: the sky sits at ``log(s * log 2)``, around
    #: +2.6 to +3.5 depending on depth, and a padded zero is ``exp(0) = 1 nJy``
    #: -- five to ten sigma below the sky, a hard black frame the data never
    #: contains.  Subtracting a constant leaves ``-grad_x E`` unchanged by the
    #: chain rule, so this costs nothing in correctness and turns the worst
    #: discontinuity in the network into a mild one.
    #:
    #: Static, like the Fourier basis and for the same reason: it is pytree
    #: metadata, so it must come from the config that travels with the
    #: checkpoint rather than from anything the optimiser or a deserialise
    #: could reach.  ``Config.input_offset`` computes it from the measured
    #: softening; it is not stored twice.
    input_offset: float = eqx.field(static=True)

    def __init__(self, config: EnergyConfig, *, input_offset: float = 0.0,
                 key: PRNGKeyArray):
        if config.sigma_scaling not in ("inverse_sigma", "none"):
            raise ValueError(f"bad sigma_scaling {config.sigma_scaling!r}")
        self.config = config
        self.input_offset = float(input_offset)
        keys = jax.random.split(key, config.n_branches + 1)
        # One embedding shared by every branch: it is a function of sigma alone,
        # so a copy per branch would be the same function learned twice -- and
        # the branches are meant to specialise by *scale*, which they do through
        # their own FiLM projections of this one embedding.
        self.embed = SigmaEmbedding(
            config.embed_dim,
            config.n_fourier,
            config.fourier_scale,
            config.activation,
            config.fourier_seed,
            key=keys[0],
        )
        self.branches = tuple(
            EnergyBranch(config, config.channels[i], config.dilations[i],
                         key=keys[i + 1])
            for i in range(config.n_branches)
        )

    # -- geometry ---------------------------------------------------------

    @property
    def n_layers(self) -> int:
        """Across every branch.  Not the geometry -- that follows the dilations."""
        return self.config.n_layers

    @property
    def n_branches(self) -> int:
        return len(self.branches)

    @property
    def receptive_radius(self) -> int:
        return self.config.receptive_radius

    @property
    def loss_margin(self) -> int:
        return self.config.loss_margin

    # -- forward ----------------------------------------------------------

    def energy_map(
        self, x: Float[Array, "c h w"], sigma: Float[Array, ""]
    ) -> Float[Array, "1 e e"]:
        """The per-cell energy density, before summation.  Useful for diagnostics.

        Same shape as the scene, because the convolutions are same-mode.  The
        scene is centred on ``input_offset`` first -- see the field.
        """
        emb = self.embed(jnp.asarray(sigma))
        h = x - self.input_offset
        total = None
        for branch in self.branches:
            # No centre-crop: same-mode convolutions leave every branch's map
            # the size of the scene, so branches of different reach already
            # agree cell for cell and simply add.
            m = branch(h, emb)
            total = m if total is None else total + m
        return total

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
