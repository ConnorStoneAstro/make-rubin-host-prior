"""The one thing every architecture in this package agrees on: a score.

``score(model, x, sigma)`` is ``grad_x log p_sigma(x)`` for a single scene, and
``batched_score`` is that vmapped.  Everything downstream -- the DSM loss, both
samplers, the diagnostics -- goes through these two and therefore does not know
or care which architecture it was handed.

The two architectures reach the same quantity by opposite routes:

* ``ConvEnergyNet`` computes a scalar ``E(x, sigma)`` and returns ``-grad_x E``.
  The score is an exact gradient, hence conservative: symmetric Jacobian,
  path-independent log-density, relative log-probabilities of scenes available
  directly.  The price is a gradient-of-a-gradient every training step.
* ``NCSNpp`` predicts ``sigma * score`` and divides.  Not conservative -- the
  Jacobian is whatever the network makes it -- and one backward pass per step.

Both put the noise level's decades in an analytic division rather than in the
weights: NCSN++ by construction, the energy through ``sigma_scaling =
"inverse_sigma"``, which sets ``E = E~ / sigma`` so that ``grad E~`` is the
O(1) quantity.  That much they already had in common before the U-Net arrived;
``nn.ncsnpp`` lists what they do not.

Keeping the interface this narrow is what lets the second be dropped in beside
the first without the loss, the samplers or the trainer changing at all.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
from jaxtyping import Array, Float


@runtime_checkable
class ScoreModel(Protocol):
    """Everything the diffusion machinery and the trainer require of a model.

    ``score`` is what the loss and the samplers use and the only one that is
    about the mathematics.  The rest exists so that nothing upstream has to ask
    which architecture it is holding: ``in_channels`` because the samplers pick
    a shape to start from, ``loss_margin`` because the loss crops with it, and
    ``describe``/``log_header`` because the two have nothing in common to report
    and an ``isinstance`` ladder in the trainer would grow a branch for every
    architecture added.
    """

    def score(
        self, x: Float[Array, "c h w"], sigma: Float[Array, ""]
    ) -> Float[Array, "c h w"]: ...

    @property
    def in_channels(self) -> int: ...

    @property
    def loss_margin(self) -> int: ...

    def describe(self, config) -> str:
        """A few lines for the run's first log line and for the console."""

    def log_header(self, config) -> dict:
        """Whatever belongs in ``log.jsonl``'s ``start`` record."""


def score(
    model: ScoreModel, x: Float[Array, "c h w"], sigma: Float[Array, ""]
) -> Float[Array, "c h w"]:
    """``grad_x log p_sigma(x)`` for one scene."""
    return model.score(x, sigma)


def batched_score(
    model: ScoreModel, x: Float[Array, "b c h w"], sigma: Float[Array, " b"]
) -> Float[Array, "b c h w"]:
    """The score of a batch, each scene at its own sigma."""
    return jax.vmap(score, in_axes=(None, 0, 0))(model, x, sigma)


def n_parameters(model: eqx.Module) -> int:
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    return sum(leaf.size for leaf in leaves)
