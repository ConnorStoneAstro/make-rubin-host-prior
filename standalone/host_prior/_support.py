"""Everything ``_net.py`` needs that is not the architecture itself.

Four small pieces, lifted from the training package and stable: the config
dataclass that describes a network, the energy base class that turns a scalar
into a score, the frozen Fourier basis the noise level is embedded in, and the
list of activations.  ``_net.py`` is generated verbatim from the trainer's
architecture file and imports exactly these names, which is what lets it be
copied rather than transcribed.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

# Smooth activations only: the score is a gradient of the network, so a
# piecewise-linear activation would give a piecewise-constant score.
ACTIVATIONS = {
    "silu": jax.nn.silu,
    "gelu": jax.nn.gelu,
    "softplus": jax.nn.softplus,
    "tanh": jnp.tanh,
    "elu": jax.nn.elu,
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
    """Random Fourier features of a scalar, with a frozen basis.

    The frequencies are a **static** field derived from a seed in the config,
    not a trainable leaf.  That is load-bearing for loading: a static field is
    pytree *metadata*, so ``tree_deserialise_leaves`` does not restore it --
    the skeleton's basis is the one that gets used.  Deriving it from the
    config, which travels with the weights, is what makes a reloaded model
    compute the same scores as the one that was saved.
    """

    freqs: tuple[float, ...] = eqx.field(static=True)

    def __init__(self, n_features: int, scale: float = 1.0, seed: int = 0):
        self.freqs = tuple(
            float(f)
            for f in scale * jax.random.normal(jax.random.key(seed), (n_features,))
        )

    def __call__(self, log_sigma):
        theta = 2.0 * jnp.pi * jnp.asarray(self.freqs) * log_sigma
        return jnp.concatenate([jnp.sin(theta), jnp.cos(theta)])


class EnergyModel(eqx.Module):
    """A model whose ``__call__`` is a scalar energy ``E(x, sigma)``.

    Supplies ``score`` as ``-grad_x E`` -- an exact gradient, hence a
    conservative field: symmetric Jacobian, path-independent log-density, and a
    potential that HMC can put in its Hamiltonian.
    """

    def score(self, x, sigma):
        return -jax.grad(lambda a: self(a, sigma))(x)


@dataclass(frozen=True)  # hashable: stored as a static field on the module
class NCSNppConfig:
    """The architecture, exactly as the trainer wrote it into ``config.json``.

    Read, never chosen: every field here comes off disk.  It is a frozen
    dataclass because it is a static field on the module, and JAX hashes the
    treedef to key its jit cache.
    """

    # These must match the trainer's NCSNppConfig defaults, and a test there
    # asserts they do: a field absent from a config.json falls back to them,
    # and two different fallbacks would build two different models from one
    # file.
    in_channels: int = 1
    nf: int = 32
    ch_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_blocks: int = 2
    activation: str = "silu"
    init_scale: float = 1e-2
    fourier_scale: float = 0.02
    fourier_seed: int = 0
    fir: bool = True
    fir_kernel: tuple[int, ...] = (1, 3, 3, 1)
    skip_rescale: bool = True
    progressive: str = "output_skip"
    progressive_input: str = "input_skip"
    combine_method: str = "cat"
    attention: bool = True
    #: ``"sum"`` is ``sum(h)/sigma``; ``"dae"`` is
    #: ``(sigma_d^2+sigma^2)/(2 sigma^2) * ||h||^2``.  Read only when the
    #: checkpoint's architecture is ``"ncsnpp_energy"``.
    energy_form: str = "sum"

    def check(self) -> None:
        if self.progressive not in ("output_skip", "none"):
            raise ValueError(f"unknown progressive {self.progressive!r}")
        if self.progressive_input not in ("input_skip", "none"):
            raise ValueError(f"unknown progressive_input {self.progressive_input!r}")
        if self.combine_method not in ("cat", "sum"):
            raise ValueError(f"unknown combine_method {self.combine_method!r}")
        if self.energy_form not in ("sum", "dae"):
            raise ValueError(f"unknown energy_form {self.energy_form!r}")
        if not self.ch_mult:
            raise ValueError("ch_mult needs at least one resolution")
        if self.nf % 2:
            raise ValueError(f"nf must be even, got {self.nf}")

    @property
    def n_levels(self) -> int:
        return len(self.ch_mult)

    @property
    def size_divisor(self) -> int:
        """The grid must be a multiple of this: it halves once per level."""
        return 2 ** (self.n_levels - 1)

    @property
    def loss_margin(self) -> int:
        return 0

    @classmethod
    def from_dict(cls, d: dict) -> "NCSNppConfig":
        """Only the fields this loader knows about.

        A config written by a newer trainer may carry more; anything unknown is
        ignored rather than refused, because a field that does not change the
        *architecture* should not stop a model loading.  A field that does
        change it will show up as a shape mismatch, loudly.

        Tuples, not lists: these are static pytree fields and must be hashable.
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}
        raw = {k: v for k, v in d.items() if k in known}
        for k in ("ch_mult", "fir_kernel"):
            if k in raw and raw[k] is not None:
                raw[k] = tuple(raw[k])
        return cls(**raw)
