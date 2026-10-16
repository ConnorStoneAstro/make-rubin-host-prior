"""Load a trained Rubin host-galaxy diffusion prior.  Nothing else.

    from host_prior import load_prior

    prior = load_prior("runs/ncsnpp-energy/final")
    x = prior.to_x(flux_njy)        # nJy -> the model's space
    u = prior.energy(x)             # scalar: HMC's Hamiltonian
    g = prior.score(x)              # -grad_x E: its leapfrog

**Vendored on purpose.**  This is a copy of just enough of ``rubin-host-prior``
to rebuild a trained model, so an inference package can use the prior without
depending on the training and extraction stack.  Dependencies: ``jax``,
``equinox``, ``numpy``.  Nothing else -- no h5py, no optax, no matplotlib, no
astropy, nothing from the LSST stack.

``_net.py`` is **generated**, byte for byte, from the trainer's architecture
file; the module field declarations in it are the checkpoint format, so their
order and static-ness decide how equinox lays out the pytree.  Do not edit it.
If the trainer's architecture changes, re-export it and copy it across --
``scripts/export_standalone.py --check`` in that repo fails when the two differ,
and ``tests/test_standalone.py`` there loads a checkpoint through both paths and
compares the numbers.

Rename the package to taste; nothing inside refers to it by name.
"""

from ._prior import LogFluxTransform, Prior, VESDE, load_prior, pflow_sample

__all__ = ["LogFluxTransform", "Prior", "VESDE", "load_prior", "pflow_sample"]
