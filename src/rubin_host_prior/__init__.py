"""A diffusion prior over Rubin host-galaxy scenes, for use in a forward model.

**To use a trained prior, you want one import:**

    from rubin_host_prior.inference import load_prior
    prior = load_prior("runs/ncsnpp-energy/final")

See ``inference`` -- it holds the model, the sigma schedule and the flux
transform together, because those three have to agree and a prior whose
transform is restated at the call site is a prior that can be used wrongly.

The pieces, in the order they are used:

``rubin``      DP2 extraction: host selection, host-centred cutouts, the artefact gate.
``data``       Shards, the log-space flux transform, pooling, augmentation, loader.
``nn``         The score networks; ``nn.score`` is the interface they share.
``diffusion``  VE SDE, denoising score matching, samplers.
``training``   Training loop, EMA, checkpoints.
``inference``  Loading a trained prior and using it.
``geometry``   Shape arithmetic for the dilated architecture.

**Three architectures**, chosen by ``Config.architecture``:

``energy``         dilated convolutions summed to a scalar; score is ``-grad_x E``.
``ncsnpp``         an NCSN++ U-Net predicting ``sigma * score`` directly.
``ncsnpp_energy``  the same U-Net read as a scalar energy.

The distinction that matters downstream is whether the score is the gradient of
something.  It is for ``energy`` and ``ncsnpp_energy``, which therefore expose a
potential; it is not for ``ncsnpp``, which is cheaper to train and cannot be
used with any sampler that needs a Hamiltonian -- HMC among them, since its
accept/reject step is on the energy and not on the score.

**Every architecture here is size-locked.**  The convolutions are same-mode and
zero-padded, so the training grid is part of the operator: a model trained on
128x128 is a prior over 128x128 scenes and is not defined on any other size.
This replaced valid convolutions, which were size-agnostic but needed ``2R``
pixels of real sky on every side of the loss region -- at ``R = 78`` that is 156
px per side, which no 512 px stamp can supply.
"""

from . import geometry
from .config import (
    BANDS,
    AugmentConfig,
    Config,
    EnergyConfig,
    PatchConfig,
    SDEConfig,
    TrainConfig,
    TransformConfig,
)

__version__ = "0.1.0"


def __getattr__(name):
    """``load_prior`` and ``Prior`` without importing them eagerly.

    ``inference`` reaches ``data`` and ``training``, which pull in h5py and
    optax; the top-level import is currently free of both and worth keeping
    that way for anything that only wants ``Config`` or ``geometry``.
    """
    if name in ("Prior", "load_prior"):
        from . import inference

        return getattr(inference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BANDS",
    "AugmentConfig",
    "Config",
    "EnergyConfig",
    "PatchConfig",
    "Prior",
    "SDEConfig",
    "TrainConfig",
    "TransformConfig",
    "geometry",
    "load_prior",
]
