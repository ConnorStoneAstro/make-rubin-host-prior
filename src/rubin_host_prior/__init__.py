"""A fully convolutional energy-based diffusion prior on Rubin host-galaxy scenes.

The pieces, in the order they are used:

``rubin``      DP2 extraction: host selection, jittered cutouts, the artefact gate.
``data``       Shards, the log-space flux transform, pooling, augmentation, loader.
``nn``         The valid-convolution energy network; ``score`` is ``-grad_x E``.
``diffusion``  VE SDE, denoising score matching, samplers.
``training``   Training loop, EMA, checkpoints.
``geometry``   Shape arithmetic -- read this before choosing a patch size.

Two design choices drive most of the rest:

*An energy model, not a score network.*  The network returns a scalar and the
score is its exact gradient, so the score is a conservative field by
construction -- a genuine score rather than an approximation to one.  The cost is
that each training step differentiates through a gradient.

*Valid convolutions only.*  No zero padding, hence no border artefacts and no
dependence on patch size; the model runs on any scene above a minimum size.  The
cost is that the summed energy under-weights pixels near the edge, so the loss
is restricted to an interior window and sampling needs a padded canvas.  See
``geometry``.
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

__all__ = [
    "BANDS",
    "AugmentConfig",
    "Config",
    "EnergyConfig",
    "PatchConfig",
    "SDEConfig",
    "TrainConfig",
    "TransformConfig",
    "geometry",
]
