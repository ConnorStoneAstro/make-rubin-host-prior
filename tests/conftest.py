import jax
import pytest

from rubin_host_prior.config import EnergyConfig
from rubin_host_prior.nn import ConvEnergyNet


#: The architecture every fixture in the suite uses.  Deliberately *not* the
#: project's default.
#:
#: The tests are not a rehearsal of the production geometry -- that is what
#: ``scripts/smoke_test.py`` is for -- and inheriting it cost twice over.  The
#: default loss margin is 72, so every batch the loader built came to
#: ``out_size + 144`` px and the suite went from four minutes to nine.  And the
#: tests broke each time the default architecture moved, which it has twice,
#: for reasons that had nothing to do with what they were testing.
#:
#: R = 2 here, so the loader carries 4 px of context instead of 72.
TINY_ENERGY = EnergyConfig(channels=((8, 8),), dilations=((1, 1),),
                           embed_dim=16, n_fourier=8)


@pytest.fixture(scope="session")
def tiny_config():
    """Two layers: R = 2, margin = 4, so a 16x16 input leaves an 8x8 interior."""
    return TINY_ENERGY


@pytest.fixture(scope="session")
def tiny_model(tiny_config):
    return ConvEnergyNet(tiny_config, key=jax.random.key(0))
