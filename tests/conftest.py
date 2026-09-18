import jax
import pytest

from rubin_host_prior.config import EnergyConfig
from rubin_host_prior.nn import ConvEnergyNet


@pytest.fixture(scope="session")
def tiny_config():
    """Two layers: R = 2, margin = 4, so a 16x16 input leaves an 8x8 interior."""
    return EnergyConfig(channels=(6, 8), embed_dim=16, n_fourier=8)


@pytest.fixture(scope="session")
def tiny_model(tiny_config):
    return ConvEnergyNet(tiny_config, key=jax.random.key(0))
