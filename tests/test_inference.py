"""Loading a trained prior and using it -- the surface a forward model sees.

Everything here is about the *seam*, not about the model: that a checkpoint is
self-describing, that the four things which have to agree come back together,
and that the two flux maps are not quietly swapped for one another.
"""

import dataclasses
import subprocess
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rubin_host_prior.config import (Config, NCSNppConfig, PatchConfig,
                                     SDEConfig)
from rubin_host_prior.inference import Prior, load_prior
from rubin_host_prior.training.checkpoint import save_checkpoint

SIZE = 32


def _write(tmp_path, architecture="ncsnpp_energy", **ncsnpp) -> Path:
    c = Config(architecture=architecture)
    c.ncsnpp = NCSNppConfig(nf=8, ch_mult=(1, 2, 2), num_blocks=1, **ncsnpp)
    c.patch = PatchConfig(native_size=128, out_size=SIZE, pool_factor=2)
    c.sde = SDEConfig(sigma_min=0.02, sigma_max=5.0, data_mean=3.0, data_std=0.7)
    c.transform = dataclasses.replace(c.transform, softening=5.0)
    model = c.build_model(jax.random.key(0))
    d = tmp_path / architecture
    save_checkpoint(d, 12345, c, model, model)
    return d


@pytest.fixture(scope="module")
def prior(tmp_path_factory) -> Prior:
    return load_prior(_write(tmp_path_factory.mktemp("ckpt")))


def test_a_checkpoint_describes_itself(prior):
    """Nothing has to be supplied again at the call site, which is the point:
    a prior whose schedule or transform were restated elsewhere could be used
    with the wrong ones and nothing would say so."""
    assert prior.step == 12345
    assert prior.size == SIZE
    assert prior.pool_factor == 2
    assert prior.sigma_min == 0.02 and prior.sigma_max == 5.0
    assert prior.transform.softening == 5.0
    assert prior.architecture == "ncsnpp_energy"
    assert prior.is_energy
    assert "32x32" in prior.summary()


def test_the_score_is_minus_the_gradient_of_the_energy_it_reports(prior):
    """The two methods are not independently derived -- but a forward model is
    entitled to assume they agree, and this is where that is checked."""
    x = jnp.asarray(np.random.default_rng(2).normal(3.0, 1.0, (1, SIZE, SIZE)),
                    jnp.float32)
    got = -jax.grad(lambda a: prior.energy(a, 1.0))(x)
    np.testing.assert_allclose(np.asarray(got), np.asarray(prior.score(x, 1.0)),
                               atol=1e-6)


@pytest.mark.parametrize("shape,energy_shape",
                         [((SIZE, SIZE), ()), ((1, SIZE, SIZE), ()),
                          ((4, 1, SIZE, SIZE), (4,))])
def test_shapes_come_back_as_they_went_in(prior, shape, energy_shape):
    x = jnp.asarray(np.random.default_rng(1).normal(3.0, 1.0, shape), jnp.float32)
    assert prior.score(x).shape == shape
    assert np.shape(prior.energy(x)) == energy_shape
    assert np.shape(prior.log_prob(x)) == energy_shape


def test_the_pixel_offset_reaches_the_model_through_both_wrappers(prior):
    """``energy`` and ``log_prob`` thread it to the same place, and the score
    -- which a sampler is calling in the same loop -- does not change."""
    x = jnp.asarray(np.random.default_rng(7).normal(3.0, 1.0, (2, 1, SIZE, SIZE)),
                    jnp.float32)
    k, n = 0.4, SIZE * SIZE

    plain = np.asarray(prior.energy(x, 1.0))
    shifted = np.asarray(prior.energy(x, 1.0, pixel_offset=k))
    np.testing.assert_allclose(plain - shifted, np.full(2, k * n), rtol=1e-4)
    np.testing.assert_allclose(np.asarray(prior.log_prob(x, 1.0, pixel_offset=k)),
                               -shifted, rtol=1e-6)

    got = -jax.grad(lambda a: prior.energy(a, 1.0, pixel_offset=k).sum())(x)
    np.testing.assert_allclose(np.asarray(got), np.asarray(prior.score(x, 1.0)),
                               atol=1e-6)


def test_a_wrong_rank_says_what_it_wanted(prior):
    with pytest.raises(ValueError, match=r"\(H, W\), \(C, H, W\) or"):
        prior.score(jnp.zeros((2, 3, 1, SIZE, SIZE)))


def test_the_two_flux_maps_are_different_and_both_right(prior):
    """``to_flux`` is ``exp(x)``, the map a forward model wants, and it is NOT
    the inverse of ``to_x``: the round trip gives ``soften(f)``.  Confusing the
    two is a quiet error -- they agree above ~10 softening scales, which is most
    of a bright galaxy and none of the sky."""
    flux = np.random.default_rng(0).normal(50.0, 20.0, (SIZE, SIZE))
    x = prior.to_x(flux)

    # the exact inverse recovers the measurement, negative sky included
    assert flux.min() < 0
    np.testing.assert_allclose(np.asarray(prior.to_flux_exact(x)), flux, atol=1e-9)

    # the model map recovers the softened flux instead, exactly
    np.testing.assert_allclose(np.asarray(prior.to_flux(x)),
                               np.asarray(prior.transform.soften(flux)), atol=1e-9)
    # ... and they differ by more than round-off where the flux is faint
    assert np.abs(np.asarray(prior.to_flux(x)) - flux).max() > 1.0
    # a scene from the prior can never be negative, which is why exp is the map
    assert np.all(np.asarray(prior.to_flux(x)) > 0)


def test_sampling_uses_the_priors_own_grid(prior):
    x = prior.sample(jax.random.key(0), n=2, n_steps=4)
    assert x.shape == (2, prior.model.in_channels, SIZE, SIZE)
    assert np.all(np.isfinite(np.asarray(x)))


def test_a_score_model_refuses_to_pretend_it_has_an_energy(tmp_path):
    """``architecture="ncsnpp"`` is not the gradient of anything, so asking for
    a potential has to fail loudly -- an HMC built on a silent zero would run
    and be wrong."""
    p = load_prior(_write(tmp_path, architecture="ncsnpp"))
    assert not p.is_energy
    x = jnp.zeros((1, SIZE, SIZE))
    assert p.score(x).shape == x.shape  # the score still works
    with pytest.raises(TypeError, match="ncsnpp_energy"):
        p.energy(x)
    with pytest.raises(TypeError):
        p.log_prob(x)


def test_loading_pulls_in_no_optional_dependency(tmp_path):
    """The extras are the LSST stack and matplotlib.  A forward model that
    imports this should need neither, and that is easy to break by adding one
    import to a package ``__init__`` -- so it is checked rather than asserted."""
    d = _write(tmp_path)
    code = (
        "import sys, json\n"
        "from rubin_host_prior.inference import load_prior\n"
        f"p = load_prior({str(d)!r})\n"
        "p.score(__import__('jax').numpy.zeros((1, 1, 32, 32)))\n"
        "heavy = {'matplotlib', 'astropy', 'pandas', 'pyarrow', 'pyvo', "
        "'requests', 'lsst', 'yaml'}\n"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules} & heavy)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("[]"), out.stdout


def test_the_top_level_import_stays_light():
    """``import rubin_host_prior`` reaches neither h5py nor optax, so something
    that only wants ``Config`` or ``geometry`` pays for neither.  ``load_prior``
    is exposed there lazily for that reason."""
    code = (
        "import sys, json, rubin_host_prior\n"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules} "
        "& {'h5py', 'optax'})))\n"
        "assert callable(rubin_host_prior.load_prior)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.splitlines()[0] == "[]", out.stdout
