"""VE SDE, the denoising loss, and the samplers."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rubin_host_prior.diffusion import (
    VESDE,
    crop_interior,
    dsm_loss,
    dsm_loss_by_sigma,
    pflow_sample,
    reverse_sde_sample,
    sample_interior,
)


class GaussianEnergy(eqx.Module):
    """Exact energy for data ``N(0, tau^2 I)``.

    ``p_sigma = N(0, (tau^2 + sigma^2) I)``, so the analytic score is
    ``-x / (tau^2 + sigma^2)``.  Gives every sampler a target with a known
    answer.
    """

    tau: float = eqx.field(static=True)

    def __call__(self, x, sigma):
        return 0.5 * jnp.sum(x**2) / (self.tau**2 + sigma**2)


def test_sigma_schedule_endpoints_and_monotonicity():
    sde = VESDE(0.02, 20.0)
    assert float(sde.sigma(0.0)) == pytest.approx(0.02)
    assert float(sde.sigma(1.0)) == pytest.approx(20.0)
    t = jnp.linspace(0, 1, 50)
    assert np.all(np.diff(np.asarray(sde.sigma(t))) > 0)
    np.testing.assert_allclose(np.asarray(sde.t_of_sigma(sde.sigma(t))), t, atol=1e-5)


def test_g2_is_the_derivative_of_sigma_squared():
    sde = VESDE(0.05, 8.0)
    t, h = 0.4, 1e-4
    numeric = (float(sde.sigma(t + h)) ** 2 - float(sde.sigma(t - h)) ** 2) / (2 * h)
    assert float(sde.g2(t)) == pytest.approx(numeric, rel=1e-4)


def test_sample_sigma_is_log_uniform():
    sde = VESDE(0.01, 10.0)
    s = np.asarray(sde.sample_sigma(jax.random.key(0), (200_000,)))
    assert s.min() >= sde.sigma_min and s.max() <= sde.sigma_max
    u = (np.log(s) - np.log(sde.sigma_min)) / (
        np.log(sde.sigma_max) - np.log(sde.sigma_min)
    )
    # Uniform in t means equal training weight per decade of noise.
    assert u.mean() == pytest.approx(0.5, abs=0.01)
    assert u.std() == pytest.approx(1 / np.sqrt(12), abs=0.01)


def test_perturb_has_the_requested_std():
    sde = VESDE()
    x = jnp.zeros((4096, 1, 8, 8))
    sigma = jnp.full((4096,), 0.37)
    noisy, eps = sde.perturb(jax.random.key(1), x, sigma)
    assert float(jnp.std(noisy)) == pytest.approx(0.37, rel=0.02)
    np.testing.assert_allclose(np.asarray(noisy), 0.37 * np.asarray(eps), rtol=1e-5)


def test_crop_interior_shapes_and_refusal():
    a = jnp.zeros((2, 1, 10, 10))
    assert crop_interior(a, 3).shape == (2, 1, 4, 4)
    assert crop_interior(a, 0).shape == a.shape
    with pytest.raises(ValueError, match="no interior"):
        crop_interior(a, 5)


def test_loss_is_one_for_a_zero_score(tiny_model):
    """With score == 0 the residual is eps, so the loss is E||eps||^2 == 1.

    A useful anchor: any trained model should sit below 1, and a model stuck at
    exactly 1 has learned nothing.
    """
    x = jax.random.normal(jax.random.key(2), (32, 1, 24, 24))

    class Zero(eqx.Module):
        loss_margin: int = eqx.field(static=True, default=4)

        def __call__(self, x, sigma):
            return jnp.asarray(0.0)

    assert float(dsm_loss(Zero(), x, jax.random.key(3), VESDE(), margin=4)) == (
        pytest.approx(1.0, abs=0.05)  # MC error ~ sqrt(2/8192)
    )


def test_analytic_score_beats_a_zero_score():
    """The loss must actually prefer the true score."""
    sde = VESDE(0.01, 10.0)
    tau = 0.8
    x = tau * jax.random.normal(jax.random.key(4), (256, 1, 16, 16))
    good = float(dsm_loss(GaussianEnergy(tau=tau), x, jax.random.key(5), sde, margin=0))

    class Zero(eqx.Module):
        def __call__(self, x, sigma):
            return jnp.asarray(0.0)

    bad = float(dsm_loss(Zero(), x, jax.random.key(5), sde, margin=0))
    assert good < bad
    # For N(0, tau^2) the optimal residual variance is tau^2/(tau^2+sigma^2),
    # averaged over the log-uniform schedule -- well under 1.
    assert good < 0.9


def test_loss_by_sigma_resolves_the_schedule():
    sde = VESDE(0.01, 10.0)
    tau = 0.8
    x = tau * jax.random.normal(jax.random.key(6), (64, 1, 32, 32))
    sigmas = sde.sigma(jnp.linspace(0.0, 1.0, 64))
    per = np.asarray(
        dsm_loss_by_sigma(
            GaussianEnergy(tau=tau), x, sigmas, jax.random.key(7), sde, margin=0
        )
    )
    assert per.shape == (64,)
    # For data N(0, tau^2) the optimal residual variance is exactly
    # tau^2 / (tau^2 + sigma^2).  Note which way round that runs: it tends to 1
    # as sigma -> 0 (tiny added noise is unidentifiable, so the best denoiser
    # learns nothing about eps) and to 0 as sigma -> infinity (x is then almost
    # pure noise, so eps is recoverable).  The loss curve against sigma therefore
    # *falls* with sigma for a well-fit model.
    expected = tau**2 / (tau**2 + np.asarray(sigmas) ** 2)
    deviation = np.abs(per - expected)
    # Each point is one example over a 32x32 interior, so its own Monte Carlo
    # error is ~sqrt(2/1024) = 4.4%.  The mean is the tight constraint; the max
    # over 64 points is allowed a few sigma.
    assert deviation.mean() < 0.05, deviation.mean()
    assert deviation.max() < 0.25, deviation.max()
    assert per[0] > 0.9 and per[-1] < 0.1


@pytest.mark.parametrize(
    "sampler,kwargs,tol",
    [
        (pflow_sample, {"n_steps": 128}, 0.02),
        (reverse_sde_sample, {"n_steps": 400, "n_corrector": 0}, 0.02),
        # The Langevin corrector has a known +O(snr^4) variance bias; see
        # reverse_sde_sample's docstring. snr=0.05 keeps it under 0.5%.
        (reverse_sde_sample, {"n_steps": 400, "n_corrector": 1, "snr": 0.05}, 0.02),
    ],
)
def test_samplers_recover_an_analytic_gaussian(sampler, kwargs, tol):
    tau, sde = 0.7, VESDE(0.01, 10.0)
    x = np.asarray(
        sampler(
            GaussianEnergy(tau=tau), jax.random.key(8), (256, 1, 32, 32), sde, **kwargs
        )
    )
    assert float(np.mean(x)) == pytest.approx(0.0, abs=0.02)
    assert float(np.std(x)) == pytest.approx(tau, rel=tol)


def test_corrector_bias_grows_with_snr():
    """Pins the documented +O(snr^4) bias of the Langevin corrector.

    Not a defect to be fixed, but a discretisation property worth having a test
    on: if it ever grows sharply, the step-size heuristic has regressed.
    """
    tau, sde = 0.7, VESDE(0.01, 10.0)
    excess = []
    for snr in (0.02, 0.10, 0.25):
        x = reverse_sde_sample(
            GaussianEnergy(tau=tau), jax.random.key(8), (256, 1, 32, 32), sde,
            n_steps=400, n_corrector=1, snr=snr,
        )
        excess.append(float(jnp.std(x)) / tau - 1.0)
    assert excess[0] < excess[1] < excess[2]
    assert abs(excess[0]) < 0.005, "bias should vanish as snr -> 0"
    assert excess[2] < 0.06


def test_pflow_heun_beats_euler_at_equal_steps():
    tau, sde = 0.7, VESDE(0.01, 10.0)
    errs = {}
    for heun in (False, True):
        x = pflow_sample(
            GaussianEnergy(tau=tau), jax.random.key(9), (512, 1, 6, 6), sde,
            n_steps=16, heun=heun,
        )
        errs[heun] = abs(float(jnp.std(x)) - tau)
    assert errs[True] < errs[False]


def test_sample_interior_returns_the_padded_middle(tiny_model):
    """Sampling must discard the border, where the score is not correct."""
    out = sample_interior(
        tiny_model, jax.random.key(10), out_size=12, n_samples=2,
        sde=VESDE(), sampler="pflow", n_steps=4,
    )
    assert out.shape == (2, 1, 12, 12)


def test_unknown_sampler_rejected(tiny_model):
    with pytest.raises(ValueError, match="unknown sampler"):
        sample_interior(tiny_model, jax.random.key(11), 12, sampler="nope")
