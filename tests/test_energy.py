"""Properties the energy model must have for its gradient to be a real score."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rubin_host_prior.config import EnergyConfig
from rubin_host_prior.nn import ConvEnergyNet, batched_score, n_parameters, score
from rubin_host_prior.nn.layers import ACTIVATIONS, get_activation


def test_energy_is_scalar_and_score_matches_input_shape(tiny_model):
    x = jax.random.normal(jax.random.key(0), (1, 20, 20))
    assert tiny_model(x, jnp.asarray(0.5)).shape == ()
    assert score(tiny_model, x, jnp.asarray(0.5)).shape == x.shape


def test_accepts_any_size_above_the_minimum(tiny_model):
    """The point of valid convolutions: no dependence on the input size."""
    for shape in [(1, 16, 16), (1, 64, 64), (1, 31, 47)]:
        x = jax.random.normal(jax.random.key(1), shape)
        assert np.isfinite(float(tiny_model(x, jnp.asarray(1.0))))
        assert score(tiny_model, x, jnp.asarray(1.0)).shape == shape


def test_score_is_conservative(tiny_model):
    """The Jacobian of the score is symmetric.

    This is the whole reason for an energy model: the score is an exact gradient,
    so its Jacobian is minus the Hessian of a scalar and is therefore symmetric.
    A freely parameterised score network satisfies this only approximately, which
    means the implied log-density is path-dependent.
    """
    size = 12
    x = jax.random.normal(jax.random.key(2), (1, size, size)) * 0.5

    def flat_score(v):
        return score(tiny_model, v.reshape(1, size, size), jnp.asarray(0.7)).ravel()

    jac = np.asarray(jax.jacfwd(flat_score)(x.ravel()))
    asym = np.abs(jac - jac.T).max()
    assert asym < 1e-4 * max(np.abs(jac).max(), 1e-8), f"asymmetry {asym}"


def test_score_is_translation_equivariant_on_the_interior(tiny_model):
    """Rolling the scene rolls the score, away from the border.

    Valid convolutions and a summed energy give exact translation equivariance;
    only the finite border breaks it, which is why the loss is cropped.
    """
    size, margin, shift = 30, tiny_model.loss_margin, 3
    x = jax.random.normal(jax.random.key(3), (1, size, size))
    s0 = np.asarray(score(tiny_model, x, jnp.asarray(1.0)))
    s1 = np.asarray(score(tiny_model, jnp.roll(x, shift, axis=-1), jnp.asarray(1.0)))
    m = margin + shift
    np.testing.assert_allclose(
        np.roll(s0, shift, axis=-1)[:, m:-m, m:-m], s1[:, m:-m, m:-m],
        rtol=2e-3, atol=1e-6,
    )


def test_energy_is_extensive_in_scene_area(tiny_model):
    """Tiling a scene multiplies the energy, because the energy is a sum.

    Not a bug -- it is what makes the model a translation-invariant prior over
    scenes of any size. But it does mean energies are only comparable between
    scenes of equal size.
    """
    x = jax.random.normal(jax.random.key(4), (1, 40, 40))
    e_one = float(tiny_model(x, jnp.asarray(1.0)))
    e_map = tiny_model.energy_map(x, jnp.asarray(1.0))
    assert e_one == pytest.approx(float(jnp.sum(e_map)), rel=1e-5)


def test_sigma_conditioning_is_spatially_constant():
    """FiLM must not introduce any spatial aggregation.

    If it did, the network's response would depend on the patch size and the
    "runs on any scene" property would quietly be false.  Test: changing sigma
    changes the energy map by an amount that is uniform once the (translation
    invariant) input is uniform.
    """
    model = ConvEnergyNet(
        EnergyConfig(channels=((6, 8),), dilations=((1, 1),), embed_dim=16, n_fourier=8, sigma_scaling="none"),
        key=jax.random.key(5),
    )
    x = jnp.full((1, 24, 24), 0.2)
    a = np.asarray(model.energy_map(x, jnp.asarray(0.1)))[0]
    b = np.asarray(model.energy_map(x, jnp.asarray(3.0)))[0]
    diff = b - a
    assert np.allclose(diff, diff.flat[0], rtol=1e-4)
    assert abs(diff.flat[0]) > 0, "sigma has no effect at all"


def test_energy_depends_on_sigma(tiny_model):
    x = jax.random.normal(jax.random.key(6), (1, 20, 20))
    values = [float(tiny_model(x, jnp.asarray(s))) for s in (0.05, 0.5, 5.0)]
    assert len({round(v, 9) for v in values}) == 3


def test_inverse_sigma_scaling_flattens_the_loss_residual_across_sigma():
    """``E = E~ / sigma`` is what lets one set of weights cover decades of sigma.

    The quantity the loss compares against ``eps`` is ``sigma * score``, and
    ``eps`` is O(1) at every noise level.  With ``E = E~ / sigma`` we get
    ``sigma * score = -grad E~``, which carries no schedule-wide trend; without
    it, ``sigma * score`` spans as many decades as sigma does, and one set of
    weights has to cover all of them.

    Note this is the opposite of what the raw score magnitudes do -- those span
    *more* decades under the scaling, and correctly so, since the true score
    really does grow like 1/sigma as sigma falls.
    """
    x = jax.random.normal(jax.random.key(7), (1, 24, 24))
    sigmas = [0.01, 0.1, 1.0, 10.0]
    spans = {}
    for mode in ("none", "inverse_sigma"):
        m = ConvEnergyNet(
            EnergyConfig(channels=((6, 8),), dilations=((1, 1),), embed_dim=16, n_fourier=8,
                         sigma_scaling=mode),
            key=jax.random.key(8),
        )
        mags = np.array([
            float(s * jnp.abs(score(m, x, jnp.asarray(s))).mean()) for s in sigmas
        ])
        spans[mode] = mags.max() / max(mags.min(), 1e-30)
    assert spans["inverse_sigma"] < 0.05 * spans["none"], spans
    assert spans["inverse_sigma"] < 10.0, spans


def test_batched_score_matches_the_loop(tiny_model):
    x = jax.random.normal(jax.random.key(9), (3, 1, 18, 18))
    sig = jnp.array([0.1, 1.0, 4.0])
    batched = np.asarray(batched_score(tiny_model, x, sig))
    for i in range(3):
        np.testing.assert_allclose(
            batched[i], np.asarray(score(tiny_model, x[i], sig[i])),
            rtol=1e-4, atol=1e-7,  # float32 reassociation under vmap
        )


def test_activations_are_smooth():
    """A C^0 activation gives a discontinuous score. Reject them by name."""
    assert "relu" not in ACTIVATIONS
    assert "leaky_relu" not in ACTIVATIONS
    with pytest.raises(ValueError, match="non-smooth|unknown"):
        get_activation("relu")


def test_head_has_no_bias(tiny_model):
    """A constant added to the energy is invisible to the score: pure gauge."""
    assert all(b.head.bias is None for b in tiny_model.branches)


def test_fourier_frequencies_are_not_trainable(tiny_model):
    """They are a static field, so weight decay cannot shrink them away."""
    leaves = jax.tree_util.tree_leaves(
        eqx.filter(tiny_model.embed.fourier, eqx.is_inexact_array)
    )
    assert leaves == []
    assert isinstance(tiny_model.embed.fourier.freqs, tuple)


def test_all_parameters_receive_gradients_at_step_zero(tiny_model):
    """Zero-initialising the head or the FiLM projection would break this."""
    from rubin_host_prior.diffusion import VESDE, dsm_loss

    x = jax.random.normal(jax.random.key(10), (4, 1, 24, 24))
    grads = eqx.filter_grad(
        lambda m: dsm_loss(m, x, jax.random.key(11), VESDE())
    )(tiny_model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    dead = [i for i, g in enumerate(leaves) if not bool(jnp.any(g != 0))]
    assert dead == [], f"leaves {dead} of {len(leaves)} got no gradient"


def test_residual_blocks_preserve_valid_shapes():
    model = ConvEnergyNet(
        EnergyConfig(channels=((8, 8, 8),), dilations=((1, 1, 1),), embed_dim=16, n_fourier=8, residual=True),
        key=jax.random.key(12),
    )
    x = jax.random.normal(jax.random.key(13), (1, 20, 20))
    assert model.energy_map(x, jnp.asarray(1.0)).shape == (1, 14, 14)
    assert n_parameters(model) > 0


# -- dilation and summed branches ------------------------------------------
#
# Two claims are being pinned here.  That a sum of branches is still one energy,
# so the score stays an exact gradient however many there are.  And that dilation
# buys reach without breaking translation equivariance -- which is the whole
# reason the long-range branch is dilated rather than pooled: a stack with total
# stride j is invariant only to shifts that are multiples of j, and the resulting
# bias is locked to the pixel lattice rather than to the image, so the other
# branch cannot cancel it and a sampling run accumulates it coherently.


def _branched(fine=(1, 1), coarse=(1, 2, 4, 1), width=4, key=0):
    cfg = EnergyConfig(
        channels=((width,) * len(fine), (width,) * len(coarse)),
        dilations=(fine, coarse),
        embed_dim=8,
        n_fourier=4,
    )
    return ConvEnergyNet(cfg, key=jax.random.key(key))


def _mute(model, i):
    """The same model with branch ``i``'s head zeroed, so it contributes no
    energy at all -- the clean way to isolate one branch's share."""
    return eqx.tree_at(
        lambda m: m.branches[i].head.weight,
        model,
        jnp.zeros_like(model.branches[i].head.weight),
    )


def test_a_sum_of_branches_is_still_one_energy():
    """E = E0 + E1 exactly, so -grad E is the sum of the branches' scores and
    every conservativity guarantee carries over unchanged."""
    model = _branched()
    x = jax.random.normal(jax.random.key(3), (1, 40, 40))
    sigma = jnp.asarray(1.0)

    both = float(model(x, sigma))
    only0 = float(_mute(model, 1)(x, sigma))
    only1 = float(_mute(model, 0)(x, sigma))
    assert both == pytest.approx(only0 + only1, rel=1e-5)

    s_both = np.asarray(score(model, x, sigma))
    s0 = np.asarray(score(_mute(model, 1), x, sigma))
    s1 = np.asarray(score(_mute(model, 0), x, sigma))
    assert s_both == pytest.approx(s0 + s1, rel=1e-4, abs=1e-7)


def test_the_short_branch_is_cropped_onto_the_long_one():
    """Both branches contribute a map of the same size, so R is the longest
    branch's and not the sum of anything."""
    model = _branched()
    assert model.receptive_radius == 8  # 1+2+4+1
    assert model.branches[0].radius == 2 and model.branches[1].radius == 8
    emap = model.energy_map(jnp.zeros((1, 40, 40)), jnp.asarray(1.0))
    assert emap.shape == (1, 40 - 16, 40 - 16)


def test_the_long_branch_actually_reaches_further():
    """Functional, not arithmetic.

    Reach is a statement about the *score*, not the energy: ``dE/dx_i`` depends
    on ``x_j`` only for ``|i - j| <= 2R``, so the Hessian of the energy is banded
    at exactly twice the radius.  Muting the long branch must shrink that band
    from ``2*8`` to ``2*2``, which is the claim "the long branch reaches
    further" with nothing left to interpret.
    """
    model = _branched()
    size = 41
    sigma = jnp.asarray(1.0)

    def bandwidth(m):
        def centre_score(xx):
            return score(m, xx, sigma)[0, size // 2, size // 2]

        g = np.asarray(jax.grad(centre_score)(jnp.zeros((1, size, size))))[0]
        yy, xx = np.mgrid[0:size, 0:size]
        reach = np.abs(g) > 1e-9
        return int(max(np.abs(yy[reach] - size // 2).max(),
                       np.abs(xx[reach] - size // 2).max()))

    assert bandwidth(_mute(model, 1)) == 2 * 2   # fine branch alone, R = 2
    assert bandwidth(model) == 2 * 8             # with the dilated branch, R = 8


def test_dilation_keeps_the_score_translation_equivariant():
    """Shift the scene by one pixel and the score shifts with it, exactly.

    This is what a pooled long-range branch would lose: with total stride j the
    energy is invariant only to shifts that are multiples of j, so a one-pixel
    shift changes the score by an amount tied to the lattice rather than to the
    image.  Here the agreement is to float32 round-off at every shift.
    """
    model = _branched()
    sigma = jnp.asarray(1.0)
    scene = jax.random.normal(jax.random.key(4), (1, 64, 64))
    base = np.asarray(score(model, scene, sigma))

    for shift in (1, 2, 3, 5):
        moved = np.asarray(score(model, jnp.roll(scene, shift, axis=-1), sigma))
        # Compare where neither crop nor the roll's wraparound interferes.
        m = model.loss_margin
        a = base[0, m:-m, m : -m - shift]
        b = moved[0, m:-m, m + shift : -m]
        assert a == pytest.approx(b, rel=2e-4, abs=1e-6), f"shift {shift}"
