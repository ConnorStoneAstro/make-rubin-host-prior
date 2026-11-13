"""The NCSN++ port: what it computes, and how it differs from the energy model.

The port itself was validated against the PyTorch reference it came from
(``AlexandreAdam/score_models``, ``dev``): weights copied module for module into
the JAX model, outputs compared at three noise levels across seven
configurations -- attention on and off, three and four resolutions, FIR and
naive resampling, both progressive modes, both combine methods, two and three
blocks per level.  Every one agreed to 1.6e-6 relative, i.e. float32 round-off.
That comparison needs torch and a checkout of the reference, so it cannot live
here; what lives here is everything that can be checked without them.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rubin_host_prior.config import Config, NCSNppConfig, PatchConfig, SDEConfig
from rubin_host_prior.diffusion import VESDE, dsm_loss, pflow_sample
from rubin_host_prior.nn import batched_score, n_parameters
from rubin_host_prior.nn.ncsnpp import (NCSNpp, fir_downsample, fir_upsample,
                                        naive_downsample)

SIZE = 32


@pytest.fixture(scope="module")
def tiny_config() -> Config:
    c = Config()
    c.architecture = "ncsnpp"
    c.ncsnpp = NCSNppConfig(nf=8, ch_mult=(1, 2, 2), num_blocks=1)
    c.patch = PatchConfig(native_size=128, out_size=SIZE, pool_factor=2)
    c.sde = SDEConfig(sigma_min=0.02, sigma_max=5.0, data_mean=0.0)
    c.transform = dataclasses.replace(c.transform, softening=1.0)
    return c


@pytest.fixture(scope="module")
def tiny_model(tiny_config) -> NCSNpp:
    return tiny_config.build_model(jax.random.key(0))


# -- FIR resampling --------------------------------------------------------


def test_fir_resampling_preserves_a_constant():
    """The gain normalisation, which is easy to get wrong by a factor of four.

    Interior only: the filter pads with zeros, so the outermost rows are pulled
    towards zero by construction -- the same border the convolutions have.
    """
    x = jnp.ones((1, 16, 16))
    for f, trim in ((fir_downsample, 2), (fir_upsample, 3)):
        interior = np.asarray(f(x))[0, trim:-trim, trim:-trim]
        np.testing.assert_allclose(interior, 1.0, atol=1e-6)


def test_fir_downsampling_kills_the_nyquist_stripe():
    """(1, 3, 3, 1) has an exact zero at the old Nyquist frequency.

    This is the point of FIR resampling.  Plain subsampling maps that stripe to
    a *constant* -- power that was at the finest scale reappears at the coarsest
    one -- which is exactly the kind of error that shows up as a wrong spectrum
    and a right-looking image.
    """
    stripe = jnp.asarray(
        ((-1.0) ** np.arange(16))[None, :, None] * np.ones((1, 16, 16)))
    subsampled = np.asarray(stripe[:, ::2, ::2])
    assert np.sqrt((subsampled ** 2).mean()) == pytest.approx(1.0)
    interior = np.asarray(fir_downsample(stripe))[0, 2:-2, 2:-2]
    np.testing.assert_allclose(interior, 0.0, atol=1e-6)


def test_fir_beats_a_box_filter_above_the_new_nyquist():
    """A 2x2 mean is an anti-alias filter, just a bad one."""
    wave = jnp.asarray(
        np.cos(2 * np.pi * np.arange(16) / 2.67)[None, :, None]
        * np.ones((1, 16, 16)))
    rms = lambda a: float(np.sqrt((np.asarray(a)[0, 2:-2, 2:-2] ** 2).mean()))
    assert rms(wave[:, ::2, ::2]) > 0.5           # aliased through at full amplitude
    assert 0.2 < rms(naive_downsample(wave)) < 0.4
    assert rms(fir_downsample(wave)) < 0.06       # ~7x better than the box


# -- the epsilon parameterisation ------------------------------------------


def test_score_is_the_output_divided_by_sigma(tiny_model):
    x = jax.random.normal(jax.random.key(1), (1, SIZE, SIZE))
    for sigma in (0.05, 1.0, 4.0):
        s = np.asarray(tiny_model.score(x, jnp.float32(sigma)))
        raw = np.asarray(tiny_model(x, jnp.float32(sigma)))
        np.testing.assert_allclose(s, raw / sigma, rtol=1e-5)


def test_the_networks_own_output_is_flat_in_sigma(tiny_model):
    """What the epsilon parameterisation buys.

    The score spans four decades across the schedule; the quantity the *weights*
    have to produce does not move at all.  Nothing in this network has to learn
    an amplitude that depends on the noise level -- the division does it.
    """
    x = jax.random.normal(jax.random.key(2), (1, SIZE, SIZE))
    sigmas = np.geomspace(0.02, 5.0, 12)
    raw = np.array([float(jnp.std(tiny_model(x * s, jnp.float32(s))))
                    for s in sigmas])
    assert raw.max() / raw.min() < 3.0
    scores = np.array([float(jnp.std(tiny_model.score(x * s, jnp.float32(s))))
                       for s in sigmas])
    assert scores.max() / scores.min() > 50.0


def test_t_of_sigma_spans_the_unit_interval(tiny_model):
    assert float(tiny_model.t_of_sigma(jnp.float32(0.02))) == pytest.approx(0.0, abs=1e-6)
    assert float(tiny_model.t_of_sigma(jnp.float32(5.0))) == pytest.approx(1.0, abs=1e-6)


def test_the_noise_embedding_is_smooth_in_sigma():
    """``fourier_scale = 0.02`` on ``t`` in [0, 1], and why that is small.

    The features are ``sin`` and ``cos`` of ``2*pi*w*x``.  Measure how far the
    unit embedding vector actually travels on its sphere as ``x`` sweeps the
    schedule: NCSN++ moves **0.14 radians in total**, so the embedding is nearly
    a straight line and neighbouring noise levels are neighbours in it.  The
    energy model's basis -- ``scale = 1.0`` on *unnormalised* ``log sigma`` --
    travels **46 radians**, seven full revolutions, and ends up with
    ``sigma_min`` and ``sigma_max`` essentially orthogonal (|cos| < 0.1).  It
    can represent any function of sigma; it cannot easily represent a smooth
    one, because nothing ties adjacent sigmas together.

    Lu & Song (arXiv:2410.11081) is the reference for why the small scale is
    deliberate rather than an oversight.
    """
    from rubin_host_prior.nn.layers import FourierFeatures

    def path_length(basis, xs):
        e = np.asarray(jax.vmap(basis)(xs))
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        return float(np.linalg.norm(np.diff(e, axis=0), axis=1).sum())

    def ends(basis, xs):
        e = np.asarray(jax.vmap(basis)(xs))
        return float(e[0] @ e[-1] / (np.linalg.norm(e[0]) * np.linalg.norm(e[-1])))

    t = jnp.linspace(0.0, 1.0, 400)
    log_sigma = jnp.linspace(np.log(0.01), np.log(20.0), 400)
    for seed in range(4):
        smooth = FourierFeatures(NCSNppConfig().nf // 2, 0.02, seed)
        assert path_length(smooth, t) < 0.5
        assert ends(smooth, t) > 0.98

        oscillatory = FourierFeatures(64, 1.0, seed)  # the energy model's basis
        assert path_length(oscillatory, log_sigma) > 20.0
        assert abs(ends(oscillatory, log_sigma)) < 0.2


def test_the_fourier_basis_is_frozen(tiny_model):
    """Static, not a leaf: the optimiser cannot reach it and AdamW cannot decay
    it towards zero.  Same rule as the energy model's -- see nn.layers."""
    leaves = jax.tree_util.tree_leaves(
        eqx.filter(tiny_model, eqx.is_inexact_array))
    freqs = np.array(tiny_model.time_fourier.freqs)
    assert not any(leaf.shape == freqs.shape and np.allclose(leaf, freqs)
                   for leaf in leaves)


# -- shape and structure ---------------------------------------------------


def test_the_scene_comes_back_the_size_it_went_in(tiny_model):
    x = jax.random.normal(jax.random.key(3), (4, 1, SIZE, SIZE))
    s = batched_score(tiny_model, x, jnp.full((4,), 1.0))
    assert s.shape == x.shape
    assert np.all(np.isfinite(np.asarray(s)))


def test_a_grid_that_does_not_halve_evenly_is_reported():
    c = Config()
    c.architecture = "ncsnpp"
    c.ncsnpp = NCSNppConfig(ch_mult=(1, 2, 2, 2))  # needs multiples of 8
    c.patch = PatchConfig(native_size=512, out_size=100, pool_factor=2)
    warnings = c.check_sizes()
    assert any("multiple of 8" in w for w in warnings), warnings


def test_progressive_can_be_switched_off(tiny_config):
    """The plain-U-Net ablation has to run, because it is the comparison that
    says whether the pyramid is what makes large scales appear."""
    c = dataclasses.replace(tiny_config)
    c.ncsnpp = dataclasses.replace(tiny_config.ncsnpp, progressive="none",
                                   progressive_input="none")
    model = c.build_model(jax.random.key(0))
    x = jax.random.normal(jax.random.key(4), (1, SIZE, SIZE))
    assert model(x, jnp.float32(1.0)).shape == x.shape


def test_the_upsampling_blocks_narrow_to_the_level_they_feed(tiny_model,
                                                            tiny_config):
    """A resampling block resamples before it convolves, so on the way up
    both of its 3x3s run on the *finer* grid.  Left at the coarser level's
    width, as the reference has it, the block into full resolution was 16% of
    the whole network at a width nothing else there uses."""
    cfg = tiny_config.ncsnpp
    widths = [cfg.nf * m for m in cfg.ch_mult]
    assert len(set(widths)) > 1, "needs a width step to mean anything"
    # ``up_levels`` runs coarsest first.
    for level, up in zip(reversed(range(len(widths))), tiny_model.up_levels):
        if level == 0:
            assert up.up is None
            continue
        fine, coarse = widths[level - 1], widths[level]
        assert up.up.conv0.weight.shape[:2] == (fine, coarse)
        assert up.up.conv1.weight.shape[:2] == (fine, fine)


def test_a_rejected_progressive_mode_says_which_ones_exist():
    with pytest.raises(ValueError, match="not ported"):
        NCSNppConfig(progressive="residual")
    with pytest.raises(ValueError, match="not ported"):
        NCSNppConfig(progressive_input="residual")


def test_building_needs_the_measured_sigma_range(tiny_config):
    c = dataclasses.replace(tiny_config)
    c.sde = SDEConfig(sigma_min=None, sigma_max=None, data_mean=0.0)
    with pytest.raises(ValueError, match="prepare_config"):
        c.build_model(jax.random.key(0))


# -- what is given up ------------------------------------------------------


def test_the_score_is_not_conservative(tiny_model):
    """The honest counterpart to the energy model's symmetric-Hessian test.

    A score that is a gradient has a symmetric Jacobian.  This one does not, and
    that is the trade being made -- no path-independent log-density, no relative
    log-probabilities of scenes.  Asserted rather than assumed, so that a later
    attempt to make this an energy model has a test that will change.
    """
    small = NCSNppConfig(nf=8, ch_mult=(1, 2), num_blocks=1, attention=False)
    model = NCSNpp(small, sigma_min=0.02, sigma_max=5.0, key=jax.random.key(0))
    x = jax.random.normal(jax.random.key(5), (1, 8, 8))
    jac = jax.jacrev(model.score)(x, jnp.float32(1.0)).reshape(64, 64)
    jac = np.asarray(jac)
    asymmetry = np.abs(jac - jac.T).max() / np.abs(jac).max()
    assert asymmetry > 1e-3


# -- it trains -------------------------------------------------------------


def test_a_few_steps_reduce_the_loss_on_a_fixed_batch(tiny_config):
    import optax

    model = tiny_config.build_model(jax.random.key(0))
    sde = VESDE.from_config(tiny_config.sde)
    x = jnp.asarray(np.random.default_rng(0).standard_normal(
        (8, 1, SIZE, SIZE)) * 0.7, jnp.float32)
    opt = optax.adam(3e-3)
    state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(model, state, key):
        loss, grads = eqx.filter_value_and_grad(dsm_loss)(model, x, key, sde, 0)
        updates, state = opt.update(grads, state,
                                    eqx.filter(model, eqx.is_inexact_array))
        return eqx.apply_updates(model, updates), state, loss

    key = jax.random.key(7)
    losses = []
    for i in range(12):
        # The same noise draw every step, so this measures fitting and not the
        # Monte Carlo scatter of the sigma draw.
        model, state, loss = step(model, state, jax.random.fold_in(key, 0))
        losses.append(float(loss))
    assert losses[0] == pytest.approx(1.0, abs=0.15)  # starts near "no score"
    assert losses[-1] < losses[0] - 0.05


def test_sampling_runs_through_the_shared_sampler(tiny_model, tiny_config):
    """The whole point of the ScoreModel interface: the sampler is unchanged."""
    sde = VESDE.from_config(tiny_config.sde)
    x = pflow_sample(tiny_model, jax.random.key(8), (2, 1, SIZE, SIZE), sde,
                     n_steps=4)
    assert x.shape == (2, 1, SIZE, SIZE)
    assert np.all(np.isfinite(np.asarray(x)))


# -- config plumbing -------------------------------------------------------


def test_both_sections_survive_a_round_trip(tiny_config):
    import json

    back = Config.from_dict(json.loads(json.dumps(tiny_config.to_dict())))
    assert back == tiny_config
    assert back.architecture == "ncsnpp"
    assert back.model_config is back.ncsnpp
    # ch_mult must come back as a tuple: it is a static pytree field, so a list
    # would make two models built from the same file structurally unequal.
    assert isinstance(back.ncsnpp.ch_mult, tuple)


def test_switching_architecture_leaves_the_other_section_alone(tiny_config):
    c = dataclasses.replace(tiny_config, architecture="energy")
    assert c.model_config is c.energy
    assert c.ncsnpp == tiny_config.ncsnpp
    assert n_parameters(c.build_model(jax.random.key(0))) > 0


def test_an_unknown_architecture_lists_the_known_ones(tiny_config):
    c = dataclasses.replace(tiny_config, architecture="unet")
    with pytest.raises(ValueError, match="energy"):
        c.build_model(jax.random.key(0))


def test_the_config_is_hashable(tiny_config):
    """It is a static field on an equinox Module, so it lands in the pytree
    *treedef* -- which JAX hashes to key its jit cache.  A plain mutable
    dataclass is unhashable and the failure surfaces far from here."""
    assert hash(tiny_config.ncsnpp) == hash(NCSNppConfig(
        **{f.name: getattr(tiny_config.ncsnpp, f.name)
           for f in dataclasses.fields(NCSNppConfig)}))


def test_a_resume_compares_the_architecture_in_use(tiny_config, tmp_path):
    """Comparing ``config.energy`` would let an NCSN++ checkpoint resume into an
    energy model whose section happened to match, and refuse a legitimate resume
    over settings the run never used."""
    import optax

    from rubin_host_prior.training.checkpoint import save_checkpoint
    from rubin_host_prior.training.trainer import _resume, make_optimizer

    model = tiny_config.build_model(jax.random.key(0))
    save_checkpoint(tmp_path, 0, tiny_config, model, model,
                    make_optimizer(tiny_config.train).init(
                        eqx.filter(model, eqx.is_inexact_array)))
    opt = make_optimizer(tiny_config.train)
    _resume(tmp_path, tiny_config, model, opt, verbose=False)  # same: fine

    other = dataclasses.replace(tiny_config, architecture="energy")
    with pytest.raises(ValueError, match="different architecture"):
        _resume(tmp_path, other, other.build_model(jax.random.key(0)), opt,
                verbose=False)

    wider = dataclasses.replace(tiny_config)
    wider.ncsnpp = dataclasses.replace(tiny_config.ncsnpp, nf=16)
    with pytest.raises(ValueError, match="different architecture"):
        _resume(tmp_path, wider, wider.build_model(jax.random.key(0)), opt,
                verbose=False)


# -- the same U-Net, read as an energy -------------------------------------


@pytest.fixture(scope="module")
def energy_config(tiny_config) -> Config:
    c = dataclasses.replace(tiny_config, architecture="ncsnpp_energy")
    c.sde = dataclasses.replace(tiny_config.sde, data_std=0.9)
    return c


def test_the_dae_prefactor_makes_the_epsilon_optimal_net_exact():
    """The derivation behind ``energy_form="dae"``, pinned.

    For data ``N(0, tau^2)`` the epsilon-optimal network is
    ``h = -sigma x / (tau^2 + sigma^2)``.  Feed that into
    ``E = c(sigma) ||h||^2 / 2`` with ``c = (tau^2 + sigma^2) / sigma^2`` and
    ``-grad_x E`` is the true score ``-x / (tau^2 + sigma^2)`` **exactly**, with
    the Jacobian term included -- which is what makes a trained ``"ncsnpp"``
    checkpoint the solution rather than merely a nearby point.

    Drop the prefactor and the score is suppressed by ``sigma^2/(tau^2+sigma^2)``:
    a factor of 6400 at ``sigma = 0.01``.  That is the dynamic range the epsilon
    parameterisation exists to remove, reintroduced.
    """
    from rubin_host_prior.nn.ncsnpp import scalar_energy

    tau = 0.8
    optimal = lambda x, s: -s * x / (tau**2 + s**2)
    x = jax.random.normal(jax.random.key(0), (1, 8, 8))

    for sigma in (0.01, 0.1, 1.0, 10.0):
        true_score = -x / (tau**2 + sigma**2)
        got = -jax.grad(lambda xx: scalar_energy(
            optimal(xx, sigma), sigma, "dae", tau))(x, )
        np.testing.assert_allclose(np.asarray(got), np.asarray(true_score),
                                   rtol=1e-5)
        # ... and what the prefactor is for.
        naive = -jax.grad(lambda xx: 0.5 * jnp.sum(optimal(xx, sigma) ** 2))(x)
        ratio = float(jnp.mean(naive / true_score))
        assert ratio == pytest.approx(sigma**2 / (tau**2 + sigma**2), rel=1e-3)


# -- the decoder removed ---------------------------------------------------


@pytest.fixture(scope="module")
def encoder_config(tiny_config) -> Config:
    c = dataclasses.replace(tiny_config, architecture="ncsnpp_encoder")
    c.sde = dataclasses.replace(tiny_config.sde, data_std=0.9)
    c.ncsnpp = dataclasses.replace(tiny_config.ncsnpp, nf=8, ch_mult=(1, 2),
                                   num_blocks=1, attention=False)
    return c


def test_the_encoder_energy_taps_every_resolution(encoder_config):
    """One map per level plus the bottleneck, each at its own grid.

    The pyramid is the architecture, not an option on it: a bottleneck-only
    energy closes 38.5% of the gap to the analytic optimum on iid Gaussian
    data where the U-Net reaches 94.6%, and the taps take it to 85.1%.
    """
    model = encoder_config.build_model(jax.random.key(0))
    x = jax.random.normal(jax.random.key(1), (1, SIZE, SIZE))
    maps = model.energy_maps(x, jnp.float32(1.0))
    assert len(maps) == len(encoder_config.ncsnpp.ch_mult) + 1
    grids = [m.shape[-1] for m in maps]
    assert grids == [SIZE, SIZE // 2, SIZE // 2]   # levels, then the bottleneck
    assert model(x, jnp.float32(1.0)).shape == ()
    assert model.score(x, jnp.float32(1.0)).shape == x.shape


def test_the_encoder_energy_score_is_conservative(encoder_config):
    """Summing scalar fields leaves a scalar, so the score is still an exact
    gradient -- which is the only reason a pyramid of taps is allowed."""
    model = encoder_config.build_model(jax.random.key(0))
    x = jax.random.normal(jax.random.key(5), (1, 8, 8))
    jac = np.asarray(jax.jacrev(model.score)(x, jnp.float32(1.0))).reshape(64, 64)
    assert np.abs(jac - jac.T).max() / np.abs(jac).max() < 1e-4


def test_coarse_levels_are_weighted_by_the_area_their_cells_cover(encoder_config):
    """Without it the finest level drowns the rest by the ratio of their cell
    counts -- measured 731x across five levels, 6x with the weighting."""
    model = encoder_config.build_model(jax.random.key(0))
    x = jax.random.normal(jax.random.key(2), (1, SIZE, SIZE))
    sigma = jnp.float32(1.0)
    maps = model.energy_maps(x, sigma)
    want = sum(float((SIZE / m.shape[-1]) ** 2) * float(
        (m - off(sigma)).sum()) for m, off in zip(maps, model.offsets))
    assert float(model(x, sigma)) == pytest.approx(want, rel=1e-4)
    # Each level's weight is the pixels one of its cells stands for.
    assert [int((SIZE / m.shape[-1]) ** 2) for m in maps] == [1, 4, 4]


def test_the_encoder_energy_is_not_size_locked(encoder_config):
    """What the U-Net cannot do: nothing reconstructs a fixed-size output, so
    any grid the downsampling divides is a valid scene."""
    model = encoder_config.build_model(jax.random.key(0))
    for size in (8, 16, 24):
        x = jnp.zeros((1, size, size))
        assert model.score(x, jnp.float32(1.0)).shape == (1, size, size)


def test_the_bottleneck_reaches_the_finest_tap(encoder_config):
    """What the context stream is for.  Without it the finest tap is a function
    of the first level's blocks alone, and nothing below can move it.

    Asked of the weights, not the pixels: GroupNorm's whole-grid statistics
    already make every feature depend on every pixel through two scalars per
    group, so a far-away pixel moving the finest map proves nothing."""
    model = encoder_config.build_model(jax.random.key(0))
    x = jax.random.normal(jax.random.key(4), (1, SIZE, SIZE))

    def reach(m):
        g = eqx.filter_grad(
            lambda m: jnp.sum(m.energy_maps(x, jnp.float32(1.0))[0]))(m)
        return (float(jnp.abs(g.mid_block2.conv0.weight).max()),
                float(jnp.abs(g.down_levels[-1].blocks[0].conv0.weight).max()))

    assert min(reach(model)) > 0
    # ... and the stream is the only way there: cut it and both vanish
    # exactly, which is the architecture this replaced.
    cut = eqx.tree_at(
        lambda m: [c.weight for c in m.context_convs], model,
        [jnp.zeros_like(c.weight) for c in model.context_convs])
    assert reach(cut) == (0.0, 0.0)


@pytest.mark.parametrize("n_blocks", [0, 2])
def test_any_depth_of_context_is_still_an_energy(encoder_config, n_blocks):
    """``context_blocks`` is how much runs on the fused features before the
    tap reads them.  None of it may cost the exact gradient or the free grid."""
    c = dataclasses.replace(encoder_config)
    c.ncsnpp = dataclasses.replace(
        encoder_config.ncsnpp, context_blocks=n_blocks)
    model = c.build_model(jax.random.key(0))
    assert [len(b) for b in model.context_blocks] == [n_blocks] * 2
    x = jax.random.normal(jax.random.key(5), (1, 8, 8))
    jac = np.asarray(jax.jacrev(model.score)(x, jnp.float32(1.0))).reshape(64, 64)
    assert np.abs(jac - jac.T).max() / np.abs(jac).max() < 1e-4
    assert model.score(jnp.zeros((1, 24, 24)), jnp.float32(1.0)).shape == (1, 24, 24)


def test_a_single_head_reads_the_whole_energy_at_full_resolution(encoder_config):
    """``progressive="none"``: one map, one offset, and every level still in
    it -- by way of the context stream, since nothing else leads there."""
    c = dataclasses.replace(encoder_config)
    c.ncsnpp = dataclasses.replace(encoder_config.ncsnpp, progressive="none")
    model = c.build_model(jax.random.key(0))
    x = jax.random.normal(jax.random.key(6), (1, SIZE, SIZE))
    sigma = jnp.float32(1.0)
    maps = model.energy_maps(x, sigma)
    assert [m.shape[-1] for m in maps] == [SIZE]
    assert len(model.offsets) == 1 and model.head_conv is None
    assert float(model(x, sigma)) == pytest.approx(
        float((maps[0] - model.offsets[0](sigma)).sum()), rel=1e-4)

    g = eqx.filter_grad(lambda m: m(x, sigma))(model)
    assert float(jnp.abs(g.mid_block2.conv0.weight).max()) > 0

    small = jax.random.normal(jax.random.key(7), (1, 8, 8))
    jac = np.asarray(jax.jacrev(model.score)(small, sigma)).reshape(64, 64)
    assert np.abs(jac - jac.T).max() / np.abs(jac).max() < 1e-4


def test_a_negative_depth_of_context_is_refused():
    with pytest.raises(ValueError, match="context_blocks"):
        NCSNppConfig(context_blocks=-1)


def test_the_sigma_offset_cannot_move_the_score(encoder_config):
    """The whole licence for subtracting it. A function of sigma alone has no
    gradient in x, so the offsets are free to be anything at all."""
    from rubin_host_prior.nn.ncsnpp import SigmaOffset

    model = encoder_config.build_model(jax.random.key(0))
    x = jax.random.normal(jax.random.key(3), (1, SIZE, SIZE))
    sigma = jnp.float32(0.3)
    before = np.asarray(model.score(x, sigma))

    wild = tuple(
        eqx.tree_at(lambda o: (o.a, o.b0, o.w, o.log_q), off,
                    (jnp.float32(-5e3), jnp.float32(1e4), off.w * -37.0,
                     off.log_q + 5.0))
        for off in model.offsets
    )
    moved = eqx.tree_at(lambda m: m.offsets, model, wild)
    np.testing.assert_array_equal(before, np.asarray(moved.score(x, sigma)))
    # ... and it does move the energy, so it is not a no-op either.
    assert float(moved(x, sigma)) != pytest.approx(float(model(x, sigma)))


def test_the_offset_bands_start_at_the_derived_values():
    """``sum_j w_j = 1/2`` per cell is the per-cell reading of the Gaussian
    identity ``sum_j w_j = D/2``, which a fit recovers to 0.1% unprompted."""
    from rubin_host_prior.nn.ncsnpp import SigmaOffset

    for n_bands in (1, 2, 4):
        off = SigmaOffset(n_bands, var=0.49)
        assert float(off.w.sum()) == pytest.approx(0.5)
        assert len(off.log_q) == n_bands
        # Spread across the spectrum, not stacked: two bands at one power are
        # one band and a wasted one.
        q = np.exp(np.asarray(off.log_q))
        if n_bands > 1:
            assert q.max() / q.min() == pytest.approx(100.0, rel=1e-3)
        # Monotone in sigma, as 0.5*sum_k log(P_k + sigma^2) is.
        vals = [float(off(jnp.float32(s))) for s in (0.01, 0.1, 1.0, 10.0)]
        assert vals == sorted(vals)

    with pytest.raises(ValueError, match="n_bands"):
        SigmaOffset(0)


def test_calibration_flattens_the_energy_and_leaves_the_score_alone(encoder_config):
    """What ``calibrate_offsets`` is for, end to end.

    The offsets get exactly zero gradient from score matching -- it cannot see
    a function of sigma alone -- so they are fitted afterwards instead of
    bolted onto the training loop.
    """
    from rubin_host_prior.nn.ncsnpp import calibrate_offsets, offset_residual

    model = encoder_config.build_model(jax.random.key(0))
    sde = VESDE.from_config(encoder_config.sde)
    x = jnp.asarray(np.random.default_rng(0).normal(3.0, 0.7, (4, 1, SIZE, SIZE)),
                    jnp.float32)
    fitted = calibrate_offsets(model, x, sde, n_sigma=8)

    # Across the ladder the model is actually trained on, and reported against
    # the energy it started from rather than against a number: what has to be
    # true is that almost all of it is gone, at *every* sigma and not on
    # average.  `after < before` was the old assertion and it passed happily
    # while the small-sigma end was out by a factor of a thousand.
    sigmas = np.asarray(sde.sigma(np.linspace(0.0, 1.0, 12)))
    before = np.array([abs(float(model(x[0], jnp.float32(s)))) for s in sigmas])
    after = np.array([abs(float(fitted(x[0], jnp.float32(s)))) for s in sigmas])
    assert (after < 0.02 * before).all(), dict(zip(sigmas.round(3), after / before))

    # Provably, not approximately: the score is untouched.
    for s in sigmas:
        np.testing.assert_array_equal(
            np.asarray(model.score(x[0], jnp.float32(s))),
            np.asarray(fitted.score(x[0], jnp.float32(s))))

    got = offset_residual(fitted, x, sde, n_sigma=8)
    assert len(got["levels"]) == len(model.offsets)
    assert all(np.isfinite(L["max_residual"]) for L in got["levels"])
    # The fit is the thing that decides all of the above, so hold it to the
    # span it is fitting rather than to an absolute residual -- the energy
    # scale is whatever training drifted to.
    assert all(L["fraction_of_span"] < 1e-2 for L in got["levels"]), got


def test_the_offset_follows_the_drift_the_sum_form_makes(encoder_config):
    """``E = sum(h) / sigma`` with an ``x``-independent part of ``h`` that the
    loss cannot see is an energy with a ``const / sigma`` term in it.

    The regression this pins: the family was ``b0 + sum_j w_j log(q_j +
    sigma^2)``, every term of which is *bounded* as ``sigma -> 0``, so it could
    not represent that drift at all.  The fit sat ~2% out at every drift scale,
    the miss was largest at the small-sigma end, and multiplied by the number
    of cells it came back as ``log_prob`` running to -1e7 as ``t -> 0``.
    """
    from rubin_host_prior.nn.ncsnpp import SigmaOffset, _fit_offset

    sde = VESDE.from_config(encoder_config.sde)
    sigmas = jnp.asarray(sde.sigma(jnp.linspace(0.0, 1.0, 24)), jnp.float32)
    off = SigmaOffset(2, var=0.49)

    # Pure drift, which is what a trained encoder very nearly is: `sigma * k`
    # measures flat to 0.1% across three decades.
    for scale in (1.0, 1e3):
        target = jnp.asarray(-6.1 * scale / sigmas, jnp.float32)
        a, b0, w, log_q = _fit_offset(off.log_q, sigmas, target)
        fitted = eqx.tree_at(lambda o: (o.a, o.b0, o.w, o.log_q), off,
                             (a, b0, w, log_q))
        got = jnp.asarray([fitted(s) for s in sigmas])
        span = float(target.max() - target.min())
        assert float(jnp.abs(got - target).max()) < 1e-5 * span
        # Recovered as the drift it is, not smeared over the log bands.
        assert float(a) == pytest.approx(-6.1 * scale, rel=1e-3)

    # The log bands still do their own job: a pure Gaussian-field curve with no
    # drift in it comes back with `a` at zero.
    q = jnp.asarray([0.5, 40.0], jnp.float32)
    target = 0.2 * jnp.log(q[0] + sigmas**2) + 0.1 * jnp.log(q[1] + sigmas**2)
    a, b0, w, log_q = _fit_offset(off.log_q, sigmas, target)
    fitted = eqx.tree_at(lambda o: (o.a, o.b0, o.w, o.log_q), off,
                         (a, b0, w, log_q))
    got = jnp.asarray([fitted(s) for s in sigmas])
    span = float(target.max() - target.min())
    assert float(jnp.abs(got - target).max()) < 5e-3 * span
    # The one that matters: no drift in the target, none in the fit.  Loose on
    # the residual because this is the case the band powers have to be *found*
    # for, and `a` can always paper over a slow band -- that it does not is
    # the statement.
    assert abs(float(a)) < 1e-2 * span

    # Rank-deficient designs, where the SVD inside `lstsq` has no unique answer
    # and its gradient is where NaN would come from: bands started stacked, a
    # single band, and a target so flat that the log columns *are* the
    # intercept.  All three have to come back finite, because the fit runs
    # unattended at every checkpoint.
    flat = jnp.full_like(sigmas, 3.0)
    for lq0, tgt in (
        (jnp.log(jnp.asarray([1.0, 1.0], jnp.float32)), target),
        (jnp.log(jnp.asarray([1.0], jnp.float32)), target),
        (off.log_q, flat),
    ):
        got = _fit_offset(lq0, sigmas, tgt)
        assert all(bool(jnp.all(jnp.isfinite(g))) for g in got), got


def test_the_encoder_refuses_the_dae_form(encoder_config):
    """``dae`` is ``||x - s(x)||^2 / 2``, which needs ``h`` to be a
    full-resolution denoising residual whose Jacobian is about ``-I/sigma``.

    A coarse tap's Jacobian is a downsampling operator and nothing drives any
    one tap to be ``sigma * score``, so the derivation does not survive here.
    Refused at build rather than silently squaring the maps.
    """
    c = dataclasses.replace(encoder_config)
    c.ncsnpp = dataclasses.replace(encoder_config.ncsnpp, energy_form="dae")
    with pytest.raises(ValueError, match="energy_form='sum'"):
        c.build_model(jax.random.key(0))
    # ... and the U-Net energy, where the derivation does hold, still takes it.
    u = dataclasses.replace(encoder_config, architecture="ncsnpp_energy")
    u.ncsnpp = dataclasses.replace(encoder_config.ncsnpp, energy_form="dae")
    assert u.build_model(jax.random.key(0)) is not None


def test_the_offsets_cost_nothing_in_training_precision(encoder_config):
    """Why calibration is about staleness and not about float32.

    Reverse mode starts from ``dE/dE = 1`` and the backward of a sum is an
    exact broadcast, so the energy's magnitude never enters the gradient.
    Shifted far enough to move E by ~1e7, the loss gradient has to be
    unchanged to the last bit -- and the offsets' own gradient has to be
    exactly zero, which is what makes them unfittable by the loss.
    """
    model = encoder_config.build_model(jax.random.key(0))
    sde = VESDE.from_config(encoder_config.sde)
    x = jnp.asarray(np.random.default_rng(0).normal(3.0, 0.7, (4, 1, SIZE, SIZE)),
                    jnp.float32)
    wild = tuple(eqx.tree_at(lambda o: o.b0, off, jnp.float32(1e4))
                 for off in model.offsets)
    moved = eqx.tree_at(lambda m: m.offsets, model, wild)
    assert abs(float(moved(x[0], jnp.float32(1.0)))) > 1e5

    g0 = eqx.filter_grad(dsm_loss)(model, x, jax.random.key(1), sde)
    g1 = eqx.filter_grad(dsm_loss)(moved, x, jax.random.key(1), sde)
    for a, b in zip(jax.tree.leaves(eqx.filter(g0, eqx.is_inexact_array)),
                    jax.tree.leaves(eqx.filter(g1, eqx.is_inexact_array))):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    for off in g0.offsets:
        assert float(jnp.abs(off.b0).max()) == 0.0
        assert float(jnp.abs(off.w).max()) == 0.0


def test_the_encoder_architecture_round_trips(encoder_config):
    import json

    back = Config.from_dict(json.loads(json.dumps(encoder_config.to_dict())))
    assert back == encoder_config
    assert back.architecture == "ncsnpp_encoder"
    assert back.ncsnpp.n_offset_bands == encoder_config.ncsnpp.n_offset_bands
    with pytest.raises(ValueError, match="n_offset_bands"):
        NCSNppConfig(n_offset_bands=0)


@pytest.mark.parametrize("form", ["sum", "dae"])
def test_the_energy_score_is_conservative(energy_config, form):
    """What the whole exercise is for.  A score that is a gradient has a
    symmetric Jacobian; ``NCSNpp``'s does not (see the test above), and HMC
    needs more than that -- it needs the potential itself for its accept/reject
    step, which only an energy supplies."""
    c = dataclasses.replace(energy_config)
    c.ncsnpp = dataclasses.replace(energy_config.ncsnpp, nf=8, ch_mult=(1, 2),
                                   num_blocks=1, attention=False,
                                   energy_form=form)
    model = c.build_model(jax.random.key(0))
    x = jax.random.normal(jax.random.key(5), (1, 8, 8))
    jac = np.asarray(jax.jacrev(model.score)(x, jnp.float32(1.0))).reshape(64, 64)
    asymmetry = np.abs(jac - jac.T).max() / np.abs(jac).max()
    assert asymmetry < 1e-4, asymmetry


@pytest.mark.parametrize("form", ["sum", "dae"])
def test_a_pixel_offset_moves_the_constant_and_nothing_else(form):
    """``E(x, sigma, k) == E(x, sigma, 0) - N k``, exactly, and the score is
    blind to it.  That is the whole contract: ``k`` depends on sigma alone, so
    at a fixed sigma it shifts every scene by the same amount -- the gauge
    freedom an unnormalised energy has anyway."""
    from rubin_host_prior.nn.ncsnpp import energy_map, scalar_energy

    h = lambda a: 0.3 * a - 0.1 * a**3  # stands in for the network
    x = jax.random.normal(jax.random.key(3), (1, 8, 8))
    sigma, k, tau = 0.3, 0.7, 0.8

    plain = scalar_energy(h(x), sigma, form, tau)
    shifted = scalar_energy(h(x), sigma, form, tau, k)
    assert float(plain - shifted) == pytest.approx(k * x.size, rel=1e-5)

    # The map sums to the energy, which is what makes "per pixel" the right
    # unit for the offset in the first place.
    assert float(jnp.sum(energy_map(h(x), sigma, form, tau))) == pytest.approx(
        float(plain), rel=1e-6)

    grad = lambda off: jax.grad(
        lambda a: scalar_energy(h(a), sigma, form, tau, off))(x)
    np.testing.assert_allclose(np.asarray(grad(0.0)), np.asarray(grad(k)),
                               rtol=1e-6)


def test_without_the_offset_float32_loses_the_difference_entirely():
    """Why the argument exists, measured rather than asserted.

    At ``sigma = 0.001`` the ``"dae"`` prefactor is ~8e5, so a 128x128 scene
    sits at ``E ~ 7e9`` where float32's spacing is already ~512.  Perturb one
    pixel by a real 8 nats of energy and the subtraction returns **0.0** -- an
    accept ratio quantised to nothing.  Subtracting the per-pixel level first
    keeps the total near zero and recovers it to a part in 600.
    """
    from rubin_host_prior.nn.ncsnpp import scalar_energy

    tau, sigma, baseline = 0.9, 0.001, 1.0
    prefactor = (tau**2 + sigma**2) / sigma**2
    base = (baseline + 1e-4 * jax.random.normal(jax.random.key(0), (1, 128, 128)))
    base = base.astype(jnp.float32)
    bumped = base.at[0, 64, 64].add(1e-5)
    assert float(scalar_energy(base, sigma, "dae", tau)) > 1e9

    # float64, from the same float32 inputs: the answer the sampler wants.
    sq = lambda a: (np.asarray(a).astype(np.float64) ** 2).sum()
    want = 0.5 * prefactor * (sq(bumped) - sq(base))
    assert want == pytest.approx(8.1, rel=0.05)

    at = lambda off: float(scalar_energy(bumped, sigma, "dae", tau, off)
                           - scalar_energy(base, sigma, "dae", tau, off))
    assert at(0.0) == 0.0
    assert at(0.5 * prefactor * baseline**2) == pytest.approx(want, rel=1e-2)


def test_both_energy_forms_descend_and_sum_descends_faster(energy_config):
    """Both train; ``"sum"`` gets further from a random init.

    That ordering is the finding, and it is the opposite of what the published
    argument for ``"dae"`` would suggest in isolation -- its gradient
    ``-c (grad h)^T h`` is proportional to ``h``, which starts at 0.08 because
    the pyramid output convolutions are initialised at ``init_scale = 1e-2``.
    The advantage ``"dae"`` has is conditional on ``h`` already being the right
    size, which is what warm-starting from a trained ``"ncsnpp"`` provides.
    See ``NCSNppEnergy`` for the full curves.
    """
    import optax

    def descend(form, n=50):
        c = dataclasses.replace(energy_config)
        c.ncsnpp = dataclasses.replace(energy_config.ncsnpp, energy_form=form)
        model = c.build_model(jax.random.key(0))
        sde = VESDE.from_config(c.sde)
        x = jnp.asarray(np.random.default_rng(0).standard_normal(
            (8, 1, SIZE, SIZE)) * 0.7, jnp.float32)
        opt = optax.adam(3e-3)
        state = opt.init(eqx.filter(model, eqx.is_inexact_array))

        @eqx.filter_jit
        def step(model, state, key):
            loss, grads = eqx.filter_value_and_grad(dsm_loss)(
                model, x, key, sde, 0)
            updates, state = opt.update(
                grads, state, eqx.filter(model, eqx.is_inexact_array))
            return eqx.apply_updates(model, updates), state, loss

        # The same noise draw every step, so this measures fitting and not the
        # Monte Carlo scatter of the sigma draw.
        key = jax.random.fold_in(jax.random.key(7), 0)
        losses = []
        for _ in range(n):
            model, state, loss = step(model, state, key)
            losses.append(float(loss))
        return losses

    by_form = {form: descend(form) for form in ("sum", "dae")}
    for form, losses in by_form.items():
        assert losses[0] == pytest.approx(1.0, abs=0.15), (form, losses[0])
        # Modest on purpose: 50 steps of a toy model. The ordering below is
        # the finding; this only has to show both are learning at all.
        assert losses[-1] < losses[0] - 0.1, (form, losses[0], losses[-1])
    assert by_form["sum"][-1] < by_form["dae"][-1]


def test_the_energy_wraps_an_unchanged_backbone(energy_config, tiny_config):
    """``.net`` is byte-for-byte an ``NCSNpp``, so the weights of a trained
    score model drop straight in.  That is the practical argument for the
    ``"dae"`` form: its solution for Gaussian data *is* those weights."""
    score_model = tiny_config.build_model(jax.random.key(0))
    energy_model = energy_config.build_model(jax.random.key(0))
    a = jax.tree_util.tree_structure(eqx.filter(score_model, eqx.is_inexact_array))
    b = jax.tree_util.tree_structure(eqx.filter(energy_model.net,
                                                eqx.is_inexact_array))
    assert a == b
    assert n_parameters(energy_model) == n_parameters(score_model)


def test_the_dae_form_refuses_to_build_without_data_std(energy_config):
    c = dataclasses.replace(energy_config)
    c.ncsnpp = dataclasses.replace(energy_config.ncsnpp, energy_form="dae")
    c.sde = dataclasses.replace(energy_config.sde, data_std=None)
    with pytest.raises(ValueError, match="data_std"):
        c.build_model(jax.random.key(0))
    # 'sum' does not use it, so it builds.
    c.ncsnpp = dataclasses.replace(energy_config.ncsnpp, energy_form="sum")
    assert c.build_model(jax.random.key(0)) is not None


def test_an_unknown_energy_form_is_rejected():
    with pytest.raises(ValueError, match="energy_form"):
        NCSNppConfig(energy_form="scalar_head")


def test_the_energy_config_round_trips(energy_config):
    import json

    c = dataclasses.replace(energy_config)
    c.ncsnpp = dataclasses.replace(energy_config.ncsnpp, energy_form="dae")
    back = Config.from_dict(json.loads(json.dumps(c.to_dict())))
    assert back == c
    assert back.architecture == "ncsnpp_energy"
    assert back.ncsnpp.energy_form == "dae"
    assert back.sde.data_std == 0.9
