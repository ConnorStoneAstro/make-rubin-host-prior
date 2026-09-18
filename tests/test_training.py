"""Training loop, EMA, checkpoints, and config serialisation."""

import dataclasses
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rubin_host_prior.config import BANDS, Config, EnergyConfig, PatchConfig
from rubin_host_prior.diffusion import VESDE, mean_dsm_loss
from rubin_host_prior.nn import ConvEnergyNet, n_parameters
from rubin_host_prior.training import (
    ema_decay_at,
    ema_update,
    load_checkpoint,
    make_optimizer,
    save_checkpoint,
    train,
)


def _config(**train_kw):
    c = Config(
        energy=EnergyConfig(channels=(8, 12), embed_dim=16, n_fourier=8),
        patch=PatchConfig(native_size=128, nominal_crop=48, out_size=16,
                          pool_factor=3),
    )
    c.transform.band_offsets = {b: 20.0 for b in BANDS}
    c.train.steps = 12
    c.train.batch_size = 4
    c.train.log_every = 4
    c.train.ckpt_every = 0
    for k, v in train_kw.items():
        setattr(c.train, k, v)
    return c


def _batches(rng, shape=(4, 1, 16, 16)):
    while True:
        yield rng.normal(size=shape).astype(np.float32) * 0.5


# -- EMA -------------------------------------------------------------------


def test_ema_decay_warms_up():
    """Without the warmup the EMA is dominated by the random init for the first
    1/(1-decay) steps -- a thousand steps of a useless copy at decay=0.999."""
    assert float(ema_decay_at(0, 0.999)) == pytest.approx(0.1)
    assert float(ema_decay_at(90, 0.999)) == pytest.approx(0.91)
    assert float(ema_decay_at(1_000_000, 0.999)) == pytest.approx(0.999)
    assert float(ema_decay_at(5, 0.5)) == pytest.approx(0.4)  # never exceeds decay


def test_ema_update_mixes_only_array_leaves(tiny_model):
    other = ConvEnergyNet(tiny_model.config, key=jax.random.key(99))
    mixed = ema_update(tiny_model, other, jnp.asarray(0.25))
    expected = 0.25 * tiny_model.head.weight + 0.75 * other.head.weight
    np.testing.assert_allclose(np.asarray(mixed.head.weight), np.asarray(expected))
    # Static fields survive untouched.
    assert mixed.config == tiny_model.config
    assert mixed.embed.fourier.freqs == tiny_model.embed.fourier.freqs


def test_ema_at_decay_zero_is_the_live_model(tiny_model):
    other = ConvEnergyNet(tiny_model.config, key=jax.random.key(98))
    mixed = ema_update(tiny_model, other, jnp.asarray(0.0))
    np.testing.assert_allclose(
        np.asarray(mixed.head.weight), np.asarray(other.head.weight)
    )


# -- optimizer -------------------------------------------------------------


def test_optimizer_warms_up_then_holds():
    cfg = _config(learning_rate=1e-3, warmup_steps=10, cosine_decay=False).train
    opt = make_optimizer(cfg)
    model = ConvEnergyNet(
        EnergyConfig(channels=(4,), embed_dim=8, n_fourier=4), key=jax.random.key(0)
    )
    params = eqx.filter(model, eqx.is_inexact_array)
    state = opt.init(params)
    grads = jax.tree_util.tree_map(lambda p: jnp.ones_like(p), params)
    steps = []
    for _ in range(20):
        updates, state = opt.update(grads, state, params)
        steps.append(float(jnp.abs(updates.head.weight).max()))
    assert steps[0] < steps[5] < steps[9]  # warming up
    assert steps[15] == pytest.approx(steps[19], rel=1e-3)  # then flat


def test_cosine_schedule_decays():
    cfg = _config(learning_rate=1e-3, warmup_steps=2, cosine_decay=True, steps=50).train
    opt = make_optimizer(cfg)
    model = ConvEnergyNet(
        EnergyConfig(channels=(4,), embed_dim=8, n_fourier=4), key=jax.random.key(0)
    )
    params = eqx.filter(model, eqx.is_inexact_array)
    state = opt.init(params)
    grads = jax.tree_util.tree_map(lambda p: jnp.ones_like(p), params)
    mags = []
    for _ in range(50):
        updates, state = opt.update(grads, state, params)
        mags.append(float(jnp.abs(updates.head.weight).max()))
    assert mags[-1] < mags[5]


# -- the loop --------------------------------------------------------------


def test_training_reduces_the_loss(tmp_path):
    """Compare *averaged* losses before and after.

    A single batch is far too noisy: the sigma draw alone swings it by O(0.1),
    because the optimal loss runs from ~1 at small sigma to ~0 at large.  The
    first ``loss_ema`` entry in the log is one batch and is not a baseline.
    """
    config = _config(steps=400, learning_rate=3e-3, log_every=50)
    model = ConvEnergyNet(config.energy, key=jax.random.key(0))
    sde = VESDE.from_config(config.sde)
    rng = np.random.default_rng(0)

    # Spatially correlated data, so there is structure to learn beyond the
    # noise: the score of white noise is trivial and would test almost nothing.
    base = rng.normal(size=(64, 1, 16, 16)).astype(np.float32)
    kern = np.ones((1, 1, 3, 3), dtype=np.float32) / 9.0
    smooth = np.stack([
        np.asarray(
            jax.scipy.signal.convolve2d(b[0], kern[0, 0], mode="same")
        )[None]
        for b in base
    ]).astype(np.float32)

    def batches():
        while True:
            yield smooth[rng.integers(0, len(smooth), 4)]

    before = mean_dsm_loss(model, batches(), 40, jax.random.key(1), sde)
    model, ema = train(
        model, batches(), config, out_dir=tmp_path / "run", sde=sde
    )
    after = mean_dsm_loss(ema, batches(), 40, jax.random.key(1), sde)
    assert np.isfinite(before) and np.isfinite(after)
    assert before == pytest.approx(1.0, abs=0.06), "untrained loss should be ~1"
    assert after < before - 0.01, f"{before:.4f} -> {after:.4f}"


def test_training_writes_a_log_and_a_final_checkpoint(tmp_path):
    config = _config()
    model = ConvEnergyNet(config.energy, key=jax.random.key(0))
    train(model, _batches(np.random.default_rng(1)), config, out_dir=tmp_path / "run")
    lines = [
        json.loads(l)
        for l in (tmp_path / "run" / "log.jsonl").read_text().splitlines()
    ]
    assert lines[0]["event"] == "start"
    assert lines[0]["loss_margin"] == 4
    assert lines[0]["n_parameters"] == n_parameters(model)
    assert (tmp_path / "run" / "final" / "ema.eqx").exists()
    assert (tmp_path / "run" / "config.json").exists()


def test_eval_logs_a_loss_curve_against_sigma(tmp_path):
    config = _config(steps=8)
    model = ConvEnergyNet(config.energy, key=jax.random.key(0))
    rng = np.random.default_rng(2)
    records = []
    train(
        model, _batches(rng), config, out_dir=tmp_path / "run",
        eval_batch=rng.normal(size=(8, 1, 16, 16)).astype(np.float32),
        eval_every=4, on_log=records.append,
    )
    evals = [r for r in records if r.get("event") == "eval"]
    assert evals
    assert len(evals[0]["loss_by_sigma"]) == 8
    assert evals[0]["sigma"][0] < evals[0]["sigma"][-1]


def test_too_small_patches_fail_immediately_with_a_useful_message(tmp_path):
    """A model with 8 layers needs > 4R pixels.  Caught from the config before
    a single batch is drawn, not on step 1 and not 10 000 steps in."""
    config = _config()
    config.energy = EnergyConfig(channels=(8,) * 8, embed_dim=16, n_fourier=8)
    model = ConvEnergyNet(config.energy, key=jax.random.key(0))
    with pytest.raises(ValueError, match=r"leave no interior"):
        train(
            model, _batches(np.random.default_rng(3)), config,
            out_dir=tmp_path / "run",
        )


def test_channel_mismatch_is_caught(tmp_path):
    config = _config()
    model = ConvEnergyNet(config.energy, key=jax.random.key(0))
    with pytest.raises(ValueError, match="channels"):
        train(
            model,
            _batches(np.random.default_rng(4), shape=(4, 3, 16, 16)),
            config,
            out_dir=tmp_path / "run",
        )


# -- checkpoints -----------------------------------------------------------


def test_checkpoint_round_trips_exactly(tmp_path):
    config = _config()
    model = ConvEnergyNet(config.energy, key=jax.random.key(7))
    ema = ConvEnergyNet(config.energy, key=jax.random.key(8))
    save_checkpoint(tmp_path / "ck", 4242, config, model, ema)

    back_ema, cfg, step = load_checkpoint(tmp_path / "ck", which="ema")
    assert step == 4242
    assert cfg.energy == config.energy
    assert cfg.transform.band_offsets == config.transform.band_offsets
    np.testing.assert_array_equal(
        np.asarray(back_ema.head.weight), np.asarray(ema.head.weight)
    )
    back_live, _, _ = load_checkpoint(tmp_path / "ck", which="model")
    np.testing.assert_array_equal(
        np.asarray(back_live.head.weight), np.asarray(model.head.weight)
    )


def test_reloaded_model_gives_identical_scores(tmp_path):
    """The point of the round trip: the prior must be byte-for-byte reusable."""
    from rubin_host_prior.nn import score

    config = _config()
    model = ConvEnergyNet(config.energy, key=jax.random.key(9))
    save_checkpoint(tmp_path / "ck", 1, config, model, model)
    back, _, _ = load_checkpoint(tmp_path / "ck")
    x = jax.random.normal(jax.random.key(10), (1, 20, 20))
    np.testing.assert_array_equal(
        np.asarray(score(model, x, jnp.asarray(0.3))),
        np.asarray(score(back, x, jnp.asarray(0.3))),
    )


def test_config_survives_json(tmp_path):
    config = _config()
    config.save(tmp_path / "c.json")
    back = Config.load(tmp_path / "c.json")
    assert back.energy == config.energy
    assert isinstance(back.energy.channels, tuple)  # JSON gives a list back
    assert back.patch == config.patch
    assert back.transform.band_offsets == config.transform.band_offsets


def test_patch_config_validates_its_own_arithmetic():
    with pytest.raises(ValueError, match="nominal_crop"):
        PatchConfig(native_size=224, nominal_crop=100, out_size=64, pool_factor=3)
    with pytest.raises(ValueError, match="native_size must be"):
        PatchConfig(native_size=100, nominal_crop=192, out_size=64, pool_factor=3)


def test_models_from_one_config_are_pytree_compatible():
    """Regression: the frozen Fourier basis lives in pytree *metadata*.

    If it were derived from the model's init key rather than from the config,
    two models would be structurally incompatible -- ``tree_map`` would fail
    (breaking the EMA) and ``tree_deserialise_leaves`` would silently leave the
    skeleton's basis in place, so a reloaded checkpoint would compute different
    scores from the same weights.
    """
    cfg = EnergyConfig(channels=(6, 8), embed_dim=16, n_fourier=8)
    a = ConvEnergyNet(cfg, key=jax.random.key(1))
    b = ConvEnergyNet(cfg, key=jax.random.key(2))
    assert a.embed.fourier.freqs == b.embed.fourier.freqs
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    ema_update(a, b, jnp.asarray(0.5))  # must not raise

    other = ConvEnergyNet(
        EnergyConfig(channels=(6, 8), embed_dim=16, n_fourier=8, fourier_seed=7),
        key=jax.random.key(1),
    )
    assert other.embed.fourier.freqs != a.embed.fourier.freqs


# -- the derived loss crop, and variable patch sizes -----------------------


def test_setup_note_is_printed_and_reports_the_derived_crop(tmp_path, capsys):
    """Requested behaviour: training announces how much it is cropping, so that
    tinkering with n_layers or kernel_size is never silent."""
    config = _config()
    config.energy = EnergyConfig(channels=(8,) * 3, embed_dim=16, n_fourier=8)
    model = ConvEnergyNet(config.energy, key=jax.random.key(0))
    train(model, _batches(np.random.default_rng(0)), config, out_dir=tmp_path / "r")
    out = capsys.readouterr().out
    assert "3 x 3x3 valid convolutions" in out
    assert "loss crop = 2R = 6 px from every side" in out
    assert "loss on interior 4x4" in out  # 16 px patches, margin 6
    assert "N + 12 px" in out


def test_setup_note_can_be_silenced(tmp_path, capsys):
    config = _config()
    model = ConvEnergyNet(config.energy, key=jax.random.key(0))
    train(model, _batches(np.random.default_rng(0)), config,
          out_dir=tmp_path / "r", verbose=False)
    assert capsys.readouterr().out == ""


def test_log_header_records_the_geometry(tmp_path):
    config = _config()
    config.patch = dataclasses.replace(config.patch, out_sizes=(16, 24))
    model = ConvEnergyNet(config.energy, key=jax.random.key(0))
    train(model, _varsize_batches(np.random.default_rng(0), (16, 24)), config,
          out_dir=tmp_path / "r", verbose=False)
    head = json.loads((tmp_path / "r" / "log.jsonl").read_text().splitlines()[0])
    assert head["loss_margin"] == 4
    assert head["receptive_radius"] == 2
    assert head["n_layers"] == 2
    assert head["training_sizes"] == [16, 24]
    assert head["interior_sizes"] == [8, 16]


def _varsize_batches(rng, sizes, batch=4):
    k = 0
    while True:
        s = sizes[k % len(sizes)]
        k += 1
        yield rng.normal(size=(batch, 1, s, s)).astype(np.float32) * 0.5


def test_variable_sizes_train_without_recompilation_errors(tmp_path):
    """One jit compilation per distinct size; all sizes share the weights."""
    config = _config(steps=12)
    config.patch = dataclasses.replace(config.patch, out_sizes=(16, 20, 24))
    model = ConvEnergyNet(config.energy, key=jax.random.key(0))
    model, ema = train(model, _varsize_batches(np.random.default_rng(1), (16, 20, 24)),
                       config, out_dir=tmp_path / "r", verbose=False)
    assert n_parameters(ema) == n_parameters(model)


def test_all_configured_sizes_are_validated_before_training_starts(tmp_path):
    """A mixed-size run must not fail thousands of steps in, when the smallest
    size first comes round."""
    config = _config()
    config.energy = EnergyConfig(channels=(8,) * 8, embed_dim=16, n_fourier=8)
    config.patch = dataclasses.replace(config.patch, out_sizes=(16, 40))
    # 40 is fine for 8 layers (needs > 32); 16 is not.
    model = ConvEnergyNet(config.energy, key=jax.random.key(0))
    with pytest.raises(ValueError, match=r"leave no interior"):
        train(model, _varsize_batches(np.random.default_rng(2), (40,)), config,
              out_dir=tmp_path / "r", verbose=False)
