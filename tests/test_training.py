"""Training loop, EMA, checkpoints, and config serialisation."""

import dataclasses
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rubin_host_prior.config import (BANDS, AugmentConfig, Config, EnergyConfig,
                                     PatchConfig)
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
        energy=EnergyConfig(channels=((8, 12),), embed_dim=16, n_fourier=8),
        patch=PatchConfig(native_size=128, nominal_crop=48, out_size=16,
                          pool_factor=3),
    )
    # The measured fields, which a real config gets from prepare_config.py and
    # which VESDE.from_config refuses to invent.
    c.transform.softening = 20.0
    c.sde.sigma_min, c.sde.sigma_max, c.sde.data_mean = 0.01, 10.0, 0.0
    c.train.steps = 12
    c.train.batch_size = 4
    c.train.log_every = 4
    c.train.n_checkpoints = 0
    for k, v in train_kw.items():
        setattr(c.train, k, v)
    return c


#: ``_config()`` is out_size 16 with R = 2, so the loader would hand the model
#: 16 + 2*loss_margin = 24 px and the loss would land on all 16.  The fakes
#: here mirror that, or they would be testing a geometry nothing produces.
def _batches(rng, shape=(4, 1, 24, 24)):
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
    expected = 0.25 * tiny_model.branches[0].head.weight + 0.75 * other.branches[0].head.weight
    np.testing.assert_allclose(np.asarray(mixed.branches[0].head.weight), np.asarray(expected))
    # Static fields survive untouched.
    assert mixed.config == tiny_model.config
    assert mixed.embed.fourier.freqs == tiny_model.embed.fourier.freqs


def test_ema_at_decay_zero_is_the_live_model(tiny_model):
    other = ConvEnergyNet(tiny_model.config, key=jax.random.key(98))
    mixed = ema_update(tiny_model, other, jnp.asarray(0.0))
    np.testing.assert_allclose(
        np.asarray(mixed.branches[0].head.weight), np.asarray(other.branches[0].head.weight)
    )


# -- optimizer -------------------------------------------------------------


def test_optimizer_warms_up_then_holds():
    cfg = _config(learning_rate=1e-3, warmup_steps=10, cosine_decay=False).train
    opt = make_optimizer(cfg)
    model = ConvEnergyNet(
        EnergyConfig(channels=((4,),), embed_dim=8, n_fourier=4), key=jax.random.key(0)
    )
    params = eqx.filter(model, eqx.is_inexact_array)
    state = opt.init(params)
    grads = jax.tree_util.tree_map(lambda p: jnp.ones_like(p), params)
    steps = []
    for _ in range(20):
        updates, state = opt.update(grads, state, params)
        steps.append(float(jnp.abs(updates.branches[0].head.weight).max()))
    assert steps[0] < steps[5] < steps[9]  # warming up
    assert steps[15] == pytest.approx(steps[19], rel=1e-3)  # then flat


def test_cosine_schedule_decays():
    cfg = _config(learning_rate=1e-3, warmup_steps=2, cosine_decay=True, steps=50).train
    opt = make_optimizer(cfg)
    model = ConvEnergyNet(
        EnergyConfig(channels=((4,),), embed_dim=8, n_fourier=4), key=jax.random.key(0)
    )
    params = eqx.filter(model, eqx.is_inexact_array)
    state = opt.init(params)
    grads = jax.tree_util.tree_map(lambda p: jnp.ones_like(p), params)
    mags = []
    for _ in range(50):
        updates, state = opt.update(grads, state, params)
        mags.append(float(jnp.abs(updates.branches[0].head.weight).max()))
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
    # The cadence is a config field, not a train() argument: a default in the
    # signature would be a second source of truth alongside TrainConfig.
    config = _config(steps=8, eval_every=4)
    model = ConvEnergyNet(config.energy, key=jax.random.key(0))
    rng = np.random.default_rng(2)
    records = []
    train(
        model, _batches(rng), config, out_dir=tmp_path / "run",
        eval_batch=rng.normal(size=(8, 1, 16, 16)).astype(np.float32),
        on_log=records.append,
    )
    evals = [r for r in records if r.get("event") == "eval"]
    assert evals
    assert len(evals[0]["loss_by_sigma"]) == 8
    assert evals[0]["sigma"][0] < evals[0]["sigma"][-1]


def test_no_out_size_is_too_small_for_the_loss_any_more(tmp_path):
    """Under the old contract a patch had to exceed 4R or the loss had no
    interior, and this configuration -- 16 px patches, R = 8 -- was a hard
    error raised before the first batch.

    The loader now carries 2R of context on every side, so the loss lands on
    the whole nominal crop whatever its size and there is nothing left to
    reject.  What used to raise now trains.
    """
    config = _config(steps=2)
    config.energy = EnergyConfig(channels=((8,) * 8,), embed_dim=16, n_fourier=8)
    assert config.patch.out_size < 4 * config.energy.receptive_radius
    model = ConvEnergyNet(config.energy, key=jax.random.key(0))
    fed = config.patch.out_size + 2 * config.energy.loss_margin
    train(model, _batches(np.random.default_rng(3), shape=(4, 1, fed, fed)),
          config, out_dir=tmp_path / "run", verbose=False)
    assert (tmp_path / "run" / "final" / "model.eqx").exists()


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
    assert cfg.transform.softening == config.transform.softening
    np.testing.assert_array_equal(
        np.asarray(back_ema.branches[0].head.weight),
        np.asarray(ema.branches[0].head.weight),
    )
    back_live, _, _ = load_checkpoint(tmp_path / "ck", which="model")
    np.testing.assert_array_equal(
        np.asarray(back_live.branches[0].head.weight), np.asarray(model.branches[0].head.weight)
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
    assert back.transform.softening == config.transform.softening


def _script(name: str):
    """Import a script by path -- ``scripts/`` is not a package."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        name.removesuffix(".py"), root / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_script_decides_a_config_value_for_itself():
    """``config.py`` is the one source of truth for defaults.

    A flag that carries its own number overrides the config on every run,
    passed or not.  That is not hypothetical: ``prepare_config.py`` assigned
    ``out_size = 64`` and ``softening_sigma = 1.0`` unconditionally, so the
    values in ``config.py`` -- and the 128 px geometry its docstring describes
    -- were never the ones that got trained.

    Asked of each parser rather than grepped out of the source: the same rule
    written as a string literal is a test of the file's formatting.
    """
    must_defer = {
        "prepare_config.py": ("softening_sigma", "pool_factor", "out_size"),
        "train.py": ("steps", "batch_size", "lr", "n_layers", "out_sizes",
                     "n_checkpoints", "n_samples", "eval_every", "eval_size"),
        "sample.py": ("n", "size", "steps"),
    }
    for name, flags in must_defer.items():
        p = _script(name).parser()
        for flag in flags:
            assert p.get_default(flag) is None, (
                f"{name} --{flag.replace('_', '-')} carries its own default, "
                f"which overrides the config whether or not it is passed"
            )


def test_the_config_builder_still_knows_where_to_read_and_write():
    """The exception, and why it is one: ``--shards`` and ``--out`` are not
    config fields but where the script runs, and in practice always the same
    two paths.  The shard default follows ``ExtractionConfig.out`` rather than
    writing the extraction's output directory down a second time."""
    from rubin_host_prior.selection import ExtractionConfig

    p = _script("prepare_config.py").parser()
    assert p.get_default("out")
    assert p.get_default("shards").startswith(ExtractionConfig.out)


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
    cfg = EnergyConfig(channels=((6, 8),), embed_dim=16, n_fourier=8)
    a = ConvEnergyNet(cfg, key=jax.random.key(1))
    b = ConvEnergyNet(cfg, key=jax.random.key(2))
    assert a.embed.fourier.freqs == b.embed.fourier.freqs
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    ema_update(a, b, jnp.asarray(0.5))  # must not raise

    other = ConvEnergyNet(
        EnergyConfig(channels=((6, 8),), embed_dim=16, n_fourier=8, fourier_seed=7),
        key=jax.random.key(1),
    )
    assert other.embed.fourier.freqs != a.embed.fourier.freqs


# -- the derived loss crop, and variable patch sizes -----------------------


def test_setup_note_is_printed_and_reports_the_derived_crop(tmp_path, capsys):
    """Requested behaviour: training announces how much it is cropping, so that
    tinkering with n_layers or kernel_size is never silent."""
    config = _config()
    config.energy = EnergyConfig(channels=((8,) * 3,), embed_dim=16, n_fourier=8)
    model = ConvEnergyNet(config.energy, key=jax.random.key(0))
    train(model, _batches(np.random.default_rng(0), shape=(4, 1, 28, 28)),
          config, out_dir=tmp_path / "r")
    out = capsys.readouterr().out
    assert "branch 0: 3 x 3x3, dilations 1x1x1" in out
    assert "loss crop = 2R = 6 px from every side" in out
    # The report is given what the loader *feeds*, 16 + 2*6, so the interior it
    # names is the nominal crop itself rather than something smaller.
    assert "patch   28x28" in out
    assert "loss on interior 16x16" in out
    assert "out_size 16: fed 28, context 6 px per side" in out
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
    # The loader feeds size + 2*margin, so the loss lands on the size itself.
    assert head["interior_sizes"] == [16, 24]


def _varsize_batches(rng, sizes, batch=4, margin=4):
    """``sizes`` are loss-region sizes; what is yielded carries the context."""
    k = 0
    while True:
        s = sizes[k % len(sizes)] + 2 * margin
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


def test_usable_size_range_no_longer_has_an_architectural_floor():
    """The lower bound used to be 4R + 1 -- below that the loss had no interior.
    The loader supplying the context removed that floor entirely; what is left
    is the stamp, which still has to hold the nominal crop itself."""
    c = Config(energy=EnergyConfig(channels=((32,) * 8,)),
               patch=PatchConfig(native_size=224, nominal_crop=192, out_size=64,
                                 pool_factor=3))
    assert c.usable_size_range() == (1, 74)   # 224 // 3
    deep = Config(energy=EnergyConfig(channels=((32,) * 20,)),
                  patch=PatchConfig(native_size=224, nominal_crop=192,
                                    out_size=64, pool_factor=3))
    # R = 20 now, and it makes no difference to what sizes can be served.
    assert deep.usable_size_range() == (1, 74)


def test_real_context_is_what_the_stamp_has_left_over():
    """How much of the 2R border is genuine sky, and where it runs out."""
    c = Config(energy=EnergyConfig(channels=((32,) * 8,)),      # R = 8, 2R = 16
               patch=PatchConfig(native_size=384, nominal_crop=288, out_size=96,
                                 pool_factor=3))
    assert c.energy.loss_margin == 16
    # 384/3 = 128 pooled in the stamp, 96 of it cropped -> 16 left each side.
    assert c.real_context(96) == pytest.approx(16.0)
    assert c.real_context(64) == pytest.approx(32.0)   # a smaller crop leaves more
    assert c.real_context(128) == pytest.approx(0.0)   # the crop fills the stamp
    assert c.real_context(160) < 0                     # it does not even fit


def test_check_sizes_flags_a_border_that_is_all_reflection():
    c = Config(energy=EnergyConfig(channels=((32,) * 8,)),
               patch=PatchConfig(native_size=384, nominal_crop=288, out_size=96,
                                 pool_factor=3))
    assert c.check_sizes() == []      # 16 real of a 16 px border: exactly enough

    c.patch = dataclasses.replace(c.patch, out_sizes=(96, 128))
    assert any("128 leaves no real context" in w for w in c.check_sizes())

    # A size that does not fit the stamp never reaches check_sizes: PatchConfig
    # refuses it outright, which is the better place for it.
    with pytest.raises(ValueError, match="needs 192 native pixels"):
        PatchConfig(native_size=96, nominal_crop=96, out_size=32,
                    pool_factor=3, out_sizes=(32, 64))


def test_check_sizes_flags_a_crop_that_is_mostly_context():
    """The efficiency warning now measures the fed patch against the loss
    region, which is where the arithmetic actually goes."""
    c = Config(energy=EnergyConfig(channels=((32,) * 20,)),   # R = 20, 2R = 40
               patch=PatchConfig(native_size=1024, nominal_crop=48, out_size=16,
                                 pool_factor=3))
    # fed 16 + 80 = 96 to train on 16: (16/96)^2 = 3%.
    assert any("only 3% of the arithmetic" in w for w in c.check_sizes())


def test_report_marks_patches_that_are_mostly_margin():
    from rubin_host_prior import geometry as geo

    text = geo.report((40, 64), ((1,) * 8,))
    assert "mostly margin" in text.split("patch   40")[1].split("\n")[0]
    assert "mostly margin" not in text.split("patch   64")[1].split("\n")[0]


def test_defaults_leave_room_for_translation():
    """Regression: native_size = out_size * pool_factor exactly (e.g. 384 for a
    128 px patch) silently disables the translation augmentation, because the
    crop then fills the whole stamp.  The defaults must not be in that state."""
    c = Config()
    assert c.patch.out_size == 128
    assert c.patch.max_translate_native > 0
    assert c.check_sizes() == []


def test_zero_translation_room_is_reported():
    c = Config(patch=PatchConfig(native_size=384, nominal_crop=384, out_size=128,
                                 pool_factor=3))
    assert c.patch.max_translate_native == 0
    assert any("no room" in w for w in c.check_sizes())


def test_no_warning_when_translation_is_switched_off():
    c = Config(patch=PatchConfig(native_size=384, nominal_crop=384, out_size=128,
                                 pool_factor=3),
               augment=AugmentConfig(translate=False))
    # A crop that fills its stamp also has no real context, which is a separate
    # and legitimate warning -- this test is about the translation one.
    assert not any("translation" in w for w in c.check_sizes())


# -- checkpoints and the samples that go with them --------------------------


def test_checkpoints_are_spread_over_the_run_not_set_by_an_interval():
    """``ckpt_every`` had to be recomputed by hand every time ``steps`` changed,
    and getting it wrong meant either one checkpoint or thousands."""
    from rubin_host_prior.config import TrainConfig

    assert TrainConfig(steps=1000, n_checkpoints=10).checkpoint_steps() == [
        100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # The last one always lands exactly on the final step.
    for steps, n in ((1000, 3), (99, 7), (200_000, 10)):
        got = TrainConfig(steps=steps, n_checkpoints=n).checkpoint_steps()
        assert len(got) == n and got[-1] == steps and got == sorted(set(got))
    # Fewer steps than checkpoints asked for gives one per step, not duplicates.
    assert TrainConfig(steps=3, n_checkpoints=10).checkpoint_steps() == [1, 2, 3]
    assert TrainConfig(steps=1000, n_checkpoints=0).checkpoint_steps() == []


def test_a_run_leaves_a_history_of_checkpoints_and_sample_grids(tmp_path):
    """Ten checkpoints over a run, each with a picture of what the model draws
    at that point -- the thing a loss curve cannot show you."""
    config = _config(steps=6, n_checkpoints=3, n_samples=4, sample_steps=2)
    model = ConvEnergyNet(config.energy, key=jax.random.key(0))
    rng = np.random.default_rng(0)
    train(model, _batches(rng), config, out_dir=tmp_path, verbose=False)

    steps = [2, 4, 6]
    kept = sorted(p.name for p in (tmp_path / "checkpoints").iterdir())
    assert kept == [f"step-{s:08d}" for s in steps]
    assert sorted(p.name for p in (tmp_path / "samples").iterdir()) == [
        f"step-{s:08d}.png" for s in steps]

    # The weights are kept every time; the optimiser state, which is two more
    # copies of them, only in `latest`.
    for s in steps:
        d = tmp_path / "checkpoints" / f"step-{s:08d}"
        assert (d / "model.eqx").exists() and (d / "ema.eqx").exists()
        assert not (d / "opt_state.eqx").exists()
    assert (tmp_path / "latest" / "opt_state.eqx").exists()

    # Every checkpoint is loadable on its own, and says which step it is.
    reloaded, _, step = load_checkpoint(tmp_path / "checkpoints" / "step-00000004")
    assert step == 4 and n_parameters(reloaded) == n_parameters(model)

    # And the log records what was drawn, so a diverged sampler is visible in
    # the log rather than only as a blank figure.
    events = [json.loads(line) for line in
              (tmp_path / "log.jsonl").read_text().splitlines()]
    ckpts = [e for e in events if e.get("event") == "checkpoint"]
    assert [e["step"] for e in ckpts] == steps
    assert all(e["sample_nonfinite"] == 0 for e in ckpts)
    assert all("sample_mean" in e and "sample_seconds" in e for e in ckpts)


def test_sampling_never_costs_a_run(tmp_path, monkeypatch):
    """Hours of training must not be lost to a diagnostic.  Sampling is the one
    step here that can exhaust device memory on its own."""
    import rubin_host_prior.diffusion.sampler as sampler

    def boom(*a, **kw):
        raise RuntimeError("RESOURCE_EXHAUSTED: out of memory")

    monkeypatch.setattr(sampler, "sample_interior", boom)
    config = _config(steps=2, n_checkpoints=1, n_samples=4, sample_steps=2)
    model = ConvEnergyNet(config.energy, key=jax.random.key(0))
    train(model, _batches(np.random.default_rng(0)), config,
          out_dir=tmp_path, verbose=False)

    assert (tmp_path / "final" / "model.eqx").exists()
    events = [json.loads(line) for line in
              (tmp_path / "log.jsonl").read_text().splitlines()]
    failed = [e for e in events if "sample_error" in e]
    assert failed and "RESOURCE_EXHAUSTED" in failed[0]["sample_error"]


# -- stopping and picking up again ------------------------------------------


def _stop_after(n_steps, shape=(4, 1, 16, 16)):
    """A batch stream that raises SIGUSR1 to this process at step ``n_steps``,
    which is what a scheduler does ahead of the wall clock."""
    import os
    import signal as signal_module

    rng = np.random.default_rng(0)
    step = 0
    while True:
        step += 1
        if step == n_steps:
            os.kill(os.getpid(), signal_module.SIGUSR1)
        yield rng.normal(size=shape).astype(np.float32) * 0.5


def test_a_chunked_run_picks_up_where_it_stopped(tmp_path):
    """The workflow this exists for: a scheduler that will not give you a long
    job, so the run is a series of short ones.

    Four things have to come back or the second chunk is a different training
    run: the weights, the EMA copy, the optimiser state -- which holds Adam's
    moments *and* the learning-rate schedule's position -- and the step.
    """
    config = _config(steps=12, n_checkpoints=0, n_samples=0)
    model = ConvEnergyNet(config.energy, key=jax.random.key(0))
    train(model, _stop_after(4), config, out_dir=tmp_path, verbose=False)

    stopped_at = json.loads((tmp_path / "latest" / "state.json").read_text())["step"]
    assert 0 < stopped_at < 12
    assert not (tmp_path / "final").exists()

    # Chunk two: same command, same config, same output directory.
    fresh = ConvEnergyNet(config.energy, key=jax.random.key(0))
    train(fresh, _batches(np.random.default_rng(1)), config, out_dir=tmp_path,
          verbose=False, resume=tmp_path / "latest")

    assert (tmp_path / "final" / "model.eqx").exists()
    events = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    headers = [e for e in events if e.get("event") == "start"]
    assert len(headers) == 2 and headers[1]["start_step"] == stopped_at
    # The second chunk began after the first, not at step 1.
    second = [e["step"] for e in events[events.index(headers[1]):] if "step" in e]
    assert min(second) > stopped_at


def test_only_latest_can_be_resumed_from(tmp_path):
    """The numbered checkpoints deliberately carry no optimiser state -- it is
    two more copies of the parameters and only the newest one is ever wanted --
    so `latest` is the one to point at."""
    config = _config(steps=8, n_checkpoints=2, n_samples=0)
    model = ConvEnergyNet(config.energy, key=jax.random.key(0))
    train(model, _batches(np.random.default_rng(0)), config,
          out_dir=tmp_path, verbose=False)

    with pytest.raises(FileNotFoundError):
        train(model, _batches(np.random.default_rng(0)), _config(
                  steps=16, n_checkpoints=0, n_samples=0),
              out_dir=tmp_path / "two", verbose=False,
              resume=tmp_path / "checkpoints" / "step-00000004")


def test_a_finished_run_cannot_be_resumed_into_nothing(tmp_path):
    """train.steps is the length of the whole run across every chunk, so a
    checkpoint already at it has nothing left to do -- said plainly rather than
    as a loop that runs zero times and writes a misleading `final`."""
    config = _config(steps=4, n_checkpoints=1, n_samples=0)
    model = ConvEnergyNet(config.energy, key=jax.random.key(0))
    train(model, _batches(np.random.default_rng(0)), config,
          out_dir=tmp_path, verbose=False)

    with pytest.raises(ValueError, match="nothing left to do"):
        train(model, _batches(np.random.default_rng(0)), config,
              out_dir=tmp_path / "again", verbose=False,
              resume=tmp_path / "latest")


def test_a_stop_signal_checkpoints_and_leaves_final_alone(tmp_path):
    """SIGUSR1 is what a scheduler sends ahead of the wall clock.  The loop
    finishes its step, writes `latest` with the optimiser state, and returns --
    and deliberately does *not* write `final`, because the run is not finished
    and that is what tells the next chunk there is more to do."""
    import os
    import signal as signal_module

    config = _config(steps=1000, n_checkpoints=0, n_samples=0)
    model = ConvEnergyNet(config.energy, key=jax.random.key(0))

    fired = {"done": False}

    def batches():
        rng = np.random.default_rng(0)
        n = 0
        while True:
            n += 1
            # Raise it to ourselves once, a few steps in.
            if n == 3 and not fired["done"]:
                fired["done"] = True
                os.kill(os.getpid(), signal_module.SIGUSR1)
            yield rng.normal(size=(4, 1, 16, 16)).astype(np.float32) * 0.5

    train(model, batches(), config, out_dir=tmp_path, verbose=False)

    assert (tmp_path / "latest" / "opt_state.eqx").exists()
    assert not (tmp_path / "final").exists(), "a stopped run is not a finished one"
    events = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    stopped = [e for e in events if e.get("event") == "stopped"]
    assert len(stopped) == 1
    assert stopped[0]["signal"] == "SIGUSR1"
    assert stopped[0]["step"] < config.train.steps

    # And the handler is put back: this is a library, not an application.
    assert signal_module.getsignal(signal_module.SIGUSR1) is signal_module.SIG_DFL
