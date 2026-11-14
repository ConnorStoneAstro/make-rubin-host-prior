"""The vendored loader under ``standalone/`` must stay in step with the trainer.

A copy of the architecture is the one piece of duplication in this repo that can
fail silently: the module field declarations in ``nn/ncsnpp.py`` *are* the
checkpoint format -- their order and their static-ness decide how equinox lays
out the pytree -- so a copy that has drifted produces a shape error at best and
a model that loads but computes something else at worst.

Two defences, both here.  ``_net.py`` is generated rather than transcribed, and
the generator is re-run and compared; and a checkpoint is loaded through both
paths and the numbers are required to be *identical*, not close.
"""

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rubin_host_prior.config import (Config, NCSNppConfig, PatchConfig,
                                     SDEConfig, TransformConfig)
from rubin_host_prior.inference import load_prior as load_full
from rubin_host_prior.training.checkpoint import save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
STANDALONE = ROOT / "standalone"
SIZE = 32


@pytest.fixture(scope="module")
def standalone_loader():
    sys.path.insert(0, str(STANDALONE))
    try:
        from host_prior import load_prior

        yield load_prior
    finally:
        sys.path.remove(str(STANDALONE))


def _checkpoint(tmp_path, architecture="ncsnpp_energy", energy_form="sum"):
    c = Config(architecture=architecture)
    c.ncsnpp = NCSNppConfig(nf=8, ch_mult=(1, 2, 2), num_blocks=1,
                            energy_form=energy_form)
    c.patch = PatchConfig(native_size=128, out_size=SIZE, pool_factor=2)
    c.sde = SDEConfig(sigma_min=0.02, sigma_max=5.0, data_mean=3.1, data_std=0.7)
    c.transform = dataclasses.replace(c.transform, softening=5.0)
    model = c.build_model(jax.random.key(3))
    d = tmp_path / f"{architecture}-{energy_form}"
    save_checkpoint(d, 4242, c, model, model)
    return d


def test_the_exported_architecture_is_not_stale():
    """``scripts/export_standalone.py --check``.  If this fails, the trainer's
    architecture changed and the vendored copy was not regenerated -- run the
    exporter without ``--check`` and commit the result."""
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_standalone.py"), "--check"],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


@pytest.mark.parametrize("architecture,energy_form", [
    ("ncsnpp", "sum"), ("ncsnpp_energy", "sum"), ("ncsnpp_energy", "dae"),
    # `ncsnpp_encoder` was missing from this list, and that is exactly how it
    # came to load with a correct score and then refuse to give up its energy:
    # the vendored `is_energy` tested `isinstance(model, NCSNppEnergy)` and the
    # encoder is a sibling of that class, not a subclass.  'sum' only -- the
    # config rejects 'dae' for this architecture, and a test below holds that.
    ("ncsnpp_encoder", "sum")])
def test_both_loaders_give_identical_numbers(standalone_loader, tmp_path,
                                             architecture, energy_form):
    """Identical, not close.  The two paths run the same code on the same
    weights, so any difference at all is a divergence rather than round-off,
    and a tolerance here would hide exactly what this is for."""
    d = _checkpoint(tmp_path, architecture, energy_form)
    full, lean = load_full(d), standalone_loader(d)

    assert lean.step == full.step == 4242
    assert lean.size == full.size == SIZE
    assert lean.pool_factor == full.pool_factor
    # Every architecture here is a potential except plain `ncsnpp`, which
    # predicts the score directly.  Stated that way round rather than naming
    # the one that is, which is what let the encoder through.
    assert lean.is_energy == full.is_energy == (architecture != "ncsnpp")
    assert lean.sigma_min == full.sigma_min
    assert lean.sigma_max == full.sigma_max

    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(3.0, 1.0, (3, 1, SIZE, SIZE)), jnp.float32)
    np.testing.assert_array_equal(np.asarray(lean.score(x, 0.7)),
                                  np.asarray(full.score(x, 0.7)))
    if full.is_energy:
        np.testing.assert_array_equal(np.asarray(lean.energy(x, 0.7)),
                                      np.asarray(full.energy(x, 0.7)))
        # The offset too: it is an inference-side argument, so this is the
        # only place the two implementations of it meet.
        np.testing.assert_array_equal(
            np.asarray(lean.energy(x, 0.7, pixel_offset=0.25)),
            np.asarray(full.energy(x, 0.7, pixel_offset=0.25)))
    np.testing.assert_array_equal(
        np.asarray(lean.sample(jax.random.key(0), n=2, n_steps=4)),
        np.asarray(full.sample(jax.random.key(0), n=2, n_steps=4)))

    flux = rng.normal(50.0, 30.0, (SIZE, SIZE))
    assert flux.min() < 0  # the regime where the two flux maps differ
    for name in ("to_x", "to_flux", "to_flux_exact"):
        a = getattr(lean, name)(flux if name == "to_x" else lean.to_x(flux))
        b = getattr(full, name)(flux if name == "to_x" else full.to_x(flux))
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)


def test_the_standalone_loader_refuses_an_architecture_it_does_not_carry(
        standalone_loader, tmp_path):
    """It has the U-Net only.  A dilated-convolution ``energy`` checkpoint must
    be named and refused, not loaded into the wrong skeleton."""
    c = Config(architecture="energy")
    c.patch = PatchConfig(native_size=512, out_size=128, pool_factor=2)
    c.sde = SDEConfig(sigma_min=0.02, sigma_max=5.0, data_mean=3.1)
    c.transform = dataclasses.replace(c.transform, softening=5.0)
    d = tmp_path / "energy"
    save_checkpoint(d, 1, c, c.build_model(jax.random.key(0)))
    with pytest.raises(ValueError, match="rubin_host_prior"):
        standalone_loader(d)


def test_the_standalone_loader_imports_nothing_it_should_not(tmp_path):
    """Three dependencies: jax, equinox, numpy.  And emphatically **not**
    ``rubin_host_prior`` -- the whole point is that it stands alone."""
    d = _checkpoint(tmp_path)
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(STANDALONE)!r})\n"
        "from host_prior import load_prior\n"
        f"p = load_prior({str(d)!r})\n"
        "p.energy(__import__('jax').numpy.zeros((1, 1, 32, 32)))\n"
        "banned = {'h5py', 'optax', 'matplotlib', 'astropy', 'pandas', "
        "'pyarrow', 'pyvo', 'requests', 'lsst', 'yaml', 'rubin_host_prior'}\n"
        "found = sorted({m.split('.')[0] for m in sys.modules} & banned)\n"
        "print(found)\n"
        "assert not found, found"
    )
    # A fresh interpreter: this test is about what the import *pulls in*, and
    # the pytest process has already imported half of it.
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=str(ROOT))
    assert out.returncode == 0, out.stdout + out.stderr[-3000:]


def test_the_vendored_config_has_every_field(standalone_loader):
    """A field the vendored copy has never heard of is dropped by its
    ``from_dict``, so the model silently falls back to a default for something
    the checkpoint specified -- which is how it came to compute a different
    ``input_offset`` from the same weights.

    Only *presence* is checked, not the values.  The trainer's defaults are
    where experiments happen, and a real checkpoint writes every field
    explicitly, so divergent defaults cannot reach a loaded model; a missing
    field can.
    """
    import dataclasses as dc

    from host_prior._support import NCSNppConfig as Lean

    from rubin_host_prior.config import NCSNppConfig as Full

    lean = {f.name for f in dc.fields(Lean)}
    full = {f.name for f in dc.fields(Full)}
    missing = sorted(full - lean)
    assert not missing, f"the vendored config is missing fields: {missing}"


def test_an_unknown_config_field_does_not_stop_a_model_loading(standalone_loader,
                                                               tmp_path):
    """A config from a newer trainer may carry fields this copy has never heard
    of.  Ignoring them is right: a field that does not change the architecture
    should not block a load, and one that does will surface as a shape error."""
    d = _checkpoint(tmp_path)
    cfg = json.loads((d / "config.json").read_text())
    cfg["ncsnpp"]["some_future_knob"] = 7
    cfg["a_whole_new_section"] = {"x": 1}
    (d / "config.json").write_text(json.dumps(cfg))
    prior = standalone_loader(d)
    assert prior.size == SIZE


def test_the_vendored_map_still_reads_a_floored_checkpoint(standalone_loader):
    """The training package dropped the floor, the pedestal and the restoration
    term; the vendored loader has to keep them, because checkpoints trained
    with them exist and a prior is only its weights *and* its map.

    So the floored map is held here against the formula written out, since
    there is no longer a second implementation to compare it with, and the
    unfloored one against the training package, which is what every new
    checkpoint means.  ``restore_bright_end`` arrived after checkpoints were
    already being written, so a config without the key must mean the map it
    was trained with and not whatever a default became.
    """
    sys.path.insert(0, str(STANDALONE))
    try:
        from host_prior._prior import LogFluxTransform as Vendored, _input_offset
    finally:
        sys.path.remove(str(STANDALONE))
    from rubin_host_prior.data.transform import LogFluxTransform

    base = {"softening": 1.572, "softening_sigma": 0.12, "floor_sigma": 1.0,
            "floor_pedestal": 0.1}
    assert Vendored(softening=1.572).restore_bright_end is False
    assert _input_offset(base) == _input_offset({**base,
                                                 "restore_bright_end": False})
    assert _input_offset(base) != _input_offset({**base,
                                                 "restore_bright_end": True})

    f = np.linspace(-30.0, 500.0, 2001)
    s = 1.572
    for on in (False, True):
        there = Vendored(softening=s, restore_bright_end=on)
        for floor, pedestal in ((13.1, 0.1), (0.0, 0.1), (4.0, None)):
            p = floor if pedestal is None else pedestal
            u = (f - floor) / s
            want = p + s * np.logaddexp(0.0, u)
            if on:
                want = want + (floor - p) / (1.0 + np.exp(-u))
            np.testing.assert_allclose(there.soften(f, floor, pedestal), want,
                                       rtol=1e-12, atol=1e-12)
            x = there.forward(f, floor=floor, pedestal=pedestal)
            np.testing.assert_allclose(x, np.log(want), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(
                there.inverse_exact(x, floor=floor, pedestal=pedestal), f,
                rtol=1e-8, atol=1e-8)

    # A checkpoint written now has a floor and no pedestal or restoration
    # key, and the vendored map with those left at their defaults is the
    # training package's, bit for bit.
    here, there = LogFluxTransform(softening=13.1), Vendored(softening=13.1)
    for floor in (0.0, 13.1, 26.2):
        np.testing.assert_array_equal(there.forward(f, floor=floor),
                                      here.forward(f, floor))
        np.testing.assert_array_equal(
            there.inverse_exact(here.forward(f, floor), floor=floor),
            here.inverse_exact(here.forward(f, floor), floor))
    assert _input_offset({"softening": 13.1, "softening_sigma": 1.0}) == \
        pytest.approx(here.sky_level)

    # A floor with no pedestal key is the floor itself: what the map was
    # before the two were split, and what the training package writes again.
    # Held to the training package's own property, because this is a static
    # field on the module and the two loaders computing it differently is a
    # different score from the same weights.
    for tr in (TransformConfig(softening=13.1),
               TransformConfig(softening=13.1, floor_sigma=0.0),
               TransformConfig(softening=4.3, softening_sigma=1 / 3,
                               floor_sigma=1.5),
               TransformConfig(softening=13.1, floor_sigma=2.0)):
        c = Config()
        c.transform = tr
        assert _input_offset(dataclasses.asdict(tr)) == c.input_offset, tr


def test_an_older_checkpoints_floor_bounds_still_read(standalone_loader, tmp_path):
    """``floor_sigma_lo``/``hi`` became one ``floor_sigma``.

    A checkpoint written before that carries the pair, and its floor was the
    *midpoint* of the two -- that is exactly what ``Config.input_offset``
    computed then, so reading it that way is a restatement rather than a
    guess.  Nothing else in the file changes, so the model must load and the
    offset must come out at the midpoint's value.
    """
    sys.path.insert(0, str(STANDALONE))
    try:
        from host_prior._prior import _floor_k, _input_offset
    finally:
        sys.path.remove(str(STANDALONE))

    new = {"softening": 26.0, "softening_sigma": 2.0, "floor_sigma": 0.5,
           "floor_pedestal": 0.1}
    old = {"softening": 26.0, "softening_sigma": 2.0, "floor_sigma_lo": 0.25,
           "floor_sigma_hi": 0.75, "floor_pedestal": 0.1}
    assert _floor_k(old) == 0.5
    assert _floor_k(new) == 0.5
    assert _input_offset(old) == _input_offset(new)

    # And a config from before the floor existed at all is still unfloored.
    assert _floor_k({"softening": 26.0}) == 0.0
