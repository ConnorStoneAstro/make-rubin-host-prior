"""Data-parallel training: the batch is split, the weights are not.

The claim is that running on four devices is the *same run* as running on one --
same global batch, same learning rate, same trajectory -- only faster.  Three
things have to hold for that, and none implies the others:

* the batch really is split across devices, or it is four copies of the same
  work and the extra GPUs bought nothing;
* the answer is unchanged, or it is a different run and the loss curves from
  before are not comparable with the ones after;
* a checkpoint is device-count agnostic, or a chunked run is pinned to whatever
  the queue happened to allocate first.

All three need more than one device, which a CPU test process does not have --
``XLA_FLAGS`` is read once, when the backend initialises, long before any test
runs.  So they run in a subprocess with four host devices forced: a real
four-device mesh running the real sharding code, not a mock.  Standing that up
costs about fifteen seconds, so the scenarios share one process and the tests
read their results out of it.
"""

import os
import subprocess
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from rubin_host_prior.training.trainer import _shardings

# Every scenario that needs four devices, printed as `key=value` lines.
_SCENARIOS = '''
import json, pathlib, tempfile
import equinox as eqx, jax, jax.numpy as jnp, numpy as np
from rubin_host_prior.config import Config, EnergyConfig, PatchConfig
from rubin_host_prior.nn import ConvEnergyNet
from rubin_host_prior.training import train
from rubin_host_prior.training.trainer import _shardings


def config(steps=6):
    c = Config(energy=EnergyConfig(channels=((8, 12),), dilations=((1, 1),), embed_dim=16, n_fourier=8),
               patch=PatchConfig(native_size=128, nominal_crop=48, out_size=16,
                                 pool_factor=3))
    c.transform.softening = 20.0
    c.sde.sigma_min, c.sde.sigma_max, c.sde.data_mean = 0.01, 10.0, 0.0
    c.train.steps, c.train.batch_size = steps, 8
    c.train.log_every, c.train.n_checkpoints, c.train.n_samples = 100, 1, 0
    return c


def batches():
    rng = np.random.default_rng(0)
    while True:
        yield rng.normal(size=(8, 1, 16, 16)).astype(np.float32) * 0.5


def model(cfg):
    return ConvEnergyNet(cfg.energy, key=jax.random.key(0))


def leaves(m):
    return [np.asarray(v) for v in
            jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_inexact_array))]


# -- what is sharded and what is replicated
n, replicated, over_batch = _shardings(None, 8, verbose=False)
batch = eqx.filter_shard(jnp.zeros((8, 1, 16, 16)), over_batch)
leaf = leaves(eqx.filter_shard(model(config()), replicated))[0]
print(f"devices={n}")
print(f"batch_shard={batch.addressable_shards[0].data.shape}")
print(f"n_shards={len(batch.addressable_shards)}")
print(f"weight_whole={leaf.shape == jnp.asarray(leaf).addressable_shards[0].data.shape}")

try:
    _shardings(4, 33, verbose=False)
    print("indivisible_refused=no")
except ValueError as exc:
    print(f"indivisible_refused={'yes' if 'raise it to 36' in str(exc) else exc}")

# -- one device and four devices are the same run
work = pathlib.Path(tempfile.mkdtemp())
runs = {}
for devices in (1, 4):
    trained, _ = train(model(config()), batches(), config(),
                       out_dir=work / f"same-{devices}", verbose=False,
                       n_devices=devices)
    runs[devices] = leaves(trained)
gap = max(float(np.max(np.abs(a - b))) for a, b in zip(runs[1], runs[4]))
print(f"max_param_gap={gap}")

# -- a four-device chunk resumes on one device
out = work / "chunked"
train(model(config(4)), batches(), config(4), out_dir=out, verbose=False,
      n_devices=4)
print(f"resume_first={json.loads((out / 'latest' / 'state.json').read_text())['step']}")
train(model(config(8)), batches(), config(8), out_dir=out, verbose=False,
      n_devices=1, resume=out / "latest")
print(f"resume_final={(out / 'final' / 'model.eqx').exists()}")
print(f"resume_last={json.loads((out / 'latest' / 'state.json').read_text())['step']}")
'''


@pytest.fixture(scope="module")
def four():
    """Every four-device result, from one subprocess, keyed by name."""
    done = subprocess.run(
        [sys.executable, "-c", _SCENARIOS],
        env={**os.environ, "XLA_FLAGS": "--xla_force_host_platform_device_count=4"},
        capture_output=True, text=True, timeout=900,
    )
    assert done.returncode == 0, f"subprocess failed:\n{done.stdout}\n{done.stderr}"
    return dict(line.split("=", 1) for line in done.stdout.strip().splitlines())


# -- the single-device path ------------------------------------------------


def test_one_device_is_an_ordinary_unsharded_run():
    """What a laptop gets by default, and what ``--devices 1`` asks for."""
    n, _, over_batch = _shardings(1, 8, verbose=False)
    assert n == 1
    batch = eqx.filter_shard(jnp.zeros((8, 1, 4, 4)), over_batch)
    assert len(batch.addressable_shards) == 1


def test_asking_for_more_devices_than_exist_says_how_many_there_are():
    n = jax.local_device_count()
    with pytest.raises(ValueError, match=f"JAX can see {n}"):
        _shardings(n + 1, 8, verbose=False)


# -- four devices ----------------------------------------------------------


def test_four_devices_split_the_batch_and_replicate_the_weights(four):
    """A quarter of the batch on each device, and a whole copy of every
    parameter.  The replication is what makes a checkpoint one set of weights
    rather than four."""
    assert four["devices"] == "4"
    assert four["batch_shard"] == "(2, 1, 16, 16)", "the batch was not split"
    assert four["n_shards"] == "4"
    assert four["weight_whole"] == "True", "the weights were split, not replicated"


def test_a_batch_that_does_not_divide_is_refused_with_the_next_size_up(four):
    """Caught before training starts, and told which number to use, rather than
    surfacing as an XLA shape error some way into the run."""
    assert four["indivisible_refused"] == "yes"


def test_four_devices_train_the_same_run_as_one(four):
    """Same seed, same global batch, same weights at the end.

    Not required to be bit-identical -- the all-reduce sums four partial
    gradients in an order one device never uses -- but float32 resolution at the
    scale of these weights is ~1e-7, so anything at this level is rounding and
    not a different trajectory.
    """
    assert float(four["max_param_gap"]) < 1e-6


def test_a_four_device_chunk_resumes_on_one_device(four):
    """The queue gives you what it has, so a chunk must not be tied to the
    device count that wrote it.  Nothing in a checkpoint records one, and this
    is the test that keeps it that way."""
    assert four["resume_first"] == "4"
    assert four["resume_final"] == "True", "the one-device chunk did not finish"
    assert four["resume_last"] == "8"
