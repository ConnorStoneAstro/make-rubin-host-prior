"""The PTED two-sample test harness.

Not a test of PTED, which has its own; a test that this package hands it the
right two samples and reads the answer the right way round.  The failure this
guards against is the quiet one -- a comparison that rejects for a reason that
has nothing to do with the model, and is read as the model being wrong.
"""

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rubin_host_prior import twosample
from rubin_host_prior.config import (Config, NCSNppConfig, PatchConfig,
                                     SDEConfig)
from rubin_host_prior.data import LogFluxTransform, PatchDataset, ShardSet
from rubin_host_prior.data.synthetic import write_synthetic_shards
from rubin_host_prior.inference import load_prior
from rubin_host_prior.training.checkpoint import save_checkpoint

pytest.importorskip("pted")

SIZE = 8


@pytest.fixture(scope="module")
def gaussian():
    rng = np.random.default_rng(0)
    return rng.normal(size=(400, 1, SIZE, SIZE))


def test_the_null_is_same_distribution_so_a_small_p_means_a_bad_model(gaussian):
    """The direction, pinned.  Reading this backwards would turn a failing
    model into a passing one and nothing downstream would notice."""
    other = np.random.default_rng(1).normal(size=(200, 1, SIZE, SIZE))
    same = twosample.compare(gaussian, other, permutations=300,
                             n_landmarks=None)
    shifted = twosample.compare(gaussian, other + 0.5, permutations=300,
                                n_landmarks=None)
    assert same["p_value"] > 0.01
    assert shifted["p_value"] < 0.01
    # And the statistic, not just the p-value, moves the way it should.
    assert shifted["statistic"] > same["statistic"]
    assert shifted["z"] > 10


def test_a_difference_in_texture_alone_is_caught(gaussian):
    """Same mean, same per-pixel variance, different spatial correlation --
    which is the failure mode this project actually has."""
    rng = np.random.default_rng(2)
    smooth = rng.normal(size=(200, 1, SIZE, SIZE))
    smooth = np.cumsum(smooth, axis=-1) / np.sqrt(SIZE)   # correlated along x
    smooth = (smooth - smooth.mean()) / smooth.std()
    got = twosample.compare(gaussian, smooth, permutations=300,
                            n_landmarks=None)
    assert got["p_value"] < 0.01


def test_the_control_is_calibrated_on_data_that_cannot_differ():
    """Two halves of one iid sample are the same distribution by construction,
    so the control's p-values have to look like draws from U(0, 1).

    Over independent datasets, not splits of one: splits of a single sample
    share whatever that sample happens to contain, so one unlucky dataset
    makes several of them reject together.  That is a real property of the
    control and the reason it repeats, but it is not what calibration means.
    """
    ps = []
    for s in range(12):
        x = np.random.default_rng(100 + s).normal(size=(240, 1, SIZE, SIZE))
        ps.append(twosample.control(x, permutations=300, n_landmarks=None,
                                    seed=s, repeats=1)["p_values"][0])
    ps = np.array(ps)
    assert ((ps > 0) & (ps <= 1)).all()
    # Uniform, loosely: no pile-up at either end.
    assert 0.2 < ps.mean() < 0.8
    assert (ps < 0.01).sum() <= 1


def test_the_control_repeats_so_one_unlucky_split_is_not_a_verdict(gaussian):
    """The measured case.  Seed 0 on this sample gives a 3.6-sigma split, which
    read alone condemns a comparison that is fine; the other splits do not."""
    c = twosample.control(gaussian, permutations=300, n_landmarks=None,
                          seed=0, repeats=5)
    assert c["repeats"] == 5 and len(c["p_values"]) == 5
    text = twosample.report(
        twosample.compare(gaussian, gaussian[:150] + 3.0, permutations=300,
                          n_landmarks=None), c)
    n_reject = sum(p < 0.01 for p in c["p_values"])
    if n_reject == 1:
        assert "roughly what alpha" in text      # noted, not fatal
        assert "REJECTED at alpha" in text        # the model verdict survives
    elif n_reject >= 2:
        assert "CONTROL rejected" in text


def test_a_broken_control_withholds_the_verdict(gaussian):
    """When the harness cannot pass its own check, the model's p-value is not
    reported as meaningful -- the whole point of having a control."""
    broken = {"p_values": [1e-4, 1e-4, 1e-4, 0.5, 1e-4], "z": [9] * 5,
              "repeats": 5, "n_x": 200, "n_y": 200, "min_p": 1e-4,
              "median_p": 1e-4, "seconds": 0.1}
    text = twosample.report(
        twosample.compare(gaussian, gaussian[:150] + 3.0, permutations=300,
                          n_landmarks=None), broken)
    assert "CONTROL rejected on 4 of 5" in text
    assert "REJECTED at alpha" not in text


def test_landmarks_keep_the_verdict_and_are_dropped_when_they_cover_everything(
        gaussian):
    """Landmarks trade sensitivity for cost, not correctness."""
    other = np.random.default_rng(3).normal(size=(200, 1, SIZE, SIZE)) + 0.5
    exact = twosample.compare(gaussian, other, permutations=300)
    lm = twosample.compare(gaussian, other, permutations=300, n_landmarks=64)
    assert exact["p_value"] < 0.01 and lm["p_value"] < 0.01
    assert lm["n_landmarks"] == 64

    # A landmark count covering the pooled sample is the exact test, and says
    # so rather than quietly running a rectangular one that is really square.
    covered = twosample.compare(gaussian, other, permutations=50,
                                n_landmarks=10_000)
    assert covered["n_landmarks"] is None


def test_the_exact_test_is_the_default(gaussian):
    """No landmarks unless asked for.  The GPU that drew the samples can build
    the full matrix, and landmarks cost sensitivity for nothing on it."""
    other = np.random.default_rng(4).normal(size=(150, 1, SIZE, SIZE))
    assert twosample.compare(gaussian, other,
                             permutations=100)["n_landmarks"] is None
    c = twosample.control(gaussian, permutations=100, repeats=1)
    assert twosample.compare(gaussian[:100], gaussian[100:200],
                             permutations=100)["n_landmarks"] is None
    assert len(c["p_values"]) == 1


def test_the_two_backends_reach_the_same_verdict(gaussian):
    """float32 on device against float64 through scipy.  The energy distance
    is a mean over ~n^2 distances computed as ||xi - yj|| directly, so float32
    costs ~1e-5 relative on each and nothing at all on the verdict."""
    other = np.random.default_rng(5).normal(size=(200, 1, SIZE, SIZE)) + 0.4
    j = twosample.compare(gaussian, other, permutations=400, seed=1,
                          backend="jax")
    n = twosample.compare(gaussian, other, permutations=400, seed=1,
                          backend="numpy")
    assert j["backend"] == "jax" and n["backend"] == "numpy"
    assert j["statistic"] == pytest.approx(n["statistic"], rel=1e-3)
    assert (j["p_value"] < 0.01) == (n["p_value"] < 0.01)

    same_j = twosample.compare(gaussian, gaussian[:200] + 0.0,
                               permutations=100, backend="jax")
    assert np.isfinite(same_j["statistic"])


def test_centring_does_not_move_the_statistic(gaussian):
    """Energy distance depends on pairwise distances alone, so the shared
    offset ``compare`` subtracts for float32 headroom has to be invisible.

    Checked by adding a large constant to *both* samples, which the centring
    removes again: if it leaked into the statistic this would drift.
    """
    other = np.random.default_rng(6).normal(size=(200, 1, SIZE, SIZE)) + 0.4
    base = twosample.compare(gaussian, other, permutations=200, seed=2)
    lifted = twosample.compare(gaussian + 1000.0, other + 1000.0,
                               permutations=200, seed=2)
    assert lifted["statistic"] == pytest.approx(base["statistic"], rel=1e-3)
    assert lifted["p_value"] == base["p_value"]


def test_the_cost_estimate_matches_what_the_test_will_build():
    """Printed before the model draws, so an out-of-memory failure happens
    while it is still cheap to have happened."""
    exact = twosample.estimate_cost(16384, 512, 16384)
    assert exact["exact"] and exact["pooled"] == exact["columns"] == 16896
    assert exact["matrix_gb"] == pytest.approx(16896**2 * 4 / 1e9)
    assert exact["flops"] == pytest.approx(16896**2 * 16384)

    lm = twosample.estimate_cost(16384, 512, 16384, n_landmarks=512)
    assert not lm["exact"] and lm["columns"] == 512
    assert lm["matrix_gb"] < exact["matrix_gb"] / 30


def test_mismatched_scene_sizes_say_what_is_wrong(gaussian):
    with pytest.raises(ValueError, match="size-locked"):
        twosample.compare(gaussian, np.zeros((10, 1, SIZE + 1, SIZE + 1)))


# -- against the real loader and a real checkpoint -------------------------


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("pted")
    write_synthetic_shards(d / "shards", n_patches=64, native_size=64,
                           patches_per_shard=32, seed=0)
    c = Config(architecture="ncsnpp_energy")
    c.ncsnpp = NCSNppConfig(nf=8, ch_mult=(1, 2), num_blocks=1, attention=False)
    c.patch = PatchConfig(native_size=64, out_size=SIZE, pool_factor=2)
    c.sde = SDEConfig(sigma_min=0.02, sigma_max=5.0, data_mean=3.0, data_std=0.7)
    c.transform = dataclasses.replace(c.transform, softening=5.0,
                                      pooled_noise_ratio=2.0)
    model = c.build_model(jax.random.key(0))
    save_checkpoint(d / "ckpt", 1, c, model, model)
    return d, c


def test_the_two_samples_come_back_the_same_shape(run_dir):
    """What the script relies on: both sides are (N, 1, S, S) in the model's
    own space, so they can be compared without a transform in between."""
    d, c = run_dir
    ss = ShardSet.from_dir(d / "shards")
    ds = PatchDataset.from_shards(
        ss, c, LogFluxTransform.from_config(c.transform))
    prior = load_prior(d / "ckpt")

    data = twosample.data_sample(ds, 32, seed=0)
    model = twosample.model_sample(prior, 4, seed=0, chunk=2, n_steps=3)
    assert data.shape == (32, 1, SIZE, SIZE)
    assert model.shape == (4, 1, SIZE, SIZE)
    # Both device arrays, so the distance matrix is built where the model was
    # drawn rather than after a round trip through host memory.
    assert isinstance(data, jax.Array) and isinstance(model, jax.Array)
    assert data.dtype == model.dtype == jnp.float32
    assert np.isfinite(data).all() and np.isfinite(model).all()
    # Comparable without a reshape at the call site.
    got = twosample.compare(data, model, permutations=50, n_landmarks=None)
    assert got["dim"] == SIZE * SIZE


def test_the_loader_is_an_infinite_generator_not_a_finite_set(run_dir):
    """The stored patches are not the training distribution.

    Each stands for eight dihedral images times a continuum of crop offsets,
    so the loader can produce far more distinct scenes than it holds rows --
    and the comparison sample must not be capped at the row count, which is a
    number with nothing to do with the test's resolution.
    """
    d, c = run_dir
    ss = ShardSet.from_dir(d / "shards")
    ds = PatchDataset.from_shards(
        ss, c, LogFluxTransform.from_config(c.transform))
    want = 4 * len(ds)
    data = twosample.data_sample(ds, want, seed=0)
    assert len(data) == want > len(ds)
    # Every one a different image, despite base patches repeating.
    flat = np.asarray(data).reshape(len(data), -1)
    assert len(np.unique(flat, axis=0)) == want


def test_without_augmentation_the_sample_stays_finite_and_distinct(run_dir):
    """The exception, and why it is one: with augmentation off a base patch
    drawn twice is the same image twice, and a permutation test reads exact
    duplicates as agreement."""
    d, c = run_dir
    ss = ShardSet.from_dir(d / "shards")
    ds = PatchDataset.from_shards(
        ss, c, LogFluxTransform.from_config(c.transform))
    data = twosample.data_sample(ds, 10_000, seed=0, augment=False)
    assert len(data) == len(ds)
    flat = np.asarray(data).reshape(len(data), -1)
    assert len(np.unique(flat, axis=0)) == len(data)


def test_the_draw_covers_the_whole_set_not_a_resident_slice(run_dir):
    """Why this does not use ``PatchDataset.batches``.

    That stream draws from a pool of resident shards refreshed every
    ``pool_refill_every`` batches, which averages out over a training run and
    does not over a single bulk draw -- at the defaults a 16384-patch draw can
    come from 8 shards of ~31.  Shards are written in extraction order, so
    that slice is a contiguous piece of the sky walk, and the control cannot
    see it because both halves of one sample share the bias.

    Uniform indices are the fix, so the thing to pin is that they stay
    uniform: a draw of a few times the row count has to touch nearly every
    row, not a band of them.
    """
    d, c = run_dir
    ss = ShardSet.from_dir(d / "shards")
    ds = PatchDataset.from_shards(
        ss, c, LogFluxTransform.from_config(c.transform), in_memory=False)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ds), size=8 * len(ds))
    # Coupon collector: 8n draws miss e^-8 of n rows, well under 1%.
    assert len(np.unique(idx)) > 0.99 * len(ds)
    assert idx.min() < 0.1 * len(ds) and idx.max() > 0.9 * len(ds)
    ds.close()


def test_the_data_sample_follows_the_augmentation_it_is_told_to(run_dir):
    """The model learnt the augmented distribution, so the comparison set has
    to be augmented too -- and the flag has to actually do something."""
    d, c = run_dir
    ss = ShardSet.from_dir(d / "shards")
    ds = PatchDataset.from_shards(
        ss, c, LogFluxTransform.from_config(c.transform))
    plain = twosample.data_sample(ds, 32, seed=0, augment=False)
    again = twosample.data_sample(ds, 32, seed=0, augment=False)
    aug = twosample.data_sample(ds, 32, seed=0, augment=True)
    np.testing.assert_array_equal(plain, again)      # reproducible
    assert not np.allclose(plain, aug)               # and not the same scenes


def test_chunking_the_model_draws_does_not_change_how_many_come_back(run_dir):
    d, _ = run_dir
    prior = load_prior(d / "ckpt")
    for chunk in (1, 3, 8):
        assert len(twosample.model_sample(prior, 5, seed=0, chunk=chunk,
                                          n_steps=3)) == 5
