"""The log transform, pooling, augmentation, shards and loader."""

import numpy as np
import pytest

from rubin_host_prior.config import BANDS, Config, PatchConfig, TransformConfig
from rubin_host_prior.data import (
    LogFluxTransform,
    PatchDataset,
    ShardSet,
    area_resample,
    block_mean,
    cache_key,
    dihedral,
    estimate_band_offsets,
    pool_to_training_grid,
    random_dihedral,
    suggest_sigma_range,
)
from rubin_host_prior.data.augment import N_DIHEDRAL
from rubin_host_prior.data.synthetic import write_synthetic_shards


# -- transform -------------------------------------------------------------


def _transform(offsets=None, **kw):
    offsets = offsets or {b: 20.0 for b in BANDS}
    return LogFluxTransform.from_config(
        TransformConfig(band_offsets=offsets, **kw)
    )


def test_forward_inverse_round_trip():
    t = _transform()
    flux = np.array([[-15.0, 0.0, 1.0, 1e2, 1e4, 1e6]], dtype=np.float64)
    band = np.array([2])
    np.testing.assert_allclose(
        t.inverse(t.forward(flux, band), band), flux, rtol=1e-10, atol=1e-8
    )


def test_zero_flux_maps_to_zero():
    """The offset makes x = 0 the (background-subtracted) sky level exactly."""
    t = _transform()
    assert t.forward(np.zeros((1, 3)), np.array([0]))[0, 0] == pytest.approx(0.0)


def test_transform_is_linear_in_the_noise_regime():
    """Near the sky level the transform is a pure rescaling, so additive
    Gaussian pixel noise stays additive and Gaussian.

    ``log1p(u) = u - u^2/2 + ...``, so the departure from linearity is ~u/2 in
    relative terms.  At 1 sky noise (u = 1/k_sigma = 0.2) that is already 10%,
    which is why the sky scatter in log space is checked separately against
    1/k_sigma rather than assumed exact.
    """
    t = _transform()
    b = t.offsets[0]
    for u, rtol in ((1e-3, 1e-3), (1e-2, 1e-2)):
        small = np.array([[-u * b, 0.0, u * b]])
        x = t.forward(small, np.array([0]))
        np.testing.assert_allclose(x, small / (b * t.log_scale), rtol=rtol)


def test_transform_is_logarithmic_in_the_bright_regime():
    t = _transform()
    b = t.offsets[0]
    bright = np.array([[1e3 * b, 1e4 * b]])
    x = t.forward(bright, np.array([0]))
    assert x[0, 1] - x[0, 0] == pytest.approx(np.log(10.0), rel=1e-3)


def test_negative_pixels_are_not_a_point_mass():
    """Half of all sky pixels are negative; log(max(f, floor)) would pile them
    onto one value. log1p(f/b) must map them to distinct, finite values."""
    t = _transform()
    b = t.offsets[0]
    neg = np.linspace(-0.8 * b, -0.01 * b, 50).reshape(1, -1)
    x = t.forward(neg, np.array([0]))
    assert np.all(np.isfinite(x))
    assert np.all(np.diff(x[0]) > 0)
    assert len(np.unique(x)) == x.size


def test_floor_clips_only_extreme_negatives():
    t = _transform(k_sigma=5.0)
    b = t.offsets[0]
    flux = np.array([[-0.5 * b, -0.9 * b, -5.0 * b]])
    x = t.forward(flux, np.array([0]))
    assert x[0, 0] > t.x_floor
    assert x[0, 1] == pytest.approx(t.x_floor)
    assert x[0, 2] == pytest.approx(t.x_floor)  # clipped, not NaN


def test_clipped_fraction_is_reported():
    t = _transform()
    flux = np.concatenate([np.zeros(99), [-10 * t.offsets[0]]]).reshape(1, -1)
    _, frac = t.forward(flux, np.array([0]), return_clipped_fraction=True)
    assert frac == pytest.approx(0.01)


def test_per_band_offsets_put_all_bands_on_a_common_footing():
    """The point of a per-band offset: the same flux-in-units-of-sky-noise maps
    to the same x in every band, so one prior can cover all six."""
    offsets = {"u": 47.0, "g": 18.0, "r": 20.0, "i": 27.0, "z": 40.0, "y": 67.0}
    t = _transform(offsets)
    xs = [
        float(t.forward(np.array([[3.0 * offsets[b]]]), np.array([i]))[0, 0])
        for i, b in enumerate(BANDS)
    ]
    assert np.allclose(xs, xs[0])


def test_missing_band_offset_is_an_error():
    with pytest.raises(ValueError, match="no offset for band"):
        LogFluxTransform.from_config(TransformConfig(band_offsets={"r": 20.0}))


def test_jacobian_matches_finite_differences():
    t = _transform()
    x = np.array([[0.3, 1.5]])
    band = np.array([1])
    h = 1e-6
    numeric = (t.inverse(x + h, band) - t.inverse(x - h, band)) / (2 * h)
    np.testing.assert_allclose(t.jacobian(x, band), numeric, rtol=1e-5)


def test_estimate_band_offsets_scales_with_pool_factor():
    """Averaging P^2 independent pixels divides the noise by P."""
    var = np.full((12, 8, 8), 144.0, dtype=np.float32)  # sky noise 12 nJy
    band = np.arange(12) % 6
    off = estimate_band_offsets(var, band, pool_factor=3, k_sigma=5.0)
    assert set(off) == set(BANDS)
    for v in off.values():
        assert v == pytest.approx(5.0 * 12.0 / 3.0)


# -- pooling ---------------------------------------------------------------


def test_block_mean_reduces_noise_by_exactly_the_factor():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1.0, (400, 24, 24))
    p = block_mean(a, 3)
    assert p.shape == (400, 8, 8)
    assert float(p.std()) == pytest.approx(1 / 3, rel=0.02)


def test_block_mean_rejects_indivisible_shapes():
    with pytest.raises(ValueError, match="not divisible"):
        block_mean(np.zeros((10, 10)), 3)


def test_area_resample_equals_block_mean_at_integer_factors():
    """The jitter path must reduce to the exact path when the factor is integral;
    otherwise scale_jitter=0 would not mean 'no interpolation'."""
    rng = np.random.default_rng(1)
    a = rng.normal(size=(4, 192, 192)).astype(np.float32)
    np.testing.assert_allclose(area_resample(a, 64), block_mean(a, 3), atol=1e-5)


def test_area_resample_conserves_the_mean():
    rng = np.random.default_rng(2)
    a = rng.normal(5.0, 1.0, (2, 190, 190))
    assert float(area_resample(a, 64).mean()) == pytest.approx(float(a.mean()), rel=1e-6)


def test_nominal_pooling_is_centred_and_exact():
    a = np.arange(224 * 224, dtype=np.float32).reshape(224, 224)
    out = pool_to_training_grid(a, out_size=64, pool_factor=3)
    expected = block_mean(a[16:208, 16:208], 3)
    np.testing.assert_allclose(out, expected)


def test_translation_uses_integer_native_pixels_only():
    """Integer native shifts give sub-output-pixel jitter with no interpolation.

    One native pixel is 1/3 of an output pixel, so this is free sub-pixel
    positional augmentation.  The test: every translated crop must be *exactly*
    a block mean of the native array at some integer offset -- if it were not,
    something had interpolated and the noise would have been altered.
    """
    rng = np.random.default_rng(3)
    a = rng.normal(size=(224, 224)).astype(np.float64)
    room = 224 - 192
    # All exact block means, indexed by their (y, x) native offset.
    exact = {
        (y, x): block_mean(a[y : y + 192, x : x + 192], 3)
        for y in range(room + 1)
        for x in range(room + 1)
    }
    seen = set()
    for seed in range(20):
        out = pool_to_training_grid(
            a, 64, 3, rng=np.random.default_rng(seed), translate=True
        )
        hits = [k for k, v in exact.items() if np.allclose(out, v, atol=1e-12)]
        assert len(hits) == 1, f"crop matched {len(hits)} exact offsets, expected 1"
        seen.add(hits[0])
    assert len(seen) > 5, f"translation produced only {len(seen)} distinct offsets"
    # Offsets must vary in both axes independently, not just along the diagonal.
    assert len({y for y, _ in seen}) > 1 and len({x for _, x in seen}) > 1


def test_scale_jitter_stays_within_the_native_stamp():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(224, 224))
    for _ in range(20):
        out = pool_to_training_grid(
            a, 64, 3, rng=rng, translate=True, scale_jitter=0.10
        )
        assert out.shape == (64, 64)


def test_pooling_refuses_a_too_small_stamp():
    with pytest.raises(ValueError, match="native pixels|not divisible"):
        pool_to_training_grid(np.zeros((100, 100)), 64, 3)


# -- augmentation ----------------------------------------------------------


def test_dihedral_is_a_group_of_order_eight():
    rng = np.random.default_rng(5)
    img = rng.normal(size=(1, 5, 5))
    elements = [dihedral(img, k) for k in range(N_DIHEDRAL)]
    assert len({e.tobytes() for e in elements}) == N_DIHEDRAL
    for i in range(N_DIHEDRAL):
        for j in range(N_DIHEDRAL):
            composed = dihedral(elements[i], j)
            assert any(np.array_equal(composed, e) for e in elements)


def test_dihedral_is_an_exact_reindexing():
    """No interpolation: the multiset of pixel values is unchanged, so the noise
    distribution and its pixel-to-pixel independence are untouched."""
    rng = np.random.default_rng(6)
    img = rng.normal(size=(1, 7, 7))
    for k in range(N_DIHEDRAL):
        np.testing.assert_array_equal(
            np.sort(dihedral(img, k).ravel()), np.sort(img.ravel())
        )


def test_dihedral_rejects_out_of_range():
    with pytest.raises(ValueError):
        dihedral(np.zeros((4, 4)), 8)


def test_random_dihedral_varies_per_example():
    rng = np.random.default_rng(7)
    batch = rng.normal(size=(64, 5, 5))
    out = random_dihedral(batch, rng)
    assert out.shape == batch.shape
    # Each example is some element of D4 applied to itself.
    for i in range(len(batch)):
        assert any(
            np.array_equal(out[i], dihedral(batch[i], k)) for k in range(N_DIHEDRAL)
        )
    assert not np.array_equal(out, batch)


# -- shards and loader -----------------------------------------------------


@pytest.fixture(scope="module")
def shard_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("shards")
    write_synthetic_shards(d, n_patches=48, native_size=112, patches_per_shard=16,
                           seed=0)
    return d


def test_shardset_spans_multiple_files(shard_dir):
    ss = ShardSet.from_dir(shard_dir)
    assert len(ss) == 48
    assert len(ss.paths) == 3
    assert ss.native_size == 112
    assert ss.bands == BANDS
    assert "DETECTED" in ss.mask_plane_dict


def test_gather_returns_rows_in_the_requested_order(shard_dir):
    """Reads are reordered per shard for efficiency; the caller must not see it."""
    ss = ShardSet.from_dir(shard_dir)
    all_images = ss.load("image")
    idx = np.array([41, 3, 17, 3, 46, 0])
    got = ss.gather(np.unique(idx), "image")
    np.testing.assert_array_equal(got, all_images[np.unique(idx)])
    scrambled = np.array([30, 5, 44, 12])
    np.testing.assert_array_equal(
        ss.gather(scrambled, "image"), all_images[scrambled]
    )


def test_shard_metadata_is_preserved(shard_dir):
    ss = ShardSet.from_dir(shard_dir)
    assert ss.meta["band_idx"].shape == (48,)
    assert np.all(ss.meta["band_idx"] < 6)
    assert np.all(np.isfinite(ss.meta["psf_sigma"]))
    assert np.all(ss.meta["sky_noise"] > 0)


def _dataset(shard_dir, out_size=32):
    ss = ShardSet.from_dir(shard_dir)
    config = Config(
        patch=PatchConfig(
            native_size=ss.native_size,
            nominal_crop=out_size * 3,
            out_size=out_size,
            pool_factor=3,
        )
    )
    config.transform.band_offsets = estimate_band_offsets(
        ss.load("variance"), ss.meta["band_idx"], 3, config.transform.k_sigma
    )
    return ss, config, PatchDataset.from_shards(
        ss, config, LogFluxTransform.from_config(config.transform)
    )


def test_dataset_batches_have_the_right_shape_and_dtype(shard_dir):
    _, config, ds = _dataset(shard_dir)
    batch = next(ds.batches(8, seed=0))
    assert batch.shape == (8, 1, 32, 32)
    assert batch.dtype == np.float32
    assert np.all(np.isfinite(batch))


def test_sky_scatter_matches_one_over_k_sigma(shard_dir):
    """The headline check that the offsets are right: after the transform, the
    sky should scatter by ~1/k_sigma in log space in every band."""
    _, config, ds = _dataset(shard_dir)
    stats = ds.stats(48)
    assert stats["sky_scatter"] == pytest.approx(
        1.0 / config.transform.k_sigma, rel=0.35
    )
    assert stats["clipped_fraction"] < 1e-3


def test_validation_batch_is_deterministic_and_unaugmented(shard_dir):
    _, _, ds = _dataset(shard_dir)
    a = ds.validation_batch(8)
    b = ds.validation_batch(8)
    np.testing.assert_array_equal(a, b)


def test_pooled_cache_round_trips(shard_dir, tmp_path):
    ss, config, ds = _dataset(shard_dir)
    path = ds.build_pooled_cache(tmp_path / "cache.h5")
    key = cache_key(config, ds.transform, ss)
    cached = PatchDataset.from_pooled_cache(
        path, config, ds.transform, expect_key=key
    )
    assert cached.mode == "pooled" and len(cached) == len(ds)
    # The cache stores the nominal (un-augmented) pooling, so it must agree.
    np.testing.assert_allclose(
        cached.make_batch(np.arange(6), augment=False),
        ds.make_batch(np.arange(6), rng=None, augment=False),
        rtol=1e-6,
    )


def test_cache_key_changes_with_the_transform(shard_dir, tmp_path):
    ss, config, ds = _dataset(shard_dir)
    path = ds.build_pooled_cache(tmp_path / "cache.h5")
    other = Config(patch=config.patch, transform=TransformConfig(
        band_offsets={b: 999.0 for b in BANDS}
    ))
    stale = cache_key(other, LogFluxTransform.from_config(other.transform), ss)
    with pytest.raises(ValueError, match="rebuild it"):
        PatchDataset.from_pooled_cache(path, other, ds.transform, expect_key=stale)


def test_dataset_refuses_shards_smaller_than_the_config(shard_dir):
    ss = ShardSet.from_dir(shard_dir)
    config = Config(patch=PatchConfig(native_size=512, nominal_crop=192,
                                      out_size=64, pool_factor=3))
    config.transform.band_offsets = {b: 20.0 for b in BANDS}
    with pytest.raises(ValueError, match="shards hold"):
        PatchDataset.from_shards(
            ss, config, LogFluxTransform.from_config(config.transform)
        )


def test_suggest_sigma_range_covers_the_data(shard_dir):
    _, _, ds = _dataset(shard_dir)
    stats = ds.stats(48)
    lo, hi = suggest_sigma_range(stats)
    assert 0 < lo < stats["sky_scatter"]
    assert hi >= stats["per_patch_range_p99"]
