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


# -- correlation length ----------------------------------------------------


def _correlated_field(n, size, w, noise, seed=0):
    """White noise smoothed by a Gaussian of width ``w``.

    The autocovariance is then a Gaussian of width ``w*sqrt(2)``, so the 1/e
    crossing sits at exactly ``2w`` -- a field with a known correlation length.
    """
    rng = np.random.default_rng(seed)
    k = np.arange(size) - size // 2
    g = np.exp(-0.5 * (k / w) ** 2)
    g /= np.sqrt((g**2).sum())
    F = np.fft.fft2(np.fft.ifftshift(np.outer(g, g)))
    x = np.real(
        np.fft.ifft2(np.fft.fft2(rng.normal(size=(n, size, size)), axes=(1, 2)) * F,
                     axes=(1, 2))
    )
    x = x / x.std()
    return x + noise * rng.normal(size=x.shape)


@pytest.mark.parametrize("w", [1.5, 3.0, 5.0])
def test_correlation_length_recovers_a_known_field(w):
    from rubin_host_prior.data import correlation_length

    r = correlation_length(_correlated_field(300, 96, w, 0.0))
    assert r["xi"] == pytest.approx(2 * w, rel=0.12)


def test_correlation_length_is_immune_to_uncorrelated_noise():
    """The whole point of renormalising at lag 1.

    Uncorrelated pixel noise is a delta at zero lag and nothing elsewhere, so it
    must not move xi at all -- but a naive 1/e crossing on the raw profile is
    dominated by it.
    """
    from rubin_host_prior.data import correlation_length

    clean = correlation_length(_correlated_field(300, 96, 3.0, 0.0))
    noisy = correlation_length(_correlated_field(300, 96, 3.0, 1.5))
    assert noisy["xi"] == pytest.approx(clean["xi"], rel=0.05)
    assert noisy["noise_fraction"] > 0.5
    naive = correlation_length(_correlated_field(300, 96, 3.0, 1.5),
                               exclude_noise=False)
    assert naive["xi"] < 0.5 * clean["xi"], "naive estimate should be badly wrong"


def test_correlation_length_uses_the_linear_not_circular_autocorrelation():
    """A circular (unpadded) autocorrelation wraps structure round the edges and
    biases xi low, which on this question is the dangerous direction."""
    from rubin_host_prior.data import autocorrelation

    x = _correlated_field(200, 64, 4.0, 0.0)
    prof = autocorrelation(x)
    assert prof[0] == pytest.approx(1.0)
    assert np.all(np.diff(prof[1:12]) < 0)  # monotone decay, no wrap-around bump


def test_streaming_accumulator_matches_the_batch_computation():
    from rubin_host_prior.data.diagnostics import AutocorrelationAccumulator
    from rubin_host_prior.data import correlation_length

    x = _correlated_field(120, 64, 3.0, 1.0)
    acc = AutocorrelationAccumulator(64)
    for patch in x:
        acc.add(patch)
    np.testing.assert_allclose(acc.result()["profile"],
                               correlation_length(x)["profile"], rtol=1e-10)
    assert acc.result()["n_patches"] == 120


def test_accumulator_skips_non_finite_patches():
    from rubin_host_prior.data.diagnostics import AutocorrelationAccumulator

    acc = AutocorrelationAccumulator(16)
    bad = np.zeros((16, 16)); bad[0, 0] = np.nan
    acc.add(bad)
    acc.add(np.random.default_rng(0).normal(size=(16, 16)))
    assert acc.result()["n_patches"] == 1


def test_accumulator_rejects_the_wrong_shape():
    from rubin_host_prior.data.diagnostics import AutocorrelationAccumulator

    with pytest.raises(ValueError, match="expected"):
        AutocorrelationAccumulator(16).add(np.zeros((8, 8)))


def test_context_advice_brackets_the_regimes():
    from rubin_host_prior.data import context_advice

    assert "comfortable" in context_advice(xi=6.0, loss_margin=16)
    assert "marginal" in context_advice(xi=16.0, loss_margin=16)
    assert "TOO SMALL" in context_advice(xi=30.0, loss_margin=16)


def test_dataset_reports_correlation_length_in_pooled_pixels(shard_dir):
    _, _, ds = _dataset(shard_dir)
    cl = ds.correlation_length(32)
    assert 0 < cl["xi"] < ds.config.patch.out_size
    assert 0.0 <= cl["noise_fraction"] <= 1.0


# -- variable patch sizes --------------------------------------------------


def test_training_sizes_are_deduplicated_with_the_reference_first():
    pc = PatchConfig(native_size=224, nominal_crop=192, out_size=64, pool_factor=3,
                     out_sizes=(48, 64, 32, 48))
    assert pc.training_sizes == (64, 32, 48)


def test_config_rejects_sizes_the_stamp_cannot_supply():
    with pytest.raises(ValueError, match="needs 288 native pixels"):
        PatchConfig(native_size=224, nominal_crop=192, out_size=64, pool_factor=3,
                    out_sizes=(96,))


def _varsize_dataset(shard_dir, out_sizes):
    ss = ShardSet.from_dir(shard_dir)
    config = Config(patch=PatchConfig(native_size=ss.native_size, nominal_crop=96,
                                      out_size=32, pool_factor=3,
                                      out_sizes=out_sizes))
    config.transform.band_offsets = estimate_band_offsets(
        ss.load("variance"), ss.meta["band_idx"], 3, config.transform.k_sigma
    )
    return ss, config, PatchDataset.from_shards(
        ss, config, LogFluxTransform.from_config(config.transform)
    )


def test_batches_cycle_sizes_round_robin(shard_dir):
    _, _, ds = _varsize_dataset(shard_dir, (16, 24, 32))
    it = ds.batches(4, seed=0)
    sizes = [next(it).shape[-1] for _ in range(9)]
    assert sizes == [32, 16, 24] * 3, sizes


def test_every_size_is_a_valid_pooled_image(shard_dir):
    _, _, ds = _varsize_dataset(shard_dir, (16, 24, 32))
    for s in (16, 24, 32):
        b = ds.make_batch(np.arange(4), rng=np.random.default_rng(0), out_size=s)
        assert b.shape == (4, 1, s, s)
        assert np.all(np.isfinite(b))


def test_validation_batch_stays_at_the_reference_size(shard_dir):
    """Otherwise validation losses are not comparable across runs or steps."""
    _, config, ds = _varsize_dataset(shard_dir, (16, 24, 32))
    assert ds.validation_batch(4).shape[-1] == config.patch.out_size


def test_pooled_cache_serves_smaller_sizes_by_sub_cropping(shard_dir, tmp_path):
    """A sub-crop of a pooled, transformed image equals the pooled transform of
    the corresponding native sub-region -- pooling is local, the transform is
    pointwise -- so the cache covers every size at or below its own."""
    ss, config, ds = _varsize_dataset(shard_dir, (16, 24, 32))
    cached = PatchDataset.from_pooled_cache(
        ds.build_pooled_cache(tmp_path / "c.h5"), config, ds.transform
    )
    for s in (16, 24, 32):
        assert cached.make_batch(np.arange(4), augment=False,
                                 out_size=s).shape == (4, 1, s, s)
    with pytest.raises(ValueError, match="cannot serve"):
        cached.make_batch(np.arange(4), augment=False, out_size=48)


def test_pooled_cache_sub_crop_matches_the_native_path(shard_dir, tmp_path):
    ss, config, ds = _varsize_dataset(shard_dir, (16,))
    cached = PatchDataset.from_pooled_cache(
        ds.build_pooled_cache(tmp_path / "c2.h5"), config, ds.transform
    )
    idx = np.arange(4)
    from_cache = cached.make_batch(idx, augment=False, out_size=16)
    full = ds.make_batch(idx, rng=None, augment=False, out_size=32)
    o = (32 - 16) // 2
    np.testing.assert_allclose(from_cache[:, 0], full[:, 0, o:o + 16, o:o + 16],
                               rtol=1e-6)


def test_small_crops_do_not_wander_off_the_host(shard_dir):
    """Translation room is capped at the reference size's room.

    Without the cap a 16 px crop would roam the whole 112 px stamp and land
    mostly on blank sky, so the data distribution would silently change with
    patch size -- which is not what varying the size is for.
    """
    from rubin_host_prior.data.pooling import pool_to_training_grid

    a = np.arange(224 * 224, dtype=np.float64).reshape(224, 224)
    corners = set()
    rng = np.random.default_rng(0)
    for _ in range(80):
        out = pool_to_training_grid(a, 16, 3, rng=rng, translate=True,
                                    max_translate=16)
        corners.add(int(out[0, 0] // 224))
    assert max(corners) - min(corners) <= 2 * 16
    centre = (224 - 48) // 2
    assert abs((max(corners) + min(corners)) / 2 - centre) <= 2


# -- the exact transform, and negative-flux headroom ------------------------


def test_forward_is_exactly_log_of_one_plus_f_over_b():
    """Pin the formula:  x = log(1 + max(f/b, r)) / c,  f = b*(exp(c*x) - 1).

    The boost that carries negative sky pixels through the logarithm is the
    ``+1`` inside ``log1p`` -- i.e. ``+b`` in flux units.  Dividing by ``b`` does
    not do it: that rescales negatives but leaves them negative.
    """
    t = _transform(log_scale=2.0, floor_ratio=-0.9)
    b = t.offsets[0]
    f = np.array([[-0.5 * b, 0.0, b, 37.0 * b]])
    expected = np.log(1.0 + f / b) / 2.0
    np.testing.assert_allclose(t.forward(f, np.array([0])), expected, rtol=1e-12)
    # and the division alone would leave every negative pixel negative
    assert np.all((f / b)[f < 0] < 0)


def test_representable_flux_range_is_bounded_below_by_minus_b():
    """``inverse`` is b*expm1(c*x) and expm1 -> -1, so the model spans
    ``(-b, +inf)``.  ``b`` is therefore a hard bound on negative flux, not just a
    numerical guard -- which is what makes ``k_sigma`` a modelling choice."""
    t = _transform()
    b = t.offsets[0]
    # Far below the floor, expm1 underflows to exactly -1 and the flux saturates
    # at -b.  That is the right behaviour: bounded, finite, no NaN -- a sampler
    # that wanders to very negative x gets "as dark as representable", not a
    # numerical blow-up.  forward() never produces x below x_floor anyway.
    assert t.inverse(np.array([[-50.0]]), np.array([0]))[0, 0] == pytest.approx(-b)
    everywhere = t.inverse(np.linspace(-60, 5, 400)[None], np.array([0]))
    assert np.all(np.isfinite(everywhere))
    assert np.all(everywhere >= -b)
    # Strictly above -b across the range the transform actually produces.
    realistic = t.inverse(np.linspace(t.x_floor, 8, 400)[None], np.array([0]))
    assert np.all(realistic > -b)


def test_floor_bites_at_minus_zero_point_nine_k_sigma():
    t = _transform(k_sigma=10.0)
    b = t.offsets[0]
    assert t.forward(np.array([[-0.9 * b]]), np.array([0]))[0, 0] == pytest.approx(
        t.x_floor
    )
    assert t.forward(np.array([[-0.89 * b]]), np.array([0]))[0, 0] > t.x_floor


@pytest.fixture(scope="module")
def dark_halo_shards(tmp_path_factory):
    """Shards carrying a 1.5-sigma background over-subtraction, kept rather than
    gated out -- the regime that decides k_sigma."""
    d = tmp_path_factory.mktemp("halo")
    write_synthetic_shards(d, n_patches=48, native_size=112, patches_per_shard=48,
                           dark_halo_sigma=1.5, seed=3)
    return d


def _halo_dataset(shard_dir, k_sigma):
    ss = ShardSet.from_dir(shard_dir)
    config = Config(patch=PatchConfig(native_size=ss.native_size, nominal_crop=96,
                                      out_size=32, pool_factor=3))
    config.transform.k_sigma = k_sigma
    config.transform.band_offsets = estimate_band_offsets(
        ss.load("variance"), ss.meta["band_idx"], 3, k_sigma
    )
    return PatchDataset.from_shards(
        ss, config, LogFluxTransform.from_config(config.transform)
    )


def test_pooling_makes_a_smooth_offset_deeper_than_the_noise(dark_halo_shards):
    """Pooling divides the noise by pool_factor but leaves a smooth offset
    untouched, so in pooled-sigma units an over-subtracted region is
    pool_factor times deeper than it was natively.  This is why k_sigma has to
    be larger than the noise alone would suggest."""
    ds = _halo_dataset(dark_halo_shards, 10.0)
    head = ds.flux_headroom(48)
    # a 1.5 sigma native halo is ~4.5 sigma pooled, plus noise and the bowl's core
    assert head["deepest"] < -4.0
    assert head["k_sigma_needed"] > 5.0


def test_too_small_k_sigma_clips_the_dark_halo(dark_halo_shards):
    small = _halo_dataset(dark_halo_shards, 5.0).stats(48)
    large = _halo_dataset(dark_halo_shards, 15.0).stats(48)
    assert small["clipped_fraction"] > 0.01, "k_sigma=5 should clip a 1.5sig halo"
    assert large["clipped_fraction"] == 0.0


def test_sky_scatter_tracks_one_over_k_sigma(shard_dir):
    """On clean patches the log-space sky scatter should be ~1/k_sigma, which is
    the check that the band offsets are right."""
    for k in (5.0, 10.0):
        ss = ShardSet.from_dir(shard_dir)
        config = Config(patch=PatchConfig(native_size=ss.native_size,
                                          nominal_crop=96, out_size=32,
                                          pool_factor=3))
        config.transform.k_sigma = k
        config.transform.band_offsets = estimate_band_offsets(
            ss.load("variance"), ss.meta["band_idx"], 3, k)
        ds = PatchDataset.from_shards(
            ss, config, LogFluxTransform.from_config(config.transform))
        assert ds.stats(48)["sky_scatter"] == pytest.approx(1.0 / k, rel=0.4)


def test_flux_headroom_needs_native_stamps(shard_dir, tmp_path):
    ss, config, ds = _dataset(shard_dir)
    cached = PatchDataset.from_pooled_cache(
        ds.build_pooled_cache(tmp_path / "h.h5"), config, ds.transform)
    with pytest.raises(ValueError, match="needs native stamps"):
        cached.flux_headroom(8)
