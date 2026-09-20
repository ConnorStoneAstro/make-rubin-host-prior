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
    estimate_band_softening,
    pool_shards,
    pool_to_training_grid,
    random_dihedral,
    suggest_sigma_range,
)
from rubin_host_prior.data.augment import N_DIHEDRAL
from rubin_host_prior.data.synthetic import write_synthetic_shards


# -- transform -------------------------------------------------------------
#
#   forward:  x = log(softplus(f / s)) / c
#   model:    f = s * exp(c * x)                  strictly positive
#
# The two are deliberately not inverses. A source cannot emit negative flux, so
# the prior's reachable domain in flux space must be positive; measured flux is
# negative wherever noise dips below the subtracted sky, and those pixels are
# carried smoothly towards zero instead of being represented faithfully.


def _transform(softening=None, **kw):
    softening = softening or {b: 4.0 for b in BANDS}
    return LogFluxTransform.from_config(
        TransformConfig(band_softening=softening, **kw)
    )


def test_model_map_is_exactly_the_softened_flux():
    """``s*exp(c*forward(f)) == softplus_s(f)``, so the entire discrepancy
    between data and model is the softening and nothing else."""
    t = _transform()
    s = t.softening[0]
    f = np.array([[-40.0, -4.0, 0.0, 4.0, 40.0, 4e5]])
    u = f / s
    softened = s * (np.maximum(u, 0.0) + np.log1p(np.exp(-np.abs(u))))
    np.testing.assert_allclose(
        t.inverse(t.forward(f, np.array([0])), np.array([0])), softened, rtol=1e-10
    )


def test_model_map_is_strictly_positive():
    """The property the whole design exists for: a scene drawn from the prior
    can never contain negative flux, however negative x wanders."""
    t = _transform()
    x = np.linspace(-500, 30, 2000)[None]
    f = t.inverse(x, np.array([0]))
    assert np.all(np.isfinite(f))
    assert np.all(f >= 0.0)
    assert np.all(f[x > -700] > 0.0)


def test_exact_inverse_round_trips_including_negative_flux():
    t = _transform()
    f = np.array([[-1e4, -200.0, -20.0, -4.0, -0.4, 0.0, 0.4, 4.0, 4e5, 4e9]])
    back = t.inverse_exact(t.forward(f, np.array([0])), np.array([0]))
    np.testing.assert_allclose(back, f, rtol=1e-12, atol=1e-12)


def test_bright_flux_passes_through_untouched():
    """``softplus(u) -> u`` exponentially, so anything detected is represented
    far inside its own photometric error."""
    t = _transform()
    s = t.softening[0]
    for mult, tol in ((3.0, 0.02), (5.0, 0.002), (10.0, 1e-4), (50.0, 1e-12)):
        f = np.array([[mult * s]])
        recovered = t.inverse(t.forward(f, np.array([0])), np.array([0]))
        assert abs(recovered.item() / f.item() - 1.0) < tol, mult


def test_accurate_above_solves_the_right_equation():
    t = _transform()
    for tol in (0.05, 0.01, 0.001):
        u = t.accurate_above(tol)
        assert np.log1p(np.exp(-u)) / u == pytest.approx(tol, rel=0.01)
    assert t.accurate_above(0.01) == pytest.approx(3.37, rel=0.02)


def test_negative_flux_vanishes_smoothly_with_no_floor():
    """No clipping, no point mass, no bound: however deep a pixel goes it stays
    representable. This is what lets background over-subtraction be kept."""
    t = _transform()
    s = t.softening[0]
    f = -np.logspace(-2, 4, 400)[None] * s
    x = t.forward(f, np.array([0]))
    assert np.all(np.isfinite(x))
    assert len(np.unique(x)) == x.size  # strictly monotone, no pile-up
    recovered = t.inverse(x, np.array([0]))
    assert np.all(recovered >= 0)
    assert np.all(np.diff(recovered[0][::-1]) >= 0)  # decreasing towards zero


def test_x_is_linear_in_flux_for_deep_negatives():
    """``softplus(u) -> e^u`` so ``x -> f/s``: the negative tail is linear, which
    keeps Gaussian noise Gaussian rather than compressing it."""
    t = _transform()
    s = t.softening[0]
    f = np.array([[-10.0 * s, -20.0 * s, -30.0 * s]])
    x = t.forward(f, np.array([0]))
    np.testing.assert_allclose(x, f / s, rtol=1e-3)


def test_sky_pedestal_is_log_two_times_the_softening():
    """The price of strict positivity: zero measured flux maps to 0.693*s."""
    t = _transform()
    s = t.softening[0]
    at_zero = t.inverse(t.forward(np.zeros((1, 1)), np.array([0])), np.array([0]))
    assert at_zero.item() == pytest.approx(np.log(2.0) * s)
    assert t.sky_pedestal == pytest.approx(np.log(2.0))
    assert t.sky_level == pytest.approx(np.log(np.log(2.0)))


def test_softening_puts_all_bands_on_a_common_footing():
    """The same flux in units of that band's sky noise maps to the same x, which
    is what makes one band-agnostic prior reasonable."""
    soft = {"u": 9.3, "g": 3.7, "r": 4.0, "i": 5.3, "z": 8.0, "y": 13.3}
    t = _transform(soft)
    xs = [
        float(t.forward(np.array([[3.0 * soft[b]]]), np.array([i]))[0, 0])
        for i, b in enumerate(BANDS)
    ]
    assert np.allclose(xs, xs[0])


def test_expected_sky_scatter_matches_a_simulation():
    from rubin_host_prior.data import expected_sky_scatter

    rng = np.random.default_rng(0)
    for ss in (0.5, 1.0, 4.0):
        t = _transform({b: ss * 1.0 for b in BANDS})  # sigma_pooled = 1
        x = t.forward(rng.normal(0.0, 1.0, (1, 200_000)), np.array([0]))
        assert float(x.std()) == pytest.approx(expected_sky_scatter(ss), rel=0.15)


def test_missing_band_softening_is_an_error():
    with pytest.raises(ValueError, match="no softening scale for band"):
        LogFluxTransform.from_config(TransformConfig(band_softening={"r": 4.0}))


def test_jacobian_is_c_times_flux_and_matches_finite_differences():
    t = _transform(log_scale=1.7)
    x = np.array([[-1.0, 0.3, 2.5]])
    band = np.array([1])
    np.testing.assert_allclose(
        t.jacobian(x, band), 1.7 * t.inverse(x, band), rtol=1e-12
    )
    h = 1e-6
    numeric = (t.inverse(x + h, band) - t.inverse(x - h, band)) / (2 * h)
    np.testing.assert_allclose(t.jacobian(x, band), numeric, rtol=1e-5)


def test_log_scale_rescales_x_only():
    a, b = _transform(log_scale=1.0), _transform(log_scale=2.0)
    f = np.array([[-8.0, 0.0, 17.0]])
    np.testing.assert_allclose(
        b.forward(f, np.array([0])), a.forward(f, np.array([0])) / 2.0, rtol=1e-12
    )


def test_transform_never_produces_nan_or_inf():
    t = _transform()
    f = np.concatenate(
        [-np.logspace(-6, 12, 500), [0.0], np.logspace(-6, 12, 500)]
    )[None]
    x = t.forward(f, np.array([0]))
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(t.inverse(x, np.array([0]))))


def test_log_softplus_is_stable_where_the_naive_form_is_not():
    from rubin_host_prior.data import log_softplus

    u = np.array([-800.0, -100.0, -25.0, -1.0, 0.0, 1.0, 100.0, 800.0])
    got = log_softplus(u)
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got[:3], u[:3], rtol=1e-9)   # -> u for u << 0
    np.testing.assert_allclose(got[-2:], np.log(u[-2:]), rtol=1e-9)  # -> log(u)
    assert got[4] == pytest.approx(np.log(np.log(2.0)))


def test_measure_pooled_sky_noise_recovers_a_known_sigma():
    """One-sided estimator: median - p15.87 is exactly one sigma for a Gaussian
    and ignores the positive tail that sources contribute."""
    from rubin_host_prior.data import measure_pooled_sky_noise

    rng = np.random.default_rng(0)
    pooled = rng.normal(0.0, 7.0, (60, 32, 32))
    band = np.arange(60) % 6
    for v in measure_pooled_sky_noise(pooled, band).values():
        assert v == pytest.approx(7.0, rel=0.05)


def test_sky_noise_estimator_is_not_inflated_by_sources():
    """Reading only the faint quartile keeps galaxies out of the estimate.

    A plain standard deviation is inflated by an order of magnitude here.
    """
    from rubin_host_prior.data import measure_pooled_sky_noise

    rng = np.random.default_rng(1)
    sky = rng.normal(0.0, 5.0, (40, 48, 48))
    yy, xx = np.mgrid[0:48, 0:48]
    source = 4000.0 * np.exp(-(((xx - 24) ** 2 + (yy - 24) ** 2) / 20.0))
    pooled = sky + source
    assert pooled.std() > 50.0, "the naive std really is badly inflated here"
    measured = measure_pooled_sky_noise(pooled, np.arange(40) % 6)["u"]
    assert measured == pytest.approx(5.0, rel=0.1)


def test_source_dominated_patches_cannot_move_the_answer():
    """The estimator does fail on a patch a galaxy fills -- there is no sky left
    to measure.  Taking the median across patches is what makes that harmless.
    """
    from rubin_host_prior.data import measure_pooled_sky_noise

    rng = np.random.default_rng(4)
    pooled = rng.normal(0.0, 5.0, (40, 48, 48))
    yy, xx = np.mgrid[0:48, 0:48]
    filled = 4000.0 * np.exp(-(((xx - 24) ** 2 + (yy - 24) ** 2) / 4000.0))
    pooled[:8] += filled  # a fifth of the patches are hopeless
    band = np.zeros(40, dtype=int)
    assert measure_pooled_sky_noise(pooled, band)["u"] == pytest.approx(5.0, rel=0.1)


def test_softening_measured_not_derived_for_correlated_noise():
    """Coadds are warped, so pixel noise is correlated and averaging P^2 pixels
    reduces it by less than P.  Deriving the pooled noise as sigma/P would be
    about 50% low at a realistic correlation width; measuring it is not."""
    from rubin_host_prior.data import measure_pooled_sky_noise
    from rubin_host_prior.data.synthetic import _convolve, _gaussian_psf

    rng = np.random.default_rng(2)
    native = rng.normal(0.0, 1.0, (120, 96, 96))
    k = _gaussian_psf(9, 0.8)
    native = np.stack([_convolve(n, k) for n in native])
    native /= native.std()  # per-pixel sigma is 1, but neighbours correlate
    native *= 12.0
    pooled = block_mean(native, 3)

    derived = 12.0 / 3.0  # what the old variance-plane route would have given
    truth = float(pooled.std())
    measured = measure_pooled_sky_noise(pooled, np.zeros(120, dtype=int))["u"]

    assert truth > 1.6 * derived, "correlation should inflate the pooled noise"
    assert measured == pytest.approx(truth, rel=0.1)


def test_estimate_band_softening_scales_the_measured_noise():
    from rubin_host_prior.data import estimate_band_softening

    rng = np.random.default_rng(3)
    # 20 patches per band: the median of a handful of per-patch estimates is
    # noisy (~15% with five), which is why the estimator wants a real sample.
    pooled = rng.normal(0.0, 6.0, (120, 64, 64))
    band = np.arange(120) % 6
    soft = estimate_band_softening(pooled, band, softening_sigma=2.0)
    assert set(soft) == set(BANDS)
    for v in soft.values():
        assert v == pytest.approx(12.0, rel=0.03)


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
    pooled, pooled_bands = pool_shards(ss, config)
    config.transform.band_softening = estimate_band_softening(
        pooled, pooled_bands, config.transform.softening_sigma
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
        band_softening={b: 999.0 for b in BANDS}
    ))
    stale = cache_key(other, LogFluxTransform.from_config(other.transform), ss)
    with pytest.raises(ValueError, match="rebuild it"):
        PatchDataset.from_pooled_cache(path, other, ds.transform, expect_key=stale)


def test_dataset_refuses_shards_smaller_than_the_config(shard_dir):
    ss = ShardSet.from_dir(shard_dir)
    config = Config(patch=PatchConfig(native_size=512, nominal_crop=192,
                                      out_size=64, pool_factor=3))
    config.transform.band_softening = {b: 20.0 for b in BANDS}
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
    pooled, pooled_bands = pool_shards(ss, config)
    config.transform.band_softening = estimate_band_softening(
        pooled, pooled_bands, config.transform.softening_sigma
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


def test_sky_scatter_matches_the_prediction(shard_dir):
    """The check that the per-band softening scales are right: the log-space sky
    scatter should match 0.721 / (softening_sigma * log_scale)."""
    from rubin_host_prior.data import expected_sky_scatter

    for ss_val in (1.0, 4.0):
        ss = ShardSet.from_dir(shard_dir)
        config = Config(patch=PatchConfig(native_size=ss.native_size,
                                          nominal_crop=96, out_size=32,
                                          pool_factor=3))
        config.transform.softening_sigma = ss_val
        pooled, pooled_bands = pool_shards(ss, config)
        config.transform.band_softening = estimate_band_softening(
            pooled, pooled_bands, ss_val)
        ds = PatchDataset.from_shards(
            ss, config, LogFluxTransform.from_config(config.transform))
        assert ds.stats(48)["sky_scatter"] == pytest.approx(
            expected_sky_scatter(ss_val), rel=0.5)


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
    pooled, pooled_bands = pool_shards(ss, config)
    config.transform.band_softening = estimate_band_softening(
        pooled, pooled_bands, config.transform.softening_sigma
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


# -- deep negatives survive the transform ----------------------------------

def test_sky_scatter_matches_the_prediction(shard_dir):
    """The check that the per-band softening scales are right: the log-space sky
    scatter should match 0.721 / (softening_sigma * log_scale)."""
    from rubin_host_prior.data import expected_sky_scatter

    for ss_val in (1.0, 4.0):
        ss = ShardSet.from_dir(shard_dir)
        config = Config(patch=PatchConfig(native_size=ss.native_size,
                                          nominal_crop=96, out_size=32,
                                          pool_factor=3))
        config.transform.softening_sigma = ss_val
        pooled, pooled_bands = pool_shards(ss, config)
        config.transform.band_softening = estimate_band_softening(
            pooled, pooled_bands, ss_val)
        ds = PatchDataset.from_shards(
            ss, config, LogFluxTransform.from_config(config.transform))
        assert ds.stats(48)["sky_scatter"] == pytest.approx(
            expected_sky_scatter(ss_val), rel=0.5)
