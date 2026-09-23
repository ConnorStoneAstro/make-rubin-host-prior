"""The log transform, pooling, augmentation, shards and loader."""

from pathlib import Path

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
    log_softplus,
    soften,
    softplus,
    pool_shards,
    pool_to_training_grid,
    random_dihedral,
    suggest_sigma_range,
)
from rubin_host_prior.data.augment import N_DIHEDRAL
from rubin_host_prior.data.synthetic import write_synthetic_shards


# -- transform -------------------------------------------------------------
#
#   forward:  x = log(s * softplus(f / s))        -> log(f) for f >> s
#   model:    f = exp(x)                          strictly positive, no band
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


R = np.array([BANDS.index("r")])


def test_every_map_is_built_from_the_one_softening_definition():
    """``soften(f) = s*softplus(f/s)`` is the definition; ``forward`` is
    ``log(soften(f)/s)/c`` and ``inverse`` is exactly ``soften``.

    ``forward`` does not literally call ``soften`` -- it goes through
    ``log_softplus``, because below ``f/s = -745`` the softened flux underflows
    to zero in float64 and its log is ``-inf``.  That is a numerical detail, and
    this is what pins the two forms together so it stays one.
    """
    from rubin_host_prior.data.transform import soften

    t = _transform()
    s = t.softening[R[0]]
    f = np.array([[-400.0, -40.0, -4.0, 0.0, 4.0, 40.0, 4e5]])

    assert t.forward(f, R) == pytest.approx(np.log(soften(f, s)))
    assert t.soften(f, R) == pytest.approx(soften(f, s))
    assert t.inverse(t.forward(f, R)) == pytest.approx(soften(f, s), rel=1e-10)

    # And the reason forward is not written that way.  At f/s = -1000 the
    # softened flux underflows to zero and the naive composition is -inf;
    # forward returns the exact limit, f/s + log(s).
    deep = np.array([[-1000.0 * s]])
    assert soften(deep, s)[0, 0] == 0.0
    assert t.forward(deep, R)[0, 0] == pytest.approx(-1000.0 + np.log(s))

    # Positive everywhere it can be represented.  exp underflows to exactly
    # zero below x = -745 in float64, which is 1e-324 nJy -- far outside
    # anything a sampler reaches, but it is a floor and it is worth knowing it
    # is there rather than believing the map is positive without qualification.
    x = np.linspace(-700, 30, 2000)[None]
    model = t.inverse(x)
    assert np.all(model > 0) and np.all(np.isfinite(model))
    assert np.all(np.diff(model[0]) > 0)  # monotone, so invertible on its image
    assert model == pytest.approx(np.exp(x))
    assert t.inverse(np.array([[-800.0]]))[0, 0] == 0.0


def test_large_fluxes_pass_through_and_negative_ones_vanish_smoothly():
    """Deliberately not an inverse pair: large fluxes come back essentially
    unchanged, negatives approach zero, and there is no floor operation anywhere
    to put a hard edge in."""
    t = _transform()
    bright = np.array([[1e3, 1e4, 1e5, 1e6]])
    assert t.inverse(t.forward(bright, R)) == pytest.approx(bright, rel=1e-3)

    negative = np.array([[-2e3, -1e3, -50.0, -5.0]])
    back = t.inverse(t.forward(negative, R))
    assert np.all(back > 0) and np.all(back < t.softening[R[0]])
    assert np.all(np.diff(back[0]) > 0)
    # inverse_exact does recover them, which is what makes the loss checkable.
    assert t.inverse_exact(t.forward(negative, R), R) == pytest.approx(
        negative, rel=1e-6)
    # Deep negatives go linear in x, which is what keeps the score finite.
    assert np.all(np.isfinite(t.forward(np.array([[-1e6, -2e6, 0.0, 1e12]]), R)))


def test_the_softening_scale_has_one_source_of_truth():
    """``softening_sigma`` decides how hard the sky is flattened, and it lived
    in two places with different values: TransformConfig said 2.0 (suppress the
    noise) while estimate_band_softening and prepare_config.py's flag both
    defaulted to the 1.0 of the earlier preserve-the-noise design, which won
    because the script assigned it unconditionally."""
    import inspect

    from rubin_host_prior.data.transform import estimate_band_softening

    assert TransformConfig().softening_sigma == 2.0
    assert (inspect.signature(estimate_band_softening)
            .parameters["softening_sigma"].default is inspect.Parameter.empty)

    script = (Path(__file__).resolve().parent.parent
              / "scripts" / "prepare_config.py").read_text()
    assert '"--softening-sigma", type=float, default=None' in script
    assert "if args.softening_sigma is not None:" in script


def test_x_is_absolute_log_flux_and_the_sky_is_what_moves():
    """The trade the transform makes.  A given flux maps to the same x in every
    band -- x is log(f), so the scene's absolute scale survives -- and the price
    is that each band's sky sits at its own log(s*log2) instead of all of them
    landing on one level.  Dividing by s inside the log would buy the common
    sky level back by making x relative, which is the wrong way round for a
    prior that has to compose with a likelihood in nJy."""
    noise = {"u": 28.0, "g": 11.0, "r": 12.0, "i": 16.0, "z": 24.0, "y": 40.0}
    t = _transform(noise)

    bright = np.array([[4e5]])
    everywhere = [t.forward(bright, np.array([i]))[0, 0] for i in range(len(BANDS))]
    assert everywhere == pytest.approx([np.log(4e5)] * len(BANDS))

    zeros = np.zeros((1, 1))
    skies = [t.forward(zeros, np.array([i]))[0, 0] for i in range(len(BANDS))]
    assert skies == pytest.approx(
        [np.log(noise[b] * np.log(2.0)) for b in BANDS])
    assert max(skies) - min(skies) == pytest.approx(np.log(40.0 / 11.0))
    # Same width about each of those levels, which is what keeps one prior over
    # all six bands sane: one sky shape at several places, not six shapes.
    for i in range(len(BANDS)):
        assert t.sky_level(np.array([i]))[0] == pytest.approx(skies[i])


def test_the_model_map_needs_no_band_and_its_jacobian_is_the_flux():
    """``inverse`` is exp(x) -- a forward model turning a scene back into nJy
    has no per-band offset to undo, which is the point of the whole change."""
    t = _transform()
    x = np.linspace(-5, 5, 50)[None]
    assert t.inverse(x) == pytest.approx(np.exp(x))
    assert t.jacobian(x) == pytest.approx(t.inverse(x))
    step = 1e-6
    fd = (t.inverse(x + step) - t.inverse(x - step)) / (2 * step)
    assert t.jacobian(x) == pytest.approx(fd, rel=1e-4)


def test_the_softening_is_always_indexed_by_the_global_band_index():
    """`band_idx` in the shards is an index into BANDS, so the softening tuple
    has to be too.  Building it over a subset silently re-bases the indexing:
    with ('r','i','z','y') measured, index 4 runs off the end and index 2
    quietly returns the i-band scale for an r-band patch."""
    partial = LogFluxTransform.from_config(
        TransformConfig(band_softening={"r": 12.0, "i": 16.0, "z": 24.0, "y": 40.0}))
    assert len(partial.softening) == len(BANDS)
    assert partial.softening[BANDS.index("r")] == 12.0
    assert partial.softening[BANDS.index("y")] == 40.0
    # Unmeasured bands are NaN, which is visible rather than wrong.
    assert np.isnan(partial.softening[BANDS.index("u")])

    # A caller that knows which bands its data has gets a named error instead.
    with pytest.raises(ValueError, match="no patches in"):
        LogFluxTransform.from_config(
            TransformConfig(band_softening={"r": 12.0}), required_bands=BANDS)
    with pytest.raises(ValueError, match="prepare_config"):
        LogFluxTransform.from_config(TransformConfig(band_softening={}),
                                     required_bands=BANDS)


def test_log_softplus_is_stable_where_the_naive_form_is_not():
    u = np.array([-800.0, -50.0, -1.0, 0.0, 1.0, 50.0, 800.0])
    got = log_softplus(u)
    assert np.all(np.isfinite(got))
    assert got[-1] == pytest.approx(np.log(800.0), rel=1e-6)
    assert got[0] == pytest.approx(-800.0, rel=1e-9)


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
    assert np.all(np.isfinite(ss.meta["sky_noise"]))
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


# -- deep negatives survive the transform ----------------------------------

def test_sky_scatter_matches_the_prediction(shard_dir):
    """The check that the per-band softening scales are right: the log-space sky
    scatter should match 0.721 / softening_sigma, in every band.  It is a check
    on the width; the per-band sky *level* differs by design."""
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


def test_shards_from_an_older_schema_are_refused(shard_dir, tmp_path):
    """An older shard still *opens* -- the metadata it lacks fills with -1 --
    which is exactly the problem, because a stale set then trains or plots
    without complaint.  Schema 1 stored variance, mask and PSF arrays; schema 2
    added four neighbour columns nothing trained on."""
    import shutil

    import h5py

    from rubin_host_prior.data.shards import SHARD_SCHEMA

    copy = tmp_path / "old"
    shutil.copytree(shard_dir, copy)
    for path in sorted(copy.glob("*.h5")):
        with h5py.File(path, "a") as f:
            del f.attrs["schema"]
    with pytest.raises(ValueError, match="shard schema 1"):
        ShardSet.from_dir(copy)
    assert SHARD_SCHEMA > 1


def test_a_shard_missing_a_metadata_column_is_refused(shard_dir, tmp_path):
    import shutil

    import h5py

    copy = tmp_path / "gappy"
    shutil.copytree(shard_dir, copy)
    with h5py.File(sorted(copy.glob("*.h5"))[0], "a") as f:
        del f["meta"]["sky_noise"]
    with pytest.raises(ValueError, match="sky_noise"):
        ShardSet.from_dir(copy)


def test_a_band_with_no_patches_gets_no_softening_scale():
    """The scale is measured, so a band the shards never saw has nothing to
    measure.  Inventing one would put its turnover wherever the guess landed."""
    from rubin_host_prior.data.transform import measure_pooled_sky_noise

    rng = np.random.default_rng(0)
    pooled = rng.normal(0.0, 12.0, (40, 16, 16))
    bands = np.zeros(40, dtype=int)          # everything in u
    got = measure_pooled_sky_noise(pooled, bands, BANDS)
    assert set(got) == {"u"} and got["u"] == pytest.approx(12.0, rel=0.1)


def test_shards_whose_bands_the_config_never_saw_are_refused(shard_dir):
    """Caught where the band can be named, rather than as a NaN that propagates
    into training or an IndexError from inside the transform."""
    ss = ShardSet.from_dir(shard_dir)
    config = Config(patch=PatchConfig(native_size=ss.native_size,
                                      nominal_crop=96, out_size=32, pool_factor=3))
    config.transform.band_softening = {"u": 12.0}
    with pytest.raises(ValueError, match="no softening scale"):
        PatchDataset.from_shards(
            ss, config, LogFluxTransform.from_config(config.transform))


def test_the_softening_is_the_identity_above_its_scale():
    """`s * softplus(f/s)` has one parameter, it is a flux, and above it the map
    is the identity: bright pixels pass through untouched and the whole
    adjustment is confined to the low and negative regime."""
    s = 24.0
    bright = np.array([5.0, 10.0, 100.0, 1e4]) * s
    assert soften(bright, s) == pytest.approx(bright, rel=1e-2)
    assert soften(np.array([10.0 * s]), s) == pytest.approx([10.0 * s], rel=1e-4)

    # Below the scale it bends over and approaches zero from above, without
    # reaching it and without a floor.
    faint = np.array([-10.0, -5.0, -2.0, -1.0, 0.0]) * s
    out = soften(faint, s)
    assert np.all(out > 0) and np.all(np.diff(out) > 0)
    assert soften(np.array([0.0]), s)[0] == pytest.approx(s * np.log(2.0))


def test_softplus_comes_from_the_library_and_agrees_across_backends():
    """numpy.logaddexp and jax.nn.softplus are the stable formulations; the
    hand-rolled branch that used to be here was only ever needed for the *log*
    of softplus, which is a different function's problem."""
    jnp = pytest.importorskip("jax.numpy")

    u = np.array([-800.0, -50.0, -1.0, 0.0, 1.0, 50.0, 800.0])
    got = softplus(u)
    assert np.all(np.isfinite(got))
    assert got[-1] == pytest.approx(800.0, rel=1e-9)   # no overflow
    assert got[3] == pytest.approx(np.log(2.0))
    assert np.allclose(np.asarray(softplus(jnp.asarray(u))), got, atol=1e-4)
    # log(softplus(u)) -> u far negative, where softplus itself underflows.
    assert log_softplus(u)[0] == pytest.approx(-800.0)
