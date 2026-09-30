"""Diagnostic figures.

These exist to make a broken pipeline obvious before GPU time is spent on it,
so the tests check that each figure is produced, carries data, and degrades
sensibly when an input is missing -- a diagnostic that silently renders empty is
worse than none at all.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rubin_host_prior import plots
from rubin_host_prior.config import Config, PatchConfig
from rubin_host_prior.data import (
    LogFluxTransform,
    PatchDataset,
    ShardSet,
    estimate_softening,
    pool_shards,
)
from rubin_host_prior.data.synthetic import write_synthetic_shards

from conftest import TINY_ENERGY


@pytest.fixture(scope="module")
def extracted(tmp_path_factory):
    d = tmp_path_factory.mktemp("plots")
    write_synthetic_shards(d / "shards", n_patches=40, native_size=112,
                           patches_per_shard=40, seed=7)
    return d


@pytest.fixture(scope="module")
def shards(extracted):
    return ShardSet.from_dir(extracted / "shards")


@pytest.fixture(scope="module")
def dataset(shards):
    config = Config(energy=TINY_ENERGY, patch=PatchConfig(native_size=shards.native_size,
                                      out_size=24,
                                      pool_factor=3, out_sizes=(16, 24)))
    pooled, bands = pool_shards(shards, config)
    config.transform.softening = estimate_softening(
        pooled, config.transform.softening_sigma
    )
    return PatchDataset.from_shards(
        shards, config, LogFluxTransform.from_config(config.transform)
    )


def _n_drawn(fig):
    """Axes that actually have an image or a patch collection drawn on them."""
    return sum(1 for ax in fig.axes if ax.images or ax.patches or ax.lines)


def test_cutouts_draws_every_requested_panel(shards, tmp_path):
    fig, path = plots.plot_cutouts(shards, n=9, out=tmp_path)
    assert path.exists() and path.stat().st_size > 0
    assert sum(1 for ax in fig.axes if ax.images) == 9
    plt.close(fig)


def test_cutouts_caps_at_the_number_of_patches(shards, tmp_path):
    fig, _ = plots.plot_cutouts(shards, n=10_000, out=tmp_path)
    assert sum(1 for ax in fig.axes if ax.images) == len(shards)
    plt.close(fig)


def test_training_batch_shows_what_the_loader_yields(dataset, tmp_path):
    fig, path = plots.plot_training_batch(dataset, n=9, out=tmp_path)
    assert path.exists()
    drawn = [ax for ax in fig.axes if ax.images]
    assert len(drawn) == 9
    # The panel is the grid: no context border, so no red square and nothing
    # discarded.  The title must still name the other cycled sizes.
    size = dataset.config.patch.out_size
    assert dataset.config.energy.loss_margin == 0
    assert drawn[0].images[0].get_array().shape[-1] == size
    assert len(drawn[0].patches) == 0
    assert "also cycles 16" in fig._suptitle.get_text()
    plt.close(fig)


def test_training_batch_marks_a_loss_margin_when_there_is_one(dataset, tmp_path):
    """The red square is not gone, it is conditional: raise the margin and the
    figure says which pixels stopped counting."""
    import dataclasses

    original = dataset.config.energy
    dataset.config.energy = dataclasses.replace(original, loss_margin=4)
    try:
        fig, _ = plots.plot_training_batch(dataset, n=4, out=tmp_path)
        drawn = [ax for ax in fig.axes if ax.images]
        box = drawn[0].patches[0]
        size = dataset.config.patch.out_size - 8
        assert box.get_width() == size and box.get_height() == size
        assert box.get_xy() == (3.5, 3.5)
        assert box.get_edgecolor()[:3] == (1.0, 0.0, 0.0)
        assert "loss on the middle" in fig._suptitle.get_text()
        plt.close(fig)
    finally:
        dataset.config.energy = original


def test_transform_figure_marks_the_predicted_sky_position(dataset, tmp_path):
    from rubin_host_prior.data import expected_sky_scatter

    fig, path = plots.plot_transform(dataset, n=2, out=tmp_path)
    assert path.exists()
    assert sum(1 for ax in fig.axes if ax.images) == 6  # 2 rows x 3 stages
    hist_ax = [ax for ax in fig.axes if ax.patches and not ax.images][0]
    # One line, at log(s*log2): a single softening scale puts every band's sky
    # in the same place.
    marked = [ln.get_xdata()[0] for ln in hist_ax.lines]
    assert marked == pytest.approx([dataset.transform.sky_level])
    assert expected_sky_scatter(dataset.config.transform.softening_sigma) > 0
    plt.close(fig)


def test_hosts_figure_uses_the_catalogue_when_present(shards, extracted, tmp_path):
    pd = pytest.importorskip("pandas")
    hosts = pd.read_parquet(extracted / "shards" / "hosts.parquet")
    with_cat, _ = plots.plot_hosts(shards, hosts, out=tmp_path)
    without, _ = plots.plot_hosts(shards, None, out=tmp_path)
    titles = {ax.get_title() for ax in with_cat.axes}
    assert "host size" in titles
    assert "host distortion" in titles
    assert "host magnitude (r)" in titles
    assert "host blendedness" in titles
    # and the shard-only panels are there either way
    for fig in (with_cat, without):
        t = {ax.get_title() for ax in fig.axes}
        assert {"patches per band", "local sky noise", "variance step"} <= t
    assert len(with_cat.axes) > len(without.axes)
    plt.close(with_cat)
    plt.close(without)


def test_hosts_figure_survives_an_all_nan_column(shards, tmp_path):
    """Extraction leaves -1/NaN where a quantity was unavailable; a panel with
    nothing in it must say so rather than raising."""
    import copy

    faked = copy.copy(shards)
    faked.meta = dict(shards.meta)
    faked.meta["variance_step"] = np.full(len(shards), np.nan)
    fig, _ = plots.plot_hosts(faked, None, out=tmp_path)
    titles = [ax.get_title() for ax in fig.axes]
    assert any("no data" in t for t in titles)
    plt.close(fig)


def test_rejections_counts_compound_reasons(tmp_path):
    pd = pytest.importorskip("pandas")
    manifest = pd.DataFrame({
        "status": ["accepted"] * 3 + ["rejected"] * 3,
        "reasons": ["", "", "", "SAT:0.01>0.0;inner_CR:0.1>0.0",
                    "zero_tol:NO_DATA", "SAT:0.02>0.0"],
        "diag_sky_noise": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
    })
    fig, path = plots.plot_rejections(manifest, out=tmp_path)
    assert path.exists()
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert set(labels) == {"SAT", "inner_CR", "zero_tol"}  # semicolons split
    plt.close(fig)


def test_rejections_handles_nothing_rejected(tmp_path):
    pd = pytest.importorskip("pandas")
    manifest = pd.DataFrame({"status": ["accepted"] * 4, "reasons": [""] * 4})
    fig, _ = plots.plot_rejections(manifest, out=tmp_path)
    assert "nothing rejected" in fig.axes[0].texts[0].get_text()
    plt.close(fig)


def test_make_all_writes_the_full_set(shards, dataset, extracted, tmp_path):
    pd = pytest.importorskip("pandas")
    hosts = pd.read_parquet(extracted / "shards" / "hosts.parquet")
    manifest = pd.DataFrame({"status": ["accepted"] * 5, "reasons": [""] * 5})
    written = plots.make_all(shards, dataset=dataset, hosts=hosts,
                             manifest=manifest, out_dir=tmp_path, n_cutouts=4)
    assert {p.name for p in written} == {
        "cutouts.png", "hosts.png", "training_batch.png", "transform.png",
        "rejections.png",
    }
    assert all(p.stat().st_size > 0 for p in written)


def test_make_all_without_a_dataset_skips_the_loader_figures(shards, tmp_path):
    written = plots.make_all(shards, out_dir=tmp_path, n_cutouts=4)
    assert {p.name for p in written} == {"cutouts.png", "hosts.png"}


def test_package_imports_without_matplotlib(monkeypatch):
    """matplotlib is lazily imported so training does not depend on it."""
    import builtins

    real = builtins.__import__

    def fail(name, *a, **k):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fail)
    with pytest.raises(ImportError, match="need matplotlib"):
        plots._plt()


# -- the diffusion, forwards and backwards ----------------------------------


def test_scale_visibility_measures_what_it_claims():
    """Two fields with known answers.

    Band-pass noise falls like ``1/scale``, so for a *white* field -- whose
    structure falls the same way -- every scale must cross at the same sigma,
    and that sigma is the field's own rms: white data dies all at once. For a
    red field, whose power is concentrated at large scales, the crossing must
    rise with scale. That contrast is the whole content of the measurement, and
    it is what says whether the galaxies in a patch outlive the stars.
    """
    from rubin_host_prior.plots import scale_visibility

    rng = np.random.default_rng(0)
    scales, n = (1, 2, 4, 8, 16), 64

    white = 3.0 * rng.normal(size=(16, n, n))
    flat = scale_visibility(white, scales=scales)["sigma_visible"]
    assert flat == pytest.approx([3.0] * len(scales), rel=0.08), flat

    # Built in Fourier, so its spectrum is exactly what the test says it is.
    ky, kx = np.meshgrid(np.fft.fftfreq(n) * n, np.fft.fftfreq(n) * n,
                         indexing="ij")
    k = np.hypot(ky, kx)
    k[0, 0] = 1.0
    phases = (rng.normal(size=(16, n, n)) + 1j * rng.normal(size=(16, n, n)))
    red = np.fft.ifft2(phases * k ** -1.2).real
    rising = scale_visibility(red, scales=scales)["sigma_visible"]
    assert all(b > a for a, b in zip(rising, rising[1:])), rising
    assert rising[-1] > 4 * rising[0], rising


def test_forward_diffusion_shows_every_level_and_the_scale_panel(dataset, tmp_path):
    from rubin_host_prior.diffusion import VESDE

    sde = VESDE(sigma_min=0.02, sigma_max=4.0)
    fig, path = plots.plot_forward_diffusion(
        dataset, sde, n=2, n_sigma=5, scales=(1, 2, 4), out=tmp_path)
    assert path.exists()
    drawn = [ax for ax in fig.axes if ax.images]
    assert len(drawn) == 2 * (5 + 1)          # a clean column plus the ladder
    size = dataset.config.patch.out_size
    assert drawn[0].images[0].get_array().shape[-1] == size
    # The SNR panel: one line per scale plus the SNR = 1 rule.
    panel = [ax for ax in fig.axes if ax.get_xlabel() == "sigma"][0]
    assert len(panel.get_lines()) == 3 + 1
    assert panel.get_xscale() == "log" and panel.get_yscale() == "log"
    plt.close(fig)


def test_reverse_trajectory_lines_up_with_the_forward_one(tiny_model, tmp_path):
    """Same ladder in both figures, so column k is the same sigma in each."""
    from rubin_host_prior.diffusion import VESDE

    sde = VESDE(sigma_min=0.05, sigma_max=2.0)
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(6, 12, 12)) * 0.5
    fig, path = plots.plot_reverse_trajectory(
        tiny_model, sde, out_size=12, n=2, n_sigma=4, n_steps=6,
        reference=reference, scales=(1, 2, 4), out=tmp_path)
    assert path.exists()
    drawn = [ax for ax in fig.axes if ax.images]
    assert len(drawn) == 2 * 4
    assert drawn[0].images[0].get_array().shape[-1] == 12
    # Columns are labelled with their sigma, descending left to right.
    titles = [float(ax.get_title()) for ax in drawn[:4] if ax.get_title()]
    assert titles == sorted(titles, reverse=True), titles
    # The width panel carries the trajectory and the forward reference.
    panel = [ax for ax in fig.axes if ax.get_xlabel() == "sigma"][0]
    assert len(panel.get_lines()) == 2
    # And the power panel carries the samples against the real patches: the
    # same total width can be spent on any mixture of scales, so matching the
    # width curve says nothing about which scales got it.
    power = [ax for ax in fig.axes
             if ax.get_xlabel() == "spatial scale (px)"][0]
    assert len(power.get_lines()) == 2
    assert power.get_xscale() == "log" and power.get_yscale() == "log"
    plt.close(fig)


def test_forward_patches_shows_the_same_scenes_either_way(dataset):
    """``--augment`` changes how the patches are presented, not which they are:
    the indices are the ones ``validation_batch`` fixes, so the figure and its
    table stay comparable between runs."""
    from rubin_host_prior.plots import forward_patches

    plain = forward_patches(dataset, 4, augment=False, seed=0)
    again = forward_patches(dataset, 4, augment=False, seed=1)
    aug = forward_patches(dataset, 4, augment=True, seed=0)

    size = dataset.config.patch.out_size
    assert plain.shape == (4, size, size) == aug.shape
    # Unaugmented is deterministic whatever the seed; augmented is not the same
    # pixels, but it is the same scenes and so the same overall level.
    np.testing.assert_array_equal(plain, again)
    assert not np.array_equal(plain, aug)
    assert aug.mean() == pytest.approx(plain.mean(), rel=0.2)


def test_band_power_beats_a_box_filter_at_attribution():
    """Why the FFT replaced a difference of box filters.

    Against a field with a *known* attenuation above 16 px, the box estimator
    reported 0.76 at 8 px where the truth is 1.00 -- sinc sidelobes smear a
    deficit a full octave down, which is exactly the kind of error that would
    have moved the crossover in a diagnosis. The periodogram attributes the
    band exactly, and it is the basis the mode-counting argument is made in.
    """
    from rubin_host_prior.plots import band_power

    n, truth = 128, 0.40
    rng = np.random.default_rng(0)
    ky, kx = np.meshgrid(np.fft.fftfreq(n) * n, np.fft.fftfreq(n) * n,
                         indexing="ij")
    k = np.hypot(ky, kx)
    k[0, 0] = 1.0
    base = (rng.normal(size=(32, n, n))
            + 1j * rng.normal(size=(32, n, n))) * k ** -1.2
    clean = np.fft.ifft2(base).real
    starved = np.fft.ifft2(base * np.where(k < n / 16.0, truth, 1.0)).real

    scales = (1, 2, 4, 8, 16, 32)
    a = band_power(clean, scales)
    b = band_power(starved, scales)
    ratio = [x / y for x, y in zip(b["rms"], a["rms"])]
    assert ratio[:3] == pytest.approx([1.0, 1.0, 1.0], abs=0.02), ratio
    assert ratio[3] == pytest.approx(1.0, abs=0.03), ratio   # 8 px untouched
    assert ratio[5] == pytest.approx(truth, abs=0.03), ratio  # 32 px starved


def test_band_power_windows_against_the_wrap():
    """The FFT assumes the patch is periodic and a stamp is not: a gradient
    across the frame is a step at the edge, and a step has power at every k.
    Unwindowed that put 540x too much into the 1 px band."""
    from rubin_host_prior.plots import band_power

    n = 128
    rng = np.random.default_rng(1)
    # A red field, which is the case that matters: its fine-scale power is
    # small, so an edge step buries it.  In a white field the 1 px band has
    # thousands of modes carrying most of the variance and the wrap is lost in
    # it -- the leak is real there too, just invisible.
    ky, kx = np.meshgrid(np.fft.fftfreq(n) * n, np.fft.fftfreq(n) * n,
                         indexing="ij")
    k = np.hypot(ky, kx)
    k[0, 0] = 1.0
    red = np.fft.ifft2((rng.normal(size=(8, n, n))
                        + 1j * rng.normal(size=(8, n, n))) * k ** -1.2).real
    ramp = red + 1.5 * np.linspace(-1, 1, n)[:, None]

    fine = 1
    clean = band_power(red, (fine,))["rms"][0]
    leaked = band_power(ramp, (fine,), window=False)["rms"][0]
    fixed = band_power(ramp, (fine,), window=True)["rms"][0]
    # ...which is why the option exists, even though it is off by default.
    assert leaked > 100 * clean, (leaked, clean)
    assert fixed == pytest.approx(clean, rel=0.05), (fixed, clean)


def test_band_power_reports_its_own_error_bar():
    """The largest band is tens of modes on a 128 px grid, so a ratio quoted
    from it without an error bar is over-claiming."""
    from rubin_host_prior.plots import band_power

    got = band_power(np.random.default_rng(0).normal(size=(16, 128, 128)),
                     (1, 8, 32))
    assert got["n_modes"] == [3535, 600, 36]
    assert all(b > a for a, b in zip(got["rel_error"], got["rel_error"][1:]))
    assert got["rel_error"][-1] == pytest.approx(1 / np.sqrt(2 * 36 * 16))


def test_the_window_would_charge_off_centre_structure_and_is_therefore_off():
    """Why ``window`` defaults to False, against the usual advice.

    A Hann window is a centred bump, so it preserves large-scale power in the
    middle of the frame and suppresses large-scale power that is not.  Take a
    host-centred stamp and randomise its Fourier phases: the power spectrum is
    identical mode for mode and only the centring is gone, so the true band
    ratio is exactly 1.000.  The window does not report that.

    It matters because this is used to compare host-centred real patches
    against model samples, which have no preferred centre -- so the bias falls
    entirely on the thing being measured.
    """
    from rubin_host_prior.plots import band_power

    n, m = 128, 32
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - n / 2, xx - n / 2)
    centred = (2.0 * np.exp(-(r / 18.0) ** 1.2)[None]
               + 0.54 * rng.normal(size=(m, n, n)))
    spec = np.abs(np.fft.fft2(centred))
    phase = rng.uniform(0, 2 * np.pi, size=spec.shape)
    scrambled = np.fft.ifft2(spec * np.exp(1j * phase)).real
    # Random phases are not Hermitian-symmetric, so `.real` drops half the
    # power -- a flat factor across every band, restored here so the comparison
    # is about the *shape* of the spectrum and nothing else.
    scrambled *= np.sqrt(centred.var(axis=(1, 2))
                         / scrambled.var(axis=(1, 2)))[:, None, None]

    scales = (1, 2, 4, 8, 16, 32)
    plain = [b / a for a, b in zip(band_power(centred, scales)["rms"],
                                   band_power(scrambled, scales)["rms"])]
    windowed = [b / a for a, b in
                zip(band_power(centred, scales, window=True)["rms"],
                    band_power(scrambled, scales, window=True)["rms"])]
    assert plain == pytest.approx([1.0] * len(scales), abs=0.05), plain
    assert windowed[-1] < 0.6, windowed        # 32 px, truth 1.0
    assert windowed[-2] < 0.8, windowed        # 16 px


def test_the_total_row_is_the_sum_of_the_bands():
    """Or the table's last line would come from a different estimator than the
    lines above it, and would not add up."""
    from rubin_host_prior.plots import band_power

    n = 128
    rng = np.random.default_rng(0)
    scales = (1, 2, 4, 8, 16, 32)

    white = rng.normal(size=(8, n, n))
    got = band_power(white, scales)
    assert got["below_scale"] == 64 and got["below_modes"] == 8
    assert got["total_rms"] == pytest.approx(
        np.sqrt(sum(v ** 2 for v in got["rms"]) + got["below_rms"] ** 2))
    # Against the per-patch variance: band_power removes each patch's own mean,
    # so a global np.std would also carry the scatter *between* patches.
    per_patch = float(np.sqrt(np.var(white, axis=(1, 2)).mean()))
    assert got["total_rms"] == pytest.approx(per_patch, rel=0.01)

    # And the reason it is reported rather than dropped: in a red field those
    # eight modes are a large share of the variance, not a rounding error.
    ky, kx = np.meshgrid(np.fft.fftfreq(n) * n, np.fft.fftfreq(n) * n,
                         indexing="ij")
    k = np.hypot(ky, kx)
    k[0, 0] = 1.0
    red = np.fft.ifft2((rng.normal(size=(8, n, n))
                        + 1j * rng.normal(size=(8, n, n))) * k ** -2.0).real
    red_bp = band_power(red, scales)
    dropped = red_bp["below_rms"] ** 2 / red_bp["total_rms"] ** 2
    assert dropped > 0.25, dropped
    assert red_bp["total_rms"] == pytest.approx(
        float(np.sqrt(np.var(red, axis=(1, 2)).mean())), rel=0.01)


def test_the_augmented_reference_is_a_different_distribution(dataset):
    """Not a quibble: the loader's translation can cut the host at the frame,
    which a shift cannot, so the spectra really differ.  A periodogram is
    translation-invariant, so this is *not* a centring effect -- it is the crop
    landing somewhere that contains less galaxy.
    """
    from rubin_host_prior.plots import band_power, forward_patches

    scales = (1, 2, 4, 8)
    centred = band_power(forward_patches(dataset, 24, augment=False), scales)
    augmented = [band_power(forward_patches(dataset, 24, augment=True, seed=s),
                            scales) for s in range(4)]

    fine = np.mean([b["rms"][0] for b in augmented]) / centred["rms"][0]
    coarse = np.mean([b["rms"][-1] for b in augmented]) / centred["rms"][-1]
    # Fine scales are sky and are unmoved by where the crop landed; the coarse
    # band is the galaxy and is not.
    assert fine == pytest.approx(1.0, abs=0.08), fine
    assert abs(coarse - 1.0) > abs(fine - 1.0), (coarse, fine)


def test_schedule_headroom_drops_bands_with_no_modes():
    """An octave narrower than the grid's mode spacing is empty, and 0/0 there
    once came back as sigma_visible = inf, a headroom of 0, and a warning
    telling you to raise sigma_max to infinity."""
    x = np.random.default_rng(0).standard_normal((8, 1, 16, 16))
    head = plots.schedule_headroom(x, 8.0, (1, 2, 4, 8))
    assert ">" not in "".join(head["band"])  # nothing coarser than 8 px on a 16 grid
    assert all(m > 0 for m in head["n_modes"])
    assert all(np.isfinite(v) for v in head["sigma_visible"])
    with pytest.raises(ValueError, match="coarsest meaningful scale is 8"):
        plots.schedule_headroom(x, 8.0, (32,))


def test_schedule_headroom_orders_by_persistence():
    """Coarse first: the row that decides whether sigma_max is big enough is
    the one with the most power per mode, and it should not have to be hunted
    for at the bottom of the table."""
    rng = np.random.default_rng(0)
    h = 32
    ky, kx = np.meshgrid(np.fft.fftfreq(h) * h, np.fft.fftfreq(h) * h,
                         indexing="ij")
    k = np.hypot(ky, kx)
    k[0, 0] = 1.0
    x = np.fft.ifft2(np.fft.fft2(rng.standard_normal((16, h, h)))
                     * k ** -1.5).real[:, None]
    head = plots.schedule_headroom(x, 4.0, (1, 2, 4, 8))
    assert head["sigma_visible"] == sorted(head["sigma_visible"], reverse=True)
    assert head["headroom"] == sorted(head["headroom"])
    # floor and headroom carry the same ordering, by construction
    assert head["pflow_ratio"] == sorted(head["pflow_ratio"])
