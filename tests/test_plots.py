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
    from rubin_host_prior.plots import _box_smooth, scale_visibility

    rng = np.random.default_rng(0)
    scales = (1, 2, 4, 8, 16)

    white = 3.0 * rng.normal(size=(4, 64, 64))
    flat = scale_visibility(white, scales=scales)["sigma_visible"]
    assert flat == pytest.approx([3.0] * len(scales), rel=0.05), flat

    red = sum(float(s) * _box_smooth(rng.normal(size=(4, 64, 64)), s)
              for s in scales)
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
    fig, path = plots.plot_reverse_trajectory(
        tiny_model, sde, out_size=12, n=2, n_sigma=4, n_steps=6,
        data_std=0.5, out=tmp_path)
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
