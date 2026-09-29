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
