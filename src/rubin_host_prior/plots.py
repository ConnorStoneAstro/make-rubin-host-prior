"""Diagnostic figures for the extraction and the loader.

The point of these is to make a broken pipeline obvious before any GPU time is
spent on it.  In rough order of how often they catch something:

``rejections``     is the gate throwing away the bright dense hosts the project
                   exists to model?  The accepted-vs-rejected magnitude
                   distribution is the single most informative panel here.
``transform``      does the log representation look the way the arithmetic says
                   it should -- sky at ``log(log 2)``, noise roughly symmetric,
                   sources well separated from it?
``training``       what the network actually receives, augmentation and all.
``cutouts``        raw stamps, in units of their own sky noise, to see the range
                   of scenes that were selected.
``hosts``          the selected population: size, magnitude, ellipticity, band,
                   sky noise, depth.

matplotlib is imported lazily so the package stays importable without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Native cutouts are shown as asinh(flux / sky noise): a stretch in sigma units,
# so bands with very different depths are directly comparable.  The low end is
# pinned rather than taken from a percentile, so the noise floor sits at the
# same place in every panel and faint structure is not drowned by it; the high
# end is per-panel but floored, so a blank patch still reads as blank.
_ASINH_VMIN = -2.0
_ASINH_VMAX_PCT = 99.8
_ASINH_VMAX_FLOOR = 3.0


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "diagnostic plots need matplotlib; pip install 'rubin-host-prior[dev]'"
        ) from exc
    return plt


def _grid(n: int):
    cols = int(np.ceil(np.sqrt(n)))
    return int(np.ceil(n / cols)), cols


def _show(ax, img, vmin=None, vmax=None, cmap="magma"):
    ax.set_axis_off()
    return ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")


def _save(fig, out: Path | None, name: str) -> Path | None:
    if out is None:
        return None
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    return path


# -- what the model has learned to draw -------------------------------------


def plot_samples(samples, step: int | None = None, out: Path | None = None,
                 name: str = "samples"):
    """A square grid of samples, in the log space the model works in.

    Deliberately the same representation and the same kind of figure as
    ``training_batch.png``, so the two can be put side by side: that comparison
    is the whole point of drawing samples during training, and it only works if
    nothing is stretched differently between them.

    One colour scale across the whole grid, from the 0.5/99.5 percentiles of all
    the samples together.  Per-panel scaling would make every sample look
    equally structured, including the ones that are noise.
    """
    plt = _plt()

    x = np.asarray(samples)
    if x.ndim == 4:                 # (B, C, H, W) -> first channel
        x = x[:, 0]
    rows, cols = _grid(len(x))
    finite = x[np.isfinite(x)]
    lo, hi = (np.percentile(finite, (0.5, 99.5)) if finite.size
              else (0.0, 1.0))

    fig, axes = plt.subplots(rows, cols, figsize=(1.35 * cols, 1.35 * rows))
    for ax, img in zip(np.ravel(np.atleast_1d(axes)), x):
        _show(ax, img, lo, hi, cmap="viridis")
    for ax in np.ravel(np.atleast_1d(axes))[len(x):]:
        ax.set_axis_off()
    title = f"{len(x)} samples, log space"
    if step is not None:
        title += f", step {step:,}"
    # Non-finite values mean the sampler diverged, which a colour scale taken
    # from the finite ones would hide completely.
    n_bad = int(np.sum(~np.isfinite(x)))
    if n_bad:
        title += f"  --  {n_bad:,} non-finite pixels"
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return fig, _save(fig, out, name)


# -- raw cutouts -----------------------------------------------------------


def plot_cutouts(shards, n: int = 100, seed: int = 0, out: Path | None = None):
    """Grid of native-resolution stamps, stretched in units of their sky noise.

    Each panel is ``asinh(flux / sky_noise)``, so a faint u-band patch and a deep
    i-band one are on the same footing and the annotation reads in sigma.
    """
    plt = _plt()
    from .config import BANDS

    rng = np.random.default_rng(seed)
    n = min(n, len(shards))
    idx = np.sort(rng.choice(len(shards), size=n, replace=False))
    images = shards.gather(idx, "image")
    band = np.asarray(shards.meta["band_idx"])[idx]
    noise = np.asarray(shards.meta["sky_noise"])[idx]

    rows, cols = _grid(n)
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.25 * rows))
    for k, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.set_axis_off()
        if k >= n:
            continue
        sigma = noise[k] if np.isfinite(noise[k]) and noise[k] > 0 else 1.0
        stretched = np.arcsinh(images[k] / sigma)
        hi = max(np.percentile(stretched, _ASINH_VMAX_PCT), _ASINH_VMAX_FLOOR)
        _show(ax, stretched, _ASINH_VMIN, hi)
        b = BANDS[band[k]] if band[k] < len(BANDS) else "?"
        ax.set_title(f"{b}  $\\sigma$={sigma:.1f} nJy", fontsize=7, pad=2)
    fig.suptitle(
        f"native cutouts, asinh(flux / sky noise)  ({len(shards)} patches total)",
        fontsize=10,
    )
    fig.tight_layout()
    return fig, _save(fig, out, "cutouts")


# -- the diffusion itself, forwards and backwards ---------------------------
#
# These two are meant to be read side by side.  ``VESDE.ladder`` gives both the
# same noise levels, so column k of one is the same sigma as column k of the
# other, and a model that has learned the score should produce at each sigma
# something whose structure matches what the forward process leaves there.  The
# column where they stop resembling each other is the sigma range to suspect.


def band_power(x: np.ndarray, scales, window: bool = False) -> dict:
    """Variance per octave band, straight off the periodogram.

    ``scales`` are in pixels; the band for ``l`` is the octave ``|k|`` from
    ``N/2l`` to ``N/l``, i.e. structure between ``l`` and ``2l`` px.

    **The FFT and not a difference of box filters**, which is what this was.
    Box filters have sinc sidelobes, so a band leaks badly into its neighbours:
    measured against a field with a *known* attenuation of 0.40 above 16 px, the
    box estimator reported 0.76 at 8 px where the truth is 1.00 -- it smeared
    the deficit a full octave down. The FFT recovered 1.000, 1.000, 0.400,
    0.400 exactly. It is also the basis the mode-counting argument is made in,
    so the measurement and the reasoning are about the same object.

**``window`` is off by default, and that is the opposite of the usual
    advice.**  A Hann window is the standard guard against the FFT's periodicity
    assumption: a gradient across the frame wraps into a step at the edge, and a
    step has power at every ``k`` -- measured on a red field, an unwindowed
    periodogram put **540x** too much power in the 1 px band.

    But a Hann window is *itself* a centred bump, so it preserves large-scale
    power that sits in the middle of the frame and suppresses large-scale power
    that does not.  Measured by taking a host-centred stamp and randomising its
    Fourier phases -- identical power spectrum mode for mode, centring
    destroyed, so the true ratio is exactly 1.000 -- the window reported
    **0.631 at 16 px and 0.434 at 32 px**, while the unwindowed periodogram gave
    1.011 and 1.002.

    That bias lands exactly where this measurement is used: the real patches are
    host-centred by construction and model samples have no preferred centre, so
    windowing charges the samples a factor of two at the largest scale for a
    difference that is not in their power spectrum.  Off by default; turn it on
    only for data with a genuine gradient, and know what it costs.

    ``n_modes`` and ``rel_error`` come back with it because the largest band is
    small -- 36 modes on a 128 px grid -- and a ratio quoted from it deserves
    its error bar.

    The bands are the octaves ``|k|`` in ``[N/2l, N/l)``, disjoint and covering
    everything down to the largest scale asked for.  Whatever sits *below* that
    comes back as ``below_rms``, and is in ``total_rms`` too: on a 128 px grid
    a scale list ending at 32 leaves only 8 modes uncovered, which sounds
    negligible and is not -- those are the modes spanning the whole frame, and
    in a field as red as a galaxy stamp they can hold a large share of the
    variance.  Silently dropping them would make ``total_rms`` a different
    number from the field's own, which is how this was found.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 4:
        x = x[:, 0]
    if x.ndim == 2:
        x = x[None]
    h, w = x.shape[-2:]
    if h != w:
        raise ValueError(f"expected square patches, got {(h, w)}")
    n_patches = len(x)
    # The mean is the k=0 mode and would swamp everything: x is log flux, so the
    # sky sits near +3 and the DC term is enormous next to the structure.
    x = x - x.mean(axis=(-2, -1), keepdims=True)
    if window:
        win = np.hanning(h)[:, None] * np.hanning(h)[None, :]
        x = x * (win / np.sqrt((win ** 2).mean()))
    power = np.abs(np.fft.fft2(x)) ** 2
    ky, kx = np.meshgrid(np.fft.fftfreq(h) * h, np.fft.fftfreq(h) * h,
                         indexing="ij")
    k = np.hypot(ky, kx)

    out = {"scales": tuple(int(s) for s in scales), "rms": [], "n_modes": [],
           "rel_error": [], "white_rms": [], "total_rms": 0.0,
           "below_scale": 0, "below_rms": 0.0, "below_modes": 0}
    for scale in out["scales"]:
        band = (k >= h / (2.0 * scale)) & (k < h / float(scale))
        m = int(band.sum())
        var = float(power[..., band].sum(-1).mean()) / h ** 4
        out["rms"].append(float(np.sqrt(var)))
        out["n_modes"].append(m)
        # Real input pairs conjugate modes, so m counts each twice.  The error
        # on a variance from m/2 independent modes over n patches is
        # sqrt(2/(m*n)); on the rms it is half that.
        out["rel_error"].append(
            float(1.0 / np.sqrt(2.0 * m * n_patches)) if m else float("inf"))
        # White noise of unit variance is flat, so its share of a band is just
        # the band's share of the modes.  Exact, and no Monte Carlo draw.
        out["white_rms"].append(float(np.sqrt(m / h ** 2)))

    # Everything larger than the biggest band asked for.  k = 0 is the mean and
    # was removed above, so this is genuine structure and not an offset.
    coarsest = max(out["scales"])
    below = (k > 0) & (k < h / (2.0 * coarsest))
    out["below_scale"] = int(2 * coarsest)
    out["below_modes"] = int(below.sum())
    out["below_rms"] = float(
        np.sqrt(float(power[..., below].sum(-1).mean()) / h ** 4))
    # The fewest modes of any row, so the row that most needs its error bar.
    out["below_rel_error"] = (
        float(1.0 / np.sqrt(2.0 * out["below_modes"] * n_patches))
        if out["below_modes"] else float("inf"))
    # Measured the same way as the bands, so the table's last row is the sum of
    # the ones above it rather than a number from a different estimator.
    out["total_rms"] = float(
        np.sqrt(sum(v ** 2 for v in out["rms"]) + out["below_rms"] ** 2))
    return out


def scale_visibility(clean: np.ndarray, scales=(1, 2, 4, 8, 16, 32),
                     window: bool = True) -> dict:
    """At which sigma does structure of each spatial scale stop being visible?

    **The question behind "the samples are all stars".**  Band-pass noise is
    flat in ``k``, so a band's share of it is just that band's share of the
    modes -- exactly, with no draw to average over.  The signal-to-noise at any
    sigma follows:

        SNR(scale, sigma) = rms(scale) / (sigma * white_rms(scale))

    and the crossing ``SNR = 1`` is the noise level above which that scale is
    gone.  Read against the schedule: a scale whose crossing sits above
    ``sigma_max`` is never resolved at any noise level the model trains on, and
    one below ``sigma_min`` is only ever learned in the last few steps.

    A white field gives the same crossing at every scale, and that crossing is
    its own rms: white data dies all at once.  Anything red outlives it at the
    large end, and by how much is the whole question.
    """
    got = band_power(clean, scales, window=window)
    return {
        "scales": got["scales"],
        "signal": got["rms"],
        "noise": got["white_rms"],
        "n_modes": got["n_modes"],
        "rel_error": got["rel_error"],
        "sigma_visible": [s / n if n > 0 else np.inf
                          for s, n in zip(got["rms"], got["white_rms"])],
    }


def _sigma_panel(ax, vis, sde, title):
    """SNR against sigma, one line per spatial scale, with the schedule marked."""
    sigmas = np.geomspace(sde.sigma_min * 0.5, sde.sigma_max * 2.0, 200)
    cmap = _plt().get_cmap("viridis")
    for i, scale in enumerate(vis["scales"]):
        snr = vis["signal"][i] / (sigmas * vis["noise"][i])
        ax.loglog(sigmas, snr, color=cmap(i / max(len(vis["scales"]) - 1, 1)),
                  label=f"{scale} px", lw=1.6)
    ax.axhline(1.0, color="k", ls=":", lw=1.0)
    ax.axvspan(sde.sigma_min, sde.sigma_max, color="0.85", zorder=0)
    ax.set_xlabel("sigma")
    ax.set_ylabel("signal / noise at this scale")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, ncol=2, title="scale", title_fontsize=6)
    ax.grid(alpha=0.25, which="both")


def forward_patches(dataset, n: int, augment: bool = False, seed: int = 0):
    """The patches the forward figure buries, and the ones its table measures.

    The same scenes either way -- the indices are the ones ``validation_batch``
    fixes -- so ``augment`` changes how they are presented and not which they
    are.

    **Which to use depends on the question.**  For a figure, unaugmented: it is
    the same picture every run, which is one less thing to hold constant. For a
    *comparison against samples*, augmented, because that is the distribution
    the model was trained to match and it is not the same distribution. The
    translation is not a mere shift -- a crop that wanders can cut the host in
    half, which no shift does -- so the spectra genuinely differ: measured on
    stamps with the default geometry, the augmented reference has **16% less**
    power at 32 px, and under 3% difference at every smaller scale. Comparing
    samples against a centred reference charges them that 16%.
    """
    rng = np.random.default_rng(12345)
    idx = np.sort(rng.choice(len(dataset), size=min(n, len(dataset)),
                             replace=False))
    return dataset.make_batch(idx, rng=np.random.default_rng(seed),
                              augment=augment)[:, 0]


def plot_forward_diffusion(
    dataset,
    sde,
    n: int = 4,
    n_sigma: int = 8,
    seed: int = 0,
    scales=(1, 2, 4, 8, 16, 32),
    augment: bool = False,
    out: Path | None = None,
):
    """Real patches buried by the forward process, and what survives where.

    The rows are patches from the loader, the first column clean and the rest at
    the ``VESDE.ladder`` noise levels, ascending.  **One noise field per patch,
    scaled** -- the marginals are identical either way, and reusing it makes the
    progression legible: the same structure is watched going under rather than a
    fresh pattern each column.

    Each column is stretched to its own percentiles.  A shared stretch would
    show the high-sigma columns as flat grey, which is true and useless; the
    question these answer is *what is still discernible here*, which is what a
    per-column stretch asks.

    The bottom panel is the quantitative version, and the one to read first: the
    noise level at which each spatial scale stops being visible.
    """
    plt = _plt()

    clean = forward_patches(dataset, n, augment, seed)
    n = len(clean)
    ladder = np.asarray(sde.ladder(n_sigma))[::-1]          # ascending
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=clean.shape)

    cols = n_sigma + 1
    fig = plt.figure(figsize=(1.45 * cols, 1.45 * n + 4.0))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.45 * n, 3.2], hspace=0.30)
    grid = outer[0].subgridspec(n, cols, hspace=0.06, wspace=0.05)
    for j in range(cols):
        panel = clean if j == 0 else clean + ladder[j - 1] * eps
        lo, hi = np.percentile(panel, (0.5, 99.5))
        for i in range(n):
            ax = fig.add_subplot(grid[i, j])
            _show(ax, panel[i], lo, hi, cmap="viridis")
            if i == 0:
                ax.set_title("clean" if j == 0 else f"{ladder[j - 1]:.3g}",
                             fontsize=7)

    vis = scale_visibility(clean, scales)
    _sigma_panel(fig.add_subplot(outer[1]), vis, sde,
                 "what survives: SNR per spatial scale against sigma "
                 "(shaded = the trained schedule; dotted = SNR 1)")
    fig.suptitle(
        f"forward diffusion: {n} real patches at {n_sigma} noise levels, "
        f"each column on its own stretch"
        + (", as training sees them" if augment else ""), fontsize=10)
    return fig, _save(fig, out, "forward_diffusion")


def plot_reverse_trajectory(
    model,
    sde,
    out_size: int,
    n: int = 4,
    n_sigma: int = 8,
    n_steps: int = 128,
    seed: int = 0,
    reference: np.ndarray | None = None,
    scales=(1, 2, 4, 8, 16, 32),
    out: Path | None = None,
):
    """How a sample comes out of the noise: the sampler, with a recorder on it.

    Columns descend the same ``VESDE.ladder`` the forward figure ascends, so the
    two line up and can be compared at matched sigma.  Each column is stretched
    to its own percentiles, for the same reason.

    The bottom left panel is the check that costs nothing and catches a
    diverging sampler: the forward marginal at ``sigma`` has width
    ``sqrt(data_var + sigma^2)``, so a trajectory whose spread departs from that
    curve is not tracking the distribution it is supposed to be reversing --
    whatever the pictures look like.

    Pass ``reference`` -- real patches, ideally the ones ``forward_patches``
    gives, so the two figures describe the same scenes -- and the bottom right
    panel asks what the left one cannot: **the same total variance can be spent
    on any mixture of scales.** A model that follows the width curve exactly and
    still draws only point sources is putting its variance in the wrong place,
    and that shows up here as a ratio rather than an impression.
    """
    import jax

    from .diffusion.sampler import pflow_trajectory

    plt = _plt()

    shape = (n, model.config.in_channels, out_size, out_size)
    states, step_sigmas = pflow_trajectory(
        model, jax.random.key(seed), shape, sde, n_steps=n_steps)
    states = np.asarray(states)[:, :, 0]                  # (steps, n, H, W)
    step_sigmas = np.asarray(step_sigmas)

    ladder = np.asarray(sde.ladder(n_sigma))              # descending
    picks = [int(np.argmin(np.abs(np.log(step_sigmas) - np.log(s))))
             for s in ladder]

    cols = len(picks)
    fig = plt.figure(figsize=(1.45 * cols, 1.45 * n + 4.0))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.45 * n, 3.2], hspace=0.30)
    grid = outer[0].subgridspec(n, cols, hspace=0.06, wspace=0.05)
    for j, k in enumerate(picks):
        panel = states[k]
        finite = panel[np.isfinite(panel)]
        lo, hi = (np.percentile(finite, (0.5, 99.5)) if finite.size else (0, 1))
        for i in range(n):
            ax = fig.add_subplot(grid[i, j])
            _show(ax, panel[i], lo, hi, cmap="viridis")
            if i == 0:
                ax.set_title(f"{step_sigmas[k]:.3g}", fontsize=7)

    bottom = outer[1].subgridspec(1, 2, wspace=0.28)
    ax = fig.add_subplot(bottom[0])
    spread = states.reshape(len(states), -1).std(axis=1)
    ax.loglog(step_sigmas, spread, lw=1.6, label="trajectory")
    data_std = None if reference is None else float(np.std(reference))
    var = 0.0 if data_std is None else data_std ** 2
    ax.loglog(step_sigmas, np.sqrt(step_sigmas ** 2 + var), "k--", lw=1.2,
              label=r"$\sqrt{\sigma^2 + \mathrm{var}(x)}$"
                    + ("" if data_std is not None else "  (data var unknown)"))
    ax.set_xlabel("sigma")
    ax.set_ylabel("std of the scene")
    ax.set_title("total width against sigma", fontsize=9)
    ax.invert_xaxis()
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which="both")

    ax2 = fig.add_subplot(bottom[1])
    got = scale_visibility(states[-1], scales)
    ax2.loglog(got["scales"], got["signal"], "o-", lw=1.6, label="samples")
    if reference is not None:
        want = scale_visibility(reference, scales)
        ax2.loglog(want["scales"], want["signal"], "s--", lw=1.4, color="k",
                   label="real patches")
        for sc, a, b in zip(got["scales"], got["signal"], want["signal"]):
            ax2.annotate(f"x{a / b:.2f}", (sc, a), textcoords="offset points",
                         xytext=(0, -12), ha="center", fontsize=6)
    ax2.set_xlabel("spatial scale (px)")
    ax2.set_ylabel("band-pass rms")
    ax2.set_title("and where that width is spent", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.25, which="both")
    fig.suptitle(
        f"reverse trajectory: {n} samples at {n_sigma} of {n_steps} steps, "
        f"sigma falling left to right, each column on its own stretch",
        fontsize=10)
    return fig, _save(fig, out, "reverse_trajectory")


# -- what the loader yields ------------------------------------------------


def plot_training_batch(dataset, n: int = 25, seed: int = 0, out: Path | None = None):
    """Grid of exactly what the network receives: pooled, log-space, augmented.

    The panel *is* the grid: same-mode convolutions score every pixel, so the
    loader feeds exactly ``out_size`` and there is no context border to show and
    no reflection to look for.

    A **red square** appears only if ``energy.loss_margin`` is non-zero, marking
    the region the loss is computed on.  At the default margin of 0 there is no
    square, because there is nothing being discarded: the model is size-locked,
    so its border is part of the operator rather than an artefact.

    A shared colour scale across panels, so the spread between patches is
    visible rather than normalised away -- the prior has to cover that spread.
    """
    plt = _plt()
    rng = np.random.default_rng(seed)
    n = min(n, len(dataset))
    idx = np.sort(rng.choice(len(dataset), size=n, replace=False))
    x = dataset.make_batch(idx, rng=rng, augment=True)[:, 0]
    lo, hi = np.percentile(x, (0.5, 99.5))

    margin = dataset.config.energy.loss_margin
    size = x.shape[-1] - 2 * margin

    rows, cols = _grid(n)
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.2 * rows))
    im = None
    for k, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.set_axis_off()
        if k < n:
            im = _show(ax, x[k], lo, hi, cmap="viridis")
            if margin > 0:
                # Inside the line is what the loss is computed on; outside it
                # is border the margin discards.  imshow puts pixel centres on
                # integers, so the edge of pixel `margin` is at margin - 0.5.
                # linewidth 1.0, not less: below about one output pixel the
                # line is antialiased into the background and reads as grey on
                # the dark parts of the panel, which is worse than no line.
                ax.add_patch(plt.Rectangle(
                    (margin - 0.5, margin - 0.5), size, size,
                    fill=False, edgecolor="red", linewidth=1.0,
                ))
    if im is not None:
        fig.colorbar(
            im,
            ax=np.atleast_1d(axes).ravel().tolist(),
            fraction=0.02,
            pad=0.01,
            label="x (log space)",
        )
    sizes = dataset.config.patch.training_sizes
    # `make_batch` was not given a size, so `size` above is the reference one;
    # the rest are cycled across batches and are not in this figure.
    also = (
        f"; also cycles {', '.join(str(s) for s in sizes[1:])}"
        if len(sizes) > 1 else ""
    )
    grid = x.shape[-1]
    crop = f", loss on the middle {size}x{size} (red)" if margin else ""
    fig.suptitle(
        f"training batch as the loader yields it: {grid}x{grid} grid{crop}, "
        f"pooled {dataset.config.patch.pool_factor}x, log space, "
        f"augmented{also}",
        fontsize=10,
    )
    return fig, _save(fig, out, "training_batch")


def plot_transform(dataset, n: int = 4, seed: int = 0, out: Path | None = None):
    """The chain from native flux to the training representation, plus the
    pixel-value histogram that says whether the transform is set up right.

    ``x`` is absolute log flux and there is one softening scale, so every band's
    sky sits at the same ``log(s * log 2)``.  The sky should pile up on that line
    with a spread near ``expected_sky_scatter``, and sources should sit clearly
    above it.
    """
    plt = _plt()
    from .data.transform import expected_sky_scatter

    rng = np.random.default_rng(seed)
    n = min(n, len(dataset))
    idx = np.sort(rng.choice(len(dataset), size=n, replace=False))
    native = dataset._native_stamps(idx)
    pooled = dataset._pool_stamps(native, rng=None, translate=False,
                                  scale_jitter=0.0)
    logged = dataset.transform.forward(pooled)
    noise = np.asarray(dataset.shards.meta["sky_noise"])[idx]

    fig = plt.figure(figsize=(10.5, 2.4 * n + 2.6))
    gs = fig.add_gridspec(n + 1, 3, height_ratios=[1] * n + [1.25])
    for k in range(n):
        sigma = noise[k] if np.isfinite(noise[k]) and noise[k] > 0 else 1.0
        for col, (img, title, cmap, asinh) in enumerate(
            (
                (
                    np.arcsinh(native[k] / sigma),
                    "native flux (asinh, $\\sigma$ units)",
                    "magma",
                    True,
                ),
                (np.arcsinh(pooled[k] / sigma), "pooled flux (asinh)", "magma", True),
                (logged[k], "x = log(s\u00b7softplus(f/s))", "viridis", False),
            )
        ):
            ax = fig.add_subplot(gs[k, col])
            if asinh:
                lo = _ASINH_VMIN
                hi = max(np.percentile(img, _ASINH_VMAX_PCT), _ASINH_VMAX_FLOOR)
            else:
                lo, hi = np.percentile(img, (0.5, 99.8))
            _show(ax, img, lo, hi, cmap=cmap)
            if k == 0:
                ax.set_title(title, fontsize=9)

    ax = fig.add_subplot(gs[n, :])
    allx = dataset.make_batch(
        np.sort(rng.choice(len(dataset), size=min(256, len(dataset)), replace=False)),
        rng=None,
        augment=False,
    )[:, 0].ravel()
    ax.hist(allx, bins=200, color="0.3")
    scatter = expected_sky_scatter(dataset.config.transform.softening_sigma)
    # One line: a single softening scale puts every band's sky at the same
    # log(s * log 2).  The predicted width is a typical one, since a band deeper
    # or shallower than the scale was measured from scatters proportionally
    # less or more about that shared level.
    sky = dataset.transform.sky_level
    ax.axvline(sky, color="crimson", lw=1.2,
               label=f"zero flux: x = log(s log 2) = {sky:.2f}")
    ax.axvspan(sky - scatter, sky + scatter, color="crimson", alpha=0.15,
               label=f"typical sky scatter $\\pm${scatter:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("x (log space)")
    ax.set_ylabel("pixels")
    ax.legend(fontsize=8)
    ax.set_title("pixel-value distribution: sky should sit in the red band", fontsize=9)
    fig.suptitle("transform chain: native flux -> pooled -> log space", fontsize=10)
    fig.tight_layout()
    return fig, _save(fig, out, "transform")


# -- the selected host population ------------------------------------------


def _hist(ax, values, bins, title, xlabel, log=False):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        ax.set_axis_off()
        ax.set_title(f"{title}\n(no data)", fontsize=8)
        return
    ax.hist(v, bins=bins, color="steelblue", edgecolor="none")
    if log:
        ax.set_xscale("log")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_ylabel("count", fontsize=8)


def plot_hosts(shards, hosts=None, band: str = "r", out: Path | None = None):
    """Summary of the population that was actually selected.

    ``hosts`` is the table written by extraction (``hosts.parquet``); without it
    the size/magnitude/ellipticity panels are skipped and only what the shards
    carry is shown.
    """
    plt = _plt()
    from .config import BANDS

    meta = shards.meta
    panels = []

    if hosts is not None and len(hosts):
        cols = set(getattr(hosts, "columns", getattr(hosts, "colnames", [])))
        # Size and shape both come from the multiband Sersic fit -- the same fit
        # the size cut is made on -- rather than from per-band adaptive moments,
        # which would be a second, PSF-convolved answer to the same question.
        if {"sersic_reff_major", "sersic_reff_minor"} <= cols:
            a = np.asarray(hosts["sersic_reff_major"], dtype=float)
            b = np.asarray(hosts["sersic_reff_minor"], dtype=float)
            panels.append(("host size", a, 40,
                           "Sersic half-light major axis (arcsec)", False))
            with np.errstate(invalid="ignore", divide="ignore"):
                q = np.where(a > 0, b / a, np.nan)
            # The distortion convention, e = (1-q^2)/(1+q^2), not the shear
            # convention (1-q)/(1+q); they differ by about a factor of two at
            # small ellipticity, enough to mislead if the axis is mislabelled.
            panels.append(("host distortion", (1 - q**2) / (1 + q**2), 40,
                           "$|e| = (1-q^2)/(1+q^2)$", False))
        fcol = f"{band}_cModelFlux"
        if fcol in cols:
            flux = np.asarray(hosts[fcol], dtype=float)
            with np.errstate(invalid="ignore", divide="ignore"):
                mag = -2.5 * np.log10(np.where(flux > 0, flux, np.nan)) + 31.4
            panels.append((f"host magnitude ({band})", mag, 40, f"{band} cModel mag", False))
        # The one panel that would have made the blob investigation
        # unnecessary: surface brightness against the sky it has to be seen
        # above.  A population piled up to the right of the line is runaway
        # fits, not galaxies.
        if {"sersic_reff_major", "sersic_reff_minor", f"{band}_cModelFlux"} <= cols:
            from rubin_host_prior.rubin.extract import host_mu_e

            panels.append(
                (
                    "host surface brightness",
                    host_mu_e(hosts, band),
                    40,
                    r"$\mu_e$ (mag/arcsec$^2$), sky $\approx$ 27",
                    False,
                )
            )
        if "sersic_index" in cols:
            # n ~ 1 is a disc, n ~ 4 an elliptical: the two kinds of host this
            # prior is meant to cover, so the balance between them matters.
            panels.append(
                (
                    "host Sersic index",
                    np.asarray(hosts["sersic_index"], dtype=float),
                    40,
                    "n (1 = exponential, 4 = de Vaucouleurs)",
                    False,
                )
            )
        bcol = f"{band}_blendedness"
        if bcol in cols:
            panels.append(
                (
                    "host blendedness",
                    np.asarray(hosts[bcol], dtype=float),
                    40,
                    "fraction of flux from neighbours",
                    False,
                )
            )

    panels.append(("local sky noise", meta["sky_noise"], 40, "nJy / native pixel", False))
    if "variance_step" in meta:
        # 1.0 is a uniform stamp; a tail above it is depth stepping across a
        # coadd cell edge, which is what the gate is set against.
        panels.append(
            (
                "variance step",
                np.asarray(meta["variance_step"], dtype=float),
                40,
                "max/min block variance floor",
                False,
            )
        )
    if "n_visits_min" in meta:
        # Exposure times are equal, so this is the depth of the shallowest cell
        # the stamp covers, straight from the coadd provenance.
        n_lo = np.asarray(meta["n_visits_min"], dtype=float)
        if np.any(n_lo > 0):
            panels.append(
                ("visits in shallowest cell", n_lo[n_lo > 0], 30, "distinct visits", False)
            )
    if "n_cells_spanned" in meta:
        # Each 150 px coadd cell has its own input visits, so depth and PSF step
        # at cell edges.  Anything above 1 means the stamp contains such a step.
        panels.append(
            (
                "coadd cells spanned",
                np.asarray(meta["n_cells_spanned"], dtype=float),
                20,
                "150 px cells per stamp",
                False,
            )
        )
    # DP2 covariates: recorded, never gated on.  Worth looking at, because if
    # INEXACT_PSF or REJECTED covers most of the accepted stamps then the PSF
    # the forward model relies on is approximate over most of the training set.
    for key, title, xlabel in (
        ("frac_no_data", "no-data fraction", "inf-variance pixels (DP2)"),
        ("frac_inexact_psf", "INEXACT_PSF fraction", "fraction of stamp"),
        ("frac_rejected", "REJECTED fraction", "fraction of stamp"),
    ):
        if key in meta:
            panels.append((title, meta[key], 40, xlabel, False))

    n_panels = len(panels) + 1  # + the band bar chart
    rows, cols = _grid(n_panels)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.7 * rows))
    flat = np.atleast_1d(axes).ravel()

    counts = shards.band_counts()
    flat[0].bar(list(BANDS), [counts[b] for b in BANDS], color="darkseagreen")
    flat[0].set_title("patches per band", fontsize=9)
    flat[0].tick_params(labelsize=7)
    flat[0].set_ylabel("count", fontsize=8)

    for ax, (title, values, bins, xlabel, logx) in zip(flat[1:], panels):
        _hist(ax, values, bins, title, xlabel, logx)
    for ax in flat[n_panels:]:
        ax.set_axis_off()
    fig.suptitle(
        f"selected host population: {len(shards)} patches"
        + (f", {len(hosts)} hosts" if hosts is not None else ""),
        fontsize=11,
    )
    fig.tight_layout()
    return fig, _save(fig, out, "hosts")


# -- the gate ---------------------------------------------------------------


def plot_rejections(manifest, band: str = "r", out: Path | None = None):
    """Why patches were thrown away, and whether the gate is biased.

    The right-hand panel is the one to look at.  If the rejected patches are
    systematically brighter or denser than the accepted ones, the training set
    is skewed against exactly the regime this project exists to model, and the
    tolerances need loosening.
    """
    plt = _plt()
    status = np.asarray(manifest["status"] if "status" in manifest else [])
    reasons = np.asarray(manifest["reasons"] if "reasons" in manifest else [""] * len(status))
    accepted = status == "accepted"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    kinds: dict[str, int] = {}
    for r in reasons[~accepted]:
        for part in str(r).split(";"):
            key = part.split(":")[0]
            if key and key != "nan":
                kinds[key] = kinds.get(key, 0) + 1
    if kinds:
        labels, values = zip(*sorted(kinds.items(), key=lambda kv: kv[1]))
        axes[0].barh(list(labels), list(values), color="indianred")
        axes[0].set_xlabel("patches rejected")
        axes[0].tick_params(labelsize=8)
    else:
        axes[0].text(0.5, 0.5, "nothing rejected", ha="center", va="center")
        axes[0].set_axis_off()
    axes[0].set_title(f"rejection reasons ({int((~accepted).sum())} of {len(status)})", fontsize=10)

    key = "diag_sky_noise"
    if key in manifest:
        v = np.asarray(manifest[key], dtype=float)
        bins = np.histogram_bin_edges(v[np.isfinite(v)], bins=40)
        axes[1].hist(
            v[accepted & np.isfinite(v)], bins=bins, alpha=0.65, label="accepted", color="steelblue"
        )
        axes[1].hist(
            v[~accepted & np.isfinite(v)],
            bins=bins,
            alpha=0.65,
            label="rejected",
            color="indianred",
        )
        axes[1].set_xlabel("sky noise at the patch (nJy)")
        axes[1].set_ylabel("count")
        axes[1].legend(fontsize=8)
        axes[1].set_title("accepted vs rejected: look for a systematic offset", fontsize=10)
    else:
        axes[1].set_axis_off()
    fig.tight_layout()
    return fig, _save(fig, out, "rejections")


# -- everything ------------------------------------------------------------


def make_all(
    shards,
    dataset=None,
    hosts=None,
    manifest=None,
    out_dir: Path | str = "diagnostics",
    band: str = "r",
    n_cutouts: int = 100,
    seed: int = 0,
) -> list[Path]:
    """Write every figure that the available inputs support."""
    plt = _plt()
    out_dir = Path(out_dir)
    written: list[Path] = []
    made = [
        plot_cutouts(shards, n=n_cutouts, seed=seed, out=out_dir),
        plot_hosts(shards, hosts, band=band, out=out_dir),
    ]
    if dataset is not None:
        made.append(plot_training_batch(dataset, n=n_cutouts, seed=seed, out=out_dir))
        made.append(plot_transform(dataset, seed=seed, out=out_dir))
    if manifest is not None and len(manifest):
        made.append(plot_rejections(manifest, band=band, out=out_dir))
    for fig, path in made:
        if path is not None:
            written.append(path)
        plt.close(fig)
    return written
