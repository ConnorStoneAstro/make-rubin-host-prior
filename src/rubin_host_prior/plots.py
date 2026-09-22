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
                   sky noise, neighbour distances.

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
    return ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                     interpolation="nearest")


def _save(fig, out: Path | None, name: str) -> Path | None:
    if out is None:
        return None
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    return path


# -- raw cutouts -----------------------------------------------------------


def plot_cutouts(shards, n: int = 25, seed: int = 0, out: Path | None = None):
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


# -- what the loader yields ------------------------------------------------


def plot_training_batch(dataset, n: int = 25, seed: int = 0, out: Path | None = None):
    """Grid of exactly what the network receives: pooled, log-space, augmented.

    A shared colour scale across panels, so the spread between patches is
    visible rather than normalised away -- the prior has to cover that spread.
    """
    plt = _plt()
    rng = np.random.default_rng(seed)
    n = min(n, len(dataset))
    idx = np.sort(rng.choice(len(dataset), size=n, replace=False))
    x = dataset.make_batch(idx, rng=rng, augment=True)[:, 0]
    lo, hi = np.percentile(x, (0.5, 99.5))

    rows, cols = _grid(n)
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.2 * rows))
    im = None
    for k, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.set_axis_off()
        if k < n:
            im = _show(ax, x[k], lo, hi, cmap="viridis")
    if im is not None:
        fig.colorbar(im, ax=np.atleast_1d(axes).ravel().tolist(),
                     fraction=0.02, pad=0.01, label="x (log space)")
    size = x.shape[-1]
    sizes = dataset.config.patch.training_sizes
    also = (f"; also cycles {', '.join(str(s) for s in sizes if s != size)}"
            if len(sizes) > 1 else "")
    fig.suptitle(
        f"training batch as the loader yields it: {size}x{size}, pooled "
        f"{dataset.config.patch.pool_factor}x, log space, augmented{also}",
        fontsize=10,
    )
    return fig, _save(fig, out, "training_batch")


def plot_transform(dataset, n: int = 4, seed: int = 0, out: Path | None = None):
    """The chain from native flux to the training representation, plus the
    pixel-value histogram that says whether the transform is set up right.

    Sky should pile up at ``x = log(log 2) ~ -0.37`` with a spread near
    ``expected_sky_scatter``; sources should sit clearly above it.
    """
    plt = _plt()
    from .data.transform import expected_sky_scatter

    rng = np.random.default_rng(seed)
    n = min(n, len(dataset))
    idx = np.sort(rng.choice(len(dataset), size=n, replace=False))
    native = dataset._native_stamps(idx)
    pooled = dataset._pool(idx, rng=None, translate=False, scale_jitter=0.0)
    logged = dataset.transform.forward(pooled, dataset.band_idx[idx])
    noise = np.asarray(dataset.shards.meta["sky_noise"])[idx]

    fig = plt.figure(figsize=(10.5, 2.4 * n + 2.6))
    gs = fig.add_gridspec(n + 1, 3, height_ratios=[1] * n + [1.25])
    for k in range(n):
        sigma = noise[k] if np.isfinite(noise[k]) and noise[k] > 0 else 1.0
        for col, (img, title, cmap, asinh) in enumerate((
            (np.arcsinh(native[k] / sigma), "native flux (asinh, $\\sigma$ units)", "magma", True),
            (np.arcsinh(pooled[k] / sigma), "pooled flux (asinh)", "magma", True),
            (logged[k], "x = log(softplus(f/s))", "viridis", False),
        )):
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
        rng=None, augment=False,
    )[:, 0].ravel()
    ax.hist(allx, bins=200, color="0.3")
    sky = dataset.transform.sky_level
    ss = dataset.config.transform.softening_sigma
    ax.axvline(sky, color="crimson", lw=1.2,
               label=f"zero flux: x = log(log 2) = {sky:.2f}")
    scatter = expected_sky_scatter(ss, dataset.config.transform.log_scale)
    ax.axvspan(sky - scatter, sky + scatter, color="crimson", alpha=0.15,
               label=f"predicted sky scatter $\\pm${scatter:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("x (log space)")
    ax.set_ylabel("pixels")
    ax.legend(fontsize=8)
    ax.set_title("pixel-value distribution: sky should sit in the red band",
                 fontsize=9)
    fig.suptitle("transform chain: native flux -> pooled -> log space", fontsize=10)
    fig.tight_layout()
    return fig, _save(fig, out, "transform")


# -- the selected host population ------------------------------------------


def _distortion(ixx, iyy, ixy):
    """``|e|`` from second moments, the distortion convention.

    ``e1 = (Ixx - Iyy)/(Ixx + Iyy)``, ``e2 = 2 Ixy/(Ixx + Iyy)``, so for an
    ellipse of axis ratio ``q`` this is ``(1 - q^2)/(1 + q^2)`` -- *not* the
    ``(1 - q)/(1 + q)`` shear convention.  They differ by roughly a factor of two
    at small ellipticity, which is enough to mislead if the axis is mislabelled.
    """
    t = ixx + iyy
    with np.errstate(invalid="ignore", divide="ignore"):
        e1, e2 = (ixx - iyy) / t, 2.0 * ixy / t
    return np.hypot(e1, e2)


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

    from rubin_host_prior.rubin.extract import host_half_light_arcsec

    meta = shards.meta
    panels = []

    if hosts is not None and len(hosts):
        cols = set(getattr(hosts, "columns", getattr(hosts, "colnames", [])))
        # DP2 second moments are per band -- there is no band-independent
        # shape_xx -- and only ugri carry them.
        mom = [f"{band}_ixx", f"{band}_iyy", f"{band}_ixy"]
        scale = float(np.nanmedian(meta["pixel_scale"])) if len(meta["pixel_scale"]) else 0.2
        if set(mom) <= cols:
            ixx, iyy, ixy = (np.asarray(hosts[c], dtype=float) for c in mom)
            panels.append(("host distortion", _distortion(ixx, iyy, ixy), 40,
                           "$|e| = (1-q^2)/(1+q^2)$", False))
            # Prefer the half-light radius, which DP2 gives directly in arcsec
            # and which the size cut is made on; fall back to the moments trace
            # converted with the pixel scale.
            size = host_half_light_arcsec(hosts)
            label = "Sersic half-light major axis (arcsec)"
            panels.append(("host size", size, 40, label, False))
        fcol = f"{band}_cModelFlux"
        if fcol in cols:
            flux = np.asarray(hosts[fcol], dtype=float)
            with np.errstate(invalid="ignore", divide="ignore"):
                mag = -2.5 * np.log10(np.where(flux > 0, flux, np.nan)) + 31.4
            panels.append((f"host magnitude ({band})", mag, 40,
                           f"{band} cModel mag", False))
        # The one panel that would have made the blob investigation
        # unnecessary: surface brightness against the sky it has to be seen
        # above.  A population piled up to the right of the line is runaway
        # fits, not galaxies.
        if {"sersic_reff_major", "sersic_reff_minor", f"{band}_cModelFlux"} <= cols:
            from rubin_host_prior.rubin.extract import host_mu_e

            panels.append(("host surface brightness", host_mu_e(hosts, band), 40,
                           r"$\mu_e$ (mag/arcsec$^2$), sky $\approx$ 27", False))
        if "sersic_index" in cols:
            # n ~ 1 is a disc, n ~ 4 an elliptical: the two kinds of host this
            # prior is meant to cover, so the balance between them matters.
            panels.append(("host Sersic index",
                           np.asarray(hosts["sersic_index"], dtype=float), 40,
                           "n (1 = exponential, 4 = de Vaucouleurs)", False))
        bcol = f"{band}_blendedness"
        if bcol in cols:
            panels.append(("host blendedness", np.asarray(hosts[bcol], dtype=float),
                           40, "fraction of flux from neighbours", False))

    panels.append(("local sky noise", meta["sky_noise"], 40, "nJy / native pixel", False))
    if "variance_step" in meta:
        # 1.0 is a uniform stamp; a tail above it is depth stepping across a
        # coadd cell edge, which is what the gate is set against.
        panels.append(("variance step", np.asarray(meta["variance_step"], dtype=float),
                       40, "max/min block variance floor", False))
    panels.append(("nearest galaxy", meta["nearest_galaxy_arcsec"], 40,
                   "arcsec", False))
    panels.append(("nearest star", meta["nearest_star_arcsec"], 40, "arcsec", False))
    panels.append(("neighbours in frame", np.asarray(meta["n_neighbours"], dtype=float),
                   30, "count within search radius", False))
    if "n_visits_min" in meta:
        # Exposure times are equal, so this is the depth of the shallowest cell
        # the stamp covers, straight from the coadd provenance.
        n_lo = np.asarray(meta["n_visits_min"], dtype=float)
        if np.any(n_lo > 0):
            panels.append(("visits in shallowest cell", n_lo[n_lo > 0], 30,
                           "distinct visits", False))
    if "n_cells_spanned" in meta:
        # Each 150 px coadd cell has its own input visits, so depth and PSF step
        # at cell edges.  Anything above 1 means the stamp contains such a step.
        panels.append(("coadd cells spanned", np.asarray(meta["n_cells_spanned"],
                                                         dtype=float),
                       20, "150 px cells per stamp", False))
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

    band_idx = np.asarray(meta["band_idx"])
    counts = [int(np.sum(band_idx == i)) for i in range(len(BANDS))]
    flat[0].bar(list(BANDS), counts, color="darkseagreen")
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
    reasons = np.asarray(
        manifest["reasons"] if "reasons" in manifest else [""] * len(status)
    )
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
    axes[0].set_title(
        f"rejection reasons ({int((~accepted).sum())} of {len(status)})", fontsize=10
    )

    key = "diag_sky_noise"
    if key in manifest:
        v = np.asarray(manifest[key], dtype=float)
        bins = np.histogram_bin_edges(v[np.isfinite(v)], bins=40)
        axes[1].hist(v[accepted & np.isfinite(v)], bins=bins, alpha=0.65,
                     label="accepted", color="steelblue")
        axes[1].hist(v[~accepted & np.isfinite(v)], bins=bins, alpha=0.65,
                     label="rejected", color="indianred")
        axes[1].set_xlabel("sky noise at the patch (nJy)")
        axes[1].set_ylabel("count")
        axes[1].legend(fontsize=8)
        axes[1].set_title("accepted vs rejected: look for a systematic offset",
                          fontsize=10)
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
    n_cutouts: int = 25,
    seed: int = 0,
) -> list[Path]:
    """Write every figure that the available inputs support."""
    plt = _plt()
    out_dir = Path(out_dir)
    written: list[Path] = []
    made = [plot_cutouts(shards, n=n_cutouts, seed=seed, out=out_dir),
            plot_hosts(shards, hosts, band=band, out=out_dir)]
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
