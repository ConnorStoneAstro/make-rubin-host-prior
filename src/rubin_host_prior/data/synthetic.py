"""Synthetic shards in DP2-like form, for testing the chain without NERSC.

Not a simulation of Rubin -- just enough structure (a Sersic host, neighbours,
point sources, a Gaussian PSF, background-subtracted Gaussian sky noise in nJy)
to exercise the loader, the transform and the training loop end to end on a
laptop, and to confirm that a trained prior does something sensible before the
real data is involved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import BANDS
from .shards import ShardWriter

#: DP2 plane names, with the locally assigned bits that extraction would record.
#: DP2 bit numbers are dynamic, so the mapping travels with the shard rather than
#: being hard-coded anywhere; these are just the values the fixture uses.
MASK_PLANES = {
    "SATURATED": 0,
    "COSMIC_RAY": 1,
    "INTERPOLATED": 2,
    "DETECTION_EDGE": 3,
    "DETECTED": 4,
    "INEXACT_PSF": 5,
    "REJECTED": 6,
}


def _sersic(size: int, flux: float, r_e: float, n: float, q: float, pa: float,
            x0: float, y0: float) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    dx, dy = x - x0, y - y0
    c, s = np.cos(pa), np.sin(pa)
    xr, yr = c * dx + s * dy, -s * dx + c * dy
    r = np.sqrt(xr**2 + (yr / q) ** 2)
    b_n = 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n)  # Ciotti & Bertin
    prof = np.exp(-b_n * ((r / r_e) ** (1.0 / n) - 1.0))
    return flux * prof / prof.sum()


def _gaussian_psf(size: int, sigma: float) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    c = (size - 1) / 2.0
    k = np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * sigma**2))
    return k / k.sum()


def _convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    s = image.shape[0] + kernel.shape[0] - 1
    f = np.fft.rfft2(image, (s, s)) * np.fft.rfft2(kernel, (s, s))
    full = np.fft.irfft2(f, (s, s))
    off = kernel.shape[0] // 2
    return full[off : off + image.shape[0], off : off + image.shape[1]]


def synthetic_patch(
    rng: np.random.Generator,
    size: int = 224,
    psf_size: int = 25,
    sky_noise: float = 12.0,
    psf_sigma: float | None = None,
    noise_correlation: float = 0.8,
    no_data_fraction: float = 0.0005,
) -> dict:
    """One background-subtracted patch in nJy, with its variance, mask and PSF.

    ``noise_correlation`` is the Gaussian width, in pixels, of the correlation
    imposed on the pixel noise -- coadds are built by warping input exposures
    onto the skymap grid, which makes neighbouring pixels share flux.  The
    per-pixel variance still matches the variance plane; only the correlation
    between pixels changes.  This matters because it breaks the assumption that
    averaging ``P**2`` pixels divides the noise by ``P``: it divides it by less.
    Set 0 for uncorrelated noise.

    ``no_data_fraction`` puts ``inf`` into that fraction of the variance plane,
    at the brightest pixels, imitating DP2's convention for "no contributing
    exposures" -- which is how the cores of saturated stars appear.  Code that
    treats non-finite variance as corruption rather than measuring it would
    reject every stamp containing a bright neighbour.
    """
    psf_sigma = psf_sigma if psf_sigma is not None else float(rng.uniform(1.4, 2.4))
    psf = _gaussian_psf(psf_size, psf_sigma)
    truth = np.zeros((size, size))

    # Host, deliberately not centred: a prior trained on centred galaxies learns
    # that galaxies are always centred.
    cx = size / 2 + rng.normal(0, size * 0.04)
    cy = size / 2 + rng.normal(0, size * 0.04)
    truth += _sersic(
        size,
        flux=float(10 ** rng.uniform(3.5, 5.5)),
        r_e=float(rng.uniform(4, 25)),
        n=float(rng.uniform(0.7, 4.0)),
        q=float(rng.uniform(0.3, 1.0)),
        pa=float(rng.uniform(0, np.pi)),
        x0=cx,
        y0=cy,
    )
    for _ in range(rng.integers(0, 4)):  # neighbour galaxies
        truth += _sersic(
            size,
            flux=float(10 ** rng.uniform(2.5, 4.0)),
            r_e=float(rng.uniform(2, 10)),
            n=float(rng.uniform(0.7, 4.0)),
            q=float(rng.uniform(0.3, 1.0)),
            pa=float(rng.uniform(0, np.pi)),
            x0=float(rng.uniform(0, size)),
            y0=float(rng.uniform(0, size)),
        )
    for _ in range(rng.integers(0, 6)):  # static point sources
        px, py = int(rng.integers(0, size)), int(rng.integers(0, size))
        truth[py, px] += float(10 ** rng.uniform(2.0, 4.5))

    image = _convolve(truth, psf)
    variance = sky_noise**2 + np.maximum(image, 0.0)  # sky + shot noise
    noise = rng.normal(0.0, 1.0, (size, size))
    if noise_correlation > 0:
        k = _gaussian_psf(min(size, 4 * int(noise_correlation) + 5), noise_correlation)
        noise = _convolve(noise, k)
        noise /= noise.std()  # keep per-pixel variance; change only correlation
    image = image + noise * np.sqrt(variance)
    mask = np.zeros((size, size), dtype=np.uint32)
    mask[image > 5 * sky_noise] |= 1 << MASK_PLANES["DETECTED"]
    # DP2 marks "no data" with inf variance, not with a plane.  Placed as a
    # saturated blob at a random position, i.e. a bright *star* -- not on the
    # host, which at r = 20-25 is nowhere near saturation.  It lands in the
    # protected central region often enough to exercise that rejection.
    if no_data_fraction > 0:
        n_bad = no_data_fraction * size * size
        radius = np.sqrt(max(n_bad, 1.0) / np.pi)
        sy, sx = rng.uniform(0, size, 2)
        yy, xx = np.mgrid[0:size, 0:size]
        blob = ((xx - sx) ** 2 + (yy - sy) ** 2) <= radius**2
        variance[blob] = np.inf
        mask[blob] |= 1 << MASK_PLANES["SATURATED"]
    # INEXACT_PSF covers a large fraction of a real DP2 coadd; recorded, not gated.
    mask[: int(0.3 * size)] |= 1 << MASK_PLANES["INEXACT_PSF"]
    return {
        "image": image.astype(np.float32),
        "variance": variance.astype(np.float32),
        "mask": mask,
        "psf": psf.astype(np.float32),
        "psf_sigma": psf_sigma,
        "center": (cx, cy),
    }


def write_synthetic_shards(
    out_dir: str | Path,
    n_patches: int = 256,
    native_size: int = 224,
    patches_per_shard: int = 128,
    sky_noise_by_band: dict[str, float] | None = None,
    noise_correlation: float = 0.8,
    no_data_fraction: float = 0.0005,
    seed: int = 0,
) -> list[Path]:
    """Write shards that look like the real thing to the loader."""
    rng = np.random.default_rng(seed)
    # Rough per-band sky noise in nJy per native pixel; u is the worst.
    noise = sky_noise_by_band or {
        "u": 28.0, "g": 11.0, "r": 12.0, "i": 16.0, "z": 24.0, "y": 40.0
    }
    hosts = []
    with ShardWriter(
        out_dir,
        native_size=native_size,
        prefix="synthetic",
        patches_per_shard=patches_per_shard,
        dataset_type="deep_coadd",
        attrs={"synthetic": 1, "sky_noise_by_band": noise,
               "release": "DP2",
               "correlated_noise": int(noise_correlation > 0),
               "background_restored": 0},
    ) as w:
        for i in range(n_patches):
            band_idx = int(rng.integers(len(BANDS)))
            band = BANDS[band_idx]
            p = synthetic_patch(
                rng, size=native_size, sky_noise=noise[band],
                noise_correlation=noise_correlation,
                no_data_fraction=no_data_fraction,
            )
            hosts.append(
                {
                    "objectId": i,
                    "coord_ra": 53.13 + float(rng.normal(0, 0.1)),
                    "coord_dec": -28.10 + float(rng.normal(0, 0.1)),
                    "r_ixx": float(10 ** rng.uniform(0.8, 2.2)),
                    "r_iyy": float(10 ** rng.uniform(0.8, 2.2)),
                    "r_ixy": float(rng.normal(0, 5)),
                    "sersic_reff_major": float(10 ** rng.uniform(0.3, 1.2)),
                    "sersic_reff_minor": float(10 ** rng.uniform(0.1, 1.0)),
                    "sersic_index": float(rng.uniform(0.5, 6.0)),
                    "refExtendedness": 1.0,
                    "refBand": "r",
                    "r_cModelFlux": float(10 ** rng.uniform(2.6, 4.6)),
                    "r_blendedness": float(rng.beta(1.2, 8)),
                }
            )
            w.add(
                p["image"],
                meta={
                    "band_idx": band_idx,
                    "x0": 0,
                    "y0": 0,
                    "ra": 53.13 + float(rng.normal(0, 0.1)),
                    "dec": -28.10 + float(rng.normal(0, 0.1)),
                    "pixel_scale": 0.2003,
                    "sky_noise": noise[band],
                    "host_id": i,
                    "host_offset_arcsec": float(rng.uniform(0, 3)),
                    "tract": 5063,
                    "patch": int(rng.integers(100)),
                    "n_neighbours": int(rng.poisson(3)),
                    "neighbour_flux_max": float(10 ** rng.uniform(2, 4.5)),
                    "nearest_galaxy_arcsec": float(rng.uniform(2, 30)),
                    "nearest_star_arcsec": float(rng.uniform(2, 30)),
                    # A 416 px stamp spans ~3x3 of the 150 px coadd cells.
                    "n_cells_spanned": int(max(1, round((native_size / 150) ** 2))),
                    "n_visits_min": int(rng.integers(8, 32)),
                    "n_visits_max": int(rng.integers(32, 40)),
                    "cell_depth_ratio": float(rng.uniform(1.0, 1.4)),
                    "variance_step": float(rng.uniform(1.0, 1.4)),
                    "frac_no_data": float(no_data_fraction),
                    "frac_inexact_psf": 0.3,
                    "frac_rejected": float(rng.uniform(0, 0.4)),
                },
            )
    _write_synthetic_hosts(Path(out_dir) / "hosts.parquet", hosts, rng)
    return w.paths


def _write_synthetic_hosts(path: Path, rows: list[dict], rng) -> None:
    """Mirror the ``hosts.parquet`` that extraction writes, so the diagnostic
    plots can be exercised end to end without the cluster."""
    import pandas as pd

    pd.DataFrame(rows).to_parquet(path, index=False)
