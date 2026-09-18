"""Synthetic shards in DP1-like units, for testing the chain without NERSC.

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

# Subset of the DP1 r29.2.0 mask plane bits (see the quality-cuts reference).
MASK_PLANES = {
    "BAD": 0,
    "SAT": 1,
    "INTRP": 2,
    "CR": 3,
    "EDGE": 4,
    "DETECTED": 5,
    "SUSPECT": 7,
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
    dark_halo_sigma: float = 0.0,
) -> dict:
    """One background-subtracted patch in nJy, with its variance, mask and PSF.

    ``dark_halo_sigma`` adds a smooth negative bowl of that depth, in units of
    the *native* sky noise, imitating DP1 background over-subtraction around a
    bright source.  Worth exercising because it is the deep-negative regime: a
    smooth offset does not average down under pooling while the noise does, so a
    bowl ``D`` sigma deep natively is ``D * pool_factor`` sigma deep in the
    pooled data.  The softplus transform carries it through without a floor,
    mapping it to a correspondingly negative ``x``.
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
    if dark_halo_sigma:
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
        rr = ((xx - cx) ** 2 + (yy - cy) ** 2) / (size / 2.0) ** 2
        image = image - dark_halo_sigma * sky_noise * np.exp(-rr)
    image = image + rng.normal(0.0, np.sqrt(variance))
    mask = np.zeros((size, size), dtype=np.uint32)
    mask[image > 5 * sky_noise] |= 1 << MASK_PLANES["DETECTED"]
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
    psf_size: int = 25,
    patches_per_shard: int = 128,
    sky_noise_by_band: dict[str, float] | None = None,
    dark_halo_sigma: float = 0.0,
    seed: int = 0,
) -> list[Path]:
    """Write shards that look like the real thing to the loader."""
    rng = np.random.default_rng(seed)
    # Rough DP1-ish per-band sky noise in nJy per native pixel; u is the worst.
    noise = sky_noise_by_band or {
        "u": 28.0, "g": 11.0, "r": 12.0, "i": 16.0, "z": 24.0, "y": 40.0
    }
    with ShardWriter(
        out_dir,
        native_size=native_size,
        psf_size=psf_size,
        mask_plane_dict=MASK_PLANES,
        prefix="synthetic",
        patches_per_shard=patches_per_shard,
        dataset_type="synthetic",
        attrs={"synthetic": 1, "sky_noise_by_band": noise},
    ) as w:
        for i in range(n_patches):
            band_idx = int(rng.integers(len(BANDS)))
            band = BANDS[band_idx]
            p = synthetic_patch(
                rng, size=native_size, psf_size=psf_size, sky_noise=noise[band],
                dark_halo_sigma=dark_halo_sigma,
            )
            w.add(
                p["image"],
                p["variance"],
                p["mask"],
                p["psf"],
                meta={
                    "band_idx": band_idx,
                    "visit": 2024110800000 + i,
                    "detector": int(rng.integers(9)),
                    "x0": 0,
                    "y0": 0,
                    "center_x": p["center"][0],
                    "center_y": p["center"][1],
                    "ra": 53.13 + float(rng.normal(0, 0.1)),
                    "dec": -28.10 + float(rng.normal(0, 0.1)),
                    "mjd": 60600.0 + float(rng.uniform(0, 60)),
                    "psf_sigma": p["psf_sigma"],
                    "pixel_scale": 0.2003,
                    "sky_noise": noise[band],
                    "host_id": i,
                    "host_offset_arcsec": float(rng.uniform(0, 3)),
                },
            )
        return w.paths
