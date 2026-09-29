"""Dataset diagnostics.

The one that matters for this project is the **correlation length**, xi: how far
apart two pixels must be before they stop being related.  In these images it is
set by the size distribution of the hosts.

It is the number that decides how much context the model needs.  The loss crop
(``2R``) is fixed by the architecture and is exactly the right margin for
*training*.  But how much larger than your region of interest a scene must be
before its middle is trustworthy is governed by xi, not by R -- the edge's
influence spreads as far as pixels remain correlated.  Empirically a margin of
roughly 1.5-3 x xi is enough; below ~1 x xi the middle of a scene is noticeably
wrong.  So compare ``2R`` against xi and make sure you are on the right side of
that.
"""

from __future__ import annotations

import numpy as np


def _radial_profile(power_sum: np.ndarray, n_patches: int, size: int,
                    max_lag: int | None) -> np.ndarray:
    """Radial profile of the unbiased autocorrelation, from summed |FFT|^2."""
    pad = 2 * size
    ac = np.fft.irfft2(power_sum / max(n_patches, 1), s=(pad, pad))
    ones = np.fft.rfft2(np.ones((size, size)), s=(pad, pad))
    count = np.fft.irfft2(ones * np.conj(ones), s=(pad, pad))
    ac = np.fft.fftshift(ac / np.maximum(count, 1.0))

    c = pad // 2
    lim = max_lag if max_lag is not None else size - 1
    yy, xx = np.mgrid[0:pad, 0:pad]
    rr = np.hypot(yy - c, xx - c)
    prof = np.empty(lim + 1)
    for i in range(lim + 1):
        sel = (rr < 0.5) if i == 0 else ((rr >= i - 0.5) & (rr < i + 0.5))
        prof[i] = ac[sel].mean()
    return prof / prof[0]


def _xi_from_profile(prof: np.ndarray, exclude_noise: bool) -> tuple[float, bool]:
    start = 1 if exclude_noise else 0
    p = prof[start:] / prof[start]
    lags = np.arange(start, start + len(p), dtype=float)
    target = float(np.exp(-1.0))
    below = np.where(p < target)[0]
    if len(below) == 0:
        return float(lags[-1]), True  # never decorrelates inside the patch
    j = int(below[0])
    if j == 0:
        return float(lags[0]), False
    p0, p1 = p[j - 1], p[j]
    frac = (p0 - target) / max(p0 - p1, 1e-12)
    return float(lags[j - 1] + frac), False


class AutocorrelationAccumulator:
    """Streaming autocorrelation, so xi can be measured over every patch.

    Only the summed power spectrum is retained, so memory is one array of the
    padded patch size regardless of how many patches are added -- which is what
    lets the extraction script measure xi over the whole run instead of over a
    subsample it had to keep in memory.
    """

    def __init__(self, size: int):
        self.size = int(size)
        self.pad = 2 * self.size
        self._power = np.zeros((self.pad, self.pad // 2 + 1))
        self.n = 0

    def add(self, patch: np.ndarray) -> None:
        a = np.asarray(patch, dtype=np.float64)
        if a.shape != (self.size, self.size):
            raise ValueError(
                f"expected {(self.size, self.size)} patches, got {a.shape}"
            )
        if not np.all(np.isfinite(a)):
            return
        f = np.fft.rfft2(a - a.mean(), s=(self.pad, self.pad))
        self._power += np.abs(f) ** 2
        self.n += 1

    def result(self, max_lag: int | None = None, exclude_noise: bool = True) -> dict:
        if self.n == 0:
            return {"xi": float("nan"), "noise_fraction": float("nan"),
                    "profile": [], "truncated": False, "n_patches": 0}
        prof = _radial_profile(self._power, self.n, self.size, max_lag)
        xi, truncated = _xi_from_profile(prof, exclude_noise)
        return {
            "xi": xi,
            "noise_fraction": float(1.0 - prof[1]) if len(prof) > 1 else float("nan"),
            "profile": prof.tolist(),
            "truncated": truncated,
            "n_patches": self.n,
        }


def autocorrelation(patches: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Radially averaged, unbiased autocorrelation profile of a stack of patches.

    Zero-padded to twice the patch size and divided by the number of overlapping
    pixels at each lag, so this is the *linear* autocorrelation.  The circular
    (unpadded) version wraps structure around the edges and biases the estimate
    towards shorter correlation lengths, which on this question would be exactly
    the wrong way to be wrong.
    """
    x = np.asarray(patches, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"expected (N, H, W) patches, got shape {x.shape}")
    n, h, w = x.shape
    if h != w:
        raise ValueError(f"expected square patches, got {(h, w)}")
    x = x - x.mean(axis=(1, 2), keepdims=True)
    f = np.fft.rfft2(x, s=(2 * h, 2 * h))
    return _radial_profile((np.abs(f) ** 2).sum(axis=0), n, h, max_lag)


def correlation_length(
    patches: np.ndarray,
    max_lag: int | None = None,
    exclude_noise: bool = True,
) -> dict:
    """Correlation length of a stack of patches, in pixels.

    ``exclude_noise`` renormalises the profile at lag 1 before looking for the
    1/e crossing.  This is essential, not cosmetic: uncorrelated pixel noise
    contributes a delta function at zero lag and nothing elsewhere, so on real
    data lag 0 sits far above lag 1 and a naive 1/e crossing returns xi ~ 1
    regardless of how big the galaxies are -- measured here, it was wrong by a
    factor of 4.  Measuring from lag 1 gives the *structural* correlation
    length, which is what the context question is about.

    Valid because exact block-mean pooling of independent pixels leaves the noise
    independent between output pixels, so lag 1 is noise-free.  With
    ``scale_jitter`` on, resampling correlates neighbours slightly and lag 1
    picks up a little noise, biasing xi marginally high.

    Returns ``xi``, the profile, and ``noise_fraction`` -- the share of the
    variance sitting in the zero-lag delta, i.e. how noise-dominated the patches
    are.
    """
    prof = autocorrelation(patches, max_lag)
    xi, truncated = _xi_from_profile(prof, exclude_noise)
    return {
        "xi": xi,
        "noise_fraction": float(1.0 - prof[1]) if len(prof) > 1 else float("nan"),
        "profile": prof.tolist(),
        "truncated": truncated,
        "n_patches": int(np.shape(patches)[0]),
    }


def reach_advice(xi: float, reach: int) -> str:
    """One line on whether the architecture's score reach covers this field.

    It used to compare ``xi`` against the *loss crop*, because under valid
    convolutions the crop and the reach were the same number (``2R``) and the
    crop was the one that cost patches.  With same-mode convolutions the crop is
    0 and the reach is still ``2R``, so the question is the honest one: can a
    pixel's score see as far as the data is correlated?
    """
    if not np.isfinite(xi) or xi <= 0:
        return "correlation length unavailable"
    ratio = reach / xi
    if ratio >= 2.0:
        verdict = "comfortable"
    elif ratio >= 1.2:
        verdict = "adequate"
    elif ratio >= 0.8:
        verdict = "marginal -- consider more reach"
    else:
        verdict = "TOO SMALL -- the score cannot see as far as the data correlates"
    return (
        f"xi = {xi:.1f} px, score reach 2R = {reach} px, "
        f"2R/xi = {ratio:.1f}  ->  {verdict}"
    )
