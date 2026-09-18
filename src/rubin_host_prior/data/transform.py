"""Flux <-> log-space pixel transform.

    x = log1p(f / b_band) / c          f = b_band * expm1(c * x)

``b_band`` is a per-band soft offset in nJy, nominally ``k_sigma`` times the
*pooled* sky noise of that band.  Three things follow, and they are the whole
reason for this choice:

* **No hard floor, no point mass.**  DP1 ``visit_image`` pixels are
  background-subtracted, so about half of all sky pixels are negative.
  ``log(max(f, floor))`` would pile 40-50% of every patch onto a single value;
  ``log1p(f / b)`` is smooth and strictly monotonic through zero instead.
* **Band-agnostic.**  Dividing by ``b_band`` before the log puts every band's sky
  level at ``x ~ 0`` with scatter ``~ 1 / k_sigma``, so u-band and y-band patches
  land in the same place and one prior can cover all six.
* **Linear where it matters, logarithmic where it must be.**  Near the noise
  floor ``x ~ f / (b * c)`` -- a pure rescaling, so additive Gaussian pixel noise
  stays additive and Gaussian.  In the bright regime it is logarithmic, which is
  what tames the ~1e4 dynamic range of a galaxy core.

The clip at ``floor_ratio`` is a numerical guard for artefacts (over-subtracted
haloes, dipoles), not a modelling choice: at ``k_sigma = 5`` a legitimate sky
pixel reaches ``f / b = -0.9`` only at 9.5 sigma.  ``forward`` reports how often
it fires so you can see if a shard is contaminated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import BANDS, TransformConfig


def _xp(a):
    """numpy or jax.numpy, whichever matches the input, so one implementation
    serves both the data loader and a jax forward model."""
    mod = type(a).__module__
    if mod.startswith("jax"):
        import jax.numpy as jnp

        return jnp
    return np


@dataclass(frozen=True)
class LogFluxTransform:
    offsets: tuple[float, ...]  # b_band in nJy, indexed like BANDS
    log_scale: float = 1.0
    floor_ratio: float = -0.9
    bands: tuple[str, ...] = BANDS

    @classmethod
    def from_config(cls, config: TransformConfig, bands: tuple[str, ...] = BANDS):
        missing = [b for b in bands if b not in config.band_offsets]
        if missing:
            raise ValueError(
                f"no offset for band(s) {missing}; run "
                f"scripts/estimate_band_offsets.py on the shards first"
            )
        return cls(
            offsets=tuple(float(config.band_offsets[b]) for b in bands),
            log_scale=config.log_scale,
            floor_ratio=config.floor_ratio,
            bands=tuple(bands),
        )

    def offset_for(self, band_index):
        xp = _xp(band_index)
        return xp.asarray(self.offsets)[band_index]

    @property
    def x_floor(self) -> float:
        """The value ``forward`` clips to; useful for plotting limits."""
        return float(np.log1p(self.floor_ratio) / self.log_scale)

    def forward(self, flux, band_index, return_clipped_fraction: bool = False):
        """nJy -> log space.  ``band_index`` broadcasts against ``flux``."""
        xp = _xp(flux)
        b = self.offset_for(xp.asarray(band_index))
        b = xp.reshape(b, xp.shape(b) + (1,) * (flux.ndim - xp.ndim(b)))
        ratio = flux / b
        clipped = ratio < self.floor_ratio
        ratio = xp.maximum(ratio, self.floor_ratio)
        x = xp.log1p(ratio) / self.log_scale
        if return_clipped_fraction:
            return x, float(xp.mean(clipped))
        return x

    def inverse(self, x, band_index):
        """log space -> nJy.  This is what a forward model calls."""
        xp = _xp(x)
        b = self.offset_for(xp.asarray(band_index))
        b = xp.reshape(b, xp.shape(b) + (1,) * (x.ndim - xp.ndim(b)))
        return b * xp.expm1(self.log_scale * x)

    def jacobian(self, x, band_index):
        """``df/dx`` at ``x``; needed if you ever want a density in flux units.

        Not needed for training or for a forward model that generates a model
        image in log space and exponentiates -- there the prior and the
        likelihood both live in log space and no Jacobian appears.
        """
        xp = _xp(x)
        b = self.offset_for(xp.asarray(band_index))
        b = xp.reshape(b, xp.shape(b) + (1,) * (x.ndim - xp.ndim(b)))
        return b * self.log_scale * xp.exp(self.log_scale * x)


def estimate_band_offsets(
    variance: np.ndarray,
    band_index: np.ndarray,
    pool_factor: int,
    k_sigma: float = 5.0,
    bands: tuple[str, ...] = BANDS,
) -> dict[str, float]:
    """``b_band = k_sigma * median pooled sky noise`` per band, in nJy.

    Averaging ``pool_factor**2`` independent pixels divides the noise by
    ``pool_factor``, hence the factor.  Visit images are in the detector frame
    and unwarped, so their pixel noise really is close to independent -- this
    estimate would be optimistic on a coadd, where warping correlates
    neighbouring pixels and pooling reduces the noise by less than
    ``pool_factor``.

    ``variance`` is per-patch native-resolution variance planes; the median over
    pixels then over patches keeps bright sources from inflating the estimate.
    """
    out: dict[str, float] = {}
    variance = np.asarray(variance)
    band_index = np.asarray(band_index)
    for i, band in enumerate(bands):
        sel = band_index == i
        if not np.any(sel):
            continue
        per_patch = np.sqrt(np.median(variance[sel], axis=(-2, -1)))
        out[band] = float(k_sigma * np.median(per_patch) / pool_factor)
    return out
