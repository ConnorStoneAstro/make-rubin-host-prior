"""Flux <-> log-space transform.

    forward:  x = log( softplus(f / s_band) ) / c        softplus(u) = log(1 + e^u)
    model:    f = s_band * exp(c * x)                    strictly positive

``s_band`` is a per-band *softening* scale in nJy, ``softening_sigma`` times the
pooled sky noise of that band.

The asymmetry between the two lines is deliberate, and is the whole point.  A
source cannot emit negative flux, so the prior's reachable domain in flux space
must be strictly positive -- hence the plain exponential, which maps all of R to
``(0, inf)``.  The data transform is therefore **not** exactly invertible, and
should not be: measured flux *is* negative wherever noise takes it below the
subtracted sky, and the right thing to do with those pixels is to let them
smoothly approach zero rather than to represent them faithfully.

``s * exp(c * forward(f))`` equals ``softplus_s(f)`` exactly, so the entire
discrepancy between the data and what the model can express is the softening and
nothing else:

* **Bright flux passes through untouched.**  ``softplus(u) -> u`` exponentially
  fast: within 1.6% at ``f = 3 s``, 0.1% at ``5 s``, and exact in double
  precision by ``10 s``.  Anything detected is represented to well under its own
  photometric error.
* **Negative flux vanishes smoothly.**  ``softplus(u) -> e^u``, so ``x -> f/s``:
  linear in flux, which keeps Gaussian pixel noise Gaussian rather than
  compressing the negative tail.  A deeply over-subtracted region maps to a very
  negative ``x``, never to a wall.
* **There is no floor anywhere.**  ``softplus`` is strictly positive on all of R,
  so no clipping, no point mass, no NaN, and no bound on how negative an input
  pixel may be.

The cost is a pedestal: the model's sky sits at ``softplus(0) * s = 0.693 * s``
rather than at zero.  That is why ``softening_sigma`` defaults to 1 -- it keeps
the pedestal at 0.69 sigma, below the noise it is replacing.  Raising it buys a
tighter, less skewed noise distribution in ``x`` at the price of a pedestal that
climbs above the noise, and of pushing the flux at which the exponential becomes
accurate proportionately higher.

An ELU-style softening was considered and rejected: ``ELU(u) + 1`` leaves a
permanent ``+s`` offset on positive flux (still 3.3% high at ``f = 30 s``),
whereas softplus converges to the identity exponentially.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np

from ..config import BANDS, TransformConfig

#: Below this, ``log(softplus(u))`` differs from ``u`` by less than 1e-9, and the
#: direct expression underflows.
_LINEAR_BELOW = -20.0
#: ``exp`` of anything above this overflows ``expm1`` in ``inverse_exact``; there
#: ``log(expm1(v)) == v`` to double precision anyway.
_EXPM1_ABOVE = 30.0


#: ``d x / d f`` at zero flux is ``0.5 / log(2) / s`` -- the softplus slope
#: (1/2) divided by softplus(0) (log 2).  So sky pixels scatter by this over
#: ``softening_sigma * log_scale`` in x.
SKY_SLOPE = 0.5 / log(2.0)


def expected_sky_scatter(softening_sigma: float, log_scale: float = 1.0) -> float:
    """Predicted spread of ``x`` for sky pixels, ``0.721 / (s_sigma * c)``.

    Compare against the measured ``PatchDataset.stats()["sky_scatter"]``: a large
    disagreement means the per-band softening scales are wrong, which would put
    the bands on different footings and break the single band-agnostic prior.
    """
    return SKY_SLOPE / (softening_sigma * log_scale)


def _xp(a):
    """numpy or jax.numpy, whichever matches the input, so one implementation
    serves both the data loader and a jax forward model."""
    if type(a).__module__.startswith("jax"):
        import jax.numpy as jnp

        return jnp
    return np


def log_softplus(u, xp=None):
    """``log(log(1 + e^u))``, stable across the whole real line.

    ``softplus`` is computed as ``max(u, 0) + log1p(exp(-|u|))`` so it never
    overflows, and the ``u < -20`` branch covers the region where softplus
    underflows to zero and its log is simply ``u``.
    """
    xp = _xp(u) if xp is None else xp
    safe = xp.maximum(u, 0.0) + xp.log1p(xp.exp(-xp.abs(u)))
    return xp.where(u < _LINEAR_BELOW, u, xp.log(xp.where(safe > 0, safe, 1.0)))


@dataclass(frozen=True)
class LogFluxTransform:
    softening: tuple[float, ...]  # s_band in nJy, indexed like BANDS
    log_scale: float = 1.0  # "c" above
    bands: tuple[str, ...] = BANDS

    @classmethod
    def from_config(cls, config: TransformConfig, bands: tuple[str, ...] = BANDS):
        missing = [b for b in bands if b not in config.band_softening]
        if missing:
            raise ValueError(
                f"no softening scale for band(s) {missing}; run "
                f"scripts/prepare_config.py on the shards first"
            )
        return cls(
            softening=tuple(float(config.band_softening[b]) for b in bands),
            log_scale=config.log_scale,
            bands=tuple(bands),
        )

    def softening_for(self, band_index):
        xp = _xp(band_index)
        return xp.asarray(self.softening)[band_index]

    def _s(self, xp, band_index, ndim):
        s = self.softening_for(xp.asarray(band_index))
        return xp.reshape(s, xp.shape(s) + (1,) * (ndim - xp.ndim(s)))

    # -- the two maps ------------------------------------------------------

    def forward(self, flux, band_index):
        """nJy -> log space.  Defined and finite for every real input."""
        xp = _xp(flux)
        return log_softplus(flux / self._s(xp, band_index, flux.ndim), xp) / self.log_scale

    def inverse(self, x, band_index):
        """log space -> nJy: ``s * exp(c * x)``.  **This is what a forward model
        calls**, and it is strictly positive by construction, so a scene drawn
        from the prior can never contain negative flux.

        It is the exact inverse of ``forward`` only for ``f >> s``; below that it
        returns the softened flux, which is the intended behaviour rather than an
        approximation error.  Use ``inverse_exact`` to undo ``forward`` exactly.

        Strictly positive for ``c * x > -745``; below that ``exp`` underflows to
        exactly zero, which is the correct limit and still not negative.  Real
        data never reaches there -- it would take a pixel hundreds of sigma below
        the sky.
        """
        xp = _xp(x)
        return self._s(xp, band_index, x.ndim) * xp.exp(self.log_scale * x)

    def inverse_exact(self, x, band_index):
        """The true inverse of ``forward``: ``s * log(expm1(exp(c * x)))``.

        For round-trip checks and for recovering the measured flux, including its
        negative values.  Agrees with ``inverse`` to double precision wherever
        the flux is more than ~10 s.
        """
        xp = _xp(x)
        cx = self.log_scale * x
        v = xp.exp(cx)
        # Three regimes.  Large v: log(expm1(v)) == v to double precision, and
        # expm1 would overflow.  Very negative cx: v underflows towards zero,
        # expm1(v) ~ v, so log(expm1(v)) ~ cx -- computing it directly would give
        # log(0) = -inf.  In between, the direct expression is fine.
        big = v > _EXPM1_ABOVE          # expm1 would overflow; log(expm1(v)) == v
        small = cx < _LINEAR_BELOW      # expm1(v) ~ v; log(expm1(v)) == c*x
        mid = xp.where(big | small, 1.0, v)
        direct = xp.log(xp.expm1(mid))
        return self._s(xp, band_index, x.ndim) * xp.where(
            big, v, xp.where(small, cx, direct)
        )

    def jacobian(self, x, band_index):
        """``df/dx`` for the model map, which is simply ``c * f``."""
        return self.log_scale * self.inverse(x, band_index)

    # -- analytic diagnostics ---------------------------------------------

    @property
    def sky_level(self) -> float:
        """``x`` at zero flux: ``log(log 2) / c``.  Where the sky sits."""
        return log(log(2.0)) / self.log_scale

    @property
    def sky_pedestal(self) -> float:
        """Model flux at zero measured flux, in units of ``s``: ``log 2``.

        The price of strict positivity.  Keep ``softening_sigma`` near 1 so this
        stays below the noise it replaces.
        """
        return log(2.0)

    def accurate_above(self, tol: float = 0.01) -> float:
        """Flux, in units of ``s``, above which ``inverse`` is within ``tol``.

        Solves ``softplus(u)/u - 1 = tol``.  Roughly 3.4 s at 1%, 5.2 s at 0.1%.
        """
        lo, hi = 1e-6, 200.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if np.log1p(np.exp(-mid)) / mid > tol:  # softplus(u)/u - 1
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


def estimate_band_softening(
    variance: np.ndarray,
    band_index: np.ndarray,
    pool_factor: int,
    softening_sigma: float = 1.0,
    bands: tuple[str, ...] = BANDS,
) -> dict[str, float]:
    """``s_band = softening_sigma * median pooled sky noise`` per band, in nJy.

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
        out[band] = float(softening_sigma * np.median(per_patch) / pool_factor)
    return out
