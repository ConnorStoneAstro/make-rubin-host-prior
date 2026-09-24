"""Flux <-> log-space transform.

    soften:   f_s = s * softplus(f / s)                  softplus(u)=log(1+e^u)
    forward:  x   = log(f_s)
    model:    f   = exp(x)                               strictly positive

**x is log flux in nJy, absolutely.**  ``softplus(u) -> u`` exponentially fast,
so ``s * softplus(f/s) -> f`` and the forward map converges to plain ``log(f)``:
a bright pixel of 4e5 nJy lands at 12.9 in every band, whatever that band's
depth.  The model map is then ``exp(x)`` with no band in it at all, which is
what a forward model in nJy wants -- there is no per-band offset to undo before
the prior composes with a likelihood.

The softening is written as ``s * softplus(f / s)`` because that form has one
parameter, it is a flux, and above it the map is the identity: bright pixels
pass through untouched and the entire adjustment is confined to the low and
negative regime.  ``exp(forward(f))`` reproduces it exactly, so the model can
express the softened data and nothing else.

``s`` is **one** softening scale in nJy for all of the data, ``softening_sigma``
times the measured pooled sky noise.  It used to be per band, and that bought
nothing once ``forward`` became ``log(f_s)``: a per-band scale only moved each
band's sky to its own ``log(s_band * log 2)``, spreading the levels over 1.27 in
x while the thing the prior is about -- the flux of the scene -- was already
band-independent.

With a single scale the sky lands at ``log(s * log 2)`` for *every* band, so the
transform now gives both an absolute flux scale and a common sky level.  What
differs between bands is the *width* of the sky about that level, which is the
honest difference: a deep band scatters less than a shallow one, and pretending
otherwise was what the per-band scale was doing.

An earlier version divided by ``s_band`` inside the logarithm.  That also gave a
common sky level, but by making x a *relative* quantity -- the same flux meant a
different x in each band, and the absolute scale of the signal was what got given
away.  Scene flux is what this prior exists to describe.

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

**The softening suppresses the sky, and that is the point.**  With ``s`` at two
sigma of the pooled noise, pixels within the noise are compressed towards a
common pedestal at ``softplus(0) * s = 1.39 sigma`` while anything detected is
untouched.  The prior is meant to describe what a galaxy looks like, not what
this particular realisation of the sky looked like, and a prior that reproduces
noise faithfully is spending capacity on a thing the likelihood already models.

``softening_sigma`` is that scale in units of the measured noise, and
is the knob.  Lower values preserve the noise distribution more faithfully at
the cost of a skewed, heavy-tailed ``x``; higher values flatten the sky harder
and push the flux at which the exponential map becomes accurate proportionately
up.  Two is a deliberate choice to suppress rather than to preserve.

An ELU-style softening was considered and rejected: ``ELU(u) + 1`` leaves a
permanent ``+s`` offset on positive flux (still 3.3% high at ``f = 30 s``),
whereas softplus converges to the identity exponentially.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np

from ..config import TransformConfig

#: ``Phi^-1(0.25) - Phi^-1(0.05)`` -- converts the faint-quartile spread of a
#: Gaussian into a sigma.
_P25_P05_TO_SIGMA = 0.9703648
#: Below this, ``log(softplus(u))`` differs from ``u`` by less than 1e-9, and the
#: direct expression underflows.
_LINEAR_BELOW = -20.0
#: ``exp`` of anything above this overflows ``expm1`` in ``inverse_exact``; there
#: ``log(expm1(v)) == v`` to double precision anyway.
_EXPM1_ABOVE = 30.0


#: ``d x / d f`` at zero flux is ``0.5 / log(2) / s`` -- the softplus slope
#: (1/2) divided by softplus(0) (log 2).  The ``log(s)`` that ``forward`` adds is
#: a constant, so it shifts the sky without widening it.
SKY_SLOPE = 0.5 / log(2.0)


def expected_sky_scatter(softening_sigma: float) -> float:
    """Predicted spread of ``x`` for sky pixels, ``0.721 / softening_sigma``.

    Exact for a band at the noise level the single scale was measured from, and
    proportionally wider or narrower for a band deeper or shallower than that:
    the scatter is ``0.721 * sigma_band / s``.  So this is the *typical* width,
    which is what ``PatchDataset.stats()["sky_scatter"]`` measures across the
    whole set.  A large disagreement means the softening scale is wrong.
    """
    return SKY_SLOPE / softening_sigma


def _xp(a):
    """numpy or jax.numpy, whichever matches the input, so one implementation
    serves both the data loader and a jax forward model."""
    if type(a).__module__.startswith("jax"):
        import jax.numpy as jnp

        return jnp
    return np


def softplus(u, xp=None):
    """``log(1 + e^u)``, from the library rather than by hand.

    ``jax.nn.softplus`` on a jax array and ``numpy.logaddexp(0, u)`` otherwise.
    Both are the stable formulation; neither overflows.
    """
    xp = _xp(u) if xp is None else xp
    if xp is not np:
        import jax.nn

        return jax.nn.softplus(u)
    return np.logaddexp(0.0, u)


def soften(flux, scale, xp=None):
    """``scale * softplus(flux / scale)``: nJy in, nJy out.

    The scale is the only parameter, and it is a flux.  Above it the map is the
    identity -- ``softplus(u) -> u`` exponentially fast, so bright pixels pass
    through with the adjustment confined to the low and negative regime, which
    is the whole point of writing it this way.  Below it the map bends over and
    approaches zero from above without ever reaching it.
    """
    xp = _xp(flux) if xp is None else xp
    return scale * softplus(flux / scale, xp)


def log_softplus(u, xp=None):
    """``log(softplus(u))``, stable across the whole real line.

    The instability here belongs to the logarithm, not to softplus: for
    ``u < -745`` softplus underflows to zero in float64 and its log is ``-inf``.
    Since ``log(softplus(u)) -> u`` in that limit, the branch below ``-20``
    returns ``u`` directly, and is exact to 1e-9 there.
    """
    xp = _xp(u) if xp is None else xp
    sp = softplus(u, xp)
    return xp.where(u < _LINEAR_BELOW, u, xp.log(xp.where(sp > 0, sp, 1.0)))


@dataclass(frozen=True)
class LogFluxTransform:
    #: The softening scale in nJy.  One number for all of the data, in every
    #: band: see the module docstring for why it is not per band.
    softening: float

    @classmethod
    def from_config(cls, config: TransformConfig):
        s = config.softening
        if s is None or not np.isfinite(s):
            raise ValueError(
                "the config has no softening scale. Run "
                "scripts/prepare_config.py on the shards; it measures one from "
                "the pooled sky noise."
            )
        if s <= 0:
            raise ValueError(
                f"softening is a flux in nJy and must be positive, not {s}"
            )
        return cls(softening=float(s))

    def soften(self, flux):
        """``s * softplus(flux / s)``: nJy in, nJy out.

        The definition the other maps are built from.  ``forward`` is
        ``log(soften(f))`` and ``inverse`` is exactly this composed with it, so
        this is the one place the softening is stated and the only shape the
        model can express.
        """
        return soften(flux, self.softening, _xp(flux))

    # -- the two maps ------------------------------------------------------

    def forward(self, flux):
        """nJy -> log space: ``log(soften(f))``, which tends to ``log(f)``.

        Takes no band.  Defined and finite for every real input.  Written as
        ``log_softplus(f/s) + log(s)`` rather than by composing ``soften`` and
        taking a logarithm, which are algebraically the same thing: below
        ``f/s = -745`` the softened flux underflows to zero in float64 and its
        log is ``-inf``, where ``log_softplus`` returns the exact limit instead.
        A test pins the two forms together.
        """
        return log_softplus(flux / self.softening, _xp(flux)) + log(self.softening)

    def inverse(self, x):
        """log space -> nJy: ``exp(x)``.  **This is what a forward model calls.**

        It takes no band, and that is the point of the transform: x is log flux
        in nJy absolutely, so turning a scene back into flux needs nothing but
        the exponential.  Strictly positive by construction, so a scene drawn
        from the prior can never contain negative flux.

        ``inverse(forward(f))`` is ``soften(f)`` exactly, at every flux -- so the
        whole discrepancy between the data and what the model can express is the
        softening and nothing else.  It is the inverse of ``forward`` only for
        ``f >> s``, which is the intended behaviour rather than an approximation
        error.  Use ``inverse_exact`` to undo ``forward`` exactly.

        Strictly positive for ``x > -745``; below that ``exp`` underflows to
        exactly zero, which is the correct limit and still not negative.  Real
        data never reaches there -- it would take a pixel hundreds of sigma below
        the sky.
        """
        return _xp(x).exp(x)

    def inverse_exact(self, x):
        """The true inverse of ``forward``: ``s * log(expm1(exp(x) / s))``.

        For round-trip checks and for recovering the measured flux, including its
        negative values.  Agrees with ``inverse`` to double precision wherever
        the flux is more than ~10 s.
        """
        xp = _xp(x)
        s = self.softening
        # ``u = log(softplus(f/s))`` -- x with the scale's constant removed,
        # which is the quantity the three regimes below are expressed in.
        cx = x - log(s)
        v = xp.exp(cx)
        # Three regimes.  Large v: log(expm1(v)) == v to double precision, and
        # expm1 would overflow.  Very negative cx: v underflows towards zero,
        # expm1(v) ~ v, so log(expm1(v)) ~ cx -- computing it directly would give
        # log(0) = -inf.  In between, the direct expression is fine.
        big = v > _EXPM1_ABOVE  # expm1 would overflow; log(expm1(v)) == v
        small = cx < _LINEAR_BELOW  # expm1(v) ~ v; log(expm1(v)) == c*x
        mid = xp.where(big | small, 1.0, v)
        direct = xp.log(xp.expm1(mid))
        return s * xp.where(big, v, xp.where(small, cx, direct))

    def jacobian(self, x):
        """``df/dx`` for the model map, which is simply ``f`` itself."""
        return self.inverse(x)

    # -- analytic diagnostics ---------------------------------------------

    @property
    def sky_level(self) -> float:
        """``x`` at zero flux: ``log(s * log 2)``.  Where the sky sits.

        The same in every band, because there is one scale.  What differs is the
        *width* about it -- a deep band scatters less than a shallow one, which
        is a real difference rather than one the transform should erase.
        """
        return log(self.softening * log(2.0))

    @property
    def sky_pedestal(self) -> float:
        """Model flux at zero measured flux, in units of ``s``: ``log 2``.

        Multiply by ``softening_sigma`` for the figure in units of the sky
        noise -- 1.39 sigma at the default 2.0.  It sits *above* the noise on
        purpose: pixels within the noise are compressed towards it, which is
        what suppressing the sky means.  An earlier version of this docstring
        said to keep it below one sigma, from when the aim was to preserve the
        noise distribution; the likelihood models the noise, so the prior
        should not.
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


def measure_pooled_sky_noise(pooled: np.ndarray) -> float:
    """Sky noise of *pooled* patches, in nJy -- measured, not derived.

    Deriving it as ``sqrt(median variance) / pool_factor`` assumes the pixel
    noise is uncorrelated, which is true of unwarped visit images and **false of
    coadds**: coaddition warps input exposures onto the skymap grid, so
    neighbouring pixels share flux and averaging ``P**2`` of them reduces the
    noise by less than ``P``.  Measured on synthetic coadd-like patches, the
    derived value underestimates the real pooled noise by a factor of two at a
    correlation width of 0.8 px.  Since this number sets the softening scale,
    getting it wrong by 2x would put the transform's turnover in the wrong place.

    The estimator reads only the faint quarter of each patch:
    ``(p25 - p5) / 0.9704``, which is exactly one sigma for a Gaussian.  Sources
    are positive, so confining the estimate to the low percentiles makes it
    immune to them until they cover more than ~75% of a patch -- a plain standard
    deviation is inflated by every galaxy present, and even a
    ``median - p16`` form drifts once a source covers a fifth of the frame.  The
    median is then taken across patches, so the few source-dominated patches that
    do defeat it cannot move the answer.

    That median wants a real sample: with five patches it scatters by ~15%, with
    twenty by ~1%.  Pass a few hundred.

    One number across every band.  It used to be one per band, which is a more
    faithful description of the data and bought the transform nothing -- see the
    module docstring.  The median is over patches of all bands together, so a
    band contributes in proportion to how much of the set it is.
    """
    pooled = np.asarray(pooled, dtype=np.float64)
    patches = pooled.reshape(len(pooled), -1)
    p25, p05 = np.percentile(patches, [25.0, 5.0], axis=1)
    per_patch = (p25 - p05) / _P25_P05_TO_SIGMA
    good = np.isfinite(per_patch) & (per_patch > 0)
    if not np.any(good):
        raise ValueError(
            "no patch had a measurable sky noise; they are probably all source "
            "or all masked"
        )
    return float(np.median(per_patch[good]))


def estimate_softening(pooled: np.ndarray, softening_sigma: float) -> float:
    """``s = softening_sigma * measured pooled sky noise``, in nJy.

    ``softening_sigma`` has no default here.  It had one, 1.0, left from when
    the aim was to preserve the noise rather than suppress it, and a second
    answer to a question ``TransformConfig`` already answers is exactly how the
    two drift apart.  Pass ``config.transform.softening_sigma``.

    Takes *pooled* patches rather than variance planes, so it holds for coadds
    as well as visit images.  Pooling needs no transform, so this can be run
    before one exists -- see ``data.dataset.pool_shards``.
    """
    return softening_sigma * measure_pooled_sky_noise(pooled)
