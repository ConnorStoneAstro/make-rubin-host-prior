"""Smoothing the sky before the floor is drawn through it.

The floor is pointwise and the sky is not flat under it.  At a 2 sigma floor
with a sharp knee, a pixel at 1.5 sigma lands around 0.5 nJy and one at 1.9
sigma around 9, against a pedestal of 0.1 -- so the 14% of sky pixels between
one and two sigma come through as graded speckle, up to five in ``x`` above
the floor, and the despeckler never looks at them because its threshold *is*
the floor.  Measured on synthetic sky that left the standard deviation of
``x`` at 0.70 where a flat sky has none.

So below the floor a pixel is replaced by a Gaussian-weighted mean of its
neighbourhood, and at or above it the pixel is left exactly alone:

    out = where(flux < threshold, gaussian(flux), flux)

A mean over ``n`` pixels of noise is ``sqrt(n)`` quieter, so sky that sat at
1.5 sigma on its own sits at a few tenths once averaged and the knee carries
it to the pedestal.  Real structure survives the same average, which is the
difference: faint outskirts just under the threshold come through as their
smoothed value rather than as holes wherever the noise dipped.  On the same
synthetic sky, with the despeckler on behind it (per cent of pixels more than
0.1 in ``x`` off the floor, white noise and noise with a neighbour correlation
of 0.27)::

    width      white    correlated
    off        12.2       12.4
    0.7 px      0.18       1.4
    1.0 px      0.01       0.26
    1.5 px      0.00       0.03

**It does not replace the despeckler, it hands it an easier job.**  A pixel
above the threshold is kept whatever its neighbours say, so the 2.3% of sky
that pokes through on its own is still there afterwards -- smoothing alone
leaves the standard deviation at 0.83 -- and what removes it is still the cut
on blob size.

**The mean is taken over everything, bright pixels included**, and just
outside a bright source that reads high: the core leaks one kernel width into
pixels that were below the threshold.  For a profile with real wings it is
small -- a factor of two in one or two pixels around a Moffat star, gone under
the noise -- and for a Gaussian-cored source it is a ring (2 sigma true, 31
after, at the contour of a 1000 sigma peak).  Masking bright pixels out of the
mean removes the ring and undershoots by about as much instead, so it is not
done.  Deciding on the *smoothed* image rather than the pixel was measured
too and is worse under correlated noise (0.31% off the floor against 0.03%).

**The other way to do this is Rubin's own footprints**: the coadd's
``DETECTED`` mask plane is "objects as present in the data" stated directly,
and everything outside it could be set to the floor.  Not done, for two
reasons: the detection is not yet accurate enough to trust as the definition
of an object, and the shards do not carry masks, so it is a re-extraction.
"""

from __future__ import annotations

import numpy as np

__all__ = ["smooth_sky"]


def smooth_sky(flux, threshold, width: float) -> np.ndarray:
    """``flux`` with every pixel below ``threshold`` replaced by a local mean.

    ``flux`` is ``(B, S, S)`` in nJy, before the log.  ``threshold`` broadcasts
    against it, so the usual call passes ``(B, 1, 1)`` and each cutout is cut
    at its own sky noise.  ``width`` is the Gaussian's standard deviation in
    pixels of this grid; zero returns the input.  Does not modify it.

    **No pixel is invented at the border.**  The kernel is truncated at the
    edge of the stamp and renormalised by the weight that fell inside, so an
    edge pixel is the mean of the neighbours it actually has.  Padding by
    reflection would feed the border a symmetry the sky does not have.
    """
    from scipy import ndimage

    flux = np.asarray(flux)
    if width <= 0:
        return flux
    smooth = ndimage.gaussian_filter(flux, (0.0, width, width),
                                     mode="constant", cval=0.0)
    inside = ndimage.gaussian_filter(np.ones(flux.shape[-2:], flux.dtype),
                                     width, mode="constant", cval=0.0)
    return np.where(flux < threshold, smooth / inside, flux)
