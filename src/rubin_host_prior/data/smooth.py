"""Blending the sky into a smoothed copy of itself before the log is taken.

The map is pointwise, and pointwise it cannot tell sky from source: a floor
drawn through raw pixels lets each one's own noise decide which side of the
knee it falls on, and with no floor at all the map is steepest exactly where
the sky is -- at zero flux ``dx/df`` is ``0.72 / s``.  Either way the sky's
noise reaches ``x`` as texture, which the prior has to spend itself learning
and will then put into its samples, and which is not a property of any galaxy.

So the image handed to the map is a weighted sum of the pixels and a
Gaussian-smoothed copy of them,

    out = lam * flux + (1 - lam) * smooth
    lam = sigmoid((smooth / sky_noise - centre) / scale)

with the weight decided by the *smoothed* flux in units of that cutout's own
sky noise.  Where the neighbourhood is empty ``lam`` is zero and the pixel is
the local mean; where it holds a source ``lam`` is one and the pixel is itself.
At the defaults (centre 1.5, scale 0.3) ``lam`` is 0.7% at zero and 99.3% at
three sigma.

**Deciding on the smoothed image is the point.**  The first version switched
on the pixel -- ``where(flux < 2 sigma, smooth, flux)`` -- and so kept the
sky's own tail past two sigma at full height: 2.3% of Gaussian sky, each pixel
standing about 1.1 in ``x`` above the smoothed sky around it, and a notch in
the histogram where the switch was.  A lone bright pixel in a quiet
neighbourhood does not move the smoothed flux, so here it is averaged away
with the rest, and a sigmoid rather than a step means no pixel sits on an
edge.  Measured on synthetic scenes -- a host and about eight neighbours,
noise with a neighbour correlation of 0.27, ``softening_sigma = 1``, width 1.5
px (std of ``x`` over empty sky; share of sky pixels more than 0.5 above the
sky's median; half the mean squared difference in ``x`` between two noise
realisations of one scene), all with no floor so that the blend is what is
being measured::

                                  sky std   speckle   noise variance
    no smoothing                   0.71     22.4%        0.470
    switch on the pixel at 2       0.28      2.7%        0.074
    blend, centre 1.5 scale 0.3    0.21      0.7%        0.045
    blend, centre 1.0 scale 0.2    0.21      1.1%        0.052

and then the floor on top of the default blend, which is what the loader
does (``TransformConfig`` has the flux each one costs)::

    floor_sigma 1, softening 1     0.06      0.0%        0.007
    floor_sigma 1, softening 1/3   0.02      0.0%        0.007

**What it costs is faint compact sources**, because smoothing dilutes a point
source about fourfold and it is the diluted peak the weight sees.  The flux
the prior is shown at the peak of an isolated point source (Gaussian PSF of
0.85 px), over the true softened peak::

    peak, sky sigma               3      5      8     12     20
    centre 1.5 scale 0.3        0.40   0.46   0.85   1.00   1.00
    centre 1.0 scale 0.2        0.46   0.78   1.00   1.01   1.00

(no floor; a floor on top takes whatever of the smoothed peak is under it.)
So ``sky_blend_sigma`` trades those against the sky: lower keeps fainter
sources and lets more of the noise through as itself.

**The weight is returned as well**, for diagnostics.  It was used once more,
to mix ``x`` towards a constant pedestal -- the sky at 1 or 0.1 nJy, with the
same ``lam`` -- which made a smooth low sky and gave back the tall step
around every object that a low sky always is.  Not done;
``data.transform`` has the account.

**The mean is taken over everything, bright pixels included**, and just
outside a bright source that reads high: the core leaks one kernel width into
pixels that were below the threshold.  For a profile with real wings it is
small -- a factor of two in one or two pixels around a Moffat star, gone under
the noise -- and for a Gaussian-cored source it is a ring.  Masking bright
pixels out of the mean removes the ring and undershoots by about as much
instead, so it is not done.

**The other way to do this is Rubin's own footprints**: the coadd's
``DETECTED`` mask plane is "objects as present in the data" stated directly,
and everything outside it could be smoothed or set to one level.  Not done,
for two reasons: the detection is not yet accurate enough to trust as the
definition of an object, and the shards do not carry masks, so it is a
re-extraction.
"""

from __future__ import annotations

import numpy as np

__all__ = ["blend_sky"]


def blend_sky(flux, sky_noise, width: float, centre: float,
              scale: float) -> tuple[np.ndarray, np.ndarray]:
    """``lam * flux + (1 - lam) * smooth``, and ``lam``.

    ``flux`` is ``(B, S, S)`` in nJy, before the log.  ``sky_noise`` broadcasts
    against it, so the usual call passes ``(B, 1, 1)`` and each cutout is
    weighed against its own noise.  ``width`` is the Gaussian's standard
    deviation in pixels of this grid.  ``centre`` and ``scale`` are in sky
    sigma: ``lam = sigmoid((smooth / sky_noise - centre) / scale)``.  Does not
    modify the input.

    **No pixel is invented at the border.**  The kernel is truncated at the
    edge of the stamp and renormalised by the weight that fell inside, so an
    edge pixel is the mean of the neighbours it actually has.  Padding by
    reflection would feed the border a symmetry the sky does not have.
    """
    from scipy import ndimage

    flux = np.asarray(flux)
    if not width > 0:
        raise ValueError(f"need a smoothing width > 0, got {width}")
    if not scale > 0:
        raise ValueError(f"need a blend scale > 0, got {scale}")
    smooth = ndimage.gaussian_filter(flux, (0.0, width, width),
                                     mode="constant", cval=0.0)
    inside = ndimage.gaussian_filter(np.ones(flux.shape[-2:], flux.dtype),
                                     width, mode="constant", cval=0.0)
    smooth = smooth / inside
    # tanh rather than the literal sigmoid: stable on both sides.
    lam = 0.5 * (np.tanh(0.5 * (smooth / sky_noise - centre) / scale) + 1.0)
    lam = lam.astype(flux.dtype, copy=False)
    return lam * flux + (1.0 - lam) * smooth, lam
