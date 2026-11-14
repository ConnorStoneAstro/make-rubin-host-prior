"""Pushing the sky's positive tail back down to the sky level.

**Off by default** -- see ``TransformConfig.despeckle_max_pixels``.  It was
written for a sky pinned to a constant and still works as described, but what
it fills with is one value, and the sky around it is no longer one.

The map itself is *pointwise*: with nothing done about it, Gaussian sky has
about 2.3% of its pixels above 2 sigma -- roughly 380 on a 128 px grid --
scattered as single pixels and pairs.  They are not features of the sky, they
are the tail of the noise, and a prior trained on them learns to put isolated
bright pixels into its samples.  ``data.smooth`` now deals with them by
weighing each pixel on its smoothed neighbourhood, which is the same idea by
another route.

A pointwise map cannot tell them from a faint compact source, because
pointwise there is no difference.  What separates them is *extent*: measured on
pooled Rubin-like stamps, 98% of the blobs above 2 sigma are four
pixels or smaller, and essentially none of the real sources are.  So the cut
here is on blob size, and it is the one place in the representation where a
pixel's neighbours decide what happens to it.

**It removes faint compact sources too, and that is not a bug to be fixed
here.** At the sizes involved a four-pixel blob of noise and a marginal
point source are the same object to any statistic; dropping both is the
decision, and ``despeckle_sigma`` is the knob that says how bright something
must be before the question is even asked.
"""

from __future__ import annotations

import numpy as np

__all__ = ["despeckle", "NOTHING_REMOVED"]

#: Eight-connectivity: diagonal neighbours count as touching.  The conservative
#: choice, because it *merges* more -- a diagonal pair is one blob of two
#: rather than two blobs of one, so fewer blobs fall under the size limit.
_NEIGHBOURHOOD = np.ones((3, 3), dtype=bool)

NOTHING_REMOVED: dict = {"n_stamps": 0, "n_blobs": 0, "n_blobs_removed": 0,
                         "n_pixels": 0, "n_pixels_removed": 0}


def despeckle(x, threshold, fill, max_pixels: int) -> tuple[np.ndarray, dict]:
    """Set every blob of ``<= max_pixels`` pixels above ``threshold`` to ``fill``.

    ``x`` is ``(B, S, S)`` in the log representation.  ``threshold`` and
    ``fill`` broadcast against it, so the usual call passes ``(B, 1, 1)`` --
    each cutout is thresholded at its own sky noise and filled at its own sky
    level.  Returns ``(x, report)`` and does not modify the input.

    ``scipy.ndimage.label`` does the connected components.  It is not a new
    dependency: ``jax`` requires ``scipy >= 1.14``, so it is already installed
    wherever this package can run at all.
    """
    from scipy import ndimage

    x = np.asarray(x)
    if max_pixels <= 0:
        return x, dict(NOTHING_REMOVED, n_stamps=len(x), n_pixels=x.size)

    thr = np.broadcast_to(np.asarray(threshold, x.dtype), x.shape)
    val = np.broadcast_to(np.asarray(fill, x.dtype), x.shape)
    out = x.copy()
    n_blobs = n_removed = n_pixels = 0
    for i in range(len(x)):
        lit = x[i] > thr[i]
        if not lit.any():
            continue
        labels, n = ndimage.label(lit, structure=_NEIGHBOURHOOD)
        # Index 0 is the background, which is not a blob and must never be
        # "small enough to remove" -- it is almost always the largest region
        # in the frame, but a frame that is lit edge to edge has none at all.
        sizes = np.bincount(labels.ravel(), minlength=n + 1)
        small = sizes <= max_pixels
        small[0] = False
        kill = small[labels]
        out[i][kill] = val[i][kill]
        n_blobs += n
        n_removed += int(small.sum())
        n_pixels += int(kill.sum())
    return out, {"n_stamps": len(x), "n_blobs": n_blobs,
                 "n_blobs_removed": n_removed, "n_pixels": x.size,
                 "n_pixels_removed": n_pixels}
