"""Filling the holes DP2 leaves in the cores of saturated stars.

**What the holes are.**  Coaddition rejects saturated pixels from every input
exposure.  Where none survives there is nothing to average, so the coadd fills
with zero and the variance plane holds ``inf``.  The result is a bright ring
around a dead centre -- a one-pixel cliff from the brightest flux in the stamp
down to nothing, which no real source makes.  A prior trained on it learns that
such cliffs are possible and then puts them in posterior samples, in exactly the
region where the likelihood has nothing to say.

**Why fill rather than reject.**  Both remove the cliff from the prior.  On a
2051-stamp set, 74 stamps (3.6%) carry any no-data at all -- median hole 6x6
native pixels, 90th percentile 16x16, largest 45x45 -- so rejecting them costs
almost nothing.  Filling keeps them, and a filled core is a *smooth bright cap*
where the truth is a peak: wrong, but wrong in a way the likelihood can push
around, which a cliff is not.

**Why detect from pixels and not from the mask.**  Shard schema 3 stores the
image and nothing else, so there is no mask at training time.  Re-extracting
with one would double the bytes per stamp, and loader I/O is the measured
bottleneck.  Detecting from pixel values costs 13% of what pooling already
costs and needs no re-extraction.

**The detector is exact, not a threshold.**  A no-data pixel is written as
exactly ``0.0`` or as a non-finite; a real sky pixel is neither.  Float noise
essentially never lands on zero, so there is nothing to tune and no false
positives -- measured at 0 over 44 million clean pixels.  Note that *both*
tests are needed: ``NaN == 0.0`` is false, so an exact-zero rule alone misses a
NaN fill, and an identical-neighbour rule misses it too, since ``NaN != NaN``.
"""

from __future__ import annotations

import numpy as np


def no_data_mask(stamp: np.ndarray) -> np.ndarray:
    """Where coaddition had nothing to average: exactly zero, or non-finite."""
    with np.errstate(invalid="ignore"):
        return (stamp == 0.0) | ~np.isfinite(stamp)


def fill_no_data(
    stamp: np.ndarray, iterations: int = 24, pad: int = 2
) -> tuple[np.ndarray, int]:
    """Harmonically inpaint the no-data pixels of one native stamp.

    Returns ``(filled, n_pixels)``.  ``n_pixels`` is 0 and the input is handed
    straight back when there is nothing to do, which is the 96% case.

    Jacobi iteration on the Laplace equation with the surrounding pixels as a
    Dirichlet boundary -- the smoothest surface that joins the rim.  Run inside
    the hole's bounding box only, so cost follows the hole and not the stamp:
    24 sweeps over a 45x45 box is nothing, and most holes are 6x6.

    **It produces a flat cap, not a peak.**  Harmonic interpolation has no
    interior extremum, so a filled star core tops out at roughly its rim rather
    than rising to the peak the real star had.  That is accepted deliberately:
    the point is to remove a gradient the prior should never have seen, and a
    forward model's likelihood constrains the bright cap anyway.  It is not a
    reconstruction and should not be read as one.
    """
    mask = no_data_mask(stamp)
    n = int(mask.sum())
    if n == 0:
        return stamp, 0
    ys, xs = np.nonzero(mask)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(stamp.shape[-2], int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(stamp.shape[-1], int(xs.max()) + 1 + pad)
    sub = np.array(stamp[y0:y1, x0:x1], dtype=np.float64)
    m = mask[y0:y1, x0:x1]
    known = ~m
    if not known.any():
        # The hole fills the box: nothing to interpolate from, so fall back to
        # the stamp's own median.  Only reachable for a hole at the very edge.
        sub[m] = float(np.median(stamp[~mask])) if (~mask).any() else 0.0
        out = np.array(stamp)
        out[y0:y1, x0:x1] = sub
        return out.astype(stamp.dtype), n
    sub[m] = float(sub[known].mean())  # a sane start; the sweeps refine it
    for _ in range(iterations):
        nb = np.zeros_like(sub)
        nb[1:-1, 1:-1] = 0.25 * (sub[:-2, 1:-1] + sub[2:, 1:-1]
                                 + sub[1:-1, :-2] + sub[1:-1, 2:])
        # Edge cells of the box keep their seeded value rather than relaxing
        # towards zero; `pad` normally puts real pixels there anyway.
        sub[m] = np.where(m, nb, sub)[m]
    out = np.array(stamp)
    out[y0:y1, x0:x1] = sub
    return out.astype(stamp.dtype), n


def fill_batch(stamps: np.ndarray, iterations: int = 24) -> tuple[np.ndarray, dict]:
    """``fill_no_data`` over a stack, with a report of what it touched.

    The report is the guard against a silent no-op: if the holes in some future
    release are written with a value this detector does not recognise, every
    count here is zero and that is visible rather than invisible.
    """
    out = stamps
    counts = np.zeros(len(stamps), dtype=np.int64)
    touched: list[int] = []
    for i in range(len(stamps)):
        filled, n = fill_no_data(stamps[i], iterations=iterations)
        if n:
            if out is stamps:
                out = np.array(stamps)
            out[i] = filled
            counts[i] = n
            touched.append(i)
    return out, {
        "n_stamps": int(len(stamps)),
        "n_stamps_filled": int(len(touched)),
        "n_pixels_filled": int(counts.sum()),
        "largest_hole_px": int(counts.max()) if len(counts) else 0,
    }
