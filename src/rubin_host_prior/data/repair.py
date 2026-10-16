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
    # Only cells with four neighbours inside the box get relaxed.  A masked
    # cell on the box border has none above or left of it, so `nb` there is the
    # zero it was initialised with -- and writing that back would put an exact
    # zero into the output, which is the very thing being repaired.  Reachable
    # whenever the hole runs to the stamp edge and `pad` cannot step outside
    # it; those cells keep their seeded value instead.
    relax = np.zeros_like(m)
    relax[1:-1, 1:-1] = m[1:-1, 1:-1]
    for _ in range(iterations):
        nb = np.zeros_like(sub)
        nb[1:-1, 1:-1] = 0.25 * (sub[:-2, 1:-1] + sub[2:, 1:-1]
                                 + sub[1:-1, :-2] + sub[1:-1, 2:])
        sub[relax] = nb[relax]
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


def _directional_max(a: np.ndarray, r: int) -> np.ndarray:
    """``(4, H, W)``: the maximum within ``r`` pixels left, right, up and down.

    Four one-sided maxima rather than one square window, because that is what
    separates a hole from an edge.  A dead core has bright flux on *every* side;
    sky just outside a bright star has it on one.  Shifts fill with ``-inf``
    rather than wrapping, so the stamp border cannot manufacture a surround.
    """
    out = np.full((4,) + a.shape, -np.inf)
    for k in range(1, r + 1):
        out[0, :, k:] = np.maximum(out[0, :, k:], a[:, :-k])      # bright to the left
        out[1, :, :-k] = np.maximum(out[1, :, :-k], a[:, k:])     # to the right
        out[2, k:, :] = np.maximum(out[2, k:, :], a[:-k, :])      # above
        out[3, :-k, :] = np.maximum(out[3, :-k, :], a[k:, :])     # below
    return out


def dark_inside_bright(
    stamp: np.ndarray, sky_noise: float, radius: int = 12,
    dark_sigma: float = 3.0, bright_sigma: float = 50.0,
) -> np.ndarray:
    """Pixels consistent with sky that are *surrounded* by something very bright.

    The hole signature, independent of what value the pipeline wrote: a real
    scene is non-negative up to noise, so a pixel at sky level with 50-sigma
    flux on all four sides within ``radius`` is neither sky nor source.

    **Surrounded, not merely near.**  Requiring brightness in one square window
    flagged the entire ring of sky around every bright star -- 778 pixels
    against a 69-pixel hole -- because sky a few pixels outside a star does have
    a bright neighbour.  It does not have one in all four directions.

    Unlike ``no_data_mask`` this needs the stamp's sky noise and two thresholds,
    which is why it is a diagnostic and not the detector: use it to find out
    what the cores are, then decide what the detector should look for.
    """
    a = np.asarray(stamp, dtype=np.float64)
    s = sky_noise if np.isfinite(sky_noise) and sky_noise > 0 else 1.0
    with np.errstate(invalid="ignore"):
        dark = ~(a > dark_sigma * s)          # non-finite counts as dark
        sides = _directional_max(np.where(np.isfinite(a), a, -np.inf), radius)
    return dark & np.all(sides > bright_sigma * s, axis=0)


def describe_dark_cores(
    stamps: np.ndarray, sky_noise: np.ndarray, radius: int = 8, n_report: int = 10
) -> dict:
    """What the dark cores in a set of stamps actually are.

    The repair detects a hole as exactly ``0.0`` or non-finite, which is what
    DP2 was understood to write.  If a hole survives the repair then that
    premise is wrong for some stamps, and guessing a replacement rule is how
    you end up with a detector that silently catches the wrong thing.  This
    reports the evidence instead.

    ``dark_inside_bright`` finds the hole *signature* -- sky-level pixels with
    a very bright neighbour -- without caring what value was written, and this
    subtracts what the current detector already sees.  **``n_pixels_missed`` is
    the number that matters**, and ``missed_value_sigma`` says what to widen the
    detector to: exactly zero means something else is wrong, a small negative
    means the pipeline wrote one rather than a zero, and a scatter about zero
    means it wrote noise.

    Thresholds are the diagnostic's, not the repair's, and deliberately
    conservative: 50 sigma for "bright" is a genuinely saturated star, not a
    galaxy core.
    """
    stamps = np.asarray(stamps)
    sigma = np.asarray(sky_noise, dtype=np.float64)
    rows = []
    for i in range(len(stamps)):
        a = np.asarray(stamps[i], dtype=np.float64)
        s = (sigma[i] if i < len(sigma) and np.isfinite(sigma[i]) and sigma[i] > 0
             else 1.0)
        suspect = dark_inside_bright(stamps[i], s, radius=radius)
        detected = no_data_mask(stamps[i])
        missed = suspect & ~detected
        vals = a[missed]
        rows.append({
            "n_suspect": int(suspect.sum()),
            "n_detected": int(detected.sum()),
            "n_missed": int(missed.sum()),
            # what the detector cannot see, in units of this cutout's sky noise
            "missed_min_sigma": float(np.nanmin(vals) / s) if vals.size else None,
            "missed_max_sigma": float(np.nanmax(vals) / s) if vals.size else None,
            "missed_median_sigma": float(np.nanmedian(vals) / s) if vals.size else None,
            "missed_exactly_zero": int((vals == 0.0).sum()) if vals.size else 0,
        })
    missed_rows = [r for r in rows if r["n_missed"]]
    return {
        "n_stamps": len(rows),
        "n_with_detected_holes": sum(1 for r in rows if r["n_detected"]),
        "n_pixels_detected": sum(r["n_detected"] for r in rows),
        # the number that decides whether the detector needs widening
        "n_with_missed_holes": len(missed_rows),
        "n_pixels_missed": sum(r["n_missed"] for r in rows),
        "missed_value_sigma": {
            "min": round(min(r["missed_min_sigma"] for r in missed_rows), 4),
            "median": round(float(np.median([r["missed_median_sigma"]
                                             for r in missed_rows])), 4),
            "max": round(max(r["missed_max_sigma"] for r in missed_rows), 4),
        } if missed_rows else None,
        "worst": sorted(missed_rows, key=lambda r: -r["n_missed"])[:n_report],
    }
