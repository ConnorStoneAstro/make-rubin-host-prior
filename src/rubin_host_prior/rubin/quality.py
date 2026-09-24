"""Artefact rejection for candidate patches.  Pure numpy -- no LSST stack.

Tuned for DP2 ``deep_coadd``.  Coaddition already handles chip edges, cosmic
rays and per-exposure electronic artefacts, so what is left is coadd-specific
and, on DP2, quite different from DP1.

**Plane names changed and bit numbers are dynamic.**  DP2 renamed ``SAT`` ->
``SATURATED``, ``CR`` -> ``COSMIC_RAY``, ``INTRP`` -> ``INTERPOLATED`` and
``EDGE`` -> ``DETECTION_EDGE``, and the bit assignments are not stable across
releases.  A DP1-era gate does not error on DP2 -- it matches nothing and passes
every stamp.  Nothing here hard-codes a bit: the packing is recorded per shard by
extraction from the coadd's own ``mask.schema``, and every plane name absent from
that mapping contributes nothing.

**No-data is carried twice over.**  DP2 has a ``NO_DATA`` plane *and* holds
``inf`` in the variance where there were no contributing exposures -- including
the cores of saturated stars.  The plane is gated at zero tolerance, following
Rubin's own guidance.  The variance is handled separately and more gently: its
fraction is measured, with a tolerance and a stricter one at the centre, because
rejecting merely because non-finite variance is *present* would throw away every
stamp containing a bright neighbour, which is the regime this project models.

**``INEXACT_PSF`` and ``REJECTED`` are not quality cuts.**  They cover a large
fraction of the DP2 coadd, so gating on them discards almost everything.  They
are recorded as per-stamp covariates so a later cut can be made from the
manifest without re-reading pixels.  ``DETECTED`` is likewise informational --
rejecting on it would reject every patch containing a galaxy.

**Neither release masks satellite trails in what you train on.**  DP2 coadds have
no ``STREAK`` plane at all.  Trails and unmasked electronics artefacts have to
come from your own detection step; nothing here will catch them.

**Nor does any plane flag a depth step.**  DP2 coadds are cell-based: each 150 px
cell is built from its own set of input visits, so where the input set changes --
at a visit edge, a dither boundary, the rim of the field -- the noise level steps
across a straight cell edge, and a stamp larger than a cell straddles it.  It is
most obvious in y, which has the fewest visits and so the largest fractional
step.  Rubin does not mark this: it is not a defect, the pixels are all real, they
are just not equally deep.  Two things record it.  The coadd's own
``provenance.contributions`` says which visits went into which cell, and since
DP2 exposures share an integration time the ratio of counts across the cells a
stamp covers *is* the step, exactly, before any pixel is read; extraction passes
it in as ``cell_depth_ratio``.  That is not the whole story, because coaddition
is inverse-variance weighted and cells with equal counts still differ by whatever
the seeing and the sky did, so the *variance plane* is measured as well --
``variance_step`` -- the ratio between
the highest and lowest block-wise variance floor across the stamp.  The floor is
a low percentile within each block so that a source, which only ever pushes
variance up, cannot fake a step, and pixels the image shows to be source are dropped
outright; blocks are sized well under a cell so at least one lands wholly inside
each.

Tolerances for ``COSMIC_RAY`` and ``INTERPOLATED`` are tighter than the
documentation recommends, deliberately.  That guidance is written for
*measurement*, where an interpolated pixel barely perturbs a flux.  Here the
model is learning a distribution over pixel values, and an interpolated pixel is
a smooth synthetic patch teaching structure that is not in the sky.  Tight is not
the same as zero, though: see ``INNER_FRAC_TOL``.
"""

from __future__ import annotations

import numpy as np

#: Measured and recorded for every stamp, never gated on.  ``INEXACT_PSF`` and
#: ``REJECTED`` cover a large fraction of the DP2 coadd, so a cut on them keeps
#: almost nothing; ``DETECTED`` marks the targets.  Having the fractions in the
#: manifest means a cut can still be made later without re-reading pixels.
COVARIATE_PLANES: tuple[str, ...] = (
    "INEXACT_PSF",
    "REJECTED",
    "DETECTED",
    "DETECTED_NEGATIVE",
    "CLIPPED",
    "SUSPECT",
)

#: Never gate on these: they mark real sources, or cover so much of the coadd
#: that a cut keeps nothing.
NEVER_REJECT: tuple[str, ...] = COVARIATE_PLANES


#: Block side for the variance floors, as a fraction of the stamp.  At the
#: nominal 416 px stamp this is 52 px: well under a 150 px coadd cell, so at
#: least one block falls wholly inside each cell and a step between cells shows
#: up as a difference between blocks; and well over a galaxy -- a 1 arcsec
#: half-light radius is 5 native px -- so no source can fill a block and lift its
#: floor.
VARIANCE_BLOCK_FRACTION: float = 1 / 8

#: Percentile taken within each block.  Low, because source Poisson variance is
#: one-sided: at the 25th percentile a bright 3 arcsec galaxy reads as a step of
#: 1.9, which would reject exactly the well-resolved hosts the set is for; at the
#: 10th it reads as 1.16.  A real step is measured identically at any percentile
#: -- the ratio of two like quantiles is unbiased -- so the low one is free.
VARIANCE_FLOOR_PERCENTILE: float = 10.0

#: Pixels brighter than sky + this many sigma are excluded from the floors.
#: In sky-limited data a source only inflates the variance appreciably once its
#: flux approaches the sky per pixel, which is a detection at S/N of order
#: sqrt(sky counts) -- far above 3 -- so this removes every pixel where the
#: source could matter, with a wide margin, and leaves the sky pixels that carry
#: the depth information.
SOURCE_NSIGMA: float = 3.0


#: A block must keep this fraction of its pixels to report a floor at all; a
#: block covered by a galaxy has no sky in it to report.
MIN_USABLE_FRACTION: float = 0.5

#: Side of the central window the host is measured in, native pixels.  16 px is
#: 3.2 arcsec at the DP2 plate scale, comfortably inside a host selected at
#: reff >= 1.5 arcsec and small enough that an empty centre cannot hide behind
#: the galaxy's own outskirts.
CENTRE_WINDOW_PX: int = 16

#: A peak counts only this far above the sky, after smoothing.
PEAK_NSIGMA: float = 5.0

#: Box smoothing applied before finding peaks, native pixels.  Without it every
#: noise wiggle riding on a bright galaxy is a local maximum above the
#: threshold, and the count measures the *area* above 5 sigma rather than the
#: number of sources -- a smooth Gaussian galaxy counted 88 peaks that way.
PEAK_SMOOTH_PX: int = 3

#: A peak must be the largest pixel within this radius.  Roughly the PSF, so two
#: sources closer than this count once.  At radius 3 a smooth galaxy counts 1,
#: the same galaxy with six stars on it counts 7, and a field of 200 stars
#: counts 185; at radius 2 the galaxy alone already counts 7.
PEAK_RADIUS_PX: int = 3


def _box_smooth(a: np.ndarray, k: int) -> np.ndarray:
    """Separable box mean over a ``k x k`` window.  Wraps at the edges, which
    the border trim in ``count_peaks`` discards."""
    out = a
    for axis in (0, 1):
        acc = np.zeros_like(out)
        for shift in range(-(k // 2), k // 2 + 1):
            acc += np.roll(out, shift, axis=axis)
        out = acc / k
    return out


def _max_filter(a: np.ndarray, r: int) -> np.ndarray:
    """Separable maximum over a ``(2r+1) x (2r+1)`` window."""
    out = a
    for axis in (0, 1):
        m = out
        for k in range(1, r + 1):
            m = np.maximum(m, np.roll(out, k, axis=axis))
            m = np.maximum(m, np.roll(out, -k, axis=axis))
        out = m
    return out


def centre_brightness(image: np.ndarray, sky_noise: float) -> float:
    """Median of the central window, in units of the sky noise.

    The catalogue says a host is there; this asks the pixels.  DP2's Sersic fits
    are not always right, and a stamp centred on a fit that found nothing is an
    empty piece of sky that passed every magnitude and surface-brightness cut on
    the way in -- the cuts are on catalogue quantities, and if the catalogue is
    wrong they cannot help.

    The coadd is background-subtracted, so sky is zero and this is a
    signal-to-noise per pixel.  The median rather than the mean, so a single
    bright neighbour clipping the window cannot stand in for a host.
    """
    h, w = image.shape
    half = min(CENTRE_WINDOW_PX, h, w) // 2
    cy, cx = h // 2, w // 2
    centre = image[cy - half:cy + half, cx - half:cx + half]
    finite = centre[np.isfinite(centre)]
    if not finite.size or not np.isfinite(sky_noise) or sky_noise <= 0:
        return float("nan")
    return float(np.median(finite) / sky_noise)


def count_peaks(image: np.ndarray, sky_noise: float) -> int:
    """How many distinct sources are in the stamp.

    A crowded star field and a galaxy are both bright and both fill the
    ``DETECTED`` plane, so no plane fraction tells them apart.  The number of
    distinct peaks does: a host is one source with a handful of companions, and
    a field that is nothing but stars has hundreds of them.

    Smooth, then keep every pixel that is the largest within ``PEAK_RADIUS_PX``
    and more than ``PEAK_NSIGMA`` above the sky.  The radius is what matters:
    the naive version -- every 8-connected local maximum above the threshold --
    counts the noise riding on a bright galaxy and so measures how much of the
    stamp is bright rather than how many sources are in it.  Measured on one
    host: 102 peaks naively, 12 with the smoothing alone, 1 as shipped.  The
    smoothing earns its place by letting the radius stay near the PSF instead of
    having to grow to suppress noise.

    Separable filters over shifted views, so no scipy and a handful of passes.
    ``np.roll`` wraps, so the border is trimmed.  A saturated flat top counts
    once per tied pixel rather than once, which is a real overcount -- but a
    saturated core is gated on its own plane long before this matters.
    """
    if not np.isfinite(sky_noise) or sky_noise <= 0:
        return -1
    a = _box_smooth(np.where(np.isfinite(image), image, 0.0).astype(float),
                    PEAK_SMOOTH_PX)
    # Smoothing over k x k independent pixels divides the noise by k.
    hit = (a >= _max_filter(a, PEAK_RADIUS_PX)) & (
        a > PEAK_NSIGMA * sky_noise / PEAK_SMOOTH_PX)
    trim = PEAK_RADIUS_PX + PEAK_SMOOTH_PX
    return int(np.count_nonzero(hit[trim:-trim, trim:-trim]))


def variance_floors(variance: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Low envelope of the sky variance on a block grid, in reading order.

    Two defences against a source being read as depth, because it adds its own
    Poisson variance and that contamination is one-sided.  First, pixels the
    ``image`` shows to be above the sky are dropped outright -- a galaxy big
    enough to fill a block would otherwise defeat any percentile.  Second, what
    is taken within a block is a low percentile rather than a mean or median.
    """
    v = np.asarray(variance, dtype=float)
    h, w = v.shape
    block = max(int(round(min(h, w) * VARIANCE_BLOCK_FRACTION)), 8)
    usable = np.isfinite(v) & (v > 0)
    im = np.asarray(image, dtype=float)
    sky = float(np.median(im[usable])) if usable.any() else 0.0
    with np.errstate(invalid="ignore"):
        usable &= np.isfinite(im) & (im < sky + SOURCE_NSIGMA * np.sqrt(v))
    floors: list[float] = []
    for y0 in range(0, h - block + 1, block):
        for x0 in range(0, w - block + 1, block):
            sl = (slice(y0, y0 + block), slice(x0, x0 + block))
            good = v[sl][usable[sl]]
            if good.size < MIN_USABLE_FRACTION * block * block:
                continue
            floors.append(float(np.percentile(good, VARIANCE_FLOOR_PERCENTILE)))
    return np.asarray(floors, dtype=float)


def variance_step(variance: np.ndarray, image: np.ndarray) -> float:
    """Ratio of the highest to the lowest block variance floor.

    1.0 is a uniform stamp; a coadd cell boundary shows as the ratio of the two
    cells' depths.  NaN when fewer than two blocks are usable, which is a lack of
    information and not a defect -- the gate treats it as a pass.
    """
    floors = variance_floors(variance, image)
    if floors.size < 2 or floors.min() <= 0:
        return float("nan")
    return float(floors.max() / floors.min())


def plane_bitmask(plane_dict: dict[str, int], names) -> int:
    """OR of the bits for ``names`` that actually exist in ``plane_dict``."""
    if isinstance(names, str):
        names = [names]
    bits = 0
    for n in names:
        if n in plane_dict:
            bits |= 1 << int(plane_dict[n])
    return bits


def plane_fraction(mask: np.ndarray, plane_dict: dict[str, int], name: str) -> float:
    """Fraction of pixels with ``name`` set; 0.0 if the plane does not exist."""
    bit = plane_bitmask(plane_dict, name)
    if bit == 0:
        return 0.0
    return float(np.mean((mask & bit) != 0))


def plane_fractions(mask: np.ndarray, plane_dict: dict[str, int]) -> dict[str, float]:
    """Every plane's fraction, for the manifest."""
    return {name: plane_fraction(mask, plane_dict, name) for name in plane_dict}


def gate(
    image: np.ndarray,
    variance: np.ndarray,
    mask: np.ndarray,
    plane_dict: dict[str, int],
    *,
    zero_tol: tuple[str, ...],
    frac_tol: dict[str, float],
    inner_frac_tol: dict[str, float],
    inner_fraction: float,
    max_no_data: float,
    max_inner_no_data: float,
    max_variance_step: float,
    max_cell_depth_ratio: float,
    min_visits: int | None,
    min_centre_sigma: float | None,
    max_peaks: int | None,
    cell_depth_ratio: float,
    n_visits: int,
) -> tuple[list[str], dict[str, float]]:
    """Return ``(rejection_reasons, diagnostics)``.  Empty reasons means accept.

    None of the tolerances have defaults.  They come from ``extraction.yaml``
    via ``PatchCuts.gate_kwargs``, which is then the only place in the codebase
    that says what a cut is; a default here would be a second answer to that
    question, and the two would drift.

    Diagnostics are returned whether or not the patch passes, so the manifest
    records them for rejected patches too -- the rejection statistics are how you
    find out whether the selection function is biased against bright, dense
    galaxy centres, which is exactly the regime this project cares about.
    """
    bad = set(frac_tol) & set(NEVER_REJECT)
    if bad:
        raise ValueError(f"refusing to gate on {sorted(bad)}: these mark real sources")

    # The documented DP1->DP2 failure mode: the planes were renamed, so a stale
    # gate matches nothing and silently passes every stamp.  Absent names are
    # meant to contribute nothing, but *all* of them being absent means the
    # mapping is wrong, not that the data is clean.
    wanted = set(zero_tol) | set(frac_tol) | set(inner_frac_tol)
    if wanted and not (wanted & set(plane_dict)):
        raise ValueError(
            f"none of the gated planes {sorted(wanted)} appear in the mask "
            f"mapping {sorted(plane_dict)}. DP2 renamed SAT->SATURATED, "
            f"CR->COSMIC_RAY, INTRP->INTERPOLATED, EDGE->DETECTION_EDGE; a "
            f"stale gate matches nothing and passes everything."
        )

    reasons: list[str] = []
    diag: dict[str, float] = {}

    if not np.all(np.isfinite(image)):
        reasons.append("nonfinite_image")

    # DP2 marks "no contributing exposures" with inf variance rather than with a
    # mask plane, and that includes the cores of saturated stars.  Measure the
    # fraction; rejecting on its mere presence would discard every stamp with a
    # bright neighbour.
    no_data = ~np.isfinite(variance) | (variance <= 0)
    frac_no_data = float(np.mean(no_data))
    diag["frac_no_data"] = frac_no_data
    if frac_no_data > max_no_data:
        reasons.append(f"no_data:{frac_no_data:.4f}>{max_no_data}")

    h, w = np.shape(mask)
    ih, iw = max(int(h * inner_fraction), 1), max(int(w * inner_fraction), 1)
    y0, x0 = (h - ih) // 2, (w - iw) // 2
    inner_slice = (slice(y0, y0 + ih), slice(x0, x0 + iw))
    inner_no_data = float(np.mean(no_data[inner_slice]))
    diag["inner_frac_no_data"] = inner_no_data
    if inner_no_data > max_inner_no_data:
        reasons.append(f"inner_no_data:{inner_no_data:.4f}>{max_inner_no_data}")

    # The exact statement of the same thing, when extraction could get it from
    # the coadd's provenance: visits share an integration time, so the ratio of
    # visit counts across the cells a stamp covers *is* the depth step, known
    # without looking at a pixel.  It is not a replacement for the measured
    # version -- coadds are inverse-variance weighted, so cells with equal counts
    # still differ by whatever the seeing and sky did -- so both run.
    # Absolute depth, as opposed to how much it varies across the stamp.  Early
    # DP2 outside the deep fields is 1-3 visits per cell, which is a different
    # sky from a deep coadd; off by default, because whether that is too shallow
    # is a judgement about the prior and not about the pixels.
    diag["n_visits_min"] = float(n_visits)
    if min_visits is not None and n_visits < min_visits:
        reasons.append(f"too_shallow:{n_visits}<{min_visits}")

    diag["cell_depth_ratio"] = float(cell_depth_ratio)
    if cell_depth_ratio > max_cell_depth_ratio:
        reasons.append(f"cell_depth:{cell_depth_ratio:.2f}>{max_cell_depth_ratio}")

    # Cell-based coadds step in depth at cell edges and nothing flags it; see the
    # module docstring.  Gate on it, because a stamp with a straight noise
    # boundary through it teaches the model that the sky does that.
    step = variance_step(variance, image)
    diag["variance_step"] = step
    if np.isfinite(step) and step > max_variance_step:
        reasons.append(f"variance_step:{step:.2f}>{max_variance_step}")

    finite = variance[~no_data]
    sky_noise = (float(np.sqrt(np.median(finite))) if finite.size
                 else float("nan"))
    diag["sky_noise"] = sky_noise

    # Is there actually a host here, and is it a galaxy?  Every cut before this
    # one is on a catalogue quantity, so a wrong Sersic fit sails through all of
    # them: the magnitude and surface-brightness limits are statements about
    # what the pipeline measured, not about what is in the pixels.
    diag["centre_sigma"] = centre_brightness(image, sky_noise)
    if (min_centre_sigma is not None and np.isfinite(diag["centre_sigma"])
            and diag["centre_sigma"] < min_centre_sigma):
        reasons.append(f"empty_centre:{diag['centre_sigma']:.2f}<{min_centre_sigma}")

    diag["n_peaks"] = float(count_peaks(image, sky_noise))
    if (max_peaks is not None and diag["n_peaks"] >= 0
            and diag["n_peaks"] > max_peaks):
        reasons.append(f"crowded:{int(diag['n_peaks'])}>{max_peaks}")

    zero_bit = plane_bitmask(plane_dict, zero_tol)
    if zero_bit and np.any(mask & zero_bit):
        present = [n for n in zero_tol if plane_fraction(mask, plane_dict, n) > 0]
        reasons.append("zero_tol:" + "+".join(present))

    for plane, tol in sorted(frac_tol.items()):
        f = plane_fraction(mask, plane_dict, plane)
        diag[f"frac_{plane}"] = f
        if f > tol:
            reasons.append(f"{plane}:{f:.4f}>{tol}")

    inner = mask[inner_slice]
    for plane, tol in sorted(inner_frac_tol.items()):
        f = plane_fraction(inner, plane_dict, plane)
        diag[f"inner_frac_{plane}"] = f
        if f > tol:
            reasons.append(f"inner_{plane}:{f:.4f}>{tol}")

    # Recorded, never gated on -- see COVARIATE_PLANES.
    for plane in COVARIATE_PLANES:
        diag[f"frac_{plane}"] = plane_fraction(mask, plane_dict, plane)

    return reasons, diag
