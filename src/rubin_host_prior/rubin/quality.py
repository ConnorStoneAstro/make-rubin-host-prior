"""Artefact rejection for candidate patches.  Pure numpy -- no LSST stack.

Two things about DP1 make the obvious approach fail.

**Half the mask planes are never set in ``visit_image``**: ``CLIPPED``,
``DETECTED_NEGATIVE``, ``INEXACT_PSF``, ``NO_DATA``, ``REJECTED``,
``SENSOR_EDGE``, ``STREAK``, ``UNMASKEDNAN``, ``VIGNETTED``.  A gate built on
them silently passes everything, so ``gate`` tests pixel finiteness and variance
positivity directly, and satellite trails have to come from the matching
``difference_image`` mask (see ``extract.StreakCache``) or from your own
detection.

**Several artefacts have no mask plane at all**: stray light, ghosts, amplifier
jumps, fringing, tree rings, crosshatch, and -- the dangerous ones here --
*dark edge* and *dark halo*, which are background **over-subtraction**.  Those
put a smooth negative bowl into exactly the low-surface-brightness regime this
project cares about, and a prior trained on them learns that galaxies sit in
negative bowls.  ``background_floor`` exists to catch them.

Tolerances differ from the DP1 documentation's recommendations on purpose.  The
published table is written for *measurement*, where an interpolated cosmic ray
is harmless because it barely perturbs a flux.  Here the model is learning a
distribution over pixel values, and an interpolated pixel is a smooth synthetic
patch that teaches the model structure that is not in the sky.  ``CR`` and
``INTRP`` are therefore tighter than the documentation suggests.  All fractions
are recorded in the manifest, so they can be loosened later without re-reading
pixels.
"""

from __future__ import annotations

import numpy as np

#: Any pixel set in these planes disqualifies the patch.  ``SENSOR_EDGE`` is
#: listed defensively -- DP1 never sets it in ``visit_image``, so it costs
#: nothing here, but it should not be *relied* on (hence ``EDGE`` alongside).
ZERO_TOL: tuple[str, ...] = ("EDGE", "ITL_DIP", "SENSOR_EDGE")

#: Maximum allowed fraction of the patch, per plane.  Stricter than the DP1
#: measurement recommendations for CR/INTRP -- see the module docstring.
FRAC_TOL: dict[str, float] = {
    "SAT": 0.0,  # saturation bleeds; the DP1 docs say exclude outright
    "BAD": 0.005,
    "SUSPECT": 0.005,
    "CR": 0.005,  # docs say "retain"; too generous for a generative model
    "INTRP": 0.02,  # docs say "retain"; likewise
    "CROSSTALK": 0.02,
}

#: Same planes, applied to the central region, where structure matters most.
INNER_FRAC_TOL: dict[str, float] = {
    "SAT": 0.0,
    "BAD": 0.0,
    "SUSPECT": 0.0,
    "CR": 0.0,
    "INTRP": 0.0,
}

#: Never gate on this: it marks real sources, and rejecting on it throws away
#: every patch that contains a galaxy.
NEVER_REJECT: tuple[str, ...] = ("DETECTED", "DETECTED_NEGATIVE")


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


def _norm_ppf(q: float) -> float:
    """Inverse standard normal CDF by bisection; avoids a scipy dependency."""
    from math import erf, sqrt

    lo, hi = -8.0, 8.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + erf(mid / sqrt(2.0))) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def background_floor(
    image: np.ndarray,
    sky_noise: float,
    mask: np.ndarray | None = None,
    plane_dict: dict[str, int] | None = None,
    n_blocks: int = 8,
    percentile: float = 25.0,
) -> dict[str, float]:
    """Locate the sky *floor*, globally and per block, in units of the sky noise.

    ``visit_image`` is background-subtracted, so the true smooth background is
    zero everywhere.  Anything significantly **below** zero is over-subtraction:
    dark edge, or the dark halo around a bright star.  Those matter more here
    than in most analyses, because they deposit a smooth negative bowl into
    exactly the low-surface-brightness regime the model is meant to represent,
    and a prior trained on them learns that galaxies sit in negative bowls.

    The estimator is a low percentile per block of a coarse grid, corrected for
    the percentile's own offset under Gaussian noise so that blank sky reads
    zero.  Deliberately **not** a fitted surface: a quadratic fit to a patch
    containing a bright galaxy absorbs the galaxy and then extrapolates strongly
    negative towards the corners, reporting a bowl that is not there --
    rejecting precisely the bright, dense hosts this project exists to model.
    A low percentile is blind to positive sources by construction and still
    tracks a real depression.

    ``min_block`` is the number to gate on.  Its noise floor is roughly
    ``-2.5 * sqrt(q(1-q)/n_pix_per_block) / phi(z_q)`` sky-noise units (about
    -0.15 for the defaults), so a threshold near -0.3 has headroom.
    """
    image = np.asarray(image, dtype=np.float64)
    h, w = image.shape
    good = np.isfinite(image)
    if mask is not None and plane_dict is not None:
        bit = plane_bitmask(plane_dict, ("SAT", "BAD", "EDGE", "INTRP"))
        if bit:
            good &= (mask & bit) == 0

    offset = _norm_ppf(percentile / 100.0) * sky_noise
    nan = {
        "sky_floor": np.nan,
        "min_block": np.nan,
        "max_block": np.nan,
        "n_background_blocks": 0,
    }
    if good.sum() < 64 or not np.isfinite(sky_noise) or sky_noise <= 0:
        return nan

    nb = max(int(n_blocks), 2)
    bh, bw = max(h // nb, 1), max(w // nb, 1)
    min_pix = max(bh * bw // 4, 16)
    blocks = []
    for by in range(0, h - bh + 1, bh):
        for bx in range(0, w - bw + 1, bw):
            sel = good[by : by + bh, bx : bx + bw]
            if sel.sum() < min_pix:
                continue
            block = image[by : by + bh, bx : bx + bw][sel]
            blocks.append((np.percentile(block, percentile) - offset) / sky_noise)
    if len(blocks) < 8:
        return nan
    return {
        "sky_floor": float(
            (np.percentile(image[good], percentile) - offset) / sky_noise
        ),
        "min_block": float(np.min(blocks)),
        "max_block": float(np.max(blocks)),
        "n_background_blocks": len(blocks),
    }


def gate(
    image: np.ndarray,
    variance: np.ndarray,
    mask: np.ndarray,
    plane_dict: dict[str, int],
    sky_noise: float | None = None,
    inner_fraction: float = 0.34,
    zero_tol: tuple[str, ...] = ZERO_TOL,
    frac_tol: dict[str, float] | None = None,
    inner_frac_tol: dict[str, float] | None = None,
    max_depression: float | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Return ``(rejection_reasons, diagnostics)``.  Empty reasons means accept.

    Diagnostics are returned whether or not the patch passes, so the manifest
    records them for rejected patches too -- the rejection statistics are how you
    find out whether the selection function is biased against bright, dense
    galaxy centres, which is exactly the regime this project cares about.
    """
    frac_tol = FRAC_TOL if frac_tol is None else frac_tol
    inner_frac_tol = INNER_FRAC_TOL if inner_frac_tol is None else inner_frac_tol
    bad = set(frac_tol) & set(NEVER_REJECT)
    if bad:
        raise ValueError(f"refusing to gate on {sorted(bad)}: these mark real sources")

    reasons: list[str] = []
    diag: dict[str, float] = {}

    # DP1 leaves NO_DATA and UNMASKEDNAN unset, so test the pixels themselves.
    if not np.all(np.isfinite(image)):
        reasons.append("nonfinite_image")
    if not np.all(np.isfinite(variance)):
        reasons.append("nonfinite_variance")
    elif np.any(variance <= 0):
        reasons.append("nonpositive_variance")

    if sky_noise is None:
        finite = variance[np.isfinite(variance) & (variance > 0)]
        sky_noise = float(np.sqrt(np.median(finite))) if finite.size else np.nan
    diag["sky_noise"] = float(sky_noise)

    zero_bit = plane_bitmask(plane_dict, zero_tol)
    if zero_bit and np.any(mask & zero_bit):
        present = [n for n in zero_tol if plane_fraction(mask, plane_dict, n) > 0]
        reasons.append("zero_tol:" + "+".join(present))

    for plane, tol in sorted(frac_tol.items()):
        f = plane_fraction(mask, plane_dict, plane)
        diag[f"frac_{plane}"] = f
        if f > tol:
            reasons.append(f"{plane}:{f:.4f}>{tol}")

    h, w = mask.shape
    ih, iw = max(int(h * inner_fraction), 1), max(int(w * inner_fraction), 1)
    y0, x0 = (h - ih) // 2, (w - iw) // 2
    inner = mask[y0 : y0 + ih, x0 : x0 + iw]
    for plane, tol in sorted(inner_frac_tol.items()):
        f = plane_fraction(inner, plane_dict, plane)
        diag[f"inner_frac_{plane}"] = f
        if f > tol:
            reasons.append(f"inner_{plane}:{f:.4f}>{tol}")

    if np.isfinite(sky_noise) and sky_noise > 0 and np.all(np.isfinite(image)):
        bg = background_floor(image, sky_noise, mask, plane_dict)
        diag.update(bg)
        # ``max_depression=None`` (the default) measures the sky floor and
        # records it without rejecting anything: the data is taken as-is,
        # background-subtraction artefacts included, and the prior is allowed to
        # learn them.  That is the right default when the artefacts are a
        # property of the current processing that a later data release will
        # improve -- you retrain rather than filter.
        #
        # Set a number to reject on it.  One-sided on purpose: a depressed sky
        # floor is over-subtraction, a raised one is starlight, and gating on
        # the magnitude would discard the brightest hosts.  Keeping depressed
        # regions costs nothing in the log transform: softplus softening has
        # no floor, so however negative a pixel goes it stays representable.
        if (
            max_depression is not None
            and np.isfinite(bg["min_block"])
            and bg["min_block"] < -max_depression
        ):
            reasons.append(
                f"background_depression:{bg['min_block']:.2f}<-{max_depression}"
            )

    return reasons, diag
