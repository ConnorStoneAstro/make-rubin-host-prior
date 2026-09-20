"""Artefact rejection for candidate patches.  Pure numpy -- no LSST stack.

Tuned for ``deep_coadd``, which is what this project trains on.  Coadds already
handle most of what made visit images awkward: chip edges, cosmic rays and other
per-exposure electronic artefacts are rejected during coaddition rather than
left for a downstream gate to find.

What remains is specific to coadds:

* **``NO_DATA`` matters here and did not before.**  A coadd patch has regions
  with no contributing exposures -- corners, gaps, the edge of the survey
  footprint -- and those pixels are not sky, they are nothing.
* **``CLIPPED`` and ``REJECTED``** mark pixels where outlier rejection fired
  during coaddition.  A little is normal; a lot means the stack disagreed with
  itself there.
* **``INEXACT_PSF``** marks where the coadd PSF model is approximate.  That
  matters for a project whose forward model needs the PSF.

Several planes are never set in ``deep_coadd`` -- ``BAD``, ``CROSSTALK``,
``DETECTED_NEGATIVE``, ``ITL_DIP``, ``NOT_DEBLENDED``, ``STREAK``,
``UNMASKEDNAN``, ``VIGNETTED`` -- so gating on them would do nothing.  They are
left out rather than listed for show.  ``plane_bitmask`` ignores names that are
absent from the mask's own dictionary, so the gate degrades safely if a later
release starts or stops setting one.

Tolerances differ from the DP1 documentation's recommendations on purpose.  The
published table is written for *measurement*, where an interpolated pixel is
harmless because it barely perturbs a flux.  Here the model is learning a
distribution over pixel values, and an interpolated pixel is a smooth synthetic
patch that teaches the model structure that is not in the sky.  ``CR`` and
``INTRP`` are therefore tighter than the documentation suggests.  All fractions
are recorded in the manifest, so they can be loosened later without re-reading
pixels.
"""

from __future__ import annotations

import numpy as np

#: Any pixel set in these planes disqualifies the patch.  ``NO_DATA`` is the
#: important one for coadds: those pixels had no contributing exposures.
ZERO_TOL: tuple[str, ...] = ("NO_DATA", "EDGE", "SENSOR_EDGE")

#: Maximum allowed fraction of the patch, per plane.  Stricter than the DP1
#: measurement recommendations for CR/INTRP -- see the module docstring.
FRAC_TOL: dict[str, float] = {
    "SAT": 0.0,  # saturation bleeds; the DP1 docs say exclude outright
    "CR": 0.005,  # mostly rejected during coaddition, so this is a safety net
    "INTRP": 0.02,  # smooth synthetic fill; bad for a generative model
    "CLIPPED": 0.02,  # outlier rejection fired during coaddition
    "REJECTED": 0.02,
    "INEXACT_PSF": 0.05,  # the forward model needs a trustworthy PSF
}

#: Same planes, applied to the central region, where structure matters most.
INNER_FRAC_TOL: dict[str, float] = {
    "SAT": 0.0,
    "CR": 0.0,
    "INTRP": 0.0,
    "CLIPPED": 0.0,
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

    return reasons, diag
