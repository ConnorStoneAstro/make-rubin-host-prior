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

Tolerances for ``COSMIC_RAY`` and ``INTERPOLATED`` are tighter than the
documentation recommends, deliberately.  That guidance is written for
*measurement*, where an interpolated pixel barely perturbs a flux.  Here the
model is learning a distribution over pixel values, and an interpolated pixel is
a smooth synthetic patch teaching structure that is not in the sky.
"""

from __future__ import annotations

import numpy as np

#: Any pixel set in these planes disqualifies the patch.  ``NO_DATA`` is one of
#: the two Rubin says to exclude outright (tutorial 202.5): no input covered the
#: pixel, so it is not sky, it is nothing.  ``DETECTION_EDGE`` means too near the
#: patch edge for the detection kernel.
ZERO_TOL: tuple[str, ...] = ("NO_DATA", "DETECTION_EDGE")

#: Maximum allowed fraction of the patch, per plane.  Stricter than the DP1
#: measurement recommendations for CR/INTRP -- see the module docstring.
FRAC_TOL: dict[str, float] = {
    # Rubin's guidance is to exclude SATURATED outright, which for *pixels* in a
    # measurement is right.  For whole training stamps it is not: on a coadd the
    # saturated core of a bright neighbour lands in a great many stamps, and a
    # scene with a bright neighbour is precisely the regime this project models,
    # so a blanket cut would reproduce the selection bias the rejection
    # statistics exist to expose.  A small fraction is kept away from the centre,
    # zero is kept at the centre where the transient goes, and the fraction is
    # recorded either way.  Tighten to 0.0 to follow the guidance literally.
    "SATURATED": 0.005,
    "COSMIC_RAY": 0.005,  # mostly rejected during coaddition; a safety net
    "INTERPOLATED": 0.02,  # smooth synthetic fill; bad for a generative model
}

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

#: Same planes, applied to the central region, where structure matters most.
INNER_FRAC_TOL: dict[str, float] = {
    "SATURATED": 0.0,
    "COSMIC_RAY": 0.0,
    "INTERPOLATED": 0.0,
}

#: Never gate on these: they mark real sources, or cover so much of the coadd
#: that a cut keeps nothing.
NEVER_REJECT: tuple[str, ...] = COVARIATE_PLANES


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
    max_no_data: float = 0.02,
    max_inner_no_data: float = 0.0,
    require_known_planes: bool = True,
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

    # The documented DP1->DP2 failure mode: the planes were renamed, so a stale
    # gate matches nothing and silently passes every stamp.  Absent names are
    # meant to contribute nothing, but *all* of them being absent means the
    # mapping is wrong, not that the data is clean.
    if require_known_planes:
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

    if sky_noise is None:
        finite = variance[~no_data]
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
