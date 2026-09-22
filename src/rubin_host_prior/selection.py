"""Every cut, in one place.

The cuts used to be scattered across four files: defaults on ``select_hosts``,
defaults on ``host_adql``, module constants in ``rubin.quality``, and command
line flags that sometimes overrode one and not the other.  Changing what counts
as a host meant editing code in several places and hoping they agreed.  This is
the whole selection, host and pixel, in one serialisable object.

    python scripts/extract_dp2_patches.py --write-selection selection.json
    $EDITOR selection.json
    python scripts/extract_dp2_patches.py --selection selection.json ...

**The cuts, and why they are these.**

*Visibility is total magnitude, not surface brightness.*  Mean surface
brightness inside the half-light ellipse decides whether a fit is real, but it
is a poor proxy for "I can see it": a tight cut on it selects *concentrated*
light, which is the opposite of what a prior over galaxy structure wants.  A
galaxy is visible when it has enough total flux, so ``max_mag`` is the primary
cut and ``max_mu_e`` is left loose, as a bound on runaway fits rather than a
selector.

*Extent is angular size referenced to the PSF.*  ``min_reff_arcsec`` is the
intrinsic half-light radius from the multiband Sersic fit; ``min_deconvolved_px``
is the same claim made against the image, ``T^2 = ((ixx+iyy)-(ixxPSF+iyyPSF))/2``,
which a point source makes exactly zero regardless of the seeing.

*The pixel cuts are tuned for shallow, ragged coverage.*  Early DP2 outside the
deep fields runs one visit per cell, and the surveyed region has edges
everywhere, so a stamp is far more likely to clip a coverage boundary than to
contain an artefact.  Gating a plane at zero tolerance when the same quantity is
also measured as a fraction just means the stricter of the two always wins and
the tolerance is decorative.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

#: m = -2.5*log10(f_nJy) + AB_ZEROPOINT
AB_ZEROPOINT = 31.4


def mag_to_flux(mag: float) -> float:
    """AB magnitude to nJy."""
    return float(10 ** ((AB_ZEROPOINT - mag) / 2.5))


def flux_to_mag(flux: float) -> float:
    """nJy to AB magnitude."""
    return float(AB_ZEROPOINT - 2.5 * __import__("math").log10(flux))


@dataclass
class HostCuts:
    """Which catalogue rows may become hosts.

    Everything here is expressed the way it is thought about -- magnitudes,
    arcseconds, mag/arcsec^2 -- and converted to the catalogue's nJy where the
    query is built.
    """

    #: Band the cuts are evaluated in.
    band: str = "r"

    #: Faint limit.  This is the visibility cut: a host fainter than this does
    #: not have the flux to show structure however large its fitted radius.
    max_mag: float = 20.5
    #: Bright limit.  Nominal only -- a saturated core is caught by the flag
    #: below, which is the real limit and does not depend on guessing where
    #: saturation sets in.
    min_mag: float = 11.5

    #: Half-light major axis of the multiband Sersic fit, arcsec.  Intrinsic,
    #: i.e. before PSF convolution.
    min_reff_arcsec: float = 1.5
    max_reff_arcsec: float = 30.0

    #: Faintest mean surface brightness inside the half-light ellipse,
    #: mag/arcsec^2.  Loose on purpose: this is a bound on fits that ran away
    #: around an invisible envelope, not a way of choosing galaxies.  One sigma
    #: of r-band sky per square arcsecond is about 27.
    max_mu_e: float | None = 25.5

    #: PSF-deconvolved moment radius, native pixels.  A point source is exactly
    #: zero; 2.5 px is 0.5 arcsec of genuine extent.
    min_deconvolved_px: float | None = 2.5

    #: Sersic index ceiling.  Above about 6 the profile has no measurable size.
    max_sersic_index: float | None = 6.0
    #: Flux-ratio extendedness, 0 or 1 in DP2.
    min_extendedness: float | None = 0.5
    #: Fraction of the flux in the footprint coming from neighbours.
    max_blendedness: float | None = None

    #: Reject a host whose core is saturated or interpolated.  The first is the
    #: real bright limit; the second is synthetic structure sitting exactly
    #: where the transient goes.
    reject_saturated_centre: bool = True
    reject_interpolated_centre: bool = True

    #: Collapse catalogue rows closer than this, in arcsec.  DP2 has no
    #: detect_isPrimary and overlapping tracts give one source two objectIds.
    dedupe_radius_arcsec: float = 0.5
    #: Draw equally from bins of equal width in log size, over the fixed range
    #: above rather than over the sample's own tail.
    size_stratified: bool = True
    n_size_bins: int = 5

    @property
    def flux_range(self) -> tuple[float, float]:
        """(faint, bright) in nJy, which is what the catalogue stores."""
        return mag_to_flux(self.max_mag), mag_to_flux(self.min_mag)

    def surface_brightness_floor(self) -> float | None:
        """nJy per arcsec^2 of half-light ellipse at ``max_mu_e``.

        So the cut can be written ``flux >= K * a * b``: multiplication only,
        which every ADQL dialect has, where ``LOG10`` and ``POWER`` are not
        guaranteed.
        """
        if self.max_mu_e is None:
            return None
        import math

        return 2.0 * math.pi * mag_to_flux(self.max_mu_e)

    def describe(self) -> str:
        """The locus the cuts define, at a few sizes.

        Size and brightness are not independent -- ``mu_e = m + 2.5 log10(2 pi a
        b)`` -- so a magnitude limit and a surface-brightness limit can quietly
        exclude each other over the range that matters.  This prints where each
        one binds.
        """
        import math

        lines = [f"  faint limit {self.max_mag:.1f} mag"
                 f"   ({self.flux_range[0]:.0f} nJy)"]
        if self.max_mu_e is not None:
            lines.append(f"  surface brightness <= {self.max_mu_e:.1f} mag/arcsec^2")
            lines.append("  reff    mag from SB   binding cut")
            for a in (1.0, 2.0, 4.0, 10.0, 20.0):
                if not (self.min_reff_arcsec <= a <= self.max_reff_arcsec):
                    continue
                b = 0.6 * a
                m_sb = self.max_mu_e - 2.5 * math.log10(2 * math.pi * a * b)
                which = "magnitude" if self.max_mag < m_sb else "surface brightness"
                lines.append(f"  {a:5.1f}\"  {m_sb:9.2f}   {which}")
        return "\n".join(lines)


@dataclass
class PatchCuts:
    """Which stamps survive, given the pixels.

    Tolerances are fractions of the stamp, and of its central region where
    structure matters most and where the transient goes.  ``None`` disables a
    cut outright; zero means no pixel of that plane is allowed.
    """

    #: Planes no pixel of which may be set.  DETECTION_EDGE means too near the
    #: patch boundary for the detection kernel to have been valid.
    #:
    #: NO_DATA is deliberately *not* here.  It was, and the same quantity is
    #: also measured from the inf-variance fraction with a 2% tolerance, so the
    #: zero-tolerance plane always won and the tolerance never did anything.  On
    #: ragged early-DP2 coverage that rejected most of the stamps that clipped a
    #: survey edge at all.
    zero_tolerance_planes: tuple[str, ...] = ("DETECTION_EDGE",)

    #: Fraction of the whole stamp, per plane.
    max_plane_fraction: dict[str, float] = field(default_factory=lambda: {
        "SATURATED": 0.005,
        "COSMIC_RAY": 0.005,
        "INTERPOLATED": 0.02,
    })
    #: Same planes over the central region.  A COSMIC_RAY pixel on a coadd is
    #: real data built from the inputs that survived; an INTERPOLATED one is
    #: invented, and a SATURATED core makes the stamp useless for the thing it
    #: is collected for.
    max_inner_plane_fraction: dict[str, float] = field(default_factory=lambda: {
        "SATURATED": 0.0,
        "COSMIC_RAY": 0.005,
        "INTERPOLATED": 0.001,
    })
    #: Linear fraction of the stamp treated as "inner".
    inner_fraction: float = 0.34

    #: Pixels with no coverage, measured from the inf-variance plane.
    max_no_data: float = 0.02
    max_inner_no_data: float = 0.0

    #: Depth uniformity.  ``cell_depth_ratio`` is exact, from the coadd
    #: provenance; ``variance_step`` is the same thing measured off the pixels
    #: and also sees weight differences the visit counts cannot.
    #:
    #: At one visit per cell -- which is most of early DP2 outside the deep
    #: fields -- a single-visit difference between neighbours is a ratio of 2 or
    #: 3, so a threshold chosen for deep coadds rejects nearly everything.
    max_cell_depth_ratio: float | None = 3.5
    max_variance_step: float | None = 4.0
    #: Absolute depth: visits in the shallowest cell the stamp covers.
    min_visits: int | None = None

    def __post_init__(self) -> None:
        # JSON has no tuples, so a loaded file would otherwise compare unequal
        # to the object it was written from.
        self.zero_tolerance_planes = tuple(self.zero_tolerance_planes)

    def gate_kwargs(self) -> dict:
        """As ``rubin.quality.gate`` takes them."""
        return {
            "zero_tol": tuple(self.zero_tolerance_planes),
            "frac_tol": dict(self.max_plane_fraction),
            "inner_frac_tol": dict(self.max_inner_plane_fraction),
            "inner_fraction": self.inner_fraction,
            "max_no_data": self.max_no_data,
            "max_inner_no_data": self.max_inner_no_data,
            "max_cell_depth_ratio": (float("inf") if self.max_cell_depth_ratio
                                     is None else self.max_cell_depth_ratio),
            "max_variance_step": (float("inf") if self.max_variance_step is None
                                  else self.max_variance_step),
            "min_visits": self.min_visits,
        }


@dataclass
class Selection:
    """Host cuts and patch cuts together: the whole selection function."""

    hosts: HostCuts = field(default_factory=HostCuts)
    patches: PatchCuts = field(default_factory=PatchCuts)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(dataclasses.asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Selection":
        raw = json.loads(Path(path).read_text())
        unknown = set(raw) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown selection sections {sorted(unknown)}")
        return cls(
            hosts=_build(HostCuts, raw.get("hosts", {})),
            patches=_build(PatchCuts, raw.get("patches", {})),
        )

    def describe(self) -> str:
        return "host cuts:\n" + self.hosts.describe()


def _build(cls, raw: dict):
    """Construct a cuts dataclass, refusing a key it does not have.

    A typo in a hand-edited selection file must not silently leave the default
    in place -- that is a cut that looks applied and is not.
    """
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"{cls.__name__} has no field(s) {sorted(unknown)}; it has "
            f"{sorted(known)}"
        )
    return cls(**raw)
