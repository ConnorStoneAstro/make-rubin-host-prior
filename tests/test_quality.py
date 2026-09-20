"""The artefact gate.

The governing requirement: reject over-subtraction artefacts while accepting
every galaxy, however bright.  A gate that rejects dense bright centres biases
the training set against precisely the regime this project exists to model.
"""

import numpy as np
import pytest

from rubin_host_prior.rubin.quality import (
    FRAC_TOL,
    NEVER_REJECT,
    gate,
    plane_bitmask,
    plane_fraction,
    plane_fractions,
)

# The DP1 r29.2.0 bit assignments.
#: DP1 r29.2.0 bit assignments.  Several of these are never set in deep_coadd
#: (BAD, CROSSTALK, DETECTED_NEGATIVE, ITL_DIP, NOT_DEBLENDED, STREAK,
#: SUSPECT, UNMASKEDNAN, VIGNETTED) -- kept in the dictionary so the tests can
#: check that gating on an unpopulated plane is a no-op rather than an error.
PLANES = {
    "BAD": 0, "SAT": 1, "INTRP": 2, "CR": 3, "EDGE": 4, "DETECTED": 5,
    "DETECTED_NEGATIVE": 6, "SUSPECT": 7, "NO_DATA": 8, "VIGNETTED": 9,
    "STREAK": 10, "CLIPPED": 11, "CROSSTALK": 12, "INEXACT_PSF": 13,
    "ITL_DIP": 14, "NOT_DEBLENDED": 15, "REJECTED": 16, "SENSOR_EDGE": 17,
    "UNMASKEDNAN": 18,
}
SIZE, SKY = 192, 12.0
_Y, _X = np.mgrid[0:SIZE, 0:SIZE]
R2 = ((_X - SIZE / 2) ** 2 + (_Y - SIZE / 2) ** 2) / (SIZE / 2) ** 2


def _scene(extra=0.0, seed=0):
    return np.random.default_rng(seed).normal(0.0, SKY, (SIZE, SIZE)) + extra


def _gate(image, mask=None, **kw):
    mask = np.zeros((SIZE, SIZE), np.uint32) if mask is None else mask
    return gate(image, np.full((SIZE, SIZE), SKY**2), mask, PLANES, **kw)


# -- plane helpers ---------------------------------------------------------


def test_plane_bitmask_ignores_absent_planes():
    assert plane_bitmask(PLANES, "SAT") == 2
    assert plane_bitmask(PLANES, ["BAD", "CR"]) == 1 | 8
    assert plane_bitmask({"BAD": 0}, "STREAK") == 0  # DP1 leaves many unset


def test_plane_fraction_of_absent_plane_is_zero():
    """DP1 never sets STREAK in deep_coadd; a gate naming it must not error or
    return NaN, it must simply contribute nothing."""
    mask = np.zeros((4, 4), np.uint32)
    assert plane_fraction(mask, {"BAD": 0}, "STREAK") == 0.0


def test_plane_fractions_covers_every_declared_plane():
    mask = np.zeros((8, 8), np.uint32)
    mask[0, 0] = 1 << PLANES["CR"]
    fracs = plane_fractions(mask, PLANES)
    assert set(fracs) == set(PLANES)
    assert fracs["CR"] == pytest.approx(1 / 64)


# -- pixel-level tests DP1 forces on us -----------------------------------


def test_nonfinite_pixels_are_caught_without_a_mask_plane():
    """NO_DATA and UNMASKEDNAN are unset in DP1, so the pixels must be tested."""
    img = _scene()
    img[5, 5] = np.nan
    assert "nonfinite_image" in _gate(img)[0]


def test_nonpositive_variance_is_caught():
    img = _scene()
    var = np.full((SIZE, SIZE), SKY**2)
    var[3, 3] = 0.0
    assert "nonpositive_variance" in gate(
        img, var, np.zeros((SIZE, SIZE), np.uint32), PLANES
    )[0]


# -- the central requirement ----------------------------------------------


@pytest.mark.parametrize(
    "label,extra",
    [
        ("blank sky", 0.0),
        ("bright galaxy", 2000 * np.exp(-R2 * 4)),
        ("faint galaxy", 60 * np.exp(-R2 * 8)),
        ("galaxy filling the stamp", 3000 * np.exp(-R2 * 0.4)),
        (
            "galaxy at the edge",
            1500 * np.exp(-(((_X - 10) ** 2 + (_Y - 90) ** 2) / 300)),
        ),
        ("two galaxies", 800 * np.exp(-R2 * 6)
         + 400 * np.exp(-(((_X - 140) ** 2 + (_Y - 40) ** 2) / 200))),
    ],
)
def test_real_scenes_are_accepted(label, extra):
    reasons, _ = _gate(_scene(extra))
    assert reasons == [], f"{label} was rejected: {reasons}"


OVER_SUBTRACTION = [
    ("dark halo, 6 sigma bowl", -6 * SKY * (1 - R2)),
    ("dark halo, 1 sigma bowl", -1.0 * SKY * (1 - R2)),
    ("dark edge gradient", -3 * SKY * (_X / SIZE)),
    ("uniform over-subtraction", -1.0 * SKY),
]

# -- mask-plane tolerances -------------------------------------------------


def test_a_single_saturated_pixel_rejects():
    """SAT is dilated to cover bleed trails; the DP1 docs say exclude outright."""
    mask = np.zeros((SIZE, SIZE), np.uint32)
    mask[10, 10] |= 1 << PLANES["SAT"]
    assert any(r.startswith("SAT") for r in _gate(_scene(), mask)[0])


def test_any_no_data_or_edge_pixel_rejects():
    """NO_DATA is the coadd-specific one: those pixels had no contributing
    exposures, so they are not sky, they are nothing."""
    for plane in ("NO_DATA", "EDGE", "SENSOR_EDGE"):
        mask = np.zeros((SIZE, SIZE), np.uint32)
        mask[0, 0] |= 1 << PLANES[plane]
        reasons = _gate(_scene(), mask)[0]
        assert any("zero_tol" in r for r in reasons), (plane, reasons)


def test_detected_is_never_a_rejection_reason():
    """Rejecting on DETECTED would throw away every patch containing a galaxy."""
    mask = np.full((SIZE, SIZE), 1 << PLANES["DETECTED"], np.uint32)
    assert _gate(_scene(), mask)[0] == []
    assert "DETECTED" in NEVER_REJECT
    assert not set(FRAC_TOL) & set(NEVER_REJECT)


def test_gating_on_detected_is_refused_loudly():
    with pytest.raises(ValueError, match="mark real sources"):
        _gate(_scene(), frac_tol={"DETECTED": 0.5})


def test_inner_region_is_stricter_than_the_whole_patch():
    """A cosmic ray 80 px from the host matters far less than one on top of it."""
    off_centre = np.zeros((SIZE, SIZE), np.uint32)
    off_centre[2:4, 2:4] |= 1 << PLANES["CR"]
    centred = np.zeros((SIZE, SIZE), np.uint32)
    centred[SIZE // 2, SIZE // 2] |= 1 << PLANES["CR"]
    assert _gate(_scene(), off_centre)[0] == []
    assert any("inner_CR" in r for r in _gate(_scene(), centred)[0])


def test_diagnostics_are_returned_for_rejected_patches_too():
    """The rejection statistics are the only way to detect a biased selection
    function, so they must be recorded even when the patch is thrown away."""
    mask = np.zeros((SIZE, SIZE), np.uint32)
    mask[10, 10] |= 1 << PLANES["SAT"]
    reasons, diag = _gate(_scene(), mask)
    assert reasons
    assert diag["frac_SAT"] > 0
    assert diag["inner_frac_CR"] == 0.0
    assert diag["sky_noise"] == pytest.approx(SKY, rel=1e-6)


def test_sky_noise_is_inferred_from_the_variance_plane():
    _, diag = _gate(_scene())
    assert diag["sky_noise"] == pytest.approx(SKY, rel=1e-6)


def test_tolerances_are_configurable():
    mask = np.zeros((SIZE, SIZE), np.uint32)
    mask[0:20, 0:20] |= 1 << PLANES["CR"]  # 1.1% of the patch, above the 0.5% default
    assert any(r.startswith("CR") for r in _gate(_scene(), mask)[0])
    assert _gate(_scene(), mask, frac_tol={"CR": 0.05})[0] == []
    # And just under the default is accepted, so the threshold is where it says.
    small = np.zeros((SIZE, SIZE), np.uint32)
    small[0:13, 0:13] |= 1 << PLANES["CR"]  # 0.46%
    assert _gate(_scene(), small)[0] == []


def test_gating_on_an_unpopulated_plane_is_a_no_op():
    """deep_coadd never sets BAD, SUSPECT, ITL_DIP and several others. A gate
    naming one must neither error nor silently pass everything else."""
    coadd_planes = {k: v for k, v in PLANES.items()
                    if k not in ("BAD", "SUSPECT", "ITL_DIP", "CROSSTALK")}
    mask = np.zeros((SIZE, SIZE), np.uint32)
    reasons, diag = gate(_scene(), np.full((SIZE, SIZE), SKY**2), mask,
                         coadd_planes, frac_tol={"BAD": 0.0, "CR": 0.005})
    assert reasons == []
    assert diag["frac_BAD"] == 0.0


def test_coadd_specific_planes_are_gated():
    """CLIPPED and REJECTED mark where outlier rejection fired during
    coaddition; a little is normal, a lot means the stack disagreed with itself."""
    for plane in ("CLIPPED", "REJECTED"):
        mask = np.zeros((SIZE, SIZE), np.uint32)
        mask[:40, :40] |= 1 << PLANES[plane]  # 4.3% of the patch
        assert any(r.startswith(plane) for r in _gate(_scene(), mask)[0]), plane
