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
    background_floor,
    gate,
    plane_bitmask,
    plane_fraction,
    plane_fractions,
)

# The DP1 r29.2.0 bit assignments.
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
    """DP1 never sets STREAK in visit_image; a gate on it must not silently
    pass everything by erroring or by NaN."""
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


@pytest.mark.parametrize("label,extra", OVER_SUBTRACTION)
def test_over_subtraction_is_measured_but_kept_by_default(label, extra):
    """Default is to take the data as-is.

    These artefacts have no mask plane in DP1 at all, and they matter: a smooth
    negative bowl sits in exactly the low-surface-brightness regime this project
    models. But they are a property of the current processing that a later data
    release will improve, so the default records them and lets the prior learn
    them rather than filtering them out. The diagnostic is still written for
    every patch, so the decision can be revisited from the manifest without
    re-reading pixels.
    """
    reasons, diag = _gate(_scene(extra))
    assert reasons == [], f"{label} should be kept by default; got {reasons}"
    assert diag["min_block"] < -0.3, f"{label} should still be measured as depressed"


@pytest.mark.parametrize("label,extra", OVER_SUBTRACTION)
def test_over_subtraction_is_rejected_when_asked(label, extra):
    reasons, _ = _gate(_scene(extra), max_depression=0.3)
    assert any("background_depression" in r for r in reasons), (
        f"{label} was accepted with max_depression=0.3; reasons={reasons}"
    )


def test_depth_of_over_subtraction_is_recorded_for_later_use():
    """Pooling divides the noise by pool_factor but leaves a smooth offset
    untouched, so a depressed region is pool_factor times deeper relative to the
    noise once pooled.  The log transform carries it through regardless --
    softplus softening has no floor -- but the depth is worth recording so the
    decision to keep these patches can be revisited from the manifest.
    """
    _, diag = _gate(_scene(-1.0 * SKY * (1 - R2)))
    assert diag["min_block"] < -0.5
    assert 3 * abs(diag["min_block"]) > 2.0  # pool_factor = 3


def test_blank_sky_noise_floor_leaves_headroom():
    """The min_block statistic must not drift close to the threshold on pure
    noise, or the gate rejects good patches at random."""
    floors = [
        background_floor(_scene(seed=s), SKY)["min_block"] for s in range(20)
    ]
    assert max(np.abs(floors)) < 0.25, f"worst blank-sky min_block {max(floors)}"


def test_positive_excursions_are_never_rejected_as_background():
    """One-sidedness, stated directly: a raised sky floor is starlight."""
    d = background_floor(_scene(2000 * np.exp(-R2 * 4)), SKY)
    assert d["min_block"] > -0.3
    assert d["max_block"] > 10


# -- mask-plane tolerances -------------------------------------------------


def test_a_single_saturated_pixel_rejects():
    """SAT is dilated to cover bleed trails; the DP1 docs say exclude outright."""
    mask = np.zeros((SIZE, SIZE), np.uint32)
    mask[10, 10] |= 1 << PLANES["SAT"]
    assert any(r.startswith("SAT") for r in _gate(_scene(), mask)[0])


def test_any_edge_or_itl_dip_pixel_rejects():
    for plane in ("EDGE", "ITL_DIP", "SENSOR_EDGE"):
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
    reasons, diag = _gate(_scene(-6 * SKY * (1 - R2)), mask)
    assert reasons
    assert diag["frac_SAT"] > 0
    assert np.isfinite(diag["min_block"])
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


def test_background_floor_degrades_rather_than_crashing():
    """Almost everything masked: return NaN and do not reject on this basis."""
    img = _scene()
    mask = np.full((SIZE, SIZE), 1 << PLANES["BAD"], np.uint32)
    d = background_floor(img, SKY, mask, PLANES)
    assert np.isnan(d["min_block"])
    reasons, diag = _gate(img, mask)
    assert not any("background" in r for r in reasons)
