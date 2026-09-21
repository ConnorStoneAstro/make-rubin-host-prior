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
    variance_step,
)

# The DP1 r29.2.0 bit assignments.
#: DP2 plane names with a locally assigned packing, which is what extraction
#: records per shard.  DP2 bit numbers are dynamic, so nothing may hard-code
#: one; these values are arbitrary and the tests must not depend on them.
PLANES = {
    "SATURATED": 0, "COSMIC_RAY": 1, "INTERPOLATED": 2, "DETECTION_EDGE": 3,
    "DETECTED": 4, "DETECTED_NEGATIVE": 5, "INEXACT_PSF": 6, "REJECTED": 7,
    "CLIPPED": 8, "SUSPECT": 9,
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
    assert plane_bitmask(PLANES, "SATURATED") == 1
    assert plane_bitmask(PLANES, ["SATURATED", "COSMIC_RAY"]) == 1 | 2
    # DP2 coadds have no STREAK plane at all
    assert plane_bitmask(PLANES, "STREAK") == 0


def test_plane_fraction_of_absent_plane_is_zero():
    """DP2 coadds carry no STREAK plane; a gate naming it must not error or
    return NaN, it must simply contribute nothing."""
    mask = np.zeros((4, 4), np.uint32)
    assert plane_fraction(mask, {"SATURATED": 0}, "STREAK") == 0.0


def test_plane_fractions_covers_every_declared_plane():
    mask = np.zeros((8, 8), np.uint32)
    mask[0, 0] = 1 << PLANES["COSMIC_RAY"]
    fracs = plane_fractions(mask, PLANES)
    assert set(fracs) == set(PLANES)
    assert fracs["COSMIC_RAY"] == pytest.approx(1 / 64)


# -- pixel-level tests DP1 forces on us -----------------------------------


def test_nonfinite_image_pixels_are_caught():
    img = _scene()
    img[5, 5] = np.nan
    assert "nonfinite_image" in _gate(img)[0]


def test_inf_variance_is_measured_not_treated_as_corruption():
    """DP2 marks "no contributing exposures" with inf variance, and that
    includes the cores of saturated stars.  Rejecting on its mere presence would
    discard every stamp with a bright neighbour -- the regime of interest."""
    img = _scene()
    var = np.full((SIZE, SIZE), SKY**2)
    var[3:6, 3:6] = np.inf
    reasons, diag = gate(img, var, np.zeros((SIZE, SIZE), np.uint32), PLANES)
    assert reasons == []
    assert diag["frac_no_data"] == pytest.approx(9 / SIZE**2)
    # sky noise must be measured from the finite pixels only
    assert diag["sky_noise"] == pytest.approx(SKY, rel=1e-6)


def test_too_much_no_data_is_rejected():
    img = _scene()
    var = np.full((SIZE, SIZE), SKY**2)
    var[:40, :40] = np.inf  # 4.3%
    assert any("no_data" in r for r in
               gate(img, var, np.zeros((SIZE, SIZE), np.uint32), PLANES)[0])


def test_no_data_at_the_centre_is_rejected_outright():
    """That is where the transient goes; missing pixels there are not
    recoverable and the model would have to invent them."""
    img = _scene()
    var = np.full((SIZE, SIZE), SKY**2)
    var[SIZE // 2, SIZE // 2] = np.inf
    assert any("inner_no_data" in r for r in
               gate(img, var, np.zeros((SIZE, SIZE), np.uint32), PLANES)[0])


def test_stale_dp1_plane_names_fail_loudly():
    """The documented DP1->DP2 failure mode: the planes were renamed, so an old
    gate matches nothing and silently passes every stamp.  That must be an
    error, not a clean bill of health."""
    with pytest.raises(ValueError, match="none of the gated planes"):
        gate(_scene(), np.full((SIZE, SIZE), SKY**2),
             np.zeros((SIZE, SIZE), np.uint32), {"SAT": 0, "CR": 1, "INTRP": 2})


def test_covariate_planes_are_recorded_but_never_gated():
    """INEXACT_PSF and REJECTED cover a large fraction of the DP2 coadd, so a
    cut on them keeps almost nothing.  Record, do not reject."""
    from rubin_host_prior.rubin.quality import COVARIATE_PLANES

    mask = np.zeros((SIZE, SIZE), np.uint32)
    for plane in ("INEXACT_PSF", "REJECTED"):
        mask[: int(0.8 * SIZE)] |= 1 << PLANES[plane]
    reasons, diag = _gate(_scene(), mask)
    assert reasons == []
    for plane in ("INEXACT_PSF", "REJECTED"):
        assert diag[f"frac_{plane}"] == pytest.approx(0.8, rel=0.02)
    assert not set(FRAC_TOL) & set(COVARIATE_PLANES)


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


def test_saturation_is_tolerated_away_from_the_centre():
    """DP1 excluded any saturation outright.  On a DP2 coadd the saturated core
    of a bright neighbour lands in a great many stamps, and a scene with a
    bright neighbour is the regime this project models -- so a small fraction
    away from the centre is kept, and the manifest records how much."""
    mask = np.zeros((SIZE, SIZE), np.uint32)
    mask[10:13, 10:13] |= 1 << PLANES["SATURATED"]
    reasons, diag = _gate(_scene(), mask)
    assert reasons == []
    assert diag["frac_SATURATED"] > 0


def test_a_lot_of_saturation_still_rejects():
    mask = np.zeros((SIZE, SIZE), np.uint32)
    mask[:40, :40] |= 1 << PLANES["SATURATED"]
    assert any(r.startswith("SATURATED") for r in _gate(_scene(), mask)[0])


def test_saturation_at_the_centre_rejects():
    mask = np.zeros((SIZE, SIZE), np.uint32)
    mask[SIZE // 2, SIZE // 2] |= 1 << PLANES["SATURATED"]
    assert any("inner_SATURATED" in r for r in _gate(_scene(), mask)[0])


def test_any_detection_edge_pixel_rejects():
    for plane in ("DETECTION_EDGE",):
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


def _centred(plane, n_px):
    """A square of ``n_px`` flagged pixels at the middle of the stamp."""
    mask = np.zeros((SIZE, SIZE), np.uint32)
    side = int(np.ceil(np.sqrt(n_px)))
    c = SIZE // 2
    mask[c:c + side, c:c + side] |= 1 << PLANES[plane]
    return mask


def test_inner_region_is_stricter_than_the_whole_patch():
    """A cosmic ray 80 px from the host matters less than one on top of it: the
    inner tolerance is 4x tighter for INTERPOLATED and 20x for SATURATED."""
    off_centre = np.zeros((SIZE, SIZE), np.uint32)
    off_centre[2:6, 2:6] |= 1 << PLANES["INTERPOLATED"]
    assert _gate(_scene(), off_centre)[0] == []
    assert any("inner_INTERPOLATED" in r
               for r in _gate(_scene(), _centred("INTERPOLATED", 16))[0])


def test_a_single_flagged_pixel_does_not_disqualify_a_stamp():
    """Regression: the inner tolerances were all zero, so one flagged pixel
    anywhere near the middle of ~20 000 rejected the stamp, and inner_COSMIC_RAY
    alone accounted for a quarter of the rejections on a real run."""
    for plane in ("COSMIC_RAY", "INTERPOLATED"):
        assert _gate(_scene(), _centred(plane, 1))[0] == []


def test_cosmic_rays_are_tolerated_where_interpolation_is_not():
    """On a coadd a COSMIC_RAY pixel is real data: the affected inputs were
    rejected during coaddition and the pixel was built from the rest, so it is
    shallower, not invented. An INTERPOLATED pixel is invented -- smooth
    synthetic fill, exactly the false structure a generative model will learn."""
    n = 10  # 0.24% of the inner region
    assert _gate(_scene(), _centred("COSMIC_RAY", n))[0] == []
    assert any("inner_INTERPOLATED" in r for r in _gate(_scene(), _centred("INTERPOLATED", n))[0])


def test_enough_cosmic_rays_still_reject():
    """Tolerated is not ignored."""
    assert any("inner_COSMIC_RAY" in r
               for r in _gate(_scene(), _centred("COSMIC_RAY", 400))[0])


def test_saturation_at_the_centre_is_never_tolerated():
    """The centre is where the transient goes; a saturated core there makes the
    stamp useless for the thing it is being collected for."""
    assert any("inner_SATURATED" in r for r in _gate(_scene(), _centred("SATURATED", 1))[0])


def test_diagnostics_are_returned_for_rejected_patches_too():
    """The rejection statistics are the only way to detect a biased selection
    function, so they must be recorded even when the patch is thrown away."""
    mask = np.zeros((SIZE, SIZE), np.uint32)
    mask[:40, :40] |= 1 << PLANES["SATURATED"]
    reasons, diag = _gate(_scene(), mask)
    assert reasons
    assert diag["frac_SATURATED"] > 0
    assert diag["inner_frac_COSMIC_RAY"] == 0.0
    assert diag["sky_noise"] == pytest.approx(SKY, rel=1e-6)


def test_sky_noise_is_inferred_from_the_variance_plane():
    _, diag = _gate(_scene())
    assert diag["sky_noise"] == pytest.approx(SKY, rel=1e-6)


def test_tolerances_are_configurable():
    mask = np.zeros((SIZE, SIZE), np.uint32)
    mask[0:20, 0:20] |= 1 << PLANES["COSMIC_RAY"]  # 1.1%, above the 0.5% default
    assert any(r.startswith("COSMIC_RAY") for r in _gate(_scene(), mask)[0])
    assert _gate(_scene(), mask, frac_tol={"COSMIC_RAY": 0.05})[0] == []
    # And just under the default is accepted, so the threshold is where it says.
    small = np.zeros((SIZE, SIZE), np.uint32)
    small[0:13, 0:13] |= 1 << PLANES["COSMIC_RAY"]  # 0.46%
    assert _gate(_scene(), small)[0] == []


def test_gating_on_an_unpopulated_plane_is_a_no_op():
    """A plane absent from this release's schema must contribute nothing -- so
    long as at least one gated plane IS present, which the loud check covers."""
    mask = np.zeros((SIZE, SIZE), np.uint32)
    reasons, diag = gate(_scene(), np.full((SIZE, SIZE), SKY**2), mask, PLANES,
                         frac_tol={"VIGNETTED": 0.0, "SATURATED": 0.005})
    assert reasons == []
    assert diag["frac_VIGNETTED"] == 0.0





# -- depth steps -----------------------------------------------------------
#
# Cell-based coadds build each 150 px cell from its own set of input visits, so
# the noise level steps across a straight cell edge and no mask plane says so.
# It is most obvious in y, which has the fewest visits and so the largest
# fractional step.


def _var(step=1.0, frac=0.5, seed=0):
    """Variance plane whose right-hand `frac` is `step` times deeper."""
    v = np.full((SIZE, SIZE), SKY**2, dtype=float)
    v[:, int(SIZE * (1 - frac)):] *= step
    # Real variance planes are themselves noisy; the floor must not be fooled.
    return v * np.random.default_rng(seed).lognormal(0.0, 0.05, v.shape)


def test_uniform_variance_has_no_step():
    assert variance_step(_var(1.0)) == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("step", [1.3, 2.0, 4.0])
def test_a_cell_boundary_is_measured_at_its_true_depth_ratio(step):
    assert variance_step(_var(step)) == pytest.approx(step, rel=0.15)


def test_a_step_is_found_wherever_it_falls():
    """The blocks are not aligned to cells, so a boundary anywhere must show."""
    for frac in (0.2, 0.35, 0.5, 0.75):
        assert variance_step(_var(2.0, frac=frac)) > 1.7


def _galaxy(sigma_px, peak_over_sky, gain=0.1, seed=0):
    """An image and the variance plane that goes with it, source Poisson and all."""
    rng = np.random.default_rng(seed)
    r2 = (_X - SIZE / 2) ** 2 + (_Y - SIZE / 2) ** 2
    gal = peak_over_sky * SKY * np.exp(-r2 / (2 * sigma_px**2))
    var = (SKY**2 + gain * gal) * rng.lognormal(0.0, 0.05, gal.shape)
    return gal + rng.normal(0.0, 1.0, gal.shape) * np.sqrt(var), var


@pytest.mark.parametrize("sigma_px", [4.2, 10.0, 20.0])
def test_a_bright_galaxy_does_not_fake_a_step(sigma_px):
    """Sources add their own Poisson variance, and it is one-sided, so it reads
    as a depth step under any statistic that is not a low envelope.  Rejecting on
    it would throw away exactly the well-resolved hosts this set is for -- and
    the biggest galaxies, which are the most wanted, would go first.

    sigma = 20 px is a galaxy wider than a whole block, which no percentile
    survives; it takes the image to tell source from depth.
    """
    image, variance = _galaxy(sigma_px, peak_over_sky=200)
    assert variance_step(variance, image) < 1.15
    reasons, _ = gate(image, variance, np.zeros((SIZE, SIZE), np.uint32), PLANES)
    assert not any(r.startswith("variance_step") for r in reasons)


def test_a_step_is_still_found_under_a_bright_galaxy():
    """Masking the source must not mask the evidence."""
    image, variance = _galaxy(10.0, peak_over_sky=200)
    variance[:, SIZE // 2:] *= 2.0
    image[:, SIZE // 2:] *= np.sqrt(2.0)
    assert variance_step(variance, image) == pytest.approx(2.0, rel=0.2)


def test_no_data_regions_do_not_read_as_depth():
    """inf variance marks no coverage, including saturated cores; a block that
    is mostly no-data has no floor to report rather than a wrong one."""
    v = _var(1.0)
    v[:40, :40] = np.inf
    assert variance_step(v) == pytest.approx(1.0, abs=0.1)


def test_gate_rejects_a_depth_step_and_records_it():
    mask = np.zeros((SIZE, SIZE), np.uint32)
    reasons, diag = gate(_scene(), _var(2.5), mask, PLANES)
    assert any(r.startswith("variance_step") for r in reasons)
    assert diag["variance_step"] == pytest.approx(2.5, rel=0.15)


def test_gate_records_the_step_even_when_it_passes():
    """The ratio is in the manifest either way, so the threshold can be retuned
    without re-reading pixels."""
    reasons, diag = gate(_scene(), _var(1.2), np.zeros((SIZE, SIZE), np.uint32), PLANES)
    assert not any(r.startswith("variance_step") for r in reasons)
    assert diag["variance_step"] == pytest.approx(1.2, rel=0.15)


def test_the_depth_gate_can_be_turned_off():
    reasons, _ = gate(_scene(), _var(4.0), np.zeros((SIZE, SIZE), np.uint32), PLANES,
                      max_variance_step=np.inf)
    assert not any(r.startswith("variance_step") for r in reasons)


def test_too_few_blocks_is_a_pass_not_a_rejection():
    """Absence of information is not a defect."""
    small = np.full((10, 10), SKY**2)
    assert np.isnan(variance_step(small))
    reasons, _ = gate(np.zeros((10, 10)), small, np.zeros((10, 10), np.uint32), PLANES)
    assert not any(r.startswith("variance_step") for r in reasons)


def test_the_exact_depth_ratio_rejects_without_looking_at_pixels():
    """Visits share an integration time, so the ratio of visit counts across the
    cells a stamp covers is the depth step -- known from provenance before a
    pixel is read."""
    clean = _var(1.0)
    mask = np.zeros((SIZE, SIZE), np.uint32)
    reasons, diag = gate(_scene(), clean, mask, PLANES, cell_depth_ratio=30 / 12)
    assert any(r.startswith("cell_depth") for r in reasons)
    assert diag["cell_depth_ratio"] == pytest.approx(2.5)


def test_equal_depth_cells_pass():
    reasons, diag = gate(_scene(), _var(1.0), np.zeros((SIZE, SIZE), np.uint32),
                         PLANES, cell_depth_ratio=30 / 29)
    assert not reasons and diag["cell_depth_ratio"] == pytest.approx(30 / 29)


def test_the_two_depth_measures_are_independent():
    """Coaddition is inverse-variance weighted, so cells with equal visit counts
    still differ by whatever the seeing and sky did. The counts cannot see that;
    the variance can. Neither subsumes the other, so both run."""
    mask = np.zeros((SIZE, SIZE), np.uint32)
    reasons, _ = gate(_scene(), _var(2.5), mask, PLANES, cell_depth_ratio=1.0)
    assert any(r.startswith("variance_step") for r in reasons)
    assert not any(r.startswith("cell_depth") for r in reasons)


def test_unknown_depth_is_not_recorded_and_not_gated():
    """Absence of provenance must not read as a depth of zero."""
    reasons, diag = gate(_scene(), _var(1.0), np.zeros((SIZE, SIZE), np.uint32),
                         PLANES, cell_depth_ratio=None)
    assert "cell_depth_ratio" not in diag
    assert not reasons


def test_a_cell_with_no_visits_at_all_is_rejected():
    reasons, _ = gate(_scene(), _var(1.0), np.zeros((SIZE, SIZE), np.uint32),
                      PLANES, cell_depth_ratio=np.inf)
    assert any(r.startswith("cell_depth") for r in reasons)
