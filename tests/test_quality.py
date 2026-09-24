"""The artefact gate.

The governing requirement: reject artefacts while accepting every galaxy,
however bright.  A gate that rejects dense bright centres biases the training
set against precisely the regime this project exists to model, so most of what
is checked here is what must *not* be thrown away.
"""

import numpy as np
import pytest

from rubin_host_prior.selection import PatchCuts
from rubin_host_prior.rubin.quality import (
    gate,
    plane_bitmask,
    plane_fraction,
    plane_fractions,
    variance_step,
)

#: DP2 plane names with a locally assigned packing, which is what extraction
#: does per shard.  DP2 bit numbers are dynamic, so nothing may hard-code one;
#: these values are arbitrary and no test may depend on them.
PLANES = {
    "SATURATED": 0, "COSMIC_RAY": 1, "INTERPOLATED": 2, "DETECTION_EDGE": 3,
    "DETECTED": 4, "DETECTED_NEGATIVE": 5, "INEXACT_PSF": 6, "REJECTED": 7,
    "CLIPPED": 8, "SUSPECT": 9,
}
SIZE, SKY = 192, 12.0
_Y, _X = np.mgrid[0:SIZE, 0:SIZE]
R2 = ((_X - SIZE / 2) ** 2 + (_Y - SIZE / 2) ** 2) / (SIZE / 2) ** 2


#: A moderate host at the centre, which every candidate stamp has: the gate now
#: asks whether one is there, so a scene of bare noise is *correctly* rejected
#: and would make every artefact test below fail for the wrong reason.
HOST = 30.0 * SKY * np.exp(-(((_X - SIZE / 2) ** 2 + (_Y - SIZE / 2) ** 2)
                             / (2 * 12.0**2)))


def _scene(extra=None, seed=0):
    """A stamp with a host in it.  Pass ``extra`` for a different one, or
    ``0.0`` for the empty sky that the centre cut exists to catch."""
    base = np.random.default_rng(seed).normal(0.0, SKY, (SIZE, SIZE))
    return base + (HOST if extra is None else extra)


#: ``gate`` has no tolerances of its own -- extraction.yaml, via PatchCuts, is
#: the only place that says what a cut is -- so a caller must supply all of
#: them.  These are the shipped values; each test overrides the one it is about.
#: ``max_variance_step`` is tightened to the old module default, which is what
#: these tests were written against.
BASELINE = {**PatchCuts().gate_kwargs(), "max_variance_step": 1.5,
            "max_cell_depth_ratio": 1.5, "cell_depth_ratio": 1.0, "n_visits": 1}


def _gate(image, mask=None, variance=None, **kw):
    mask = np.zeros((SIZE, SIZE), np.uint32) if mask is None else mask
    variance = np.full((SIZE, SIZE), SKY**2) if variance is None else variance
    return gate(image, variance, mask, PLANES, **{**BASELINE, **kw})


def _flagged(plane, n_px, centred=True):
    mask = np.zeros((SIZE, SIZE), np.uint32)
    side = int(np.ceil(np.sqrt(n_px)))
    o = SIZE // 2 if centred else 2
    mask[o:o + side, o:o + side] |= 1 << PLANES[plane]
    return mask


# -- plane bookkeeping ------------------------------------------------------


def test_absent_planes_contribute_nothing_and_do_not_error():
    """DP2 coadds carry no STREAK plane at all; naming one must be a no-op
    rather than an error or a NaN."""
    mask = np.zeros((4, 4), np.uint32)
    assert plane_bitmask(PLANES, "STREAK") == 0
    assert plane_bitmask(PLANES, ["SATURATED", "COSMIC_RAY"]) == 1 | 2
    assert plane_fraction(mask, PLANES, "STREAK") == 0.0
    assert set(plane_fractions(mask, PLANES)) == set(PLANES)


def test_stale_dp1_plane_names_fail_loudly():
    """DP2 renamed SAT->SATURATED, CR->COSMIC_RAY, INTRP->INTERPOLATED,
    EDGE->DETECTION_EDGE.  A DP1-era gate does not error on DP2 -- it matches
    nothing and passes every stamp, which is the worst possible outcome."""
    with pytest.raises(ValueError, match="renamed"):
        gate(_scene(), np.full((SIZE, SIZE), SKY**2),
             np.zeros((SIZE, SIZE), np.uint32), {"SAT": 0, "CR": 1}, **BASELINE)


def test_gating_on_a_plane_that_marks_real_sources_is_refused():
    with pytest.raises(ValueError, match="mark real sources"):
        _gate(_scene(), frac_tol={"DETECTED": 0.5})


# -- what must be accepted --------------------------------------------------


@pytest.mark.parametrize("label,extra", [
    ("bright galaxy", 900.0 * np.exp(-R2 * 40)),
    ("steep centre", 4000.0 * np.exp(-R2 * 300)),
])
def test_real_scenes_are_accepted(label, extra):
    assert _gate(_scene(extra))[0] == [], label


def test_blank_sky_is_not_a_real_scene():
    """It used to be in the list above.  A stamp with nothing at its centre is
    exactly what a wrong Sersic fit delivers, and every cut before the gate is
    on a catalogue quantity that such a fit satisfies."""
    assert any(r.startswith("empty_centre") for r in _gate(_scene(0.0))[0])


def test_a_single_flagged_pixel_does_not_disqualify_a_stamp():
    """The inner tolerances were all zero, so one flagged pixel among ~20 000
    rejected the stamp, and inner_COSMIC_RAY alone accounted for a quarter of
    the rejections on a real run."""
    for plane in ("COSMIC_RAY", "INTERPOLATED"):
        assert _gate(_scene(), _flagged(plane, 1))[0] == []


def test_detected_pixels_are_recorded_and_never_a_reason():
    """DETECTED marks the targets; rejecting on it rejects the whole sample."""
    mask = np.zeros((SIZE, SIZE), np.uint32)
    mask[R2 < 0.3] |= 1 << PLANES["DETECTED"]
    mask |= 1 << PLANES["INEXACT_PSF"]  # covers most of the DP2 coadd
    reasons, diag = _gate(_scene(), mask)
    assert reasons == []
    assert diag["frac_DETECTED"] > 0 and diag["frac_INEXACT_PSF"] == 1.0


# -- what must be rejected --------------------------------------------------


def test_the_planes_that_disqualify_a_stamp():
    """Saturation is tolerated away from the centre -- a bright neighbour lands
    in a great many stamps and that is the regime being modelled -- but not at
    the centre, where the transient goes, and not in quantity."""
    assert _gate(_scene(), _flagged("SATURATED", 4, centred=False))[0] == []
    assert any("inner_SATURATED" in r
               for r in _gate(_scene(), _flagged("SATURATED", 1))[0])
    assert any("SATURATED" in r
               for r in _gate(_scene(), _flagged("SATURATED", 4000,
                                                 centred=False))[0])
    # DETECTION_EDGE is zero-tolerance anywhere.
    assert any("zero_tol" in r
               for r in _gate(_scene(), _flagged("DETECTION_EDGE", 1,
                                                 centred=False))[0])


def test_cosmic_rays_are_tolerated_where_interpolation_is_not():
    """On a coadd a COSMIC_RAY pixel is real data: the affected inputs were
    rejected during coaddition and the pixel built from the rest, so it is
    shallower, not invented.  An INTERPOLATED pixel is invented."""
    assert _gate(_scene(), _flagged("COSMIC_RAY", 10))[0] == []
    assert any("inner_INTERPOLATED" in r
               for r in _gate(_scene(), _flagged("INTERPOLATED", 10))[0])
    assert any("inner_COSMIC_RAY" in r
               for r in _gate(_scene(), _flagged("COSMIC_RAY", 400))[0])


def test_no_data_is_measured_not_treated_as_corruption():
    """DP2 marks no coverage with inf variance, including saturated cores.
    Rejecting on its mere presence would discard every stamp with a bright
    neighbour."""
    variance = np.full((SIZE, SIZE), SKY**2)
    variance[:2, :2] = np.inf
    assert _gate(_scene(), variance=variance)[0] == []

    variance = np.full((SIZE, SIZE), SKY**2)
    variance[: SIZE // 2] = np.inf
    reasons, diag = _gate(_scene(), variance=variance)
    assert any(r.startswith("no_data") for r in reasons)
    assert diag["frac_no_data"] == pytest.approx(0.5)


def test_diagnostics_are_returned_for_rejected_stamps_too():
    """The rejection statistics are the only way to see a biased selection
    function, so they have to survive the rejection."""
    reasons, diag = _gate(_scene(), _flagged("INTERPOLATED", 4000))
    assert reasons and {"sky_noise", "frac_no_data", "variance_step"} <= set(diag)


# -- depth ------------------------------------------------------------------
#
# Cell-based coadds build each 150 px cell from its own input visits, so the
# noise steps across a straight cell edge and no mask plane says so.


def _var(step=1.0, frac=0.5, seed=0):
    v = np.full((SIZE, SIZE), SKY**2, dtype=float)
    v[:, int(SIZE * (1 - frac)):] *= step
    return v * np.random.default_rng(seed).lognormal(0.0, 0.05, v.shape)


def _step(variance, image=None):
    """``variance_step`` needs the image: the source masking is not optional,
    because a galaxy wider than a block defeats any percentile.  Flat zeros
    stand in where the test is about the variance plane alone."""
    return variance_step(variance,
                         np.zeros_like(variance) if image is None else image)


def _galaxy(sigma_px, peak_over_sky=200, gain=0.1, seed=0):
    """An image and the variance plane that goes with it, source Poisson too."""
    rng = np.random.default_rng(seed)
    r2 = (_X - SIZE / 2) ** 2 + (_Y - SIZE / 2) ** 2
    gal = peak_over_sky * SKY * np.exp(-r2 / (2 * sigma_px**2))
    var = (SKY**2 + gain * gal) * rng.lognormal(0.0, 0.05, gal.shape)
    return gal + rng.normal(0.0, 1.0, gal.shape) * np.sqrt(var), var


@pytest.mark.parametrize("step", [1.0, 1.3, 2.0, 4.0])
def test_a_depth_step_is_measured_at_its_true_ratio(step):
    assert _step(_var(step)) == pytest.approx(step, rel=0.15, abs=0.1)


def test_a_step_is_found_wherever_it_falls_and_survives_a_scene():
    """The blocks are not aligned to cells, so a boundary anywhere must show --
    and masking the source must not mask the evidence."""
    for frac in (0.2, 0.35, 0.5, 0.75):
        assert _step(_var(2.0, frac=frac)) > 1.7
    image, variance = _galaxy(10.0)
    variance[:, SIZE // 2:] *= 2.0
    image[:, SIZE // 2:] *= np.sqrt(2.0)
    assert _step(variance, image) == pytest.approx(2.0, rel=0.2)


@pytest.mark.parametrize("sigma_px", [4.2, 10.0, 20.0])
def test_a_bright_galaxy_does_not_fake_a_step(sigma_px):
    """Source Poisson variance is one-sided, so it reads as a depth step under
    any statistic that is not a low envelope -- and the biggest galaxies, the
    ones most wanted, would be rejected first.  sigma = 20 px is wider than a
    whole block, which no percentile survives; it takes the image."""
    image, variance = _galaxy(sigma_px)
    assert _step(variance, image) < 1.15
    assert not any(r.startswith("variance_step")
                   for r in _gate(image, variance=variance)[0])


def test_no_data_regions_do_not_read_as_depth():
    v = _var(1.0)
    v[:40, :40] = np.inf
    assert _step(v) == pytest.approx(1.0, abs=0.1)


def test_absence_of_information_is_a_pass_not_a_rejection():
    small = np.full((10, 10), SKY**2)
    assert np.isnan(_step(small))
    assert not any(r.startswith("variance_step") for r in
                   gate(np.zeros((10, 10)), small,
                        np.zeros((10, 10), np.uint32), PLANES, **BASELINE)[0])


def test_the_measured_step_gates_and_is_always_recorded():
    """Recorded either way, so the threshold can be retuned from the manifest
    without re-reading pixels."""
    assert any(r.startswith("variance_step")
               for r in _gate(_scene(), variance=_var(2.5))[0])
    reasons, diag = _gate(_scene(), variance=_var(1.2))
    assert not reasons and diag["variance_step"] == pytest.approx(1.2, rel=0.15)
    assert not any(r.startswith("variance_step") for r in
                   _gate(_scene(), variance=_var(4.0),
                         max_variance_step=np.inf)[0])


def test_the_exact_depth_ratio_gates_without_looking_at_pixels():
    """Visits share an integration time, so the ratio of visit counts across the
    cells a stamp covers is the depth step, known from provenance."""
    assert any(r.startswith("cell_depth")
               for r in _gate(_scene(), cell_depth_ratio=30 / 12)[0])
    assert not _gate(_scene(), cell_depth_ratio=30 / 29)[0]
    # inf is a cell with no visits at all: the worst case, not an excuse.
    assert any(r.startswith("cell_depth")
               for r in _gate(_scene(), cell_depth_ratio=np.inf)[0])
    # Recorded either way, so the threshold can be retuned from the manifest.
    assert _gate(_scene(), cell_depth_ratio=1.4)[1]["cell_depth_ratio"] == 1.4


def test_the_two_depth_measures_are_independent():
    """Coaddition is inverse-variance weighted, so cells with equal visit counts
    still differ by whatever the seeing and sky did.  The counts cannot see
    that; the variance can.  Neither subsumes the other."""
    reasons, _ = _gate(_scene(), variance=_var(2.5), cell_depth_ratio=1.0)
    assert any(r.startswith("variance_step") for r in reasons)
    assert not any(r.startswith("cell_depth") for r in reasons)


def test_absolute_depth_is_recorded_and_gated_only_on_request():
    """Early DP2 outside the deep fields is 1-3 visits per cell, a different sky
    from a deep coadd.  Whether that is too shallow is a judgement about the
    prior, so it is recorded always and gated only when asked."""
    _, diag = _gate(_scene(), n_visits=2)
    assert diag["n_visits_min"] == 2
    assert any(r.startswith("too_shallow")
               for r in _gate(_scene(), n_visits=2, min_visits=10)[0])
    assert not _gate(_scene(), n_visits=30, min_visits=10)[0]


# -- is a host there, and is it a galaxy? -----------------------------------


def _star_field(n=200, seed=1):
    rng = np.random.default_rng(seed)
    out = rng.normal(0.0, SKY, (SIZE, SIZE))
    for _ in range(n):
        y, x = rng.integers(8, SIZE - 8, 2)
        out += 40 * SKY * np.exp(-(((_X - x) ** 2 + (_Y - y) ** 2) / (2 * 1.5**2)))
    return out


def test_an_empty_centre_is_caught_by_the_pixels_not_the_catalogue():
    """Every host cut runs on a catalogue quantity, so a Sersic fit that found
    nothing passes the magnitude and surface-brightness limits and arrives as a
    stamp of empty sky.  This is the same question asked of the pixels."""
    from rubin_host_prior.rubin.quality import centre_brightness

    assert centre_brightness(_scene(0.0), SKY) == pytest.approx(0.0, abs=0.5)
    assert centre_brightness(_scene(), SKY) > 20

    assert any(r.startswith("empty_centre") for r in _gate(_scene(0.0))[0])
    assert not any(r.startswith("empty_centre") for r in _gate(_scene())[0])
    # Recorded either way, so a threshold can be retuned from the manifest.
    assert _gate(_scene())[1]["centre_sigma"] > 20
    assert _gate(_scene(0.0))[1]["centre_sigma"] == pytest.approx(0.0, abs=0.5)
    # None disables it rather than meaning zero.
    assert not any(r.startswith("empty_centre")
                   for r in _gate(_scene(0.0), min_centre_sigma=None)[0])


def test_a_crowded_field_is_told_from_a_galaxy_by_counting_sources():
    """Both are bright and both fill the DETECTED plane, so no mask fraction
    separates them.  The number of distinct peaks does."""
    from rubin_host_prior.rubin.quality import count_peaks

    assert count_peaks(_scene(0.0), SKY) == 0
    assert count_peaks(_scene(), SKY) == 1        # one host, not its bright area
    assert count_peaks(_star_field(), SKY) > 100

    assert any(r.startswith("crowded") for r in _gate(_star_field(), max_peaks=100)[0])
    assert not any(r.startswith("crowded") for r in _gate(_scene(), max_peaks=100)[0])
    assert _gate(_scene())[1]["n_peaks"] == 1
    # None disables it rather than meaning zero.
    assert not any(r.startswith("crowded")
                   for r in _gate(_star_field(), max_peaks=None)[0])


def test_the_prominence_radius_is_what_stops_a_galaxy_counting_as_a_crowd():
    """The naive version -- every 8-connected local maximum above 5 sigma --
    counts noise riding on a bright galaxy, so it measures how much of the stamp
    is bright rather than how many sources are in it.  Measured on one host:
    102 peaks naively, 12 with the smoothing alone, 1 as shipped.

    The radius does most of the work and the smoothing is what lets the radius
    stay small; neither alone is the reason, which is worth pinning down since
    the docstring first claimed it was the smoothing.
    """
    from rubin_host_prior.rubin import quality

    host = _scene()
    assert quality.count_peaks(host, SKY) == 1

    def with_(**over):
        keep = {k: getattr(quality, k) for k in over}
        try:
            for k, v in over.items():
                setattr(quality, k, v)
            return quality.count_peaks(host, SKY)
        finally:
            for k, v in keep.items():
                setattr(quality, k, v)

    assert with_(PEAK_SMOOTH_PX=1, PEAK_RADIUS_PX=1) > 50   # the naive count
    assert with_(PEAK_RADIUS_PX=1) > 5                      # smoothing alone
    assert with_(PEAK_SMOOTH_PX=1) == 1                     # radius alone
