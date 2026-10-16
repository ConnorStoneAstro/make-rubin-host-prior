"""Filling the holes DP2 leaves in saturated star cores."""

import numpy as np
import pytest

from rubin_host_prior.data.repair import fill_batch, fill_no_data, no_data_mask


def _star(side=64, radius=5.0, peak=2000.0, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-side // 2:side // 2, -side // 2:side // 2]
    img = (peak * np.exp(-(x**2 + y**2) / 40.0)
           + rng.normal(0, 1, (side, side))).astype(np.float32)
    hole = (x**2 + y**2) < radius**2
    return img, hole


def test_the_detector_catches_both_fill_values_and_nothing_else():
    """A no-data pixel is exactly 0.0 or non-finite; a real sky pixel is
    neither.  Both tests are needed -- ``NaN == 0.0`` is false, so an
    exact-zero rule alone would miss a NaN fill entirely."""
    img, hole = _star()
    clean = img.copy()
    assert not no_data_mask(clean).any()

    for value in (0.0, np.nan, np.inf, -np.inf):
        a = img.copy()
        a[hole] = value
        got = no_data_mask(a)
        assert got[hole].all(), value
        assert not got[~hole].any(), value


def test_filling_removes_the_cliff_without_touching_the_rest():
    img, hole = _star()
    before = img.copy()
    before[hole] = 0.0
    filled, n = fill_no_data(before)

    assert n == int(hole.sum())
    # the cliff is gone: the steepest step across the hole drops to the star's
    # own gradient
    row = before.shape[0] // 2
    assert np.abs(np.diff(filled[row])).max() < 0.4 * np.abs(np.diff(before[row])).max()
    # nothing outside the hole's padded box moved
    assert np.allclose(filled[~hole][:100], before[~hole][:100])
    # and the fill sits in the right regime: bright, not sky
    assert filled[hole].min() > np.median(img) * 10


def test_a_filled_core_is_a_flat_cap_and_not_a_reconstruction():
    """Harmonic interpolation has no interior extremum, so the core tops out at
    roughly its rim instead of rising to the peak the real star had.  Pinned
    because it is a deliberate limitation, not an accident -- the point is to
    remove a gradient, and the likelihood shapes the cap."""
    img, hole = _star()
    truth_peak = img[hole].max()
    punched = img.copy()
    punched[hole] = 0.0
    filled, _ = fill_no_data(punched)
    assert filled[hole].max() < truth_peak        # never recovers the peak
    rim = img[(~hole)][np.argsort(-img[~hole])][:50].mean()
    assert 0.5 * rim < filled[hole].mean() < 2.0 * rim


def test_a_clean_stamp_is_handed_straight_back():
    """The 96% case: no copy, no work."""
    img, _ = _star()
    out, n = fill_no_data(img)
    assert n == 0
    assert out is img


def test_the_batch_report_is_the_guard_against_a_silent_no_op():
    """If a future release writes its holes with a value the detector does not
    know, every count here stays zero -- which is visible, where a silently
    unfilled batch would not be."""
    img, hole = _star()
    punched = img.copy()
    punched[hole] = 0.0
    stamps = np.stack([img, punched, img])
    out, report = fill_batch(stamps)
    assert report["n_stamps"] == 3
    assert report["n_stamps_filled"] == 1
    assert report["n_pixels_filled"] == int(hole.sum())
    assert report["largest_hole_px"] == int(hole.sum())
    assert np.array_equal(out[0], img) and np.array_equal(out[2], img)

    _, clean_report = fill_batch(np.stack([img, img]))
    assert clean_report["n_stamps_filled"] == 0


def test_a_hole_at_the_edge_is_filled_and_not_re_zeroed():
    """The relaxation used to write zeros back.

    A masked cell on the border of the bounding box has no neighbour above or
    left of it, so the Jacobi stencil there was the zero it was initialised
    with -- and assigning that put an exact 0.0 into the output, which is the
    very thing being repaired.  Only reachable when the hole runs to the stamp
    edge and the pad cannot step outside it, which is why it survived a test
    that only checked the result was finite.
    """
    img = (100.0 + np.random.default_rng(0).normal(0, 1, (32, 32))).astype(np.float32)
    img[:6, :6] = 0.0
    filled, n = fill_no_data(img)
    assert n == 36
    assert np.all(np.isfinite(filled))
    # nothing the detector would pick up a second time
    assert not no_data_mask(filled).any()
    # and the fill sits at the surrounding level, not at zero
    assert filled[:6, :6].min() > 50.0


def test_a_hole_at_the_edge_does_not_crash():
    img, _ = _star()
    img[:3, :3] = 0.0
    filled, n = fill_no_data(img)
    assert n == 9
    assert np.all(np.isfinite(filled))
    assert not no_data_mask(filled).any()


def test_a_hole_filling_its_whole_box_falls_back_to_the_median():
    """Only reachable for a hole flush against the border, where the pad finds
    no real pixels to interpolate from."""
    img = np.ones((16, 16), dtype=np.float32) * 5.0
    img[:, :] = 0.0
    filled, n = fill_no_data(img)
    assert n == 256
    assert np.all(np.isfinite(filled))


# -- the diagnostic, for when a hole survives the repair --------------------


def test_dark_inside_bright_finds_a_hole_whatever_value_was_written():
    """``no_data_mask`` knows two fill values.  This knows none, and asks the
    geometric question instead -- so it can say whether a surviving hole is a
    hole the detector cannot see, or not a hole at all."""
    from rubin_host_prior.data.repair import dark_inside_bright

    img, hole = _star(peak=4000.0)
    n = int(hole.sum())
    for value in (0.0, -0.4, np.nan, "noise"):
        a = img.copy()
        a[hole] = (np.random.default_rng(1).normal(0, 1, n) if value == "noise"
                   else value)
        found = int(dark_inside_bright(a, 1.0).sum())
        assert found > 0.6 * n, (value, found, n)


def test_dark_inside_bright_does_not_flag_sky_beside_a_star():
    """The reason it tests four directions and not one window.

    Sky a few pixels outside a bright star does have a bright neighbour, so a
    single square window flagged the whole ring around every star -- 778 pixels
    against a 69-pixel hole.  It does not have a bright neighbour in all four
    directions.
    """
    from rubin_host_prior.data.repair import dark_inside_bright

    img, _ = _star(peak=4000.0)                      # a star with no hole
    assert dark_inside_bright(img, 1.0).sum() == 0
    rng = np.random.default_rng(0)
    assert dark_inside_bright(rng.normal(0, 1, (64, 64)), 1.0).sum() == 0


def test_describe_dark_cores_separates_seen_from_unseen():
    from rubin_host_prior.data.repair import describe_dark_cores

    img, hole = _star(peak=4000.0)
    seen = img.copy(); seen[hole] = 0.0              # the detector sees this
    unseen = img.copy(); unseen[hole] = -0.4         # it does not see this
    report = describe_dark_cores(np.stack([seen, unseen, img]), np.ones(3))

    assert report["n_stamps"] == 3
    assert report["n_with_detected_holes"] == 1
    assert report["n_with_missed_holes"] == 1
    assert report["n_pixels_missed"] > 0
    # and it says what to widen the detector to
    assert report["missed_value_sigma"]["max"] < 0
