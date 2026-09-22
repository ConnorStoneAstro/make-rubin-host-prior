"""Extraction logic that does not need the LSST stack.

These cover the parts of the DP2 port most likely to go wrong silently: the
object-table columns (DP2 moved second moments per band and dropped
``detect_isPrimary`` entirely), deduplication across overlapping tracts, and the
PSF moments, which DP2's PSF object cannot compute for you.
"""

import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from rubin_host_prior.rubin.extract import (
    OBJECT_COLUMNS,
    OBJECT_BAND_COLUMNS,
    PHOTOMETRY_BANDS,
    SIGMA_TO_FWHM,
    _adaptive_moments,
    _data_id_dict,
    BANDS,
    build_host_catalogue,
    coadd_refs_for_tract,
    SKYMAP,
    object_refs_for_tract,
    read_component,
    missing_cells,
    _fits_in_patch,
    _sky_to_pixel,
    _stamp_box,
    unmask,
    with_positions,
    host_adql,
    TAP_URL_FALLBACK,
    TOKEN_ENV_VARS,
    _BearerForPrefix,
    discover_tap_url,
    rsp_token,
    check_tap_scope,
    find_token,
    token_info,
    run_adql,
    _within_radius,
    _write_manifest,
    cell_visit_counts,
    cells_in_stamp,
    dedupe_hosts,
    host_half_light_arcsec,
    host_trace_radius_px,
    next_batch,
    select_hosts,
    stamp_depth,
)

Table = pytest.importorskip("astropy.table").Table


def _flux_for(reff_arcsec, rng, mu_e=22.0):
    """Total flux of a galaxy of this size at an ordinary surface brightness.

    Size and flux are not independent on the sky, and a fixture that pretends
    they are hides the fact that a flux ceiling set for small galaxies wipes out
    a large size cut.  m = mu_e - 2.5 log10(2 pi reff^2), then nJy at zp 31.4.
    """
    mu = rng.normal(mu_e, 0.7, len(reff_arcsec))
    mag = mu - 2.5 * np.log10(2 * np.pi * np.maximum(reff_arcsec, 1e-3) ** 2)
    return 10 ** ((31.4 - mag) / 2.5)


def _catalogue(n=3000, seed=0, band="r", tract=5063):
    rng = np.random.default_rng(seed)
    ixx = 10 ** rng.uniform(0.4, 3.2, n)
    iyy = 10 ** rng.uniform(0.4, 3.2, n)
    reff = np.sqrt(0.5 * (ixx + iyy)) * 0.2 * 1.177  # trace px -> half-light arcsec
    return Table({
        "objectId": np.arange(n),
        "coord_ra": 53.13 + rng.normal(0, 0.1, n),
        "coord_dec": -28.10 + rng.normal(0, 0.1, n),
        "refExtendedness": np.ones(n),
        "refBand": [band] * n,
        "tract": np.full(n, tract),
        "patch": rng.integers(0, 100, n),
        f"{band}_cModelFlux": _flux_for(reff, rng),
        f"{band}_blendedness": rng.beta(1.2, 8, n),
        f"{band}_ixx": ixx,
        f"{band}_iyy": iyy,
        f"{band}_ixy": rng.normal(0, 5, n),
        # The multiband Sersic fit: one morphology for all six bands, so these
        # carry no band prefix.
        "sersic_reff_major": reff,
        "sersic_reff_minor": 0.7 * reff,
        "sersic_index": rng.uniform(0.5, 6.0, n),
        "sersic_unknown_flag": np.zeros(n, bool),
        "sersic_no_data_flag": np.zeros(n, bool),
    })


def test_object_columns_do_not_ask_for_dp1_only_names():
    """DP2 has no band-independent shape_xx and no detect_* columns at all.
    Requesting a column that does not exist fails the whole read."""
    assert not any(c.startswith("shape_") for c in OBJECT_COLUMNS)
    assert not any(c.startswith("detect") for c in OBJECT_COLUMNS)
    assert {"objectId", "coord_ra", "coord_dec", "refExtendedness"} <= set(OBJECT_COLUMNS)


def test_selecting_on_an_unknown_band_is_refused():
    """Requesting a column that does not exist fails the whole read, so the band
    is checked before any of them is built."""
    with pytest.raises(ValueError, match="photometry"):
        select_hosts(_catalogue(), band="H")


def test_trace_radius_uses_per_band_moments():
    t = _catalogue(50)
    trace = host_trace_radius_px(t, "r")
    expected = np.sqrt(0.5 * (np.asarray(t["r_ixx"]) + np.asarray(t["r_iyy"])))
    np.testing.assert_allclose(trace, expected)


def test_missing_moments_raise_rather_than_silently_skipping():
    """The old code quietly fell back to an unstratified draw when the columns
    were absent -- which on DP2 would have been every single time, leaving a
    sample dominated by the smallest, faintest galaxies."""
    t = _catalogue(50)
    del t["r_ixx"]
    with pytest.raises(KeyError, match="per band"):
        host_trace_radius_px(t, "r")
    with pytest.raises(KeyError):
        select_hosts(t, band="r", n_hosts=10)


# -- deduplication ---------------------------------------------------------


def test_dedupe_collapses_the_same_source_under_two_ids():
    """Tracts and patches overlap, and with no detect_isPrimary a source in an
    overlap appears twice -- across two tracts under two different objectIds, so
    an id-only dedupe would miss it."""
    t = Table({
        "objectId": [1, 2, 3],
        "coord_ra": [53.1000, 53.10002, 53.2000],
        "coord_dec": [-28.1000, -28.10001, -28.2000],
    })
    out = dedupe_hosts(t, radius_arcsec=0.5)
    assert len(out) == 2
    assert 3 in out["objectId"].tolist()


def test_dedupe_keeps_genuinely_distinct_close_pairs():
    """A real close pair beyond the radius must survive."""
    t = Table({
        "objectId": [1, 2],
        "coord_ra": [53.1000, 53.1000],
        "coord_dec": [-28.1000, -28.1000 + 2.0 / 3600.0],  # 2 arcsec apart
    })
    assert len(dedupe_hosts(t, radius_arcsec=0.5)) == 2


def test_dedupe_also_drops_repeated_ids():
    t = Table({"objectId": [7, 7, 8],
               "coord_ra": [53.1, 53.9, 54.2],
               "coord_dec": [-28.1, -28.9, -27.2]})
    assert sorted(dedupe_hosts(t)["objectId"].tolist()) == [7, 8]


def test_select_hosts_dedupes_before_selecting():
    t = _catalogue(200)
    doubled = Table(np.concatenate([t.as_array(), t.as_array()]))
    assert len(select_hosts(doubled, band="r")) == len(select_hosts(t, band="r"))


def _sized_catalogue(n, scale=0.25, seed=1, size_mult=25.0, tract=5063):
    """A catalogue with a steep size distribution, as a real one has."""
    rng = np.random.default_rng(seed)
    trace_sq = size_mult * 10 ** rng.exponential(scale, n)
    reff = np.sqrt(trace_sq) * 0.2 * 1.177  # trace px -> half-light arcsec
    return Table({
        "objectId": np.arange(n),
        "coord_ra": 53.13 + rng.normal(0, 0.3, n),
        "coord_dec": -28.10 + rng.normal(0, 0.3, n),
        "refExtendedness": np.ones(n),
        "tract": np.full(n, tract),
        "patch": rng.integers(0, 100, n),
        "r_cModelFlux": _flux_for(reff, rng),
        "r_ixx": trace_sq, "r_iyy": trace_sq, "r_ixy": np.zeros(n),
        "sersic_reff_major": reff,
        "sersic_reff_minor": 0.7 * reff,
        "sersic_index": rng.uniform(0.5, 6.0, n),
        "sersic_unknown_flag": np.zeros(n, bool),
        "sersic_no_data_flag": np.zeros(n, bool),
    })


# -- host selection --------------------------------------------------------


def test_size_stratification_actually_stratifies():
    """Regression for a bug that made this a no-op.

    Drawing equally from *quantile* bins is exactly a uniform sample, because
    quantile bins hold equal numbers by construction -- so the original
    implementation stratified nothing at all while claiming to. Equal-width bins
    in log size hold wildly unequal numbers, and an equal draw from each is what
    actually gets well-resolved hosts into the sample.
    """
    t = _sized_catalogue(4000, seed=1)
    kw = dict(band="r", min_reff_arcsec=None)  # isolate stratification from the cut
    parent = host_half_light_arcsec(select_hosts(t, **kw))
    big = np.percentile(parent, 90)

    strat = host_half_light_arcsec(select_hosts(t, n_hosts=50, seed=0, **kw))
    flat = host_half_light_arcsec(
        select_hosts(t, n_hosts=50, seed=0, size_stratified=False, **kw)
    )
    assert np.mean(strat > big) > 4 * np.mean(flat > big)
    assert np.median(strat) > np.median(flat)


def test_stratification_cannot_invent_large_galaxies():
    """Asking for the whole population must return the whole population, not a
    size-skewed subset of it -- the gain is bounded by what exists."""
    t = _sized_catalogue(600, seed=2)
    kw = dict(band="r", min_reff_arcsec=None)
    everything = len(select_hosts(t, **kw))
    assert len(select_hosts(t, n_hosts=everything, **kw)) == everything


def test_stratification_still_returns_the_number_asked_for():
    """Sparse bins at the large end leave the quota unfilled; the remainder is
    topped up rather than silently returning fewer hosts."""
    t = _sized_catalogue(3000, scale=0.3, seed=3)
    kw = dict(band="r", min_reff_arcsec=None)
    available = len(select_hosts(t, **kw))
    want = available // 2
    assert len(select_hosts(t, n_hosts=want, seed=0, **kw)) == want


def test_point_like_objects_are_dropped():
    t = _catalogue(500)
    t["r_ixx"] = np.full(len(t), 1.0)
    t["r_iyy"] = np.full(len(t), 1.0)  # trace radius 1.0 px, below the PSF
    assert len(select_hosts(t, band="r")) == 0


def test_flux_range_is_respected():
    # Size cut off: these fluxes belong to small galaxies, and with both cuts on
    # the sample is empty -- which is the interaction the warning above is for.
    t = _catalogue(2000)
    out = select_hosts(t, band="r", flux_range=(1000.0, 5000.0),
                       min_reff_arcsec=None)
    flux = np.asarray(out["r_cModelFlux"])
    assert len(out) and flux.min() > 1000.0 and flux.max() <= 5000.0


# -- PSF moments -----------------------------------------------------------


@pytest.mark.parametrize("sx,sy", [(2.0, 2.0), (2.5, 1.5), (4.0, 3.0)])
def test_adaptive_moments_are_exact_on_a_gaussian(sx, sy):
    n = 61
    yy, xx = np.mgrid[0:n, 0:n]
    g = np.exp(-((xx - 30) ** 2 / (2 * sx**2) + (yy - 30) ** 2 / (2 * sy**2)))
    m = _adaptive_moments(g)
    assert m["psf_ixx"] == pytest.approx(sx**2, rel=1e-3)
    assert m["psf_iyy"] == pytest.approx(sy**2, rel=1e-3)
    assert m["psf_sigma"] == pytest.approx(np.sqrt(sx * sy), rel=1e-3)
    assert m["psf_fwhm"] == pytest.approx(m["psf_sigma"] * SIGMA_TO_FWHM)


def test_adaptive_moments_resist_wings():
    """DP2's PSF object has no moment methods, so these are measured from the
    kernel -- and unweighted moments of a kernel with wings are badly wrong."""
    n = 61
    yy, xx = np.mgrid[0:n, 0:n]
    core = np.exp(-((xx - 30) ** 2 + (yy - 30) ** 2) / (2 * 2.0**2))
    withwings = core + 0.002 * np.exp(-np.hypot(xx - 30, yy - 30) / 12.0)
    unweighted = (withwings * (xx - 30) ** 2).sum() / withwings.sum()
    assert unweighted > 10.0, "the naive estimator really is this bad"
    assert _adaptive_moments(withwings)["psf_ixx"] == pytest.approx(4.0, rel=0.05)


def test_adaptive_moments_degrade_cleanly_on_an_empty_kernel():
    m = _adaptive_moments(np.zeros((21, 21)))
    assert all(np.isnan(v) for v in m.values())


# -- data ids --------------------------------------------------------------


class _ModernDataCoordinate:
    """daf_butler >= v27: exposes ``.mapping``/``.required``, is not a Mapping.

    With ``__getitem__`` but no ``keys``, ``dict()`` falls through to sequence
    iteration and asks for element 0, which is the ``KeyError: 0`` seen on the
    stack rather than anything that names the real problem.
    """

    def __init__(self, values):
        self._values = dict(values)
        self.mapping = MappingProxyType(self._values)
        self.required = MappingProxyType(self._values)

    def __getitem__(self, key):
        return self._values[key]

    def __str__(self):
        return f"{{{', '.join(f'{k}: {v}' for k, v in self._values.items())}}}"


class _LegacyDataCoordinate(Mapping):
    """daf_butler < v27, where ``dict(data_id)`` worked."""

    def __init__(self, values):
        self._values = dict(values)

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


DATA_ID = {"band": "r", "skymap": "lsst_cells_v2", "tract": 5063, "patch": 17}


def test_plain_dict_on_a_modern_data_id_is_the_bug_we_are_fixing():
    with pytest.raises(KeyError):
        dict(_ModernDataCoordinate(DATA_ID))


@pytest.mark.parametrize("cls", [_ModernDataCoordinate, _LegacyDataCoordinate])
def test_data_id_dict_works_on_both_butler_generations(cls):
    assert _data_id_dict(cls(DATA_ID)) == DATA_ID


def test_data_id_dict_falls_back_to_required_when_mapping_is_absent():
    coord = _ModernDataCoordinate(DATA_ID)
    del coord.mapping
    assert _data_id_dict(coord) == DATA_ID


def test_data_id_dict_keeps_provenance_when_nothing_works():
    class _Opaque:
        def __getitem__(self, key):
            raise KeyError(key)

        def __str__(self):
            return "opaque-data-id"

    out = _data_id_dict(_Opaque())
    assert out == {"repr": "opaque-data-id"}


def test_extracted_fields_survive_the_conversion():
    fields = _data_id_dict(_ModernDataCoordinate(DATA_ID))
    assert str(fields.get("band", "?")) == "r"
    assert int(fields.get("tract", -1)) == 5063
    assert int(fields.get("patch", -1)) == 17


# -- host size -------------------------------------------------------------


def _sersic(reff, unknown=False, no_data=False, minor=None):
    reff = np.atleast_1d(np.asarray(reff, float))
    n = len(reff)
    return Table({
        "sersic_reff_major": reff,
        "sersic_reff_minor": np.full(n, 0.6) * reff if minor is None else minor,
        "sersic_unknown_flag": np.full(n, unknown),
        "sersic_no_data_flag": np.full(n, no_data),
    })


def test_size_comes_from_the_multiband_sersic_fit():
    """One morphology fit to all six bands at once, so the column carries no band
    prefix and does not inherit the band-to-band scatter of a per-band fit."""
    assert "sersic_reff_major" in OBJECT_COLUMNS
    assert not any(c.startswith("{b}_sersic") or "reff" in c
                   for c in OBJECT_BAND_COLUMNS)
    assert host_half_light_arcsec(_sersic([3.2, 5.0])) == pytest.approx([3.2, 5.0])


def test_a_failed_sersic_fit_is_not_a_size():
    """The flags are the only thing separating a fit from whatever was left in
    the column when it failed, and a size cut compares NaN away but not junk."""
    assert np.isnan(host_half_light_arcsec(_sersic([4.0], unknown=True))[0])
    assert np.isnan(host_half_light_arcsec(_sersic([4.0], no_data=True))[0])


def test_nonsensical_radii_are_nan():
    out = host_half_light_arcsec(_sersic([np.nan, 0.0, -1.0, 4.0]))
    assert np.isnan(out[:3]).all() and out[3] == 4.0


def test_the_minor_axis_is_available_but_not_the_default():
    t = _sersic([5.0])
    assert host_half_light_arcsec(t, axis="minor")[0] == pytest.approx(3.0)
    with pytest.raises(ValueError):
        host_half_light_arcsec(t, axis="circularised")


def test_passing_a_band_fails_loudly():
    """The fit is multiband. A call written against the old per-band signature
    must not be silently reinterpreted."""
    with pytest.raises(ValueError, match="axis"):
        host_half_light_arcsec(_sersic([5.0]), "r")


def test_a_missing_sersic_column_is_refused_not_guessed():
    t = _sersic([5.0])
    t.remove_column("sersic_reff_major")
    with pytest.raises(KeyError, match="sersic"):
        host_half_light_arcsec(t)


def test_all_six_bands_carry_photometry():
    """Corrects an earlier reading of the rendered HTML schema page, which is
    large enough that an excerpt gives a confidently wrong answer. The schema
    YAML has u..y for cModelFlux, ixx and sersicFlux alike."""
    assert PHOTOMETRY_BANDS == ("u", "g", "r", "i", "z", "y")


def test_small_hosts_are_cut():
    """The catalogue is dominated by galaxies a pixel or two across, which carry
    no structure for a prior to learn from."""
    t = _sized_catalogue(3000, scale=0.6, seed=7)
    out = select_hosts(t, band="r", min_reff_arcsec=3.0)
    assert len(out) and np.all(host_half_light_arcsec(out) >= 3.0)
    assert len(out) < len(select_hosts(t, band="r", min_reff_arcsec=None))


def test_the_size_cut_can_be_turned_off():
    t = _sized_catalogue(3000, scale=0.6, seed=7)
    loose = select_hosts(t, band="r", min_reff_arcsec=None)
    assert np.nanmin(host_half_light_arcsec(loose)) < 3.0


def test_hosts_without_a_sersic_fit_do_not_pass_the_size_cut():
    """NaN is not a size. It must not compare its way through the cut."""
    t = _sized_catalogue(400, scale=0.6, seed=8)
    t["sersic_no_data_flag"][:200] = True
    out = select_hosts(t, band="r", min_reff_arcsec=3.0)
    assert len(out) and np.all(np.asarray(out["objectId"]) >= 200)


# -- per-cell depth --------------------------------------------------------
#
# DP2 exposures share an integration time, so the number of visits in a cell is
# the depth of that cell, and the ratio across the cells a stamp covers is the
# step -- exactly, and without reading a pixel.  provenance.contributions is the
# only route: deep_coadd_input_summary is documented as patch-level and says
# outright that it does not record which visits went into each cell.


class _Grid:
    def __init__(self, cell=150):
        self.cell = cell

    def index_of(self, x, y):
        return SimpleNamespace(i=int(x) // self.cell, j=int(y) // self.cell)


def _coadd(contributions=None, grid=True):
    prov = None if contributions is None else SimpleNamespace(contributions=contributions)
    return SimpleNamespace(grid=_Grid() if grid else None, provenance=prov)


def _contributions(rows, cols=("cell_i", "cell_j")):
    """rows: (i, j, visit, detector) tuples."""
    a = np.asarray(rows, dtype=np.int64)
    return Table({cols[0]: a[:, 0], cols[1]: a[:, 1],
                  "visit": a[:, 2], "detector": a[:, 3]})


def test_visits_are_counted_per_cell():
    t = _contributions([(0, 0, 10, 1), (0, 0, 11, 1), (0, 0, 12, 1), (1, 0, 10, 1)])
    assert cell_visit_counts(_coadd(t)) == {(0, 0): 3, (1, 0): 1}


def test_a_visit_split_across_detectors_counts_once():
    """One row per (visit, detector, cell), so a visit whose detector boundary
    crosses a cell appears twice. It is still one visit of depth."""
    t = _contributions([(0, 0, 10, 1), (0, 0, 10, 2), (0, 0, 11, 1)])
    assert cell_visit_counts(_coadd(t)) == {(0, 0): 2}


@pytest.mark.parametrize("cols", [("cell_i", "cell_j"), ("cell_x", "cell_y"), ("i", "j")])
def test_the_cell_columns_are_resolved_by_trial(cols):
    """The API documents the table as {visit, detector, cell} without pinning the
    column names, and CellIJ does not survive into an astropy column as one
    object."""
    t = _contributions([(0, 0, 10, 1), (0, 0, 11, 1)], cols=cols)
    assert cell_visit_counts(_coadd(t)) == {(0, 0): 2}


def test_unrecognised_columns_degrade_rather_than_crash(caplog):
    t = Table({"cell_index": [0, 0], "visit": [10, 11]})
    with caplog.at_level("WARNING"):
        assert cell_visit_counts(_coadd(t)) == {}
    assert "cell_index" in caplog.text


def test_missing_provenance_is_not_an_error():
    assert cell_visit_counts(_coadd(None)) == {}
    assert cell_visit_counts(_coadd(Table({"visit": []}))) == {}


def test_cells_in_stamp_covers_the_rectangle_not_just_the_corners():
    """A 416 px stamp spans 3x3 cells of 150 px; the middle one has no corner in
    it and would be missed by a corner-only span."""
    cells = cells_in_stamp(_coadd(), x=225, y=225, size=416)
    assert len(cells) == 9 and (1, 1) in cells


def test_a_stamp_inside_one_cell_spans_one():
    assert cells_in_stamp(_coadd(), x=75, y=75, size=64) == [(0, 0)]


def test_stamp_depth_is_the_range_over_the_cells_covered():
    counts = {(0, 0): 30, (0, 1): 12, (1, 0): 28}
    assert stamp_depth(counts, [(0, 0), (0, 1), (1, 0)]) == (12, 30)


def test_a_cell_absent_from_the_table_contributed_nothing():
    assert stamp_depth({(0, 0): 30}, [(0, 0), (9, 9)]) == (0, 30)


def test_depth_is_unknown_rather_than_zero_without_provenance():
    """-1 is 'not measured'; 0 would claim a cell with no visits in it."""
    assert stamp_depth({}, [(0, 0)]) == (-1, -1)
    assert stamp_depth({(0, 0): 30}, []) == (-1, -1)


# -- working towards a target ----------------------------------------------
#
# n_hosts counts hosts, not cutouts: each yields at most one per band and the
# gate rejects a share of those. n_patches is the target, and extraction keeps
# drawing fresh hosts until it has them.


def test_excluded_hosts_are_not_offered_again():
    """Without this the top-up loop re-offers the same objects forever."""
    t = _sized_catalogue(400, scale=0.6, seed=11)
    first = select_hosts(t, band="r", n_hosts=20, seed=0)
    ids = {int(i) for i in first["objectId"]}
    second = select_hosts(t, band="r", n_hosts=20, seed=0, exclude_ids=ids)
    assert len(second) == 20
    assert not (ids & {int(i) for i in second["objectId"]})


def test_repeated_rounds_walk_the_catalogue_to_exhaustion():
    t = _sized_catalogue(300, scale=0.6, seed=12)
    available = len(select_hosts(t, band="r"))
    tried, rounds = set(), 0
    while rounds < 50:
        rounds += 1
        got = select_hosts(t, band="r", n_hosts=25, seed=rounds, exclude_ids=tried)
        if not len(got):
            break
        tried |= {int(i) for i in got["objectId"]}
    assert len(tried) == available and rounds <= 50


def test_the_next_batch_is_sized_from_the_observed_yield():
    """25 cutouts from 100 hosts, 90 still wanted -> ~360 hosts plus headroom."""
    assert next_batch(100, 100, 25, 90) == pytest.approx(468, abs=1)


def test_a_round_that_yielded_nothing_widens_rather_than_dividing_by_zero():
    assert next_batch(100, 100, 0, 90) == 400
    assert next_batch(None, 0, 0, 90) == 64


def test_the_batch_never_collapses_to_nothing():
    """A shortfall of one must not ask for zero hosts and spin."""
    assert next_batch(100, 100, 100, 1) >= 16
    assert next_batch(100, 100, 100, 0) >= 16


# -- accounting ------------------------------------------------------------


def test_every_reason_is_counted_not_just_the_first():
    """gate returns reasons in a fixed order, so counting only the first blames
    whichever check runs early. A plane gated last accounted for a quarter of a
    real run's rejections without appearing in the counts at all."""
    records = [
        {"status": "rejected", "reasons": "cell_depth:2.1>1.5;inner_INTERPOLATED:0.3>0.0"},
        {"status": "rejected", "reasons": "inner_INTERPOLATED:0.4>0.0"},
        {"status": "accepted"},
    ]
    out = _write_manifest(Path(tempfile.mkdtemp()), records, [])
    assert out["rejection_counts"] == {"inner_INTERPOLATED": 2, "cell_depth": 1}
    assert out["first_rejection_counts"] == {"cell_depth": 1, "inner_INTERPOLATED": 1}
    assert out["n_rejected"] == 2 and out["n_attempts"] == 3


def test_a_reason_is_not_double_counted_within_one_stamp():
    records = [{"status": "rejected", "reasons": "no_data:0.1>0.02;no_data:0.1>0.02"}]
    out = _write_manifest(Path(tempfile.mkdtemp()), records, [])
    assert out["rejection_counts"] == {"no_data": 1}


def test_diagnostic_percentiles_come_from_every_attempt():
    """Thresholds should be chosen from what the field looks like, accepted
    stamps included, not from the rejected tail alone."""
    records = [{"status": "accepted", "diag_cell_depth_ratio": 1.0 + i / 100}
               for i in range(100)]
    out = _write_manifest(Path(tempfile.mkdtemp()), records, [])
    pct = out["diagnostic_percentiles"]["cell_depth_ratio"]
    assert pct["p50"] == pytest.approx(1.495, abs=0.01)
    assert pct["p95"] == pytest.approx(1.94, abs=0.02)


def test_percentiles_skip_diagnostics_that_were_never_measured():
    out = _write_manifest(Path(tempfile.mkdtemp()), [{"status": "accepted"}], [])
    assert out["diagnostic_percentiles"] == {}


def test_hosts_are_restricted_to_the_field_being_swept():
    """The object table is a whole tract, ~1.7 deg across; a 0.3 deg sweep covers
    under a tenth of it. Hosts outside the disc land in patches that are never
    loaded and vanish without even a rejection record."""
    t = _sized_catalogue(2000, seed=21)  # spread over ~0.3 deg in each axis
    field = _within_radius(t, 53.13, -28.10, 0.1)
    assert 0 < len(field) < len(t)
    r = np.asarray(field["coord_ra"]); d = np.asarray(field["coord_dec"])
    cosd = np.cos(np.deg2rad(-28.10))
    assert np.all(np.hypot((r - 53.13) * cosd, d + 28.10) <= 0.1 + 1e-12)


def test_the_field_cut_keeps_everything_when_the_radius_is_generous():
    t = _sized_catalogue(500, seed=22)
    assert len(_within_radius(t, 53.13, -28.10, 10.0)) == len(t)


# -- the whole footprint ---------------------------------------------------
#
# Big galaxies are rare per square degree, so a 3" cut on a 0.3 deg field finds
# almost nothing. Hosts come from every object table instead, cut per tract so
# that what is held is the host list rather than the footprint.


class _Ref:
    def __init__(self, tract, patch=None, band=None):
        d = {"skymap": "lsst_cells_v2", "tract": tract}
        if patch is not None:
            d["patch"] = patch
        if band is not None:
            d["band"] = band
        self.dataId = _LegacyDataCoordinate(d)


class _Butler:
    """Enough butler to drive the catalogue scan."""

    def __init__(self, tables, coadds=None):
        self.tables = tables  # {tract: Table}
        self.coadds = coadds or {}
        self.reads = []

    def query_datasets(self, kind, data_id=None, where="", bind=None,
                       limit=None, **kw):
        data_id = data_id or {}
        tract = data_id.get("tract")
        if kind == "object":
            tracts = [tract] if tract is not None else sorted(self.tables)
            refs = [_Ref(t) for t in tracts if t in self.tables]
            return refs[:limit] if limit else refs
        return list(self.coadds.get(tract, []))

    def get(self, ref, parameters=None):
        tract = _data_id_dict(ref.dataId)["tract"]
        self.reads.append(tract)
        t = self.tables[tract]
        cols = (parameters or {}).get("columns")
        return t[[c for c in cols if c in t.colnames]] if cols else t


def _footprint(n_tracts=4, per_tract=300, seed=30):
    return {1000 + i: _sized_catalogue(per_tract, scale=0.6, seed=seed + i,
                                      tract=1000 + i)
            for i in range(n_tracts)}


def test_the_catalogue_is_built_from_every_object_table():
    tables = _footprint()
    butler = _Butler(tables)
    pool = build_host_catalogue(butler, source="butler", min_reff_arcsec=3.0)
    assert set(butler.reads) == set(tables)
    assert len(pool) and np.all(host_half_light_arcsec(pool) >= 3.0)


def test_only_survivors_are_kept_in_memory():
    """An object table is ~700k rows and the footprint is ~1000 of them, so the
    cuts have to run per tract rather than on a concatenation of all of them."""
    tables = _footprint()
    pool = build_host_catalogue(_Butler(tables), source="butler", min_reff_arcsec=3.0)
    assert len(pool) < sum(len(t) for t in tables.values())


def test_sampling_does_not_happen_during_the_scan():
    """A stratified draw has to see the whole pool; drawing per tract would
    stratify within tracts and not across them. Asking for a sample during the
    scan is refused rather than quietly applied per tract."""
    with pytest.raises(TypeError):
        build_host_catalogue(_Butler(_footprint()), source="butler", min_reff_arcsec=None, n_hosts=5)


def test_limit_tracts_bounds_a_test_run():
    tables = _footprint(n_tracts=6)
    butler = _Butler(tables)
    build_host_catalogue(butler, source="butler", min_reff_arcsec=None, limit_tracts=2)
    assert len(set(butler.reads)) == 2


def test_a_tract_that_will_not_read_does_not_end_the_scan():
    tables = _footprint(n_tracts=3)
    butler = _Butler(tables)
    bad = sorted(tables)[1]
    real_get = butler.get

    def get(ref, parameters=None):
        if _data_id_dict(ref.dataId)["tract"] == bad:
            raise RuntimeError("corrupt")
        return real_get(ref, parameters)

    butler.get = get
    pool = build_host_catalogue(butler, source="butler", min_reff_arcsec=3.0)
    assert len(pool) and bad not in set(np.asarray(pool["tract"], dtype=int))


def test_the_catalogue_is_cached_and_reused(tmp_path):
    """Scanning a thousand object tables takes minutes; it should happen once."""
    tables = _footprint()
    butler = _Butler(tables)
    first = build_host_catalogue(butler, source="butler", min_reff_arcsec=3.0,
                                 cache=tmp_path / "hosts.parquet")
    n_reads = len(butler.reads)
    second = build_host_catalogue(butler, source="butler", min_reff_arcsec=3.0,
                                  cache=tmp_path / "hosts.parquet")
    assert len(butler.reads) == n_reads  # nothing re-read
    assert len(second) == len(first)


def test_nothing_surviving_anywhere_is_an_error_not_an_empty_run():
    tables = _footprint()
    with pytest.raises(RuntimeError, match="min_reff_arcsec"):
        build_host_catalogue(_Butler(tables), source="butler", min_reff_arcsec=1e6)


def test_coadd_refs_come_from_the_patches_hosts_are_in():
    """A patch holds one or two hosts under a cut this selective, so sweeping
    every patch overlapping a field loads a great many that hold none."""
    coadds = {7: [_Ref(7, patch=p, band="r") for p in range(20)]}
    butler = _Butler({}, coadds)
    got = coadd_refs_for_tract(butler, 7, [3, 11], bands=BANDS)
    assert sorted(int(_data_id_dict(r.dataId)["patch"]) for r in got) == [3, 11]


def test_asking_for_no_patches_queries_nothing():
    butler = _Butler({}, {7: [_Ref(7, patch=0, band="r")]})
    assert coadd_refs_for_tract(butler, 7, [], bands=BANDS) == []


def test_the_flux_ceiling_is_high_enough_for_the_size_cut(caplog):
    """A 3" half-light radius at an ordinary 22 mag/arcsec^2 is r ~ 17.6, nine
    times brighter than the old 36000 nJy ceiling. A ceiling set for a 1"
    population annihilates the size cut, so the conflict is called out."""
    t = _sized_catalogue(2000, scale=0.6, seed=31)
    with caplog.at_level("WARNING"):
        select_hosts(t, band="r", min_reff_arcsec=3.0, flux_range=(360.0, 36000.0))
    assert "fighting the size cut" in caplog.text


# -- reading only the stamp ------------------------------------------------


class _Components:
    """A butler that serves components and bbox reads, and counts both."""

    def __init__(self, serve_components=True, serve_bbox=True):
        self.serve_components = serve_components
        self.serve_bbox = serve_bbox
        self.component_reads = []
        self.whole_reads = 0
        self.bbox_reads = 0

    def get(self, what, dataId=None, parameters=None):
        if isinstance(what, str) and "." in what:
            if not self.serve_components:
                raise RuntimeError("components not served here")
            self.component_reads.append(what.split(".", 1)[1])
            return SimpleNamespace(name=what)
        if parameters and "bbox" in parameters:
            if not self.serve_bbox:
                raise RuntimeError("bbox reads not served here")
            self.bbox_reads += 1
            return SimpleNamespace(bbox=parameters["bbox"])
        self.whole_reads += 1
        return SimpleNamespace(whole=True)


def test_components_are_read_without_touching_pixels():
    """A patch is 4100 px square and a stamp is 416, so characterising a patch
    through components and then reading only the stamp is two orders of
    magnitude less I/O -- and hosts are too thinly spread for a second stamp to
    amortise a whole read against."""
    butler = _Components()
    ref = _Ref(5063, patch=12, band="r")
    for role in ("wcs", "bbox", "psf", "grid", "provenance"):
        assert read_component(butler, ref, role) is not None
    assert butler.component_reads == ["sky_projection", "bbox", "psf", "grid",
                                      "provenance"]
    assert butler.whole_reads == 0


def test_a_refused_component_is_none_rather_than_an_exception():
    """The caller falls back to the whole patch; it must not have to catch."""
    butler = _Components(serve_components=False)
    assert read_component(butler, _Ref(5063, patch=12, band="r"), "wcs") is None


def test_visit_counts_work_off_a_component_or_a_coadd():
    """provenance can come from a component read or off a patch read whole."""
    rows = [(0, 0, 10, 1), (0, 0, 11, 1)]
    prov = SimpleNamespace(contributions=_contributions(rows))
    assert cell_visit_counts(prov) == {(0, 0): 2}
    assert cell_visit_counts(SimpleNamespace(provenance=prov)) == {(0, 0): 2}


def test_cells_work_off_a_grid_or_a_coadd():
    grid = _Grid()
    assert cells_in_stamp(grid, 225, 225, 416) == \
        cells_in_stamp(SimpleNamespace(grid=grid), 225, 225, 416)


# -- TAP -------------------------------------------------------------------
#
# The host cuts are a selection, and a selection is what a query service is for:
# the footprint is ~10^9 rows and the survivors ~10^4, so filtering there rather
# than here is the difference between moving the survivors and moving the
# catalogue. The pixels are the opposite case -- those are already local.


class _Job:
    def __init__(self, service, query):
        self.service, self.query = service, query
        self.phase = "PENDING"
        self.deleted = False

    def run(self):
        self.phase = "EXECUTING"

    def wait(self, phases=(), timeout=None):
        self.phase = self.service.end_phase

    def raise_if_error(self):
        if self.phase == "ERROR":
            raise RuntimeError("ADQL error: bad column")

    def fetch_result(self):
        return SimpleNamespace(to_table=lambda: self.service.table)

    def delete(self):
        self.deleted = True


class _Tap:
    def __init__(self, table, end_phase="COMPLETED"):
        self.table, self.end_phase = table, end_phase
        self.jobs = []

    def submit_job(self, query):
        job = _Job(self, query)
        self.jobs.append(job)
        return job


def test_the_selective_cuts_go_into_the_query():
    q = host_adql(bands=("r",), min_reff_arcsec=3.0, flux_range=(360.0, 3.0e6))
    assert "sersic_reff_major >= 3.0" in q
    assert "r_cModelFlux > 360.0" in q and "r_cModelFlux <= 3000000.0" in q
    assert "FROM dp2.Object" in q
    assert "sersic_reff_major" in q.split("FROM")[0]  # and comes back as a column


def test_boolean_flags_stay_out_of_the_query():
    """How a boolean column compares is backend-specific, and a wrong guess
    silently returns nothing. They cost nothing to apply locally."""
    q = host_adql(bands=("r",))
    assert "sersic_unknown_flag" in q.split("FROM")[0]  # fetched
    assert "sersic_unknown_flag" not in q.split("WHERE")[1]  # not compared


def test_the_query_does_not_sort():
    """Sorting is expensive on a shared service and the stratified draw has to
    happen locally anyway."""
    assert "ORDER BY" not in host_adql()


def test_a_region_is_optional():
    assert "CONTAINS" not in host_adql()
    assert "CONTAINS" in host_adql(ra=53.13, dec=-28.1, radius_deg=0.3)


def test_top_bounds_a_test_query():
    assert host_adql(top=25).startswith("SELECT TOP 25 ")
    assert host_adql().startswith("SELECT objectId")


def test_tap_results_still_get_the_local_cuts():
    """The service applied the numeric cuts; the Sersic failure flags, the
    point-source cross-check and the cross-tract dedupe are not expressible
    there."""
    table = _sized_catalogue(200, scale=0.6, seed=40)
    table["sersic_no_data_flag"][:100] = True
    pool = build_host_catalogue(tap_service=_Tap(table), min_reff_arcsec=3.0)
    assert len(pool) and np.all(np.asarray(pool["objectId"]) >= 100)


def test_the_tap_job_is_deleted_even_when_it_fails():
    """An abandoned job sits on a shared service."""
    service = _Tap(_sized_catalogue(50, seed=41), end_phase="ERROR")
    with pytest.raises(RuntimeError):
        build_host_catalogue(tap_service=service, min_reff_arcsec=3.0)
    assert service.jobs[0].deleted


def test_a_job_that_ends_in_an_unexpected_phase_is_an_error():
    service = _Tap(_sized_catalogue(50, seed=42), end_phase="ABORTED")
    with pytest.raises(RuntimeError, match="ABORTED"):
        run_adql(service, "SELECT 1")


def test_tap_results_can_be_cached_for_an_offline_run(tmp_path):
    """A batch node may have no network. Query once where there is one, cache,
    and extraction runs from the cache with no service at all."""
    service = _Tap(_sized_catalogue(200, scale=0.6, seed=43))
    build_host_catalogue(tap_service=service, min_reff_arcsec=3.0,
                         cache=tmp_path / "hosts.parquet")
    pool = build_host_catalogue(tap_service=None, butler=None,
                                min_reff_arcsec=3.0,
                                cache=tmp_path / "hosts.parquet")
    assert len(pool) and len(service.jobs) == 1


def test_an_unknown_source_is_refused():
    with pytest.raises(ValueError, match="tap"):
        build_host_catalogue(source="qserv")


def test_the_butler_scan_needs_a_butler():
    with pytest.raises(ValueError, match="butler"):
        build_host_catalogue(source="butler")


# -- reaching TAP from off the RSP -----------------------------------------
#
# TAP is an IVOA standard and the RSP endpoint is an ordinary TAP service behind
# a bearer token, so pyvo speaks to it directly. lsst.rsp exists only on the RSP
# itself, where it wraps exactly this.


def test_the_token_comes_from_the_environment(monkeypatch):
    for var in TOKEN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ACCESS_TOKEN", "gt-secret")
    assert rsp_token() == "gt-secret"


def test_the_token_can_come_from_a_file(monkeypatch, tmp_path):
    for var in TOKEN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    path = tmp_path / "tok"
    path.write_text("gt-from-disk\n")
    monkeypatch.setattr("rubin_host_prior.rubin.extract.TOKEN_PATHS", (str(path),))
    assert rsp_token() == "gt-from-disk"


def test_a_missing_token_says_how_to_get_one(monkeypatch):
    for var in TOKEN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("rubin_host_prior.rubin.extract.TOKEN_PATHS", ())
    with pytest.raises(RuntimeError, match="data.lsst.cloud"):
        rsp_token()


def test_the_token_goes_only_to_the_service():
    """A session header follows redirects: one redirect off-host and the token
    has been handed to whoever answered."""
    auth = _BearerForPrefix("gt-secret", ["https://data.lsst.cloud/api/tap"])

    def header_for(url):
        req = SimpleNamespace(url=url, headers={})
        return auth(req).headers.get("Authorization")

    assert header_for("https://data.lsst.cloud/api/tap") == "Bearer gt-secret"
    assert header_for("https://data.lsst.cloud/api/tap/async?x=1") == "Bearer gt-secret"
    assert header_for("https://data.lsst.cloud/api/other") is None
    assert header_for("https://elsewhere.example/api/tap") is None
    # A prefix match must not be a string-prefix match on a different host.
    assert header_for("https://data.lsst.cloud/api/tap-evil") is None


def test_the_endpoint_is_discovered(monkeypatch):
    doc = {"datasets": {"dp2": {"services": {"tap": {"url": "https://x/api/tap"}}}}}
    monkeypatch.setattr(
        "rubin_host_prior.rubin.extract.requests.get",
        lambda *a, **k: SimpleNamespace(raise_for_status=lambda: None,
                                        json=lambda: doc),
    )
    assert discover_tap_url("dp2") == "https://x/api/tap"


def test_discovery_failure_falls_back_to_a_known_endpoint(monkeypatch):
    def boom(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr("rubin_host_prior.rubin.extract.requests.get", boom)
    assert discover_tap_url("dp2") == TAP_URL_FALLBACK


def test_an_unlisted_release_falls_back_rather_than_crashing(monkeypatch):
    monkeypatch.setattr(
        "rubin_host_prior.rubin.extract.requests.get",
        lambda *a, **k: SimpleNamespace(raise_for_status=lambda: None,
                                        json=lambda: {"datasets": {}}),
    )
    assert discover_tap_url("dp7") == TAP_URL_FALLBACK


# -- why a token was rejected ----------------------------------------------


def _response(status=200, payload=None):
    return SimpleNamespace(
        status_code=status, ok=200 <= status < 300,
        json=lambda: payload if payload is not None else {},
    )


def test_the_token_source_is_reported_but_never_the_token(monkeypatch, caplog):
    """ACCESS_TOKEN is a generic name that other software sets too, and picking
    up somebody else's value looks exactly like a rejected RSP token."""
    for var in TOKEN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ACCESS_TOKEN", "gt-abcdef.ghijklmnop")
    with caplog.at_level("INFO"):
        rsp_token()
    assert "$ACCESS_TOKEN" in caplog.text
    assert "gt-..." in caplog.text
    assert "abcdef" not in caplog.text and "ghijklmnop" not in caplog.text


def test_find_token_reports_a_file_source(monkeypatch, tmp_path):
    for var in TOKEN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    path = tmp_path / "tok"
    path.write_text("gt-x.y\n")
    monkeypatch.setattr("rubin_host_prior.rubin.extract.TOKEN_PATHS", (str(path),))
    assert find_token() == ("gt-x.y", str(path))


def test_a_rejected_token_says_so_before_any_query(monkeypatch):
    monkeypatch.setattr("rubin_host_prior.rubin.extract.requests.get",
                        lambda *a, **k: _response(401))
    with pytest.raises(RuntimeError, match="expired, revoked"):
        token_info("gt-nope")


def _check(info, monkeypatch=None):
    import rubin_host_prior.rubin.extract as ex
    real = ex.requests.get
    ex.requests.get = lambda *a, **k: _response(200, info)
    try:
        check_tap_scope("gt-x")
    finally:
        ex.requests.get = real


def test_a_valid_token_without_the_tap_scope_is_caught_early():
    """Otherwise this surfaces as a bare 401 from inside a TAP job submission,
    which says nothing about what to fix."""
    info = {"username": "cstone", "scopes": ["read:image", "exec:notebook"]}
    with pytest.raises(RuntimeError, match="read:tap"):
        _check(info)


def test_a_token_with_the_scope_passes():
    _check({"username": "cstone", "scopes": ["read:tap", "read:image"]})


def test_being_unable_to_check_does_not_block_a_run(monkeypatch):
    """Failing to check a token is not the same as the token being bad."""
    def boom(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr("rubin_host_prior.rubin.extract.requests.get", boom)
    assert token_info("gt-x") == {}
    check_tap_scope("gt-x")  # must not raise


def test_an_unexpected_status_is_not_treated_as_a_rejection(monkeypatch):
    monkeypatch.setattr("rubin_host_prior.rubin.extract.requests.get",
                        lambda *a, **k: _response(503))
    assert token_info("gt-x") == {}


# -- rows that have no position --------------------------------------------


def test_a_non_finite_position_is_not_in_the_patch():
    """Two ways a position arrives non-finite: a catalogue row with no
    coordinates, or a projection of somewhere the patch does not cover. Both
    mean 'not here', and neither is worth ending a run over."""
    box = SimpleNamespace(contains=lambda x, y: True)
    assert _fits_in_patch(box, np.nan, 100.0, 416) is False
    assert _fits_in_patch(box, 100.0, np.inf, 416) is False
    assert _fits_in_patch(box, 100.0, 100.0, 416) is True


def test_a_stamp_is_never_centred_on_a_non_finite_position():
    with pytest.raises(ValueError, match="cannot centre"):
        _stamp_box(np.nan, 10.0, 416)


def test_a_projection_that_refuses_a_position_gives_nan_not_an_exception():
    class _Wcs:
        def sky_to_pixel(self, coord):
            raise ValueError("outside the projection")

    xs, ys = _sky_to_pixel(_Wcs(), [53.1], [-28.1])
    assert np.isnan(xs[0]) and np.isnan(ys[0])


def test_non_finite_input_coordinates_never_reach_the_projection():
    class _Wcs:
        def sky_to_pixel(self, coord):
            raise AssertionError("should not have been called")

    xs, ys = _sky_to_pixel(_Wcs(), [np.nan], [np.nan])
    assert np.isnan(xs[0]) and np.isnan(ys[0])


def test_null_floats_from_tap_become_nan():
    """np.asarray on a masked column hands back the raw buffer with no hint that
    part of it is not data."""
    t = Table({"coord_ra": np.ma.array([53.1, 0.0], mask=[False, True])})
    out = unmask(t)
    assert out["coord_ra"][0] == 53.1 and np.isnan(out["coord_ra"][1])


def test_a_null_integer_is_flagged_and_made_unmatchable(caplog):
    """A garbage `patch` would file a host under a patch it is nowhere near."""
    t = Table({"patch": np.ma.array([3, 999], mask=[False, True])})
    with caplog.at_level("WARNING"):
        out = unmask(t)
    assert out["patch"][0] == 3 and out["patch"][1] == -1
    assert "patch" in caplog.text


def test_rows_without_a_position_are_dropped_with_a_count(caplog):
    t = Table({"coord_ra": [53.1, np.nan, 53.3],
               "coord_dec": [-28.1, -28.2, np.nan]})
    with caplog.at_level("WARNING"):
        out = with_positions(t)
    assert len(out) == 1 and "2 host candidate" in caplog.text


def test_refs_are_constrained_by_data_id_not_by_a_where_clause():
    """The expression language bit once and silently: in `where="tract = :tract"`
    the bind key shadows the dimension of the same name, so it resolved as
    `tract = tract` -- true for every row. The query returned the whole repo,
    truncated at 20000, and hosts were matched against same-numbered patches in
    other tracts, which projected ~200000 px away."""
    seen = {}

    class _Recording(_Butler):
        def query_datasets(self, kind, data_id=None, **kw):
            seen.update(kind=kind, data_id=data_id, kw=kw)
            return []

    _Recording({}, {}).query_datasets  # keep the class used
    b = _Recording({}, {})
    coadd_refs_for_tract(b, 5063, [1, 2], bands=BANDS)
    assert seen["data_id"] == {"skymap": SKYMAP, "tract": 5063}
    assert not seen["kw"].get("where")
    assert seen["kw"].get("limit") is None


def test_refs_from_the_wrong_tract_are_dropped_and_reported(caplog):
    """Patch indices repeat across tracts, so a patch filter alone lets a ref
    from anywhere through -- which is exactly how the bug showed up."""
    coadds = {5063: [_Ref(5063, patch=3, band="r"), _Ref(99, patch=3, band="r")]}

    class _Leaky(_Butler):
        def query_datasets(self, kind, data_id=None, **kw):
            return list(coadds[5063])

    with caplog.at_level("WARNING"):
        got = coadd_refs_for_tract(_Leaky({}, {}), 5063, [3], bands=BANDS)
    assert len(got) == 1
    assert int(_data_id_dict(got[0].dataId)["tract"]) == 5063
    assert "not constraining tract" in caplog.text


def test_bands_are_filtered_client_side_too():
    coadds = {7: [_Ref(7, patch=1, band=b) for b in ("u", "r", "z")]}

    class _All(_Butler):
        def query_datasets(self, kind, data_id=None, **kw):
            return list(coadds[7])

    got = coadd_refs_for_tract(_All({}, {}), 7, [1], bands=("r", "z"))
    assert sorted(str(_data_id_dict(r.dataId)["band"]) for r in got) == ["r", "z"]


# -- patches with missing cells --------------------------------------------


class _CellIJ:
    """CellGridBounds.missing holds these, not tuples."""

    def __init__(self, i, j):
        self.i, self.j = i, j


class _Bounds:
    """CellGridBounds: the populated region, minus individually missing cells."""

    def __init__(self, x0, x1, y0, y1, missing=()):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.missing = set(missing)

    def contains(self, x, y):
        if not (self.x0 <= x < self.x1 and self.y0 <= y < self.y1):
            return False
        return (x // 150, y // 150) not in self.missing


def test_a_stamp_must_fit_the_cell_grid_not_just_the_image():
    """A patch at the edge of coverage has cells that were never built: its
    image bbox is the full patch while its cell grid covers only part of it, and
    slicing outside that raises rather than returning empty pixels."""
    image_bbox = _Bounds(24000, 27300, 17850, 21150)
    cell_bounds = _Bounds(26100, 27150, 17850, 21150)
    x, y = 24905.0, 20236.0  # inside the image, outside the cells
    assert _fits_in_patch(image_bbox, x, y, 416) is True
    assert _fits_in_patch(cell_bounds, x, y, 416) is False


def test_a_missing_cell_inside_a_stamp_is_caught_separately():
    """A corner test catches a stamp hanging off the edge of coverage but not a
    hole in the middle of one, and slicing across either raises."""
    bounds = SimpleNamespace(missing=[_CellIJ(5, 5)])
    covered = [(4, 4), (4, 5), (5, 4), (5, 5)]
    assert missing_cells(bounds, covered) == [(5, 5)]
    assert missing_cells(bounds, [(1, 1), (1, 2)]) == []


def test_a_patch_with_no_missing_cells_costs_nothing():
    assert missing_cells(SimpleNamespace(missing=frozenset()), [(0, 0)]) == []
    assert missing_cells(None, [(0, 0)]) == []


def test_the_stamp_corners_are_what_is_tested():
    """A centre inside the grid is not enough; the whole stamp has to be."""
    bounds = _Bounds(0, 3300, 0, 3300)
    assert _fits_in_patch(bounds, 1650.0, 1650.0, 416) is True
    assert _fits_in_patch(bounds, 100.0, 1650.0, 416) is False
