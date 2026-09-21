"""Extraction logic that does not need the LSST stack.

These cover the parts of the DP2 port most likely to go wrong silently: the
object-table columns (DP2 moved second moments per band and dropped
``detect_isPrimary`` entirely), deduplication across overlapping tracts, and the
PSF moments, which DP2's PSF object cannot compute for you.
"""

from collections.abc import Mapping
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from rubin_host_prior.rubin.extract import (
    OBJECT_COLUMNS,
    PHOTOMETRY_BANDS,
    SIGMA_TO_FWHM,
    _adaptive_moments,
    _data_id_dict,
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


def _catalogue(n=3000, seed=0, band="r"):
    rng = np.random.default_rng(seed)
    ixx = 10 ** rng.uniform(0.4, 2.4, n)
    iyy = 10 ** rng.uniform(0.4, 2.4, n)
    reff = np.sqrt(0.5 * (ixx + iyy)) * 0.2 * 1.177  # trace px -> half-light arcsec
    return Table({
        "objectId": np.arange(n),
        "coord_ra": 53.13 + rng.normal(0, 0.1, n),
        "coord_dec": -28.10 + rng.normal(0, 0.1, n),
        "refExtendedness": np.ones(n),
        "refBand": [band] * n,
        f"{band}_cModelFlux": 10 ** rng.uniform(2.6, 4.5, n),
        f"{band}_blendedness": rng.beta(1.2, 8, n),
        f"{band}_ixx": ixx,
        f"{band}_iyy": iyy,
        f"{band}_ixy": rng.normal(0, 5, n),
        # Half-light radius tracks the moments trace, as it does on the sky, so
        # the two size cuts are consistent with each other.
        f"{band}_cModel_exp_reff_major": reff,
        f"{band}_cModel_exp_reff_minor": 0.7 * reff,
        f"{band}_cModel_dev_reff_major": reff,
        f"{band}_cModel_dev_reff_minor": 0.7 * reff,
        f"{band}_cModel_fracDev": rng.uniform(0, 1, n),
    })


# -- object table columns --------------------------------------------------


def test_object_columns_do_not_ask_for_dp1_only_names():
    """DP2 has no band-independent shape_xx and no detect_* columns at all.
    Requesting a column that does not exist fails the whole read."""
    assert not any(c.startswith("shape_") for c in OBJECT_COLUMNS)
    assert not any(c.startswith("detect") for c in OBJECT_COLUMNS)
    assert {"objectId", "coord_ra", "coord_dec", "refExtendedness"} <= set(OBJECT_COLUMNS)


def test_photometry_bands_exclude_z_and_y():
    """Coadd images exist in all six bands, but the Object table carries
    photometry and shapes only for ugri."""
    assert PHOTOMETRY_BANDS == ("u", "g", "r", "i")
    for b in ("z", "y"):
        assert b not in PHOTOMETRY_BANDS


def test_selecting_on_a_band_without_photometry_is_refused():
    with pytest.raises(ValueError, match="no DP2 Object photometry"):
        select_hosts(_catalogue(), band="z")


# -- second moments --------------------------------------------------------


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


def _sized_catalogue(n, scale=0.25, seed=1, size_mult=1.0):
    """A catalogue with a steep size distribution, as a real one has."""
    rng = np.random.default_rng(seed)
    trace_sq = size_mult * 10 ** rng.exponential(scale, n)
    reff = np.sqrt(trace_sq) * 0.2 * 1.177  # trace px -> half-light arcsec
    return Table({
        "objectId": np.arange(n),
        "coord_ra": 53.13 + rng.normal(0, 0.3, n),
        "coord_dec": -28.10 + rng.normal(0, 0.3, n),
        "refExtendedness": np.ones(n),
        "r_cModelFlux": 10 ** rng.uniform(2.6, 4.5, n),
        "r_ixx": trace_sq, "r_iyy": trace_sq, "r_ixy": np.zeros(n),
        "r_cModel_exp_reff_major": reff,
        "r_cModel_exp_reff_minor": 0.7 * reff,
        "r_cModel_dev_reff_major": reff,
        "r_cModel_dev_reff_minor": 0.7 * reff,
        "r_cModel_fracDev": rng.uniform(0, 1, n),
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
    parent = host_half_light_arcsec(select_hosts(t, **kw), "r")
    big = np.percentile(parent, 90)

    strat = host_half_light_arcsec(select_hosts(t, n_hosts=50, seed=0, **kw), "r")
    flat = host_half_light_arcsec(
        select_hosts(t, n_hosts=50, seed=0, size_stratified=False, **kw), "r"
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
    t = _catalogue(2000)
    out = select_hosts(t, band="r", flux_range=(1000.0, 5000.0))
    flux = np.asarray(out["r_cModelFlux"])
    assert flux.min() > 1000.0 and flux.max() <= 5000.0


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


def _two_component(exp, dev, frac, band="r"):
    n = len(exp)
    return Table({
        f"{band}_cModel_exp_reff_major": np.asarray(exp, float),
        f"{band}_cModel_dev_reff_major": np.asarray(dev, float),
        f"{band}_cModel_fracDev": np.asarray(frac, float),
    })


def test_half_light_follows_the_component_that_has_the_flux():
    """DP2 publishes no combined cModel radius, and whichever component carries
    no flux has a radius to match. fracDev is the fit's own statement of how the
    flux divides, so weighting by it gives the runaway component no say."""
    t = _two_component(exp=[2.0, 2.0, 2.0], dev=[8.0, 8.0, 8.0], frac=[0.0, 1.0, 0.5])
    assert host_half_light_arcsec(t, "r") == pytest.approx([2.0, 8.0, 5.0])


def test_half_light_is_neither_the_larger_nor_the_smaller_component():
    """Taking max would admit small galaxies whose unconstrained component ran
    away; taking min would reject large ones whose component collapsed."""
    t = _two_component(exp=[0.4], dev=[9.0], frac=[0.05])
    r = float(host_half_light_arcsec(t, "r")[0])
    assert 0.4 < r < 9.0 and r == pytest.approx(0.83, abs=0.01)


def test_a_missing_component_shifts_the_weight_rather_than_zeroing_it():
    t = _two_component(exp=[3.0], dev=[np.nan], frac=[0.8])
    assert host_half_light_arcsec(t, "r") == pytest.approx([3.0])


def test_half_light_is_nan_when_neither_component_was_fit():
    t = _two_component(exp=[np.nan], dev=[np.nan], frac=[0.5])
    assert np.isnan(host_half_light_arcsec(t, "r")[0])


def test_a_missing_fracDev_is_refused_not_guessed():
    t = _two_component(exp=[1.0], dev=[1.0], frac=[0.5])
    t.remove_column("r_cModel_fracDev")
    with pytest.raises(KeyError, match="fracDev"):
        host_half_light_arcsec(t, "r")


def test_minor_axis_is_available_but_not_the_default():
    t = _catalogue(50)
    major = host_half_light_arcsec(t, "r", axis="major")
    minor = host_half_light_arcsec(t, "r", axis="minor")
    assert np.all(minor < major)
    with pytest.raises(ValueError):
        host_half_light_arcsec(t, "r", axis="circularised")


def test_small_hosts_are_cut():
    """The catalogue is dominated by galaxies a pixel or two across, which carry
    no structure for a prior to learn from."""
    t = _sized_catalogue(3000, scale=0.6, seed=7)
    out = select_hosts(t, band="r", min_reff_arcsec=1.0)
    assert len(out) and np.all(host_half_light_arcsec(out, "r") >= 1.0)
    assert len(out) < len(select_hosts(t, band="r", min_reff_arcsec=None))


def test_the_size_cut_can_be_turned_off():
    t = _sized_catalogue(3000, scale=0.6, seed=7)
    loose = select_hosts(t, band="r", min_reff_arcsec=None)
    assert np.nanmin(host_half_light_arcsec(loose, "r")) < 1.0


def test_hosts_without_a_cmodel_fit_do_not_pass_the_size_cut():
    """NaN is not a size. It must not compare its way through the cut."""
    t = _sized_catalogue(400, scale=0.6, seed=8)
    for col in ("r_cModel_exp_reff_major", "r_cModel_dev_reff_major"):
        t[col][:200] = np.nan
    out = select_hosts(t, band="r", min_reff_arcsec=1.0)
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
