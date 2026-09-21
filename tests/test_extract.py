"""Extraction logic that does not need the LSST stack.

These cover the parts of the DP2 port most likely to go wrong silently: the
object-table columns (DP2 moved second moments per band and dropped
``detect_isPrimary`` entirely), deduplication across overlapping tracts, and the
PSF moments, which DP2's PSF object cannot compute for you.
"""

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import pytest

from rubin_host_prior.rubin.extract import (
    OBJECT_COLUMNS,
    PHOTOMETRY_BANDS,
    SIGMA_TO_FWHM,
    _adaptive_moments,
    _data_id_dict,
    dedupe_hosts,
    host_trace_radius_px,
    select_hosts,
)

Table = pytest.importorskip("astropy.table").Table


def _catalogue(n=3000, seed=0, band="r"):
    rng = np.random.default_rng(seed)
    return Table({
        "objectId": np.arange(n),
        "coord_ra": 53.13 + rng.normal(0, 0.1, n),
        "coord_dec": -28.10 + rng.normal(0, 0.1, n),
        "refExtendedness": np.ones(n),
        "refBand": [band] * n,
        f"{band}_cModelFlux": 10 ** rng.uniform(2.6, 4.5, n),
        f"{band}_blendedness": rng.beta(1.2, 8, n),
        f"{band}_ixx": 10 ** rng.uniform(0.4, 2.4, n),
        f"{band}_iyy": 10 ** rng.uniform(0.4, 2.4, n),
        f"{band}_ixy": rng.normal(0, 5, n),
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


# -- host selection --------------------------------------------------------


def test_size_stratification_actually_stratifies():
    """Regression for a bug that made this a no-op.

    Drawing equally from *quantile* bins is exactly a uniform sample, because
    quantile bins hold equal numbers by construction -- so the original
    implementation stratified nothing at all while claiming to. Equal-width bins
    in log size hold wildly unequal numbers, and an equal draw from each is what
    actually gets well-resolved hosts into the sample.
    """
    rng = np.random.default_rng(1)
    n = 4000
    size = 10 ** rng.exponential(0.25, n)  # steep, as a real catalogue is
    t = Table({
        "objectId": np.arange(n),
        "coord_ra": 53.13 + rng.normal(0, 0.3, n),
        "coord_dec": -28.10 + rng.normal(0, 0.3, n),
        "refExtendedness": np.ones(n),
        "r_cModelFlux": 10 ** rng.uniform(2.6, 4.5, n),
        "r_ixx": size, "r_iyy": size, "r_ixy": np.zeros(n),
    })
    parent = host_trace_radius_px(select_hosts(t, band="r"), "r")
    big = np.percentile(parent, 90)

    strat = host_trace_radius_px(select_hosts(t, band="r", n_hosts=50, seed=0), "r")
    flat = host_trace_radius_px(
        select_hosts(t, band="r", n_hosts=50, seed=0, size_stratified=False), "r"
    )
    assert np.mean(strat > big) > 4 * np.mean(flat > big)
    assert np.median(strat) > np.median(flat)


def test_stratification_cannot_invent_large_galaxies():
    """Asking for the whole population must return the whole population, not a
    size-skewed subset of it -- the gain is bounded by what exists."""
    rng = np.random.default_rng(2)
    n = 600
    size = 10 ** rng.exponential(0.25, n)
    t = Table({
        "objectId": np.arange(n),
        "coord_ra": 53.13 + rng.normal(0, 0.3, n),
        "coord_dec": -28.10 + rng.normal(0, 0.3, n),
        "refExtendedness": np.ones(n),
        "r_cModelFlux": 10 ** rng.uniform(2.6, 4.5, n),
        "r_ixx": size, "r_iyy": size, "r_ixy": np.zeros(n),
    })
    everything = len(select_hosts(t, band="r"))
    assert len(select_hosts(t, band="r", n_hosts=everything)) == everything


def test_stratification_still_returns_the_number_asked_for():
    """Sparse bins at the large end leave the quota unfilled; the remainder is
    topped up rather than silently returning fewer hosts."""
    rng = np.random.default_rng(3)
    n = 3000
    size = 10 ** rng.exponential(0.3, n)
    t = Table({
        "objectId": np.arange(n),
        "coord_ra": 53.13 + rng.normal(0, 0.3, n),
        "coord_dec": -28.10 + rng.normal(0, 0.3, n),
        "refExtendedness": np.ones(n),
        "r_cModelFlux": 10 ** rng.uniform(2.6, 4.5, n),
        "r_ixx": size, "r_iyy": size, "r_ixy": np.zeros(n),
    })
    available = len(select_hosts(t, band="r"))
    want = available // 2
    assert len(select_hosts(t, band="r", n_hosts=want, seed=0)) == want


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
