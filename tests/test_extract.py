"""Extraction: the sweep end to end, and the pieces that fail quietly.

Most of what can go wrong here only goes wrong against a butler -- a data id
that stopped being a Mapping, a bind key that shadowed a dimension, a component
that turned out to be a property -- so the centre of this file is a real sweep
over ``tests/fakes``.  The rest covers the handful of pure functions whose
failures would not be obvious from a traceback.
"""

import numpy as np
import pytest

import rubin_host_prior.rubin.extract as ex
from rubin_host_prior.rubin.extract import (
    OBJECT_BAND_COLUMNS,
    OBJECT_COLUMNS,
    PHOTOMETRY_BANDS,
    SKYMAP,
    TOKEN_ENV_VARS,
    _BearerForPrefix,
    _data_id_dict,
    _write_manifest,
    build_host_catalogue,
    cell_visit_counts,
    cells_in_stamp,
    coadd_refs_for_tract,
    discover_tap_url,
    extract_patches,
    find_token,
    host_adql,
    host_half_light_arcsec,
    missing_cells,
    next_batch,
    run_adql,
    select_hosts,
    stamp_depth,
    unmask,
    with_positions,
)

import fakes

Table = pytest.importorskip("astropy.table").Table
pytest.importorskip("pandas")


# -- fixtures ---------------------------------------------------------------


def _catalogue(tract, ra0, dec0, n=40, seed=1, patches=(0,), reff_range=(0.0, 0.5)):
    """Hosts placed well inside the given patches of one tract."""
    rng = np.random.default_rng(seed)
    patch = rng.choice(patches, n)
    row, col = np.divmod(patch, 10)
    px = col * fakes.PATCH + rng.uniform(400, fakes.PATCH - 400, n)
    py = row * fakes.PATCH + rng.uniform(400, fakes.PATCH - 400, n)
    cosd = np.cos(np.deg2rad(dec0))
    reff = 10 ** rng.uniform(*reff_range, size=n)
    return Table({
        "objectId": np.arange(n) + tract * 10000,
        "coord_ra": ra0 + px * fakes.PIXEL_SCALE / cosd,
        "coord_dec": dec0 + py * fakes.PIXEL_SCALE,
        "refExtendedness": np.ones(n),
        "refBand": ["r"] * n,
        "tract": np.full(n, tract),
        "patch": patch.astype(int),
        "sersic_reff_major": reff,
        "sersic_reff_minor": 0.6 * reff,
        "sersic_index": rng.uniform(0.5, 6.0, n),
        "sersic_unknown_flag": np.zeros(n, bool),
        "sersic_no_data_flag": np.zeros(n, bool),
        "sersic_chi2_reduced": np.ones(n),
        **{f"{b}_{c}": v for b in PHOTOMETRY_BANDS
           for c, v in (("cModelFlux", 10 ** rng.uniform(5.0, 5.8, n)),
                        ("blendedness", rng.beta(1.2, 8, n)),
                        ("ixx", np.full(n, 30.0)),
                        ("iyy", np.full(n, 30.0)),
                        ("ixy", np.zeros(n)),
                        ("ixxPSF", np.full(n, 4.0)),
                        ("iyyPSF", np.full(n, 4.0)),
                        ("pixelFlags_saturatedCenter", np.zeros(n, bool)),
                        ("pixelFlags_interpolatedCenter", np.zeros(n, bool)))},
    })


TRACTS = {5063: (53.13, -28.10), 5064: (55.00, -28.10)}


@pytest.fixture
def butler(monkeypatch):
    objects = {t: _catalogue(t, *c, seed=t) for t, c in TRACTS.items()}
    fakes.install(monkeypatch, ex)
    return fakes.FakeButler(TRACTS, objects)


def _run(butler, tmp_path, **kw):
    kw.setdefault("bands", ("r", "i"))
    kw.setdefault("n_hosts", 40)
    kw.setdefault("n_patches", 12)
    return extract_patches(butler, out_dir=tmp_path, host_source="butler",
                           native_size=416, seed=0, **kw)


# -- the sweep --------------------------------------------------------------


def test_a_run_produces_the_stamps_asked_for_reading_only_their_pixels(
        butler, tmp_path):
    """Whole-patch reads are ~100x the I/O, and the fake butler refuses them, so
    a regression to the old fallback path fails here rather than on the cluster.
    """
    summary = _run(butler, tmp_path)

    assert summary["n_accepted"] == 12
    assert butler.reads["bbox"] == 12 and butler.reads["whole"] == 0
    # Components are cheap and per patch, not per stamp.
    assert butler.reads["component"] < butler.reads["bbox"]

    from rubin_host_prior.data.shards import ShardSet

    shards = ShardSet.from_dir(tmp_path / "shards")
    assert len(shards) == 12
    assert shards.load("image").shape == (12, 416, 416)
    assert (tmp_path / "hosts.parquet").exists()
    assert (tmp_path / "manifest.parquet").exists()
    assert summary["n_rejected"] == 0


def test_only_the_image_is_stored(butler, tmp_path):
    """The prior is a distribution over pixels; variance, mask and PSF are what
    the gate is made of and are dropped once it has run."""
    import h5py

    _run(butler, tmp_path)
    with h5py.File(sorted((tmp_path / "shards").glob("*.h5"))[0], "r") as f:
        assert set(f) == {"image", "meta"}
        assert "psf" not in f and "variance" not in f
        # but what the gate saw survives as scalars
        assert {"sky_noise", "variance_step", "frac_no_data"} <= set(f["meta"])


def test_every_stamp_lands_where_it_was_asked_for(butler, tmp_path):
    """The guard against DP2's two pixel-origin conventions: mixing them
    displaces a stamp by up to a patch, which still looks like plausible sky."""
    import pandas as pd

    _run(butler, tmp_path)
    manifest = pd.read_parquet(tmp_path / "manifest.parquet")
    assert (manifest["centre_sep_arcsec"] < 1.0).all()


def test_a_host_is_not_extracted_twice_in_the_same_band(butler, tmp_path):
    """Tracts and patches overlap, so a host near a boundary is covered more
    than once and would otherwise be silently weighted up in training."""
    import pandas as pd

    _run(butler, tmp_path, n_patches=None, max_patches=None, n_hosts=8)
    manifest = pd.read_parquet(tmp_path / "manifest.parquet")
    accepted = manifest[manifest["status"] == "accepted"]
    assert not accepted.duplicated(subset=["host_id", "band"]).any()


def test_stamps_over_cells_that_were_never_built_are_rejected(monkeypatch,
                                                              tmp_path):
    """A patch at the edge of coverage has an image bbox bigger than its cell
    grid, and slicing outside it raises rather than returning empty pixels."""
    import pandas as pd

    objects = {t: _catalogue(t, *c, seed=t) for t, c in TRACTS.items()}
    fakes.install(monkeypatch, ex)
    # Knock out every cell of patch 0 in one tract.
    gone = [fakes.CellIJ(i, j) for i in range(fakes.CELLS_PER_PATCH)
            for j in range(fakes.CELLS_PER_PATCH)]
    butler = fakes.FakeButler(TRACTS, objects, missing={(5063, 0): gone})

    summary = _run(butler, tmp_path, n_patches=None, n_hosts=40)
    manifest = pd.read_parquet(tmp_path / "manifest.parquet")
    assert summary["n_accepted"] > 0
    # Nothing from the emptied tract, and nothing crashed getting there.
    assert (manifest[manifest["status"] == "accepted"]["tract"] == 5064).all()


def test_a_shallow_coadd_is_rejected_when_asked(monkeypatch, tmp_path):
    """Early DP2 outside the deep fields is 1-3 visits per cell."""
    objects = {t: _catalogue(t, *c, seed=t) for t, c in TRACTS.items()}
    fakes.install(monkeypatch, ex)
    butler = fakes.FakeButler(TRACTS, objects, n_visits=2)

    summary = _run(butler, tmp_path, n_patches=None,
                   gate_kwargs={"min_visits": 10})
    assert summary["n_accepted"] == 0
    assert "too_shallow" in summary["rejection_counts"]


def test_a_component_the_repo_will_not_serve_ends_the_run(butler, tmp_path):
    """Falling back to a whole-patch read cost two orders of magnitude in I/O,
    silently, for several runs.  Crash instead."""
    real = butler.get

    def no_provenance(what, dataId=None, parameters=None):
        if isinstance(what, str) and what.endswith("provenance"):
            raise RuntimeError("not served")
        return real(what, dataId, parameters)

    butler.get = no_provenance
    with pytest.raises(RuntimeError, match="provenance"):
        _run(butler, tmp_path)


# -- talking to the butler --------------------------------------------------


def test_tract_queries_are_constrained_by_data_id(butler):
    """`where="tract = :tract"` resolved as `tract = tract` -- the bind key
    shadows the dimension -- so the query returned the whole repo, truncated at
    20000, and hosts matched same-numbered patches in other tracts."""
    seen = {}
    butler.query_datasets = lambda kind, data_id=None, **kw: (
        seen.update(data_id=data_id, kw=kw) or []
    )
    coadd_refs_for_tract(butler, 5063, [1], bands=("r",))
    assert seen["data_id"] == {"skymap": SKYMAP, "tract": 5063}
    assert not seen["kw"].get("where")


def test_refs_from_another_tract_are_dropped_and_reported(butler, caplog):
    """Patch indices repeat across tracts, so a patch filter alone lets a ref
    from anywhere through."""
    butler.query_datasets = lambda kind, data_id=None, **kw: [
        fakes.Ref(skymap=SKYMAP, tract=5063, patch=3, band="r"),
        fakes.Ref(skymap=SKYMAP, tract=99, patch=3, band="r"),
    ]
    with caplog.at_level("WARNING"):
        got = coadd_refs_for_tract(butler, 5063, [3], bands=("r",))
    assert len(got) == 1 and "not constraining tract" in caplog.text


def test_a_data_id_is_read_through_mapping():
    """DataCoordinate stopped being a Mapping in daf_butler v27: dict() on one
    falls through to sequence iteration and dies with KeyError: 0."""
    coord = fakes.DataCoordinate({"band": "r", "tract": 5063, "patch": 7})
    assert _data_id_dict(coord) == {"band": "r", "tract": 5063, "patch": 7}
    with pytest.raises(KeyError):
        dict(coord)


# -- cells and depth --------------------------------------------------------


def test_visits_are_counted_per_cell_not_per_row():
    """One row per (visit, detector, cell), so a visit whose detector boundary
    crosses a cell appears twice.  It is still one visit of depth."""
    rows = [(0, 0, 10, 1), (0, 0, 10, 2), (0, 0, 11, 1), (1, 0, 10, 1)]
    a = np.asarray(rows, dtype=np.int64)
    table = Table({"cell_i": a[:, 0], "cell_j": a[:, 1],
                   "visit": a[:, 2], "detector": a[:, 3]})
    counts = cell_visit_counts(fakes.Provenance(table))
    assert counts == {(0, 0): 2, (1, 0): 1}
    assert stamp_depth(counts, [(0, 0), (1, 0)]) == (1, 2)
    # A cell absent from the table contributed nothing; no table at all is
    # "not measured", which is a different thing.
    assert stamp_depth(counts, [(0, 0), (9, 9)]) == (0, 2)
    assert stamp_depth({}, [(0, 0)]) == (-1, -1)


def test_unrecognised_contribution_columns_end_the_run():
    table = Table({"cell_index": [0, 0], "visit": [10, 11]})
    with pytest.raises(RuntimeError, match="CONTRIB_CELL_COLUMNS"):
        cell_visit_counts(fakes.Provenance(table))


def test_the_cells_a_stamp_covers_include_the_middle_ones():
    """A 416 px stamp spans 3x3 cells of 150 px; the centre one has no corner
    in it and a corner-only span would miss it."""
    bounds = fakes.CellGridBounds(fakes.Box(fakes.Interval(0, 3300),
                                            fakes.Interval(0, 3300)))
    cells = cells_in_stamp(bounds, 225, 225, 416)
    assert len(cells) == 9 and (1, 1) in cells
    assert missing_cells(fakes.CellGridBounds(bounds.bbox,
                                              [fakes.CellIJ(1, 1)]), cells) == [(1, 1)]


# -- host selection ---------------------------------------------------------


def _hosts(n=3000, scale=0.6, seed=1, tract=5063):
    rng = np.random.default_rng(seed)
    trace_sq = 25.0 * 10 ** rng.exponential(scale, n)
    reff = np.sqrt(trace_sq) * 0.2 * 1.177
    mag = rng.normal(22.0, 0.7, n) - 2.5 * np.log10(2 * np.pi * reff**2)
    return Table({
        "objectId": np.arange(n),
        "coord_ra": 53.13 + rng.normal(0, 0.3, n),
        "coord_dec": -28.10 + rng.normal(0, 0.3, n),
        "refExtendedness": np.ones(n),
        "tract": np.full(n, tract), "patch": rng.integers(0, 100, n),
        "r_cModelFlux": 10 ** ((31.4 - mag) / 2.5),
        "r_blendedness": rng.beta(1.2, 8, n),
        "r_ixx": trace_sq, "r_iyy": trace_sq, "r_ixy": np.zeros(n),
        "r_ixxPSF": np.full(n, 4.0), "r_iyyPSF": np.full(n, 4.0),
        "r_pixelFlags_saturatedCenter": np.zeros(n, bool),
        "r_pixelFlags_interpolatedCenter": np.zeros(n, bool),
        "sersic_reff_major": reff, "sersic_reff_minor": 0.7 * reff,
        "sersic_index": rng.uniform(0.5, 6.0, n),
        "sersic_unknown_flag": np.zeros(n, bool),
        "sersic_no_data_flag": np.zeros(n, bool),
    })


def test_size_comes_from_the_multiband_sersic_fit():
    """One morphology fit to all six bands, so the column carries no band
    prefix and needs no blending across components."""
    assert "sersic_reff_major" in OBJECT_COLUMNS
    assert not any("reff" in c for c in OBJECT_BAND_COLUMNS)

    t = Table({"sersic_reff_major": [3.2, 5.0, 4.0, np.nan, -1.0],
               "sersic_reff_minor": [1.9, 3.0, 2.4, 1.0, 1.0],
               "sersic_unknown_flag": [False, False, True, False, False],
               "sersic_no_data_flag": [False, False, False, False, False]})
    got = host_half_light_arcsec(t)
    assert got[:2] == pytest.approx([3.2, 5.0])
    # A failed fit, a NaN and a nonsensical radius are all "no size", because a
    # size cut compares NaN away but not whatever was left in the column.
    assert np.isnan(got[2:]).all()
    assert host_half_light_arcsec(t, axis="minor")[0] == pytest.approx(1.9)
    with pytest.raises(ValueError, match="axis"):
        host_half_light_arcsec(t, "r")  # the old per-band signature


def test_small_hosts_are_cut_and_the_rest_are_spread_over_size():
    """The catalogue is dominated by galaxies a pixel or two across, which carry
    no structure to learn.  Stratification must use equal-width bins in log
    size: quantile bins hold equal numbers by construction, so drawing equally
    from them is exactly a uniform sample and stratifies nothing."""
    t = _hosts(seed=1)
    kept = select_hosts(t, band="r", min_reff_arcsec=3.0)
    assert len(kept) and np.all(host_half_light_arcsec(kept) >= 3.0)
    assert len(kept) < len(select_hosts(t, band="r", min_reff_arcsec=None))

    kw = dict(band="r", min_reff_arcsec=None)
    parent = host_half_light_arcsec(select_hosts(t, **kw))
    big = np.percentile(parent, 90)
    strat = host_half_light_arcsec(select_hosts(t, n_hosts=50, seed=0, **kw))
    flat = host_half_light_arcsec(
        select_hosts(t, n_hosts=50, seed=0, size_stratified=False, **kw))
    # The bug this guards was exactly a no-op: quantile bins gave ratio 1.0.
    assert np.mean(strat > big) > 2 * np.mean(flat > big)
    assert np.median(strat) > 1.4 * np.median(flat)


def test_duplicates_and_already_tried_hosts_are_not_offered():
    """There are no detect_* columns in DP2, and overlapping tracts give one
    source two different objectIds, so sky coincidence is the only dedupe."""
    t = _hosts(200, seed=2)
    doubled = Table(np.concatenate([t.as_array(), t.as_array()]))
    assert len(select_hosts(doubled, band="r")) == len(select_hosts(t, band="r"))

    first = select_hosts(t, band="r", n_hosts=20, seed=0)
    ids = {int(i) for i in first["objectId"]}
    second = select_hosts(t, band="r", n_hosts=20, seed=0, exclude_ids=ids)
    assert len(second) == 20 and not (ids & {int(i) for i in second["objectId"]})


def test_the_top_up_loop_terminates_and_sizes_itself():
    """25 cutouts from 100 hosts with 90 still wanted is ~360 hosts plus
    headroom; a round that yielded nothing must widen rather than divide by
    zero; and the batch must never collapse to nothing and spin."""
    assert next_batch(100, 100, 25, 90) == pytest.approx(468, abs=1)
    assert next_batch(100, 100, 0, 90) == 400
    assert next_batch(100, 100, 100, 0) >= 16


# -- catalogue sources ------------------------------------------------------


def test_the_butler_scan_keeps_only_survivors(butler):
    """An object table is ~700k rows and the footprint ~1000 of them, so the
    cuts run per tract rather than on a concatenation of all of them."""
    pool = build_host_catalogue(butler, source="butler")
    assert 0 < len(pool) <= sum(len(t) for t in butler.objects.values())
    with pytest.raises(ValueError, match="butler"):
        build_host_catalogue(source="butler")
    with pytest.raises(ValueError, match="tap"):
        build_host_catalogue(source="qserv")


def test_the_selective_cuts_go_into_the_adql_and_the_rest_do_not():
    q = host_adql(bands=("r",), min_reff_arcsec=3.0, flux_range=(360.0, 3e6))
    select, _, where = q.partition("WHERE")
    assert "sersic_reff_major >= 3.0" in where
    assert "r_cModelFlux > 360.0" in where and "FROM dp2.Object" in q
    # Fetched but compared locally: how a boolean compares in ADQL is
    # backend-specific and a wrong guess silently returns nothing.
    assert "sersic_unknown_flag" in select and "sersic_unknown_flag" not in where
    # Sorting burdens a shared service and the stratified draw is local anyway.
    assert "ORDER BY" not in q
    assert "CONTAINS" in host_adql(ra=53.13, dec=-28.1, radius_deg=0.3)
    assert host_adql(top=25).startswith("SELECT TOP 25 ")


def test_tap_results_get_the_local_cuts_and_can_be_cached(tmp_path):
    """The service applied the numeric cuts; the Sersic flags, the point-source
    cross-check and the cross-tract dedupe are not expressible there.  A batch
    node may have no network, so the answer has to survive to disk."""
    table = _hosts(200, seed=4)
    table["sersic_no_data_flag"][:100] = True
    service = _Tap(table)
    pool = build_host_catalogue(tap_service=service,
                                cache=tmp_path / "hosts.parquet")
    assert len(pool) and np.all(np.asarray(pool["objectId"]) >= 100)

    again = build_host_catalogue(tap_service=None,
                                 cache=tmp_path / "hosts.parquet")
    assert len(again) == len(pool) and len(service.jobs) == 1


def test_a_failed_tap_job_still_deletes_itself():
    """An abandoned job sits on a shared service."""
    service = _Tap(_hosts(50, seed=5), end_phase="ERROR")
    with pytest.raises(RuntimeError):
        run_adql(service, "SELECT 1")
    assert service.jobs[0].deleted


class _Job:
    def __init__(self, service, query):
        self.service, self.query, self.phase, self.deleted = service, query, "P", False

    def run(self):
        pass

    def wait(self, phases=(), timeout=None):
        self.phase = self.service.end_phase

    def raise_if_error(self):
        if self.phase == "ERROR":
            raise RuntimeError("ADQL error")

    def fetch_result(self):
        return type("R", (), {"to_table": lambda _: self.service.table})()

    def delete(self):
        self.deleted = True


class _Tap:
    def __init__(self, table, end_phase="COMPLETED"):
        self.table, self.end_phase, self.jobs = table, end_phase, []

    def submit_job(self, query):
        self.jobs.append(_Job(self, query))
        return self.jobs[-1]


# -- getting to TAP at all --------------------------------------------------


def test_the_token_is_found_and_never_logged(monkeypatch, caplog):
    """ACCESS_TOKEN is a generic name other software sets too, and picking up
    somebody else's value looks exactly like a rejected RSP token."""
    for var in TOKEN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(ex, "TOKEN_PATHS", ())
    with pytest.raises(RuntimeError, match="data.lsst.cloud"):
        find_token()

    monkeypatch.setenv("ACCESS_TOKEN", "gt-abcdef.ghijkl")
    assert find_token() == ("gt-abcdef.ghijkl", "$ACCESS_TOKEN")
    with caplog.at_level("INFO"):
        ex.rsp_token()
    assert "gt-..." in caplog.text and "abcdef" not in caplog.text


def test_the_token_goes_only_to_the_service():
    """A session header follows redirects: one redirect off-host and the token
    has been handed to whoever answered."""
    auth = _BearerForPrefix("gt-secret", ["https://data.lsst.cloud/api/tap"])

    def header(url):
        return auth(type("R", (), {"url": url, "headers": {}})()).headers.get(
            "Authorization")

    assert header("https://data.lsst.cloud/api/tap/async") == "Bearer gt-secret"
    assert header("https://data.lsst.cloud/api/other") is None
    assert header("https://elsewhere.example/api/tap") is None
    assert header("https://data.lsst.cloud/api/tap-evil") is None


def test_the_endpoint_is_discovered_and_a_missing_one_is_fatal(monkeypatch):
    doc = {"datasets": {"dp2": {"services": {"tap": {"url": "https://x/api/tap"}}}}}
    monkeypatch.setattr(ex.requests, "get", lambda *a, **k: type("R", (), {
        "raise_for_status": lambda _: None, "json": lambda _: doc})())
    assert discover_tap_url("dp2") == "https://x/api/tap"
    with pytest.raises(RuntimeError, match="no TAP service"):
        discover_tap_url("dp7")


# -- bookkeeping ------------------------------------------------------------


def test_every_rejection_reason_is_counted_not_just_the_first(tmp_path):
    """gate returns reasons in a fixed order, so counting only the first blames
    whichever check runs early -- a plane gated last accounted for a quarter of
    a real run's rejections without appearing in the counts at all."""
    records = [
        {"status": "rejected", "reasons": "cell_depth:2>1.5;inner_INTERPOLATED:0.3>0"},
        {"status": "rejected", "reasons": "inner_INTERPOLATED:0.4>0"},
        {"status": "accepted", "diag_variance_step": 1.1},
    ]
    out = _write_manifest(tmp_path, records, [])
    assert out["rejection_counts"] == {"inner_INTERPOLATED": 2, "cell_depth": 1}
    assert out["first_rejection_counts"] == {"cell_depth": 1, "inner_INTERPOLATED": 1}
    assert out["n_rejected"] == 2 and out["n_attempts"] == 3


def test_null_catalogue_values_do_not_become_data():
    """np.asarray on a masked column hands back the raw buffer with no hint that
    part of it is not data: harmless NaN for a float, but for an integer like
    patch it is whatever was in memory, filing a host under a patch it is
    nowhere near."""
    t = Table({"coord_ra": np.ma.array([53.1, 0.0], mask=[False, True]),
               "coord_dec": np.ma.array([-28.1, -28.2], mask=[False, False]),
               "patch": np.ma.array([3, 999], mask=[False, True])})
    out = unmask(t)
    assert np.isnan(out["coord_ra"][1]) and out["patch"][1] == -1
    assert len(with_positions(out)) == 1


def test_surface_brightness_is_what_separates_a_galaxy_from_a_runaway_fit():
    """Nothing else in the selection requires a host to be visible.  At a 3"
    half-light radius the old 360 nJy floor admitted mu_e = 29.4, fainter than a
    sigma of sky per square arcsecond, where the Sersic fit is degenerate and
    walks off to a large radius around an invisible envelope.  Those rows pass
    every size cut and arrive as point-like blobs."""
    # A real galaxy and a runaway: same radius, four magnitudes apart.
    t = Table({
        "objectId": [1, 2], "coord_ra": [53.1, 53.3], "coord_dec": [-28.1, -28.3],
        "refExtendedness": [1.0, 1.0],
        "tract": [5063, 5063], "patch": [0, 0],
        "sersic_reff_major": [3.0, 3.0], "sersic_reff_minor": [1.8, 1.8],
        "sersic_index": [1.0, 1.0],
        "sersic_unknown_flag": [False, False],
        "sersic_no_data_flag": [False, False],
        "r_cModelFlux": [3.3e5, 3.6e2],
        "r_ixx": [30.0, 30.0], "r_iyy": [30.0, 30.0], "r_ixy": [0.0, 0.0],
        "r_ixxPSF": [4.0, 4.0], "r_iyyPSF": [4.0, 4.0],
        "r_pixelFlags_saturatedCenter": [False, False],
        "r_pixelFlags_interpolatedCenter": [False, False],
    })
    # Over the half-light *ellipse*, a*b, not a circle of radius a: at
    # b = 0.6a that is 0.55 mag/arcsec^2 brighter than the circular figure.
    mu = ex.host_mu_e(t, "r")
    assert mu[0] == pytest.approx(21.4, abs=0.1)
    assert mu[1] == pytest.approx(28.8, abs=0.1)
    kept = select_hosts(t, band="r", flux_range=(1.0, 1e9))
    assert [int(i) for i in kept["objectId"]] == [1]


def test_the_point_source_cut_is_referenced_to_the_psf():
    """A star's raw moments are whatever the seeing was -- 2.0 px at median DP2
    seeing -- so min_trace_px = 1.75 rejected nothing.  Deconvolved, a point
    source is exactly zero."""
    t = Table({"r_ixx": [4.0, 30.0], "r_iyy": [4.0, 30.0],
               "r_ixxPSF": [4.0, 4.0], "r_iyyPSF": [4.0, 4.0]})
    got = ex.host_deconvolved_px(t, "r")
    assert got[0] == 0.0 and got[1] == pytest.approx(np.sqrt(26.0))
    with pytest.raises(KeyError, match="ixxPSF"):
        ex.host_deconvolved_px(Table({"r_ixx": [1.0], "r_iyy": [1.0]}), "r")


def test_a_saturated_or_interpolated_core_disqualifies_a_host():
    """The real bright limit, and the real reason to drop a core: an
    interpolated centre is synthetic structure exactly where the transient
    goes."""
    t = _hosts(40, seed=9)
    t["r_pixelFlags_saturatedCenter"][:10] = True
    t["r_pixelFlags_interpolatedCenter"][10:20] = True
    kept = select_hosts(t, band="r")
    assert len(kept) and np.all(np.asarray(kept["objectId"]) >= 20)


def test_stratification_bins_do_not_follow_the_sample_tail():
    """Edges taken from the data's own min and max hand whole bins to whatever
    tail exists -- with runaway fits, that means stratification preferentially
    selects them."""
    q = host_adql(bands=("r",))
    assert "sersic_reff_major <= 12.0" in q  # the tail is bounded server-side

    t = _hosts(2000, seed=10)
    t["sersic_reff_major"][:5] = 200.0  # a handful of absurd fits
    t["sersic_reff_minor"][:5] = 120.0
    kept = select_hosts(t, band="r", n_hosts=100, seed=0)
    assert np.all(host_half_light_arcsec(kept) <= 12.0)


def test_the_butler_and_tap_column_lists_are_not_the_same():
    """TAP's dp2.Object view serves derived columns the pipeline never wrote --
    {band}_cModelMag among them -- and asking the butler parquet for one fails
    the whole read.  So the two callers get two lists."""
    from rubin_host_prior.rubin.extract import neighbour_columns

    butler_side = set(neighbour_columns(("r", "i")))
    assert not any("Mag" in c for c in butler_side)
    assert not any("Mag" in c for c in host_adql().split("FROM")[0])
    # The neighbour index needs a position, an id, an extendedness and a flux,
    # and reading more than that is what broke.
    assert butler_side == {"objectId", "coord_ra", "coord_dec",
                           "refExtendedness", "r_cModelFlux", "i_cModelFlux"}


def test_a_column_the_parquet_lacks_says_why(butler):
    """The butler's own message names the column but not the reason."""
    def missing(what, dataId=None, parameters=None):
        raise ValueError("Column x_cModelMag ... not available in parquet file.")

    butler.get = missing
    with pytest.raises(RuntimeError, match="not the same table"):
        ex._neighbour_index(butler, 5063, ["x_cModelMag"], ("r",))
