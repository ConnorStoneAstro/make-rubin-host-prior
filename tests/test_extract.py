"""Extraction: the host-major walk end to end, and the pieces that fail quietly.

Most of what can go wrong here only goes wrong against a butler -- a data id
that stopped being a Mapping, a bind key that shadowed a dimension, a component
that turned out to be a property -- so the centre of this file is a real walk
over ``tests/fakes``.  The rest covers the handful of pure functions whose
failures would not be obvious from a traceback.

Deliberately not exhaustive.  The host cuts run on the TAP service, so a test of
them here would be a test of the fake; what is checked is that the right cut
reaches the ADQL.  Everything else leans on the error messages, which name the
cause and what to do about it.
"""

from pathlib import Path

import numpy as np
import pytest

import rubin_host_prior.rubin.extract as ex
from rubin_host_prior.selection import HostCuts, PatchCuts, Selection
from rubin_host_prior.rubin.extract import (
    PHOTOMETRY_BANDS,
    SKYMAP,
    TOKEN_ENV_VARS,
    _BearerForPrefix,
    _data_id_dict,
    _write_manifest,
    build_host_catalogue,
    cell_visit_counts,
    cells_in_stamp,
    coadd_refs,
    dedupe_hosts,
    describe_summary,
    discover_tap_url,
    extract_patches,
    find_token,
    host_adql,
    missing_cells,
    run_adql,
    select_hosts,
    stamp_depth,
    unmask,
)

import fakes

Table = pytest.importorskip("astropy.table").Table
pytest.importorskip("pandas")


# -- fixtures ---------------------------------------------------------------


def _catalogue(tract, ra0, dec0, n=40, seed=1, patches=(0,)):
    """Hosts placed well inside the given patches of one tract.

    They arrive already cut, because on a real run they do: the ADQL applied the
    size, magnitude and surface-brightness limits on the service.
    """
    rng = np.random.default_rng(seed)
    patch = rng.choice(patches, n)
    row, col = np.divmod(patch, 10)
    px = col * fakes.PATCH + rng.uniform(400, fakes.PATCH - 400, n)
    py = row * fakes.PATCH + rng.uniform(400, fakes.PATCH - 400, n)
    cosd = np.cos(np.deg2rad(dec0))
    reff = 10 ** rng.uniform(0.3, 1.0, n)
    return Table({
        "objectId": np.arange(n) + tract * 10000,
        "coord_ra": ra0 + px * fakes.PIXEL_SCALE / cosd,
        "coord_dec": dec0 + py * fakes.PIXEL_SCALE,
        "refExtendedness": np.ones(n),
        "tract": np.full(n, tract),
        "patch": patch.astype(int),
        "sersic_reff_major": reff,
        "sersic_reff_minor": 0.6 * reff,
        "sersic_index": rng.uniform(0.5, 6.0, n),
        "sersic_unknown_flag": np.zeros(n, bool),
        "sersic_no_data_flag": np.zeros(n, bool),
        **{f"{b}_{c}": v for b in PHOTOMETRY_BANDS
           for c, v in (("cModelFlux", 10 ** rng.uniform(5.0, 5.8, n)),
                        ("blendedness", rng.beta(1.2, 8, n)),
                        ("pixelFlags_saturatedCenter", np.zeros(n, bool)),
                        ("pixelFlags_interpolatedCenter", np.zeros(n, bool)))},
    })


TRACTS = {5063: (53.13, -28.10), 5064: (55.00, -28.10)}


def _make(monkeypatch, patches=(0,), **butler_kw):
    """A fake butler and a TAP service that agree about where the hosts are."""
    from astropy.table import vstack

    objects = {t: _catalogue(t, *c, seed=t, patches=patches)
               for t, c in TRACTS.items()}
    fakes.install(monkeypatch, ex)
    butler = fakes.FakeButler(TRACTS, objects, **butler_kw)
    return butler, fakes.FakeTap(vstack(list(objects.values())))


@pytest.fixture
def butler(monkeypatch):
    made = _make(monkeypatch)
    made[0].tap = made[1]
    return made[0]


def _run(butler, tmp_path, **kw):
    kw.setdefault("bands", ("r", "i"))
    kw.setdefault("n_stamps", 12)
    return extract_patches(butler, tmp_path, tap_service=butler.tap,
                           native_size=416, seed=0, **kw)


# -- the walk ---------------------------------------------------------------


def test_a_run_produces_the_stamps_asked_for_reading_only_their_pixels(
        butler, tmp_path):
    """Whole-patch reads are ~100x the I/O, and the fake butler refuses them, so
    a regression to the old fallback path fails here rather than on the cluster.
    """
    summary = _run(butler, tmp_path)

    assert summary["counts"]["stamps_accepted"] == 12
    assert butler.reads["bbox"] == 12 and butler.reads["whole"] == 0
    assert summary["counts"]["stamps_rejected"] == 0

    from rubin_host_prior.data.shards import ShardSet

    shards = ShardSet.from_dir(tmp_path / "shards")
    assert len(shards) == 12
    assert shards.load("image").shape == (12, 416, 416)
    assert (tmp_path / "hosts.parquet").exists()
    assert (tmp_path / "manifest.parquet").exists()


def test_a_host_that_fails_costs_only_its_place_in_the_queue(monkeypatch,
                                                             tmp_path):
    """The walk draws deeper into the catalogue rather than returning at the
    first failures, and says so when the catalogue runs out before the target.
    """
    gone = [fakes.CellIJ(i, j) for i in range(fakes.CELLS_PER_PATCH)
            for j in range(fakes.CELLS_PER_PATCH)]
    butler, tap = _make(monkeypatch, missing={(5063, 0): gone})
    butler.tap = tap

    # More than the live tract can supply -- 40 hosts x 2 bands is 80 -- so the
    # walk has to cover the 40 dead hosts too before it gives up.
    summary = _run(butler, tmp_path, n_stamps=120)
    counts = summary["counts"]
    assert counts["stamps_accepted"] == 80
    assert counts["hosts_tried"] == 80          # the whole catalogue
    assert counts["hosts_too_near_the_edge_of_coverage"] == 40
    assert counts["stamps_requested"] == 120    # and it is honest about it

    import pandas as pd

    manifest = pd.read_parquet(tmp_path / "manifest.parquet")
    assert (manifest[manifest["status"] == "accepted"]["tract"] == 5064).all()
    assert "off_the_grid" in summary["rejection_counts"]


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
        # and the neighbour covariates nothing trained on are gone
        assert "n_neighbours" not in f["meta"]


def test_stamps_are_centred_on_the_host(butler, tmp_path):
    """Decentring belongs in the loader, which crops at a random offset every
    epoch, rather than being fixed once at extraction time."""
    import pandas as pd

    _run(butler, tmp_path)
    manifest = pd.read_parquet(tmp_path / "manifest.parquet")
    hosts = pd.read_parquet(tmp_path / "hosts.parquet")
    accepted = manifest[manifest["status"] == "accepted"].merge(
        hosts[["objectId", "coord_ra", "coord_dec"]],
        left_on="host_id", right_on="objectId")
    assert len(accepted)
    # The guard against DP2's two pixel-origin conventions: mixing them
    # displaces a stamp by up to a patch, which still looks like plausible sky.
    assert (manifest["centre_sep_arcsec"] < 1.0).all()

    from rubin_host_prior.data.shards import ShardSet

    shards = ShardSet.from_dir(tmp_path / "shards")
    # The recorded stamp centre is the host's catalogue position exactly.
    for ra, dec in zip(shards.meta["ra"], shards.meta["dec"]):
        match = np.hypot((accepted["coord_ra"] - ra) * np.cos(np.deg2rad(dec)),
                         accepted["coord_dec"] - dec).min() * 3600.0
        assert match < 1e-6


def test_a_shallow_coadd_is_rejected_when_asked(monkeypatch, tmp_path):
    """Early DP2 outside the deep fields is 1-3 visits per cell."""
    butler, tap = _make(monkeypatch, n_visits=2)
    butler.tap = tap
    summary = extract_patches(
        butler, tmp_path, tap_service=tap, native_size=416, bands=("r",),
        n_stamps=4, selection=Selection(patches=PatchCuts(min_visits=10)))
    assert summary["counts"]["stamps_accepted"] == 0
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


def test_a_catalogue_that_points_nowhere_says_so_early(butler, tmp_path):
    """The Object table's `patch` and the deep_coadd dataId `patch` being
    different numberings looks exactly like a field with no coverage: every
    stamp is rejected and an empty set is written without complaint."""
    butler.query_datasets = lambda *a, **k: []
    with pytest.raises(RuntimeError, match="same numbering"):
        _run(butler, tmp_path)


# -- talking to the butler --------------------------------------------------


def test_components_are_read_once_per_patch_not_once_per_host(butler, tmp_path):
    """303 of a real run's 398 seconds were these three reads, because the
    host-major walk pays per host what the old patch-major sweep amortised.
    The walk is ordered so that hosts sharing a patch are consecutive, which is
    what lets a one-patch cache catch every repeat."""
    from astropy.table import vstack

    # Two hosts per patch, so the second must cost nothing.
    objects = {t: _catalogue(t, *c, n=4, seed=t, patches=(0,))
               for t, c in TRACTS.items()}
    fakes.install(monkeypatch_noop(), ex)
    b = fakes.FakeButler(TRACTS, objects)
    extract_patches(b, tmp_path, tap_service=fakes.FakeTap(
        vstack(list(objects.values()))), native_size=416, bands=("r", "i"),
        n_stamps=None, seed=0)

    # Two patches (one per tract) x two bands x three roles.  Per host it would
    # be eight hosts x two bands x three, four times as many.
    assert b.reads["component"] == 2 * 2 * 3
    assert b.reads["bbox"] == 8 * 2      # one per host per band, as it must be
    assert b.reads["whole"] == 0


class monkeypatch_noop:
    """fakes.install only needs something with setattr."""

    def setattr(self, obj, name, value):
        setattr(obj, name, value)


def test_coadd_queries_are_constrained_by_data_id(butler):
    """`where="tract = :tract"` resolved as `tract = tract` -- the bind key
    shadows the dimension -- so the query returned the whole repo, truncated at
    20000, and hosts matched same-numbered patches in other tracts."""
    seen = {}
    butler.query_datasets = lambda kind, data_id=None, **kw: (
        seen.update(data_id=data_id, kw=kw) or []
    )
    coadd_refs(butler, 5063, 1, bands=("r",))
    assert seen["data_id"] == {"skymap": SKYMAP, "tract": 5063, "patch": 1}
    assert not seen["kw"].get("where")


def test_a_ref_from_the_wrong_patch_ends_the_run(butler):
    """Patch indices repeat across tracts, so an unconstrained query hands back
    refs that look right and are cut from somewhere else entirely."""
    butler.query_datasets = lambda kind, data_id=None, **kw: [
        fakes.Ref(skymap=SKYMAP, tract=99, patch=3, band="r"),
    ]
    with pytest.raises(RuntimeError, match="wrong piece of sky"):
        coadd_refs(butler, 5063, 3, bands=("r",))


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
    # A cell absent from the table contributed nothing at all, which is the
    # worst case, not a missing measurement.
    assert stamp_depth(counts, [(0, 0), (9, 9)]) == (0, 2)


def test_provenance_that_cannot_be_read_ends_the_run():
    """An empty or unrecognised contributions table used to leave the measured
    variance step as the only detector, silently."""
    with pytest.raises(RuntimeError, match="CONTRIB_CELL_COLUMNS"):
        cell_visit_counts(fakes.Provenance(
            Table({"cell_index": [0, 0], "visit": [10, 11]})))
    with pytest.raises(RuntimeError, match="built from visits"):
        cell_visit_counts(fakes.Provenance(Table({"visit": []})))


def test_the_cells_a_stamp_covers_include_the_middle_ones():
    """A 416 px stamp spans 3x3 cells of 150 px; the centre one has no corner
    in it and a corner-only span would miss it."""
    bounds = fakes.CellGridBounds(fakes.Box(fakes.Interval(0, 3300),
                                            fakes.Interval(0, 3300)))
    cells = cells_in_stamp(bounds, 225, 225, 416)
    assert len(cells) == 9 and (1, 1) in cells
    assert missing_cells(fakes.CellGridBounds(bounds.bbox,
                                              [fakes.CellIJ(1, 1)]), cells) == [(1, 1)]


# -- the host list ----------------------------------------------------------


def test_the_selective_cuts_go_into_the_adql_and_the_rest_do_not():
    """This is where the host cuts are applied.  They are not applied again
    after the rows arrive, so what is in this query is what the catalogue is."""
    q = host_adql(bands=("r",), cuts=HostCuts(min_reff_arcsec=3.0))
    select, _, where = q.partition("WHERE")
    assert "sersic_reff_major >= 3.0" in where
    # Magnitudes become fluxes: the ADQL has no LOG10 it can be trusted with,
    # and surface brightness becomes a flux against an area for the same reason.
    assert f"r_cModelFlux > {HostCuts().flux_range[0]:.1f}" in where
    assert "sersic_reff_major * sersic_reff_minor" in where
    assert "FROM dp2.Object" in q
    # Fetched but compared locally: how a boolean compares in ADQL is
    # backend-specific and a wrong guess silently returns nothing.
    assert "sersic_unknown_flag" in select and "sersic_unknown_flag" not in where
    # One size measurement, so no second moments are asked for at all.
    assert "_ixx" not in q and "_ixxPSF" not in q
    # Sorting burdens a shared service and the draw is local anyway.
    assert "ORDER BY" not in q
    assert "CONTAINS" in host_adql(ra=53.13, dec=-28.1, radius_deg=0.3)
    assert host_adql(top=25).startswith("SELECT TOP 25 ")


def test_what_the_query_cannot_do_is_done_locally(monkeypatch, tmp_path):
    """A failed Sersic fit leaves whatever was in the column, which would
    otherwise compare its way through the size cut the service applied; and an
    interpolated centre is synthetic structure exactly where the transient goes.
    """
    table = _catalogue(5063, 53.13, -28.10, n=40, seed=9)
    table["sersic_no_data_flag"][:10] = True
    table["r_pixelFlags_saturatedCenter"][10:20] = True
    kept = select_hosts(table, HostCuts(reject_interpolated_centre=False))
    assert len(kept) and np.all(np.asarray(kept["objectId"]) >= 5063 * 10000 + 20)


def test_a_galaxy_listed_twice_is_not_weighted_up():
    """DP2 has no detect_isPrimary and tracts overlap, so a galaxy in an overlap
    is listed under two different objectIds and would yield two near-identical
    cutouts."""
    t = Table({"objectId": [1, 2, 3],
               "coord_ra": [53.1, 53.1 + 1e-5, 53.5],
               "coord_dec": [-28.1, -28.1, -28.1]})
    assert [int(i) for i in dedupe_hosts(t, 0.5)["objectId"]] == [1, 3]


def test_tap_results_are_cached_so_a_batch_node_needs_no_network(tmp_path):
    table = _catalogue(5063, 53.13, -28.10, n=60, seed=4)
    service = fakes.FakeTap(table)
    cache = tmp_path / "hosts.parquet"
    pool = build_host_catalogue(tap_service=service, cache=cache)
    assert len(pool)

    again = build_host_catalogue(tap_service=None, cache=cache)
    assert len(again) == len(pool) and len(service.queries) == 1


def test_a_cache_built_with_other_cuts_is_refused(tmp_path):
    """A cache has the cuts baked into it. Reusing one after editing the config
    is a cut that looks applied and is not, which is the failure this whole
    file is arranged against."""
    cache = tmp_path / "hosts.parquet"
    table = _catalogue(5063, 53.13, -28.10, n=20, seed=4)
    build_host_catalogue(tap_service=fakes.FakeTap(table), cache=cache,
                         cuts=HostCuts(min_reff_arcsec=1.5))
    with pytest.raises(RuntimeError, match="different query"):
        build_host_catalogue(tap_service=fakes.FakeTap(table), cache=cache,
                             cuts=HostCuts(min_reff_arcsec=3.0))

    cache.with_suffix(".sql").unlink()
    with pytest.raises(RuntimeError, match="predates this check"):
        build_host_catalogue(tap_service=fakes.FakeTap(table), cache=cache)


def test_a_failed_tap_job_still_deletes_itself():
    """An abandoned job sits on a shared service."""
    job = fakes._FakeJob(Table({"a": [1]}))
    job.phase = "ERROR"
    service = type("S", (), {"submit_job": lambda _s, _q: job})()
    with pytest.raises(RuntimeError, match="ERROR"):
        run_adql(service, "SELECT 1")
    assert job.deleted


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


def test_surface_brightness_is_what_separates_a_galaxy_from_a_runaway_fit():
    """Nothing else in the selection requires a host to be visible.  At a 3"
    half-light radius the old 360 nJy floor admitted mu_e = 29.4, fainter than a
    sigma of sky per square arcsecond, where the Sersic fit is degenerate and
    walks off to a large radius around an invisible envelope.  Those rows pass
    every size cut and arrive as point-like blobs."""
    # A real galaxy and a runaway: same radius, four magnitudes apart.  Over the
    # half-light *ellipse*, a*b, not a circle of radius a: at b = 0.6a that is
    # 0.55 mag/arcsec^2 brighter than the circular figure.
    t = Table({"sersic_reff_major": [3.0, 3.0], "sersic_reff_minor": [1.8, 1.8],
               "r_cModelFlux": [3.3e5, 3.6e2]})
    mu = ex.host_mu_e(t, "r")
    assert mu[0] == pytest.approx(21.4, abs=0.1)
    assert mu[1] == pytest.approx(28.8, abs=0.1)
    # And the ADQL turns the same limit into a flux against an area.
    assert HostCuts(max_mu_e=25.5).surface_brightness_floor() == pytest.approx(
        2 * np.pi * 10 ** ((31.4 - 25.5) / 2.5))


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
    out = _write_manifest(tmp_path, records)
    assert out["rejection_counts"] == {"inner_INTERPOLATED": 2, "cell_depth": 1}
    assert out["n_rejected"] == 2 and out["n_attempts"] == 3


def test_the_summary_says_what_it_counts(butler, tmp_path):
    """A host is a catalogue object; a stamp is one cutout of one host in one
    band.  n_hosts_tried and n_hosts_attempted read as the same thing and were
    not, and stamps_attempted minus rejections never gave back the hosts asked
    for because those are different units."""
    summary = _run(butler, tmp_path)
    c = summary["counts"]

    assert c["stamps_attempted"] == c["stamps_accepted"] + c["stamps_rejected"]
    assert c["hosts_tried"] <= c["host_candidates_in_catalogue"]
    # One host yields up to len(bands) stamps, which is the whole confusion.
    assert c["stamps_attempted"] == 2 * c["hosts_tried"]
    assert c["stamps_requested"] == 12

    text = describe_summary(summary)
    assert "hosts walked" in text and "stamps attempted" in text
    assert "one host x one band" in text
    assert "time:" in text


def test_the_run_reports_where_its_time_went_and_how_deep_the_field_is(
        monkeypatch, tmp_path, caplog):
    """Half an hour with no output is a run you cannot tune.  And the depth line
    used to describe a single patch while reading like a property of the run, so
    it moved whenever anything perturbed the RNG stream."""
    butler, tap = _make(monkeypatch, patches=(0, 1, 2))
    butler.tap = tap
    with caplog.at_level("INFO"):
        summary = _run(butler, tmp_path, n_stamps=40)

    seconds = summary["seconds"]
    # The component reads are timed one role at a time, not as a single total:
    # they were 76% of a real run and "component reads: 303s" does not say which
    # of the three to attack.
    assert {"host catalogue", "stamp pixels", "read: wcs",
            "read: cell grid (psf)", "read: provenance"} <= set(seconds)
    assert all(v >= 0 for v in seconds.values())

    dist = summary["visits_per_cell"]
    assert dist["n"] > fakes.CELLS_PER_PATCH ** 2, "should span more than one patch"
    assert dist["min"] <= dist["p50"] <= dist["max"]
    assert "visits each" in caplog.text


# -- one file holds every cut ----------------------------------------------


def test_the_run_is_described_entirely_by_one_file(tmp_path):
    """The script has no defaults of its own, so nothing about a run can differ
    between the file and the command line."""
    import yaml

    from rubin_host_prior.selection import ExtractionConfig

    shipped = ExtractionConfig.load(
        Path(__file__).resolve().parent.parent / "extraction.yaml")
    assert shipped.out and shipped.stamps.bands and shipped.hosts.band == "r"
    assert shipped.patches.gate_kwargs()["max_no_data"] > 0

    path = tmp_path / "e.yaml"
    shipped.save(path)
    assert ExtractionConfig.load(path) == shipped

    raw = yaml.safe_load(path.read_text())
    raw["hosts"]["max_magnitude"] = 19.0     # near miss for max_mag
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="max_magnitude"):
        ExtractionConfig.load(path)

    raw.pop("hosts")
    raw["host"] = {}                          # near miss for a whole section
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unknown section"):
        ExtractionConfig.load(path)


def test_the_gate_has_no_tolerances_of_its_own():
    """extraction.yaml is the only place that says what a cut is.  A default in
    the gate would be a second answer, and the two would drift."""
    import inspect

    from rubin_host_prior.rubin.quality import gate

    required = {n for n, p in inspect.signature(gate).parameters.items()
                if p.default is inspect.Parameter.empty}
    assert {"zero_tol", "frac_tol", "inner_frac_tol", "max_variance_step",
            "max_cell_depth_ratio", "max_no_data"} <= required
    assert set(PatchCuts().gate_kwargs()) | {"image", "variance", "mask",
                                             "plane_dict", "cell_depth_ratio",
                                             "n_visits"} == set(
        inspect.signature(gate).parameters)


def test_disabling_a_patch_cut_means_disabled_not_zero():
    cuts = PatchCuts(min_visits=5, max_variance_step=None)
    assert cuts.gate_kwargs()["max_variance_step"] == float("inf")
    assert cuts.gate_kwargs()["min_visits"] == 5


def test_magnitude_is_the_visibility_cut_and_becomes_a_flux():
    """Total magnitude, not surface brightness.  A tight mu_e cut selects
    *concentrated* light, which is the opposite of what a prior over galaxy
    structure wants."""
    cuts = HostCuts(max_mag=20.5)
    faint, bright = cuts.flux_range
    assert faint == pytest.approx(22909, rel=1e-3)   # r = 20.5
    assert bright > faint
    assert f"{faint:.1f}" in host_adql(cuts=cuts)


def test_describe_shows_which_cut_binds_at_each_size():
    """Size and brightness are not independent -- mu_e = m + 2.5log10(2 pi a b) --
    so a magnitude limit and a surface-brightness limit can quietly exclude each
    other over the range that matters."""
    text = HostCuts(max_mag=20.5, max_mu_e=25.5).describe()
    assert "magnitude" in text and "surface brightness" in text
    # Bright and small: magnitude binds.  Large: surface brightness binds.
    rows = [l.split() for l in text.splitlines() if l.strip().startswith(("2.0", "20.0"))]
    assert rows[0][-1] == "magnitude" and rows[-1][-1] == "brightness"


def test_no_data_is_gated_once_not_twice():
    """It was both a zero-tolerance plane and a measured fraction with a 2%
    tolerance, so the plane always won and the tolerance never did anything --
    which on ragged coverage rejected most stamps that clipped a survey edge."""
    cuts = PatchCuts()
    assert "NO_DATA" not in cuts.zero_tolerance_planes
    assert cuts.gate_kwargs()["max_no_data"] > 0
