"""Build a patch training set from DP2 ``deep_coadd`` images.

Plain Python against the Butler as a data-access layer -- no ``pipetask``, no
BPS.  Import-safe without the LSST stack: everything stack-specific is imported
lazily by ``_lsst()``, so the rest of the package (and the test suite) works on a
laptop.

DP2 is not DP1 with more data.  It replaces ``lsst.afw.image`` with
``lsst.images``: a ``deep_coadd`` is a ``CellCoadd``, getters became attributes,
mask planes were renamed and are read through a schema, and there are two
different pixel-origin conventions.  DP1 code does not error on DP2, it
misbehaves quietly.  The three places that bite:

* **Two pixel origins.**  ``sky_projection`` works in *tract* coordinates,
  ``astropy_wcs`` in *patch-local*.  Mixing them misplaces a position by up to a
  full patch (~4000 px) -- far enough to land on the wrong galaxy, close enough
  to look plausible.  Tract coordinates are used throughout, which is what
  ``Box.factory`` and ``bbox.contains`` expect, and every stamp centre is
  additionally projected back to the sky and compared against the position asked
  for (``_verify_centre``).  That guard costs nothing and turns a whole class of
  silent geometry error into a loud one.
* **Tracts and patches overlap at their edges**, so a host near a boundary is
  covered by two patches and would otherwise be extracted twice in the same
  band.  Accepted ``(host, band)`` pairs are tracked and repeats skipped.
* **Coadds are cell-based** -- a 22x22 grid of 150-pixel cells, each built from a
  different set of input visits.  Depth and PSF are therefore piecewise constant
  with steps at cell edges, and a stamp larger than 150 native pixels *will*
  straddle them.  ``n_cells_spanned`` is recorded per stamp so the effect stays
  measurable.
* **Mask planes are dynamic.**  Bit numbers are not stable across releases, so
  the mask is repacked into a ``uint32`` using a mapping derived from the
  coadd's own ``mask.schema`` and that mapping is stored with every shard.
  Nothing downstream hard-codes a bit.
* **Variance holds ``inf``** where there were no contributing exposures,
  including the cores of saturated stars.  That is measured as a fraction, not
  treated as corruption -- see ``quality.gate``.

Other DP2 specifics this file depends on:

* ``coadd[box]`` returns a **view**; it must be ``.copy()``-ed before the parent
  is released.  ``Box.factory`` takes ``[y, x]`` -- numpy order, not ``(x, y)``.
* **Pixels are already nanojanskys** and variance is nJy^2.  No calibration step
  belongs here.
* DP2 coadds are **over-subtracted around extended galaxies**, and the
  background is recoverable via ``backgrounds`` / ``apply_background``.  This
  module takes the image **as delivered**, without restoring it, and records
  that choice in every shard as ``background_restored=0`` so a future training
  set made the other way is distinguishable.
* A patch is loaded **once** and every stamp that falls inside it is sliced from
  that one read.  The obvious galaxy-by-galaxy loop reloads the same patch for
  each host and dominates the runtime.

Everything above is taken from the verified DP2 reference.  The few items still
marked WARN are the ones it does not cover: the object-table column names (taken
from DP1 and assumed unchanged) and the DP2 field inventory.  Accessors go
through ``_attr``, which reports what an object actually offers if a name is ever
wrong, so a mismatch is a clear error rather than a crash.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np

from ..config import BANDS
from ..data.diagnostics import AutocorrelationAccumulator
from ..data.shards import ShardWriter
from .quality import gate

log = logging.getLogger(__name__)

_STACK: SimpleNamespace | None = None

#: Both are literally ``"dp2"`` -- simpler than DP1's ``LSSTComCam/DP1``.
#: Off-platform, confirm the alias with ``Butler.get_known_repos()`` or
#: ``$DAF_BUTLER_REPOSITORY_INDEX``.
REPO = "dp2"
COLLECTION = "dp2"
SKYMAP = "lsst_cells_v2"

DATASET_TYPE = "deep_coadd"
#: Coadds are tiled by patch, so this is the predicate that finds one.
COADD_REGION = "patch.region OVERLAPS :region"

#: Attribute names on a ``CellCoadd``.  DP2 turned DP1's getters into attributes
#: and camelCase into snake_case.  Access goes through ``_attr``, which reports
#: what the object actually offers if a name is ever wrong.
DP2_ATTRS = {
    "image": "image",
    "variance": "variance",
    "mask": "mask",
    "psf": "psf",
    "wcs": "sky_projection",
    "bbox": "bbox",
    "schema": "schema",
    "origin": "yx0",
    "grid": "grid",
}

#: DP2 offers three WCS representations and they do **not** share a pixel
#: origin: ``sky_projection`` is in *tract* coordinates, ``astropy_wcs`` in
#: *patch-local*.  Mixing them is a position error of up to a full patch
#: (~4000 px) -- far enough to land in the wrong galaxy, close enough to look
#: plausible.  This module uses tract coordinates throughout, which is both the
#: precise representation and the frame ``Box.factory`` and ``bbox.contains``
#: expect.  ``yx0`` converts to patch-local if ever needed.
PIXEL_ORIGIN = "tract"

#: Maximum separation between where a stamp actually landed and where it was
#: asked for, in arcsec.  A pixel-origin mix-up is off by far more than this.
CENTRE_TOLERANCE_ARCSEC = 1.0

#: Minimal column subset of the per-tract ``object`` table (1248 columns, so
#: always subset).  Verified against the DP2 SDM schema.
#:
#: Note what is *absent* relative to DP1: there are no band-independent
#: ``shape_xx``/``shape_yy``/``shape_xy`` columns -- second moments are per band
#: -- and there are no ``detect_*`` columns at all, so ``detect_isPrimary`` is
#: not available for dropping deblend and overlap duplicates.  See
#: ``dedupe_hosts``.
#:
#: The ``sersic_*`` columns are the multiband fit and carry no band prefix: one
#: morphology fit to all six bands at once, which is why it is the size
#: measurement here.  ``sersic_reff_major`` is in **arcsec** (unlike
#: ``sersic_reff_x``, which is in pixels) and is the radius *before* convolution
#: with the PSF, so it is the galaxy's intrinsic size rather than its observed
#: extent.
OBJECT_COLUMNS = [
    "objectId",
    "coord_ra",
    "coord_dec",
    "refExtendedness",
    "refBand",
    "tract",
    "patch",
    "sersic_reff_major",
    "sersic_reff_minor",
    "sersic_index",
    # Booleans marking a fit that failed or had nothing to fit.  Without them a
    # failed fit contributes whatever happened to be in the column.
    "sersic_unknown_flag",
    "sersic_no_data_flag",
    "sersic_chi2_reduced",
]

#: Bands for which the DP2 Object table carries photometry and shapes.  Verified
#: against the schema YAML (``sdm_schemas``, ``drp_base.yaml``), which is the
#: source to use: the rendered HTML schema page is large enough that reading it
#: in excerpts gives a confidently wrong answer about which bands exist.
PHOTOMETRY_BANDS = ("u", "g", "r", "i", "z", "y")

#: Added per band in ``PHOTOMETRY_BANDS``.  ``_ixx``/``_iyy``/``_ixy`` are
#: Gaussian-weighted adaptive moments in pixel^2, kept for ellipticity and as a
#: non-parametric size cross-check.  Size itself comes from the band-independent
#: ``sersic_reff_major`` in ``OBJECT_COLUMNS``, not from anything per band.
OBJECT_BAND_COLUMNS = [
    "{b}_cModelFlux",
    "{b}_cModelFluxErr",
    "{b}_blendedness",
    "{b}_ixx",
    "{b}_iyy",
    "{b}_ixy",
]

#: ECDFS, still in tract 5063 as on DP1, and the field the DP2 tutorials use
#: throughout.  ELAISS1 (10.26, -44.49) and EDFS (59.10, -48.73) also appear.
#: WARN: a full DP2 field/depth inventory is not in the tutorials -- check
#: dp2.lsst.io before choosing a field on cadence grounds.
ECDFS = (53.13, -28.10)


def _lsst() -> SimpleNamespace:
    """Import the LSST stack once, lazily."""
    global _STACK
    if _STACK is None:
        import lsst.sphgeom as sphgeom
        from lsst.daf.butler import Butler
        from lsst.images import Box

        _STACK = SimpleNamespace(Butler=Butler, Box=Box, sphgeom=sphgeom)
    return _STACK


def _attr(obj, role: str):
    """Fetch a ``CellCoadd`` attribute by role, reporting alternatives on failure."""
    name = DP2_ATTRS[role]
    try:
        return getattr(obj, name)
    except AttributeError:
        available = sorted(a for a in dir(obj) if not a.startswith("_"))
        raise AttributeError(
            f"{type(obj).__name__} has no attribute {name!r} (role {role!r}). "
            f"It offers: {available}. Fix DP2_ATTRS[{role!r}] in rubin/extract.py "
            f"against references/dp2-images-api.md."
        ) from None


def _data_id_dict(data_id) -> dict[str, object]:
    """A ``DataCoordinate`` as a plain dict, across daf_butler versions.

    ``DataCoordinate`` stopped being a ``Mapping`` in daf_butler v27, so
    ``dict(data_id)`` no longer takes the mapping path: it falls through to
    *sequence* iteration, asks for ``data_id[0]``, and dies with ``KeyError: 0``
    -- an error that names neither the object nor the problem.  ``.mapping``
    (every dimension) and ``.required`` (the required ones) are the
    replacements; the old path stays for older stacks.
    """
    for attr in ("mapping", "required"):
        view = getattr(data_id, attr, None)
        if view is not None:
            try:
                return {str(k): v for k, v in dict(view).items()}
            except (TypeError, ValueError, KeyError):
                pass
    try:
        return {str(k): v for k, v in dict(data_id).items()}
    except (TypeError, ValueError, KeyError):
        # Worth keeping the provenance even unparsed.  The caller reads only
        # band/tract/patch out of this and has defaults for all three.
        return {"repr": str(data_id)}


# -- repo ------------------------------------------------------------------


def open_butler(repo: str = REPO, collection: str = COLLECTION):
    """Open the DP2 repo read-only.  Never open a shared mirror writeable."""
    butler = _lsst().Butler(repo, collections=collection)
    if butler is None:
        raise RuntimeError(f"could not open butler repo {repo!r}")
    return butler


# -- masks -----------------------------------------------------------------


def pack_mask(mask, max_planes: int = 32) -> tuple[np.ndarray, dict[str, int]]:
    """Flatten a DP2 plane-based mask into a ``uint32`` plus its own mapping.

    A DP2 mask pixel is a short byte array, not a single integer, so
    ``mask.array & bit`` does not work at all; ``mask.get(name)`` returns a plain
    boolean plane and is the only sane way in.  Bit numbers are assigned
    dynamically and are not stable across releases, so the bits used here are
    local and travel with the shard.  Nothing downstream hard-codes one.  Planes
    beyond ``max_planes`` are dropped, with a warning -- a silently truncated
    mask would be worse than a loud one.
    """
    schema = _attr(mask, "schema")
    # ``schema`` iterates plane objects and can yield ``None`` for unused slots;
    # ``schema.names`` is the clean list.
    names = [n for n in getattr(schema, "names", schema) if n]
    if len(names) > max_planes:
        log.warning(
            "mask has %d planes, packing only the first %d: %s dropped",
            len(names), max_planes, names[max_planes:],
        )
        names = names[:max_planes]
    packed = None
    mapping: dict[str, int] = {}
    for bit, name in enumerate(names):
        plane = np.asarray(mask.get(name), dtype=bool)
        if packed is None:
            packed = np.zeros(plane.shape, dtype=np.uint32)
        packed |= plane.astype(np.uint32) << np.uint32(bit)
        mapping[str(name)] = bit
    if packed is None:
        raise ValueError("mask reported no planes")
    return packed, mapping


# -- host selection --------------------------------------------------------


def host_columns(bands: Sequence[str] = BANDS,
                 extra: Sequence[str] = ()) -> list[str]:
    """The column subset to read.  The table has 1248 columns, so this is not
    optional, and asking for one that does not exist fails the whole read."""
    usable = [b for b in bands if b in PHOTOMETRY_BANDS]
    unknown = [b for b in bands if b not in PHOTOMETRY_BANDS]
    if unknown:
        raise ValueError(
            f"band(s) {unknown} are not DP2 bands; choose from {PHOTOMETRY_BANDS}"
        )
    columns = list(OBJECT_COLUMNS)
    for b in usable:
        columns += [c.format(b=b) for c in OBJECT_BAND_COLUMNS]
    return columns + list(extra)


def find_object_refs(butler, ra: float | None = None, dec: float | None = None,
                     radius_deg: float | None = None, limit: int | None = None):
    """``object`` table refs: every one in the repo, or those near a position.

    One per tract.  With no position this is the whole DP2 footprint, which is
    around a thousand tables -- see ``build_host_catalogue`` for what that costs.
    """
    if radius_deg is None or ra is None or dec is None:
        refs = list(butler.query_datasets("object", limit=limit))
        if not refs:
            raise RuntimeError("no object tables in this repo")
        return refs
    region = _lsst().sphgeom.Region.from_ivoa_pos(
        f"CIRCLE {float(ra)} {float(dec)} {float(radius_deg)}"
    )
    refs = list(butler.query_datasets(
        "object", where="tract.region OVERLAPS :region",
        bind={"region": region}, limit=limit,
    ))
    if not refs:
        raise RuntimeError(
            f"no object table within {radius_deg} deg of ({ra}, {dec})"
        )
    return refs


def build_host_catalogue(
    butler,
    bands: Sequence[str] = BANDS,
    band: str = "r",
    ra: float | None = None,
    dec: float | None = None,
    radius_deg: float | None = None,
    limit_tracts: int | None = None,
    cache: str | Path | None = None,
    report_every: int = 25,
    **cuts,
):
    """Host candidates from every object table in reach, cut but not sampled.

    The cuts run per tract and only the survivors are kept, so what is held in
    memory is the host list rather than the footprint.  That matters: an object
    table is ~700k rows, and the whole DP2 coadd footprint is around a thousand
    of them.  Sampling deliberately does *not* happen here -- a stratified draw
    has to see the whole pool, or it stratifies within tracts and not across
    them.

    Reading a thousand tables is minutes to tens of minutes even with column
    pruning, so pass ``cache`` and it is done once.  The cache is keyed by
    nothing: if the cuts change, delete it.
    """
    from astropy.table import vstack

    if cache is not None:
        cache = Path(cache)
        if cache.exists():
            table = _read_table(cache)
            log.info("host catalogue: %d candidates from cache %s",
                     len(table), cache)
            return table

    refs = find_object_refs(butler, ra, dec, radius_deg, limit=limit_tracts)
    columns = host_columns(bands)
    log.info("building host catalogue from %d object table(s)%s", len(refs),
             "" if radius_deg is None else f" within {radius_deg} deg")

    kept: list = []
    n_rows = 0
    for i, ref in enumerate(refs, start=1):
        try:
            table = butler.get(ref, parameters={"columns": columns})
        except Exception as exc:
            log.warning("object table %s unreadable (%r); skipping", ref.dataId, exc)
            continue
        n_rows += len(table)
        if radius_deg is not None and ra is not None:
            table = _within_radius(table, ra, dec, radius_deg)
        if not len(table):
            continue
        survivors = select_hosts(table, band=band, n_hosts=None, **cuts)
        if len(survivors):
            kept.append(survivors)
        if i % report_every == 0 or i == len(refs):
            log.info("  %d/%d tables, %d rows scanned, %d candidates so far",
                     i, len(refs), n_rows, sum(len(t) for t in kept))

    if not kept:
        raise RuntimeError(
            f"no host passed the cuts in {len(refs)} object table(s) covering "
            f"{n_rows} rows; loosen min_reff_arcsec or host_flux_range"
        )
    pool = vstack(kept, metadata_conflicts="silent") if len(kept) > 1 else kept[0]
    # Tracts overlap, so a host in an overlap appears in two tables under two
    # different objectIds.  The per-tract dedupe cannot see that; this can.
    pool = dedupe_hosts(pool)
    log.info("host catalogue: %d candidates from %d rows across %d tables",
             len(pool), n_rows, len(refs))
    if cache is not None:
        _write_table(cache.with_suffix(""), pool)
        log.info("cached the host catalogue at %s", cache)
    return pool


def dedupe_hosts(table, radius_arcsec: float = 0.5):
    """Drop objects that are the same source seen twice.

    DP2 has no ``detect_isPrimary`` -- no ``detect_*`` columns at all -- and
    tracts and patches overlap at their edges, so a source in an overlap region
    appears more than once and, across two tracts, under two different
    ``objectId``s.  Deduplicating on id alone would not catch that, so near
    coincidences on the sky are collapsed too.  Without this a galaxy in an
    overlap region is silently weighted up in the training set.
    """
    ra = np.asarray(table["coord_ra"], dtype=float)
    dec = np.asarray(table["coord_dec"], dtype=float)
    ids = np.asarray(table["objectId"])
    keep = np.zeros(len(table), dtype=bool)
    seen_ids: set = set()
    # Sort by declination so the sky search only has to look at a local window.
    order = np.argsort(dec)
    kept_ra: list[float] = []
    kept_dec: list[float] = []
    r_deg = radius_arcsec / 3600.0
    for i in order:
        if ids[i] in seen_ids:
            continue
        duplicate = False
        for j in range(len(kept_dec) - 1, -1, -1):
            if kept_dec[j] < dec[i] - r_deg:
                break
            cosd = max(np.cos(np.deg2rad(dec[i])), 1e-6)
            if np.hypot((kept_ra[j] - ra[i]) * cosd, kept_dec[j] - dec[i]) < r_deg:
                duplicate = True
                break
        if duplicate:
            continue
        keep[i] = True
        seen_ids.add(ids[i])
        kept_ra.append(ra[i])
        kept_dec.append(dec[i])
    return table[keep]


def _colnames(table) -> set[str]:
    """Column names of an astropy Table or a pandas DataFrame."""
    return set(getattr(table, "colnames", None) or getattr(table, "columns", []))


def host_trace_radius_px(table, band: str) -> np.ndarray:
    """Trace radius in pixels from the per-band adaptive moments.

    DP2 has no band-independent ``shape_xx``; second moments live in
    ``{band}_ixx`` / ``{band}_iyy``, in pixel^2.  Raises rather than returning
    NaN if they are absent, because every caller uses this to avoid a sample
    dominated by the smallest, faintest galaxies, and silently losing that is
    worse than stopping.
    """
    needed = [f"{band}_ixx", f"{band}_iyy"]
    have = _colnames(table)
    if not set(needed) <= have:
        raise KeyError(
            f"{needed} not in the object table. DP2 second moments are per band "
            f"(there is no shape_xx), and only {PHOTOMETRY_BANDS} carry them. "
            f"Columns present: {sorted(c for c in have if '_i' in c)[:12]}"
        )
    ixx = np.asarray(table[needed[0]], dtype=float)
    iyy = np.asarray(table[needed[1]], dtype=float)
    return np.sqrt(np.maximum(0.5 * (ixx + iyy), 0.0))


def host_half_light_arcsec(table, axis: str = "major") -> np.ndarray:
    """Half-light radius in arcsec, from the multiband Sersic fit.

    ``sersic_reff_major`` carries no band prefix: it is one morphology fit to all
    six bands at once, so it does not inherit the band-to-band scatter of a
    per-band fit and does not have to be blended across components the way
    cModel's separate exponential and de Vaucouleurs radii do.

    Two things to know about what it means.  It is in **arcsec** -- unlike
    ``sersic_reff_x``, which is the same quantity in pixels -- and it is measured
    *before* convolution with the PSF, so it is the galaxy's intrinsic size
    rather than its observed extent.  The seen object is a little larger.

    NaN where the fit failed, had no data, or is absent, so a failure cannot
    compare its way through a size cut.
    """
    if axis not in ("major", "minor"):
        raise ValueError(f"axis must be 'major' or 'minor', not {axis!r}")
    col = f"sersic_reff_{axis}"
    have = _colnames(table)
    if col not in have:
        raise KeyError(
            f"{col!r} not in the object table. It is the multiband Sersic fit and "
            f"carries no band prefix. Columns present: "
            f"{sorted(c for c in have if 'sersic' in c or 'reff' in c)[:12]}"
        )
    r = np.asarray(table[col], dtype=float)
    ok = np.isfinite(r) & (r > 0)
    for flag in ("sersic_unknown_flag", "sersic_no_data_flag"):
        if flag in have:
            ok &= ~np.asarray(table[flag], dtype=bool)
    return np.where(ok, r, np.nan)


def select_hosts(
    table,
    band: str = "r",
    flux_range: tuple[float, float] = (360.0, 3.0e6),
    max_blendedness: float | None = None,
    n_hosts: int | None = None,
    seed: int = 0,
    exclude_ids: set[int] | None = None,
    size_stratified: bool = True,
    min_reff_arcsec: float = 3.0,
    min_trace_px: float = 1.75,
    dedupe_radius_arcsec: float = 0.5,
    n_size_bins: int = 5,
):
    """Extended objects in a flux range, stratified by apparent size.

    ``flux_range`` bounds are 360 nJy (r = 25.0) to 3e6 nJy (r = 15.2).  The
    ceiling is high because it has to be: a galaxy with a 3" half-light radius
    and an ordinary effective surface brightness of 22 mag/arcsec^2 has r ~ 17.6,
    nine times brighter than the 36000 nJy ceiling this used to carry.  That
    ceiling was set for a 1" population and would have annihilated the size cut.
    Saturated cores are the gate's job, not this one's.

    ``min_reff_arcsec`` is the real size cut, on ``sersic_reff_major`` from the
    multiband Sersic fit.  The catalogue is dominated by galaxies a pixel or two
    across, which carry no structure for a prior to learn, and they would
    otherwise be most of the sample.  At the DP2 pixel of 0.2 arcsec the default
    3 arcsec is 15 native pixels of half-light radius, 5 after the 3x pooling,
    with the visible galaxy running several half-light radii beyond that.

    That is a demanding cut: galaxies this large are rare, so a small field will
    not supply many of them and ``--radius-deg`` is the knob that matters more
    than ``--n-hosts``.  The count surviving is logged, split by whether the
    object was too small or simply had no usable Sersic fit.

    ``refExtendedness`` is a hard 0/1 threshold on a flux ratio, so it is
    unreliable near the faint limit -- a cut on it alone at r > 23 admits a lot
    of faint stars.  The size cross-check drops the point-like contaminants.
    (DP2 also offers continuous ``{band}_sizeExtendedness`` and
    ``{band}_model_extendedness``, either of which would be a better primary cut
    if the sample turns out to need one.)

    ``size_stratified`` draws equally from bins of equal *width* in log half-light
    radius, so the sample is spread over size rather than following the
    catalogue, in which small faint galaxies vastly outnumber well-resolved ones.
    Note this must be equal-width bins: drawing equally from quantile bins is
    exactly a uniform sample, since quantile bins hold equal numbers by
    construction.  If the size columns are missing this **raises** rather than
    quietly falling back to a uniform draw.

    ``exclude_ids`` drops hosts that have already been tried, so repeated calls
    walk through the catalogue rather than re-offering the same objects.

    ``min_trace_px`` is a second, non-parametric size floor from the adaptive
    moments, kept as a cross-check against a runaway cModel fit.  It is a DP1-era
    ComCam PSF size; check it against the DP2 PSF before leaning on it.
    """
    rng = np.random.default_rng(seed)
    if band not in PHOTOMETRY_BANDS:
        raise ValueError(
            f"band {band!r} has no DP2 Object photometry; choose from "
            f"{PHOTOMETRY_BANDS}"
        )
    flux_col = f"{band}_cModelFlux"
    t = dedupe_hosts(table, dedupe_radius_arcsec)

    keep = np.ones(len(t), dtype=bool)
    if exclude_ids:
        # Topping up towards a target: these have already been tried, and
        # offering them again would either duplicate a stamp or re-earn the same
        # rejection.
        keep &= ~np.isin(np.asarray(t["objectId"], dtype=np.int64),
                         np.fromiter(exclude_ids, dtype=np.int64,
                                     count=len(exclude_ids)))
    if "refExtendedness" in t.colnames:
        ext = np.asarray(t["refExtendedness"], dtype=float)
        keep &= np.isfinite(ext) & (ext > 0.5)
    flux = np.asarray(t[flux_col], dtype=float)
    in_flux = np.isfinite(flux) & (flux > flux_range[0]) & (flux <= flux_range[1])
    keep &= in_flux
    if max_blendedness is not None and f"{band}_blendedness" in t.colnames:
        bl = np.asarray(t[f"{band}_blendedness"], dtype=float)
        keep &= ~(np.isfinite(bl) & (bl > max_blendedness))

    trace = host_trace_radius_px(t, band)
    keep &= np.isfinite(trace) & (trace > min_trace_px)

    reff = host_half_light_arcsec(t)
    if min_reff_arcsec is not None:
        big_enough = np.isfinite(reff) & (reff >= min_reff_arcsec)
        # Split the loss, because "no cModel fit" and "genuinely small" are very
        # different statements about the selection function.
        log.info(
            "half-light cut at %.2f\": %d of %d survive; %d dropped as smaller, "
            "%d for having no usable Sersic fit",
            min_reff_arcsec, int((keep & big_enough).sum()), int(keep.sum()),
            int((keep & np.isfinite(reff) & ~big_enough).sum()),
            int((keep & ~np.isfinite(reff)).sum()),
        )
        # Size and flux are not independent: a galaxy with a 3" half-light
        # radius and any ordinary surface brightness is bright, so a ceiling set
        # for small galaxies quietly annihilates a large size cut.  Rather than
        # assume a surface brightness, notice when the two cuts are nearly
        # disjoint and say so.
        n_size_only = int((big_enough & ~in_flux).sum())
        n_both = int((keep & big_enough).sum())
        if n_size_only and n_both < 0.2 * n_size_only:
            log.warning(
                "%d objects pass the %.2f\" size cut but fail the flux range "
                "%s, against %d that pass both: the flux ceiling is fighting the "
                "size cut. Galaxies this large are bright -- raise "
                "host_flux_range[1]",
                n_size_only, min_reff_arcsec, flux_range, n_both,
            )
        keep &= big_enough

    t = t[keep]
    reff = reff[keep]
    if n_hosts is None or n_hosts >= len(t):
        return t
    if not size_stratified:
        return t[rng.choice(len(t), size=n_hosts, replace=False)]

    # Bins equally spaced in log size, NOT quantiles.  Quantile bins hold equal
    # numbers by construction, so drawing equally from each is exactly a uniform
    # sample and stratifies nothing -- which is what this used to do.  Equal-width
    # bins hold wildly unequal numbers, so an equal draw from each is what
    # actually flattens the size distribution and gets well-resolved hosts into
    # the sample.
    log_size = np.log10(np.maximum(reff, 1e-6))
    edges = np.linspace(log_size.min(), log_size.max(), n_size_bins + 1)
    edges[-1] += 1e-9
    per_bin = max(n_hosts // n_size_bins, 1)
    picks: list[int] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.where((log_size >= lo) & (log_size < hi))[0]
        if len(idx) == 0:
            continue
        picks += list(rng.choice(idx, size=min(per_bin, len(idx)), replace=False))
    # Sparse bins at the large end leave the quota unfilled; top up uniformly
    # from whatever is left rather than returning fewer hosts than asked for.
    picks = list(dict.fromkeys(picks))
    if len(picks) < n_hosts:
        rest = np.setdiff1d(np.arange(len(t)), np.array(picks, dtype=int))
        extra = min(n_hosts - len(picks), len(rest))
        if extra:
            picks += list(rng.choice(rest, size=extra, replace=False))
    return t[np.array(picks[:n_hosts])]


# -- image discovery -------------------------------------------------------


def coadd_refs_for_tract(butler, tract: int, patches: Iterable[int],
                         bands: Sequence[str] = BANDS) -> list:
    """``deep_coadd`` refs for the patches of one tract that hold a host.

    Driven by the host list rather than by a disc on the sky.  With a size cut
    this selective a patch holds one or two hosts, so sweeping every patch that
    overlaps a field loads a great many that hold none; asking for the patches
    the hosts are actually in does not.
    """
    patches = {int(p) for p in patches}
    if not patches:
        return []
    where = "skymap = :skymap AND tract = :tract"
    if bands is not None and len(bands) < len(BANDS):
        where += " AND band.name IN (" + ", ".join(f"'{b}'" for b in bands) + ")"
    refs = butler.query_datasets(
        DATASET_TYPE, where=where,
        bind={"skymap": SKYMAP, "tract": int(tract)},
        explain=False,
    )
    return [r for r in refs
            if int(_data_id_dict(r.dataId).get("patch", -1)) in patches]


def object_refs_for_tract(butler, tract: int) -> list:
    """The ``object`` table of one tract."""
    return list(butler.query_datasets(
        "object", where="skymap = :skymap AND tract = :tract",
        bind={"skymap": SKYMAP, "tract": int(tract)}, explain=False,
    ))


# -- geometry --------------------------------------------------------------


def _sky_to_pixel(wcs, ra, dec) -> tuple[np.ndarray, np.ndarray]:
    """(ra, dec) in degrees -> fractional **tract** pixel (x, y).

    ``sky_projection.sky_to_pixel`` takes a SkyCoord and returns an object with
    ``.x``/``.y``.  It is called one position at a time because a vectorised form
    is not documented; the cost is negligible beside a patch read.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    ra = np.atleast_1d(np.asarray(ra, dtype=float))
    dec = np.atleast_1d(np.asarray(dec, dtype=float))
    xs = np.empty(ra.size)
    ys = np.empty(ra.size)
    for i in range(ra.size):
        xy = wcs.sky_to_pixel(
            SkyCoord(ra=ra[i] * u.deg, dec=dec[i] * u.deg, frame="icrs")
        )
        xs[i], ys[i] = float(xy.x), float(xy.y)
    return xs, ys


def _pixel_to_sky(wcs, x: float, y: float) -> tuple[float, float]:
    """**Tract** pixel -> (ra, dec) in degrees."""
    sky = wcs.pixel_to_sky(x=float(x), y=float(y))
    return float(sky.ra.deg), float(sky.dec.deg)


def _verify_centre(wcs, x: float, y: float, ra: float, dec: float) -> float:
    """Separation in arcsec between where a stamp landed and where it was asked for.

    The guard against DP2's two pixel-origin conventions.  A mix-up displaces a
    position by up to a full patch -- far enough to land on a different galaxy,
    close enough that the stamp still looks like a plausible piece of sky.  Round
    tripping every centre turns that into an immediate, loud failure.
    """
    got_ra, got_dec = _pixel_to_sky(wcs, x, y)
    cosd = np.cos(np.deg2rad(dec))
    return float(np.hypot((got_ra - ra) * cosd, got_dec - dec) * 3600.0)


def _stamp_box(x: float, y: float, size: int):
    """A ``size`` x ``size`` box centred on tract pixel ``(x, y)``.

    ``Box.factory`` is indexed ``[y, x]`` -- numpy order, the opposite of DP1's
    ``Box2I(x, y)``.  A transposed stamp is square and will not error; it will
    just be wrong.
    """
    Box = _lsst().Box
    iy, ix = int(round(y)), int(round(x))
    half = size // 2
    return Box.factory[iy - half : iy - half + size, ix - half : ix - half + size]


def _fits_in_patch(bbox, x: float, y: float, size: int) -> bool:
    """Both stamp corners inside the patch, in tract coordinates.

    Slicing off the edge truncates or raises depending on the path, and stamps of
    inconsistent size would poison the training set.  Rejecting is cheaper than
    repairing; the rejection is logged so the selection function stays
    measurable.
    """
    ix, iy = int(round(x)), int(round(y))
    half = size // 2
    return bool(
        bbox.contains(x=ix - half, y=iy - half)
        and bbox.contains(x=ix - half + size - 1, y=iy - half + size - 1)
    )


def cells_in_stamp(coadd, x: float, y: float, size: int) -> list[tuple[int, int]]:
    """Every ``(i, j)`` cell index the stamp covers.  Empty if the grid is absent."""
    try:
        grid = _attr(coadd, "grid")
        half = size // 2
        corners = [
            grid.index_of(x=int(round(x)) + dx, y=int(round(y)) + dy)
            for dx in (-half, half - 1)
            for dy in (-half, half - 1)
        ]
        ii = [c.i for c in corners]
        jj = [c.j for c in corners]
        return [(i, j)
                for i in range(min(ii), max(ii) + 1)
                for j in range(min(jj), max(jj) + 1)]
    except Exception as exc:
        log.debug("cell grid unavailable: %s", exc)
        return []


#: Candidate spellings of the cell index in ``provenance.contributions``.  The
#: API documents the table as ``{visit, detector, cell}`` without pinning the
#: column names, and ``CellIJ`` does not survive into an astropy column as one
#: object, so the pair is resolved by trial and the real names are logged if none
#: of these match.
CONTRIB_CELL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("cell_i", "cell_j"),
    ("cell_x", "cell_y"),
    ("i", "j"),
    ("x", "y"),
)


def cell_visit_counts(coadd) -> dict[tuple[int, int], int]:
    """Distinct visits contributing to each ``(i, j)`` cell.

    This is the quantity that makes a depth step, and it is exact: DP2 exposures
    share an integration time, so a cell built from 12 visits is simply shallower
    than its neighbour built from 30, and the noise steps across the edge between
    them.  No mask plane says so, but ``CellCoadd.provenance.contributions`` is a
    table of ``{visit, detector, cell}`` -- which observation went into which
    cell -- so counting it gives the step before a single pixel is examined.

    Note ``deep_coadd_input_summary`` is *not* an alternative: Rubin documents it
    as patch-level and says outright that it does not record which visit-detector
    images contributed to each cell.

    Empty if provenance is missing or its columns are not what is expected, which
    leaves the measured ``variance_step`` as the only detector rather than
    failing the run.
    """
    prov = getattr(coadd, "provenance", None)
    contributions = getattr(prov, "contributions", None) if prov is not None else None
    if contributions is None or len(contributions) == 0:
        log.debug("no coadd provenance; falling back to the measured variance step")
        return {}

    names = list(getattr(contributions, "colnames", None)
                 or getattr(contributions, "columns", []))
    cell_cols = next((c for c in CONTRIB_CELL_COLUMNS if set(c) <= set(names)), None)
    if cell_cols is None or "visit" not in names:
        log.warning(
            "provenance.contributions has columns %s; expected a visit column and "
            "one of %s. Per-cell depth is unavailable and only the measured "
            "variance step will catch depth boundaries. Add the real spelling to "
            "CONTRIB_CELL_COLUMNS in rubin/extract.py.",
            names, [list(c) for c in CONTRIB_CELL_COLUMNS],
        )
        return {}

    # One row per (visit, detector, cell), so a visit crossing a detector
    # boundary inside a cell appears twice; count distinct visits, not rows.
    keys = np.stack([
        np.asarray(contributions[cell_cols[0]], dtype=np.int64),
        np.asarray(contributions[cell_cols[1]], dtype=np.int64),
        np.asarray(contributions["visit"], dtype=np.int64),
    ], axis=1)
    cells, counts = np.unique(np.unique(keys, axis=0)[:, :2], axis=0,
                              return_counts=True)
    return {(int(i), int(j)): int(n) for (i, j), n in zip(cells, counts)}


def stamp_depth(counts: dict[tuple[int, int], int],
                cells: list[tuple[int, int]]) -> tuple[int, int]:
    """``(min, max)`` visit count over the cells a stamp covers; ``(-1, -1)`` if
    unknown.  Cells missing from the table contributed nothing and count as 0."""
    if not counts or not cells:
        return -1, -1
    n = [counts.get(c, 0) for c in cells]
    return min(n), max(n)


# -- the driver ------------------------------------------------------------


def next_batch(batch: int | None, n_hosts: int, gained: int, shortfall: int,
               headroom: float = 1.3, blind_growth: int = 4,
               floor: int = 16) -> int:
    """How many hosts to ask for in the next round.

    Sized from the yield actually observed rather than from an assumption, since
    the yield depends on the field, the band set and how tight the gate is, none
    of which are known in advance.  A round that produced nothing says the
    estimate is useless, not that the field is empty, so widen the net instead of
    dividing by zero.

    Always at least ``floor``, so the last few cutouts do not cost a round each.
    """
    if n_hosts <= 0 or gained <= 0:
        return max((batch or floor) * blind_growth, floor)
    per_host = gained / n_hosts
    return max(int(np.ceil(max(shortfall, 0) / per_host * headroom)), floor)


def extract_patches(
    butler,
    out_dir: str | Path,
    ra: float = ECDFS[0],
    dec: float = ECDFS[1],
    radius_deg: float | None = None,
    bands: Sequence[str] = BANDS,
    native_size: int = 416,
    n_hosts: int | None = 8000,
    n_patches: int | None = None,
    max_rounds: int = 8,
    host_cache: str | Path | None = None,
    limit_tracts: int | None = None,
    jitter_arcsec: float = 4.0,
    host_flux_range: tuple[float, float] = (360.0, 3.0e6),
    max_blendedness: float | None = None,
    min_reff_arcsec: float = 3.0,
    patches_per_shard: int = 1024,
    max_patches: int | None = None,
    neighbour_radius_arcsec: float = 30.0,
    gate_kwargs: dict | None = None,
    seed: int = 0,
    prefix: str = "patches",
) -> dict:
    """Extract patches near selected hosts and write shards plus a manifest.

    The loop is organised **by patch, not by host**: one query finds every coadd
    overlapping the field, and each is loaded once and sliced for every host that
    falls inside it.  Iterating hosts instead reloads the same patch repeatedly
    and dominates the runtime.

    Positions are **jittered** around each host rather than centred on it.  A
    prior trained on centred galaxies learns that galaxies are always centred,
    which is useless for a transient that can sit anywhere in the scene.

    ``n_hosts`` is a number of *hosts*; each can yield up to one cutout per band,
    and the gate rejects a good share of those, so it does not set the size of
    the training set.  ``n_patches`` does: it is a target number of accepted
    cutouts, and extraction keeps drawing fresh batches of hosts -- sized from
    the yield it has actually observed -- until it has them, the catalogue runs
    out, or ``max_rounds`` is reached.  ``max_patches`` remains a hard stop that
    never tops up.

    Topping up is not free of consequences: whatever the gate rejects, it
    rejects preferentially, so a set filled by several rounds is drawn deeper
    into the catalogue than one filled by the first.  ``rejection_counts`` in the
    summary is the thing to read before deciding that is acceptable.

    Every attempt is recorded in the manifest, rejections included, with the
    reason and the diagnostics.  Those statistics *are* the selection function,
    and the bias they reveal -- against dense bright centres -- is the regime this
    project exists to model.
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_kwargs = dict(gate_kwargs or {})
    # ``n_patches`` is a target to work towards; ``max_patches`` is a hard stop.
    # Both end the sweep at the same place, only ``n_patches`` tops up.
    target = n_patches if n_patches is not None else max_patches

    field = build_host_catalogue(
        butler, bands=bands, band="r" if "r" in bands else bands[0],
        ra=ra, dec=dec, radius_deg=radius_deg, limit_tracts=limit_tracts,
        cache=host_cache,
        flux_range=host_flux_range, max_blendedness=max_blendedness,
        min_reff_arcsec=min_reff_arcsec,
    )

    records: list[dict] = []
    neighbour_rows: list[dict] = []
    mask_mapping: dict[str, int] | None = None
    writer: ShardWriter | None = None
    n_accepted = 0
    acf = AutocorrelationAccumulator(native_size)
    # Tracts and patches overlap at their edges, so a host near a boundary is
    # covered by more than one patch and would otherwise be extracted twice in
    # the same band -- duplicates that a training set would silently weight up.
    seen: set[tuple[int, str]] = set()
    tried: set[int] = set()
    component_reads_failed = False
    depth_logged = False
    depth_checked = False
    depth_usable = True

    n_refs = 0
    neighbour_columns = host_columns(bands)

    def _sweep(hosts, host_id, tgt_ra, tgt_dec, r_jit):
        """One pass for one batch of hosts, tract by tract.

        Tract-major, not patch-major: the object table is per tract, so the
        neighbour index can be built once per tract and thrown away, which is
        what makes a footprint-wide host list affordable.  Within a tract only
        the patches that actually hold a host are asked for.
        """
        nonlocal writer, mask_mapping, n_accepted, component_reads_failed
        nonlocal depth_logged, depth_checked, depth_usable, n_refs

        by_patch: dict[tuple[int, int], list[int]] = {}
        for h, (t, pa) in enumerate(zip(np.asarray(hosts["tract"], dtype=int),
                                        np.asarray(hosts["patch"], dtype=int))):
            by_patch.setdefault((int(t), int(pa)), []).append(h)
        tracts = sorted({t for t, _ in by_patch})
        # Shuffled, because the sweep stops the moment the target is reached;
        # left in order it would fill the set from one corner of the footprint.
        rng.shuffle(tracts)

        for tract in tracts:
            want = [pa for (t, pa) in by_patch if t == tract]
            refs = coadd_refs_for_tract(butler, tract, want, bands)
            rng.shuffle(refs)
            n_refs += len(refs)
            neighbours = _neighbour_index(butler, tract, neighbour_columns, bands)
            for ref in refs:
                data_id = ref.dataId
                fields = _data_id_dict(data_id)
                band_name = str(fields.get("band", "?"))
                base = {
                    "dataId": json.dumps({k: str(v) for k, v in fields.items()}),
                    "band": band_name,
                    "tract": int(fields.get("tract", -1)),
                    "patch": int(fields.get("patch", -1)),
                }

                # Component reads move no pixels, so the patch is only loaded if a
                # host actually lands in it.  With ~10^6 coadds that matters.  If
                # this repo will not serve components, fall back to whole patches:
                # slower, but the alternative is rejecting every ref in the field
                # and finding out at the end of the run.
                coadd = None
                try:
                    wcs = butler.get(f"{DATASET_TYPE}.sky_projection", dataId=data_id)
                except Exception as exc:
                    if not component_reads_failed:
                        log.warning(
                            "component read of %s.sky_projection failed (%r); loading "
                            "whole patches instead, which is slower but equivalent",
                            DATASET_TYPE, exc,
                        )
                        component_reads_failed = True
                    try:
                        coadd = butler.get(ref)
                    except Exception as exc2:
                        records.append({**base, "status": "rejected",
                                        "reasons": f"read_failed:{exc2!r}"[:120]})
                        continue
                    wcs = _attr(coadd, "wcs")

                # Only the hosts the catalogue assigned to this patch.  Testing
                # every host against every patch is quadratic and unaffordable once
                # the host list spans the footprint rather than one field.
                candidates = [
                    h for h in by_patch.get((tract, base["patch"]), [])
                    if (int(host_id[h]), band_name) not in seen
                ]
                if not candidates:
                    continue
                xs, ys = _sky_to_pixel(wcs, tgt_ra[candidates], tgt_dec[candidates])

                if coadd is not None:
                    bbox = _attr(coadd, "bbox")
                else:
                    try:
                        bbox = butler.get(f"{DATASET_TYPE}.bbox", dataId=data_id)
                    except Exception:
                        coadd = butler.get(ref)
                        bbox = _attr(coadd, "bbox")

                inside = [
                    (h, x, y) for h, x, y in zip(candidates, xs, ys)
                    if _fits_in_patch(bbox, x, y, native_size)
                ]
                if not inside:
                    continue
                if coadd is None:
                    try:
                        coadd = butler.get(ref)
                    except Exception as exc:
                        records.append({**base, "status": "rejected",
                                        "reasons": f"read_failed:{exc!r}"[:120]})
                        continue
                psf_model = _attr(coadd, "psf")
                # Once per patch: which visits went into which cell.  This is the
                # depth step stated exactly, rather than inferred from the noise.
                visit_counts = cell_visit_counts(coadd)
                if visit_counts and not depth_logged:
                    log.info("per-cell visit counts available: %d cells, %d-%d visits",
                             len(visit_counts), min(visit_counts.values()),
                             max(visit_counts.values()))
                    depth_logged = True
                log.debug("patch %s band %s: %d hosts", base["patch"], band_name,
                          len(inside))

                for h, x, y in inside:
                    if target is not None and n_accepted >= target:
                        raise _Done
                    rec = {**base, "host_id": int(host_id[h]),
                           "host_offset_arcsec": float(r_jit[h])}

                    sep = _verify_centre(wcs, x, y, float(tgt_ra[h]), float(tgt_dec[h]))
                    rec["centre_sep_arcsec"] = sep
                    if not np.isfinite(sep) or sep > CENTRE_TOLERANCE_ARCSEC:
                        rec.update(status="rejected", reasons=f"centre_mismatch:{sep:.2f}")
                        records.append(rec)
                        continue

                    # `coadd[box]` is a VIEW; copy or the whole parent patch stays
                    # pinned in memory, which defeats the point of a stamp.
                    stamp = coadd[_stamp_box(x, y, native_size)].copy()
                    image = np.asarray(_attr(stamp, "image").array, dtype=np.float32)
                    if image.shape != (native_size, native_size):
                        rec.update(status="rejected", reasons=f"clipped:{image.shape}")
                        records.append(rec)
                        continue
                    variance = np.asarray(_attr(stamp, "variance").array, dtype=np.float32)
                    packed, mapping = pack_mask(_attr(stamp, "mask"))
                    if mask_mapping is None:
                        mask_mapping = mapping
                    elif mapping != mask_mapping:
                        rec.update(status="rejected", reasons="mask_schema_changed")
                        records.append(rec)
                        continue

                    cells = cells_in_stamp(coadd, x, y, native_size)
                    if visit_counts and cells and not depth_checked:
                        # The grid's (i, j) and the provenance table's cell columns
                        # are two independent conventions, and nothing guarantees
                        # they agree on which one is x.  If they are transposed every
                        # lookup misses, every stamp reads as zero-visit, and the run
                        # rejects everything for the most confusing possible reason.
                        depth_checked = True
                        depth_usable = any(c in visit_counts for c in cells)
                        if not depth_usable:
                            log.warning(
                                "none of the cells a stamp covers %s appear in "
                                "provenance.contributions (which has e.g. %s): the "
                                "cell index conventions do not match, so per-cell "
                                "depth is ignored for this run and only the measured "
                                "variance step will catch depth boundaries",
                                cells[:4], sorted(visit_counts)[:4],
                            )
                    n_lo, n_hi = stamp_depth(visit_counts if depth_usable else {}, cells)
                    if n_lo > 0:
                        depth_ratio = n_hi / n_lo
                    elif n_lo == 0:
                        depth_ratio = np.inf  # a cell with no visits at all
                    else:
                        depth_ratio = None  # provenance unavailable
                    reasons, diag = gate(image, variance, packed, mask_mapping,
                                         cell_depth_ratio=depth_ratio, **gate_kwargs)
                    rec.update({f"diag_{k}": v for k, v in diag.items()})
                    psf = psf_bundle(psf_model, x, y)
                    if psf is None:
                        reasons = list(reasons) + ["psf_unavailable"]
                    if reasons:
                        rec.update(status="rejected", reasons=";".join(reasons))
                        records.append(rec)
                        continue

                    if writer is None:
                        writer = ShardWriter(
                            out_dir / "shards",
                            native_size=native_size,
                            psf_size=psf["psf"].shape[0],
                            mask_plane_dict=mask_mapping,
                            prefix=prefix,
                            patches_per_shard=patches_per_shard,
                            dataset_type=DATASET_TYPE,
                            attrs={
                                "release": "DP2",
                                "skymap": SKYMAP,
                                "field_ra": ra,
                                "field_dec": dec,
                                "bands": json.dumps(list(bands)),
                                "jitter_arcsec": jitter_arcsec,
                                "flux_units": "nJy",
                                "correlated_noise": 1,  # coadds are warped
                                "pixel_origin": PIXEL_ORIGIN,
                                # DP2 coadds get a final background subtraction that
                                # over-subtracts around extended galaxies, and it can
                                # be restored with apply_background('pretty').  These
                                # are as delivered.
                                "background_restored": 0,
                            },
                        )

                    nb = [] if neighbours is None else neighbours.near(
                        float(tgt_ra[h]), float(tgt_dec[h]), neighbour_radius_arcsec,
                        band_name, host_id=int(host_id[h]),
                    )
                    others = [n for n in nb if not n["is_host"]]
                    gal = [n["sep_arcsec"] for n in others if n["extendedness"] > 0.5]
                    star = [n["sep_arcsec"] for n in others if n["extendedness"] <= 0.5]
                    y0, x0 = _origin(stamp)
                    acf.add(image)
                    writer.add(
                        image, variance, packed, psf["psf"],
                        meta={
                            "band_idx": BANDS.index(band_name) if band_name in BANDS else 255,
                            "x0": x0,
                            "y0": y0,
                            "center_x": x,
                            "center_y": y,
                            "ra": float(tgt_ra[h]),
                            "dec": float(tgt_dec[h]),
                            "psf_sigma": psf["psf_sigma"],
                            "psf_fwhm": psf["psf_fwhm"],
                            "psf_ixx": psf["psf_ixx"],
                            "psf_iyy": psf["psf_iyy"],
                            "psf_ixy": psf["psf_ixy"],
                            "pixel_scale": _pixel_scale(wcs, x, y),
                            "sky_noise": diag.get("sky_noise", np.nan),
                            "host_id": int(host_id[h]),
                            "host_offset_arcsec": float(r_jit[h]),
                            "tract": base["tract"],
                            "patch": base["patch"],
                            "n_cells_spanned": len(cells) if cells else -1,
                            "n_visits_min": n_lo,
                            "n_visits_max": n_hi,
                            "n_neighbours": len(others),
                            "neighbour_flux_max": float(
                                max([n["flux"] for n in others], default=np.nan)
                            ),
                            "nearest_galaxy_arcsec": float(min(gal, default=np.nan)),
                            "nearest_star_arcsec": float(min(star, default=np.nan)),
                            "frac_no_data": diag.get("frac_no_data", np.nan),
                            "variance_step": diag.get("variance_step", np.nan),
                            "cell_depth_ratio": diag.get("cell_depth_ratio", np.nan),
                            "frac_inexact_psf": diag.get("frac_INEXACT_PSF", np.nan),
                            "frac_rejected": diag.get("frac_REJECTED", np.nan),
                        },
                    )
                    for n in others:
                        neighbour_rows.append({"patch_index": n_accepted, **n})
                    rec.update(status="accepted", patch_index=n_accepted)
                    records.append(rec)
                    seen.add((int(host_id[h]), band_name))
                    n_accepted += 1
                del coadd

    host_tables = []
    batch = n_hosts
    rounds = 0
    while True:
        rounds += 1
        hosts = select_hosts(
            field,
            band="r" if "r" in bands else bands[0],
            flux_range=host_flux_range,
            max_blendedness=max_blendedness,
            min_reff_arcsec=min_reff_arcsec,
            n_hosts=batch,
            exclude_ids=tried,
            seed=seed + rounds,
        )
        if not len(hosts):
            log.info("no untried hosts left in the catalogue after %d round(s)",
                     rounds - 1)
            break
        tried.update(int(i) for i in hosts["objectId"])
        host_tables.append(hosts)
        log.info("round %d: %d hosts selected from %d candidates",
                 rounds, len(hosts), len(field))

        host_ra = np.asarray(hosts["coord_ra"], dtype=float)
        host_dec = np.asarray(hosts["coord_dec"], dtype=float)
        host_id = np.asarray(hosts["objectId"], dtype=np.int64)
        # Jitter once per host, not once per (host, patch): the same physical
        # scene should be cut the same way in every band.
        r_jit = jitter_arcsec * np.sqrt(rng.uniform(size=len(hosts)))
        th_jit = rng.uniform(0, 2 * np.pi, size=len(hosts))
        cosd = np.maximum(np.cos(np.deg2rad(host_dec)), 1e-6)
        tgt_ra = host_ra + r_jit * np.cos(th_jit) / 3600.0 / cosd
        tgt_dec = host_dec + r_jit * np.sin(th_jit) / 3600.0

        before = n_accepted
        try:
            _sweep(hosts, host_id, tgt_ra, tgt_dec, r_jit)
        except _Done:
            log.info("reached the target of %d cutouts", target)
            break
        gained = n_accepted - before
        log.info("round %d: %d cutouts from %d hosts (%.2f per host), %d total",
                 rounds, gained, len(hosts), gained / max(len(hosts), 1), n_accepted)

        if target is None or n_accepted >= target:
            break
        if rounds >= max_rounds:
            log.warning(
                "stopping after %d rounds with %d of %d cutouts; raise max_rounds, "
                "widen --radius-deg, or loosen the gate -- read rejection_counts "
                "in the summary first", rounds, n_accepted, target,
            )
            break
        batch = next_batch(batch, len(hosts), gained, target - n_accepted)

    if host_tables:
        hosts = _stack_tables(host_tables)
    if target is not None and n_accepted < target:
        log.warning("produced %d of the %d cutouts asked for", n_accepted, target)

    # A host inside the field can still produce no attempt at all: its stamp may
    # not fit inside any one patch.  That is part of the selection function too,
    # and it is invisible in the manifest, which only has rows for attempts.
    attempted = {int(r["host_id"]) for r in records if r.get("host_id") is not None}
    never = len(tried) - len(attempted)

    paths = writer.close() if writer is not None else []
    acf_result = acf.result()
    _write_table(out_dir / "hosts", hosts)
    summary = _write_manifest(out_dir, records, neighbour_rows)
    summary.update(
        release="DP2",
        n_accepted=n_accepted,
        n_shards=len(paths),
        shards=[str(p) for p in paths],
        n_requested=target,
        n_rounds=rounds,
        n_hosts_tried=len(tried),
        n_hosts_attempted=len(attempted),
        n_hosts_no_stamp_fitted=never,
        field_radius_deg=radius_deg,
        n_host_candidates=len(field),
        n_hosts=len(hosts),
        n_coadd_patches=n_refs,
        dataset_type=DATASET_TYPE,
        mask_plane_dict=mask_mapping,
        correlation_length_native_flux_px=acf_result["xi"],
        correlation_length_noise_fraction=acf_result["noise_fraction"],
        correlation_length_n_patches=acf_result["n_patches"],
        correlation_profile_native_flux=acf_result["profile"],
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _within_radius(table, ra: float, dec: float, radius_deg: float):
    """Catalogue rows inside a disc on the sky.  Flat-sky; exact enough under a
    degree, where the error is a part in 10^5."""
    r = np.asarray(table["coord_ra"], dtype=float)
    d = np.asarray(table["coord_dec"], dtype=float)
    cosd = np.maximum(np.cos(np.deg2rad(dec)), 1e-6)
    return table[np.hypot((r - ra) * cosd, d - dec) <= radius_deg]


def _stack_tables(tables):
    """Concatenate the per-round host tables, or return the one there is."""
    if not tables:
        return None
    if len(tables) == 1:
        return tables[0]
    try:
        from astropy.table import vstack

        return vstack(tables, join_type="exact")
    except Exception as exc:  # pragma: no cover - astropy is present in the stack
        log.warning("could not stack %d host tables (%s); recording the first",
                    len(tables), exc)
        return tables[0]


def _origin(stamp) -> tuple[int, int]:
    """``(y0, x0)`` of a stamp in tract coordinates, or ``(-1, -1)``.

    Mandatory, not optional: without the origin a saved stamp cannot be mapped
    back to the sky.
    """
    try:
        yx0 = _attr(stamp, "origin")
        return int(yx0.y), int(yx0.x)
    except Exception as exc:
        log.debug("no stamp origin: %s", exc)
        return -1, -1


def _pixel_scale(wcs, x: float, y: float) -> float:
    """Arcsec per pixel, measured by stepping one pixel through the WCS itself.

    Avoids assuming any particular WCS introspection API.
    """
    try:
        ra0, dec0 = _pixel_to_sky(wcs, x, y)
        ra1, dec1 = _pixel_to_sky(wcs, x + 1.0, y)
        cosd = np.cos(np.deg2rad(dec0))
        return float(np.hypot((ra1 - ra0) * cosd, dec1 - dec0) * 3600.0)
    except Exception:
        return float("nan")


def psf_bundle(psf, x: float, y: float) -> dict | None:
    """Coadd PSF kernel and its moments at tract pixel ``(x, y)``.

    ``compute_kernel_image`` is the convolution kernel a forward model wants
    (``compute_stellar_image`` is the one to compare against an observed star).

    **The DP2 PSF object carries no shape or moment methods at all** -- unlike
    DP1's ``afw`` PSF, ``CellPointSpreadFunction`` offers no ``computeShape``.
    The moments are therefore measured from the kernel here.  The tutorial uses
    GalSim HSM; this uses adaptive moments computed directly, to avoid a
    dependency for one number and to give the same answer on any release.
    """
    try:
        kernel = np.asarray(psf.compute_kernel_image(x=float(x), y=float(y)).array,
                            dtype=np.float32)
    except Exception as exc:
        log.debug("PSF kernel unavailable at (%.1f, %.1f): %s", x, y, exc)
        return None
    if not np.all(np.isfinite(kernel)) or kernel.sum() <= 0:
        return None
    return {"psf": kernel, **_adaptive_moments(kernel)}


#: FWHM / sigma for a Gaussian.
SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _adaptive_moments(image: np.ndarray, max_iter: int = 40,
                      tol: float = 1e-8) -> dict:
    """Gaussian-weighted second moments, iterated to self-consistency.

    Unweighted moments of a PSF kernel are dominated by its wings and by
    whatever noise is out there, and can diverge outright.  Weighting by a
    Gaussian matched to the profile and iterating fixes that: for a Gaussian
    image of covariance ``M`` weighted by ``W``, the weighted covariance is
    ``(M^-1 + W^-1)^-1``, so at the fixed point ``W = M`` the measurement reads
    ``M/2`` and the update is simply twice the weighted moments.
    """
    a = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
    total = a.sum()
    if not np.isfinite(total) or total <= 0:
        return {"psf_sigma": np.nan, "psf_fwhm": np.nan, "psf_ixx": np.nan,
                "psf_iyy": np.nan, "psf_ixy": np.nan}
    yy, xx = np.mgrid[0 : a.shape[0], 0 : a.shape[1]].astype(np.float64)
    cy, cx = (a * yy).sum() / total, (a * xx).sum() / total
    ixx = iyy = max(float(a.shape[0]) / 6.0, 1.0) ** 2
    ixy = 0.0
    for _ in range(max_iter):
        det = ixx * iyy - ixy**2
        if not np.isfinite(det) or det <= 0:
            break
        dx, dy = xx - cx, yy - cy
        chi2 = (iyy * dx**2 - 2 * ixy * dx * dy + ixx * dy**2) / det
        w = a * np.exp(-0.5 * np.clip(chi2, 0, 200))
        wsum = w.sum()
        if wsum <= 0:
            break
        cx_new = (w * xx).sum() / wsum
        cy_new = (w * yy).sum() / wsum
        dxn, dyn = xx - cx_new, yy - cy_new
        nxx = 2.0 * (w * dxn**2).sum() / wsum
        nyy = 2.0 * (w * dyn**2).sum() / wsum
        nxy = 2.0 * (w * dxn * dyn).sum() / wsum
        shift = max(abs(nxx - ixx), abs(nyy - iyy), abs(nxy - ixy))
        cx, cy, ixx, iyy, ixy = cx_new, cy_new, nxx, nyy, nxy
        if shift < tol:
            break
    det = max(ixx * iyy - ixy**2, 0.0)
    sigma = float(det**0.25)
    return {"psf_sigma": sigma, "psf_fwhm": sigma * SIGMA_TO_FWHM,
            "psf_ixx": float(ixx), "psf_iyy": float(iyy), "psf_ixy": float(ixy)}


class _Done(Exception):
    """Internal: stop the nested extraction loops at max_patches."""


def _neighbour_index(butler, tract: int, columns: Sequence[str],
                     bands: Sequence[str]):
    """Neighbour index for one tract, or ``None`` if its table will not read.

    Neighbours must come from the *whole* tract, not from the host pool: a host
    is interesting precisely because of what sits near it, and almost nothing
    near it passed the host cuts.  One table per tract, held only while that
    tract is being swept.
    """
    refs = object_refs_for_tract(butler, tract)
    if not refs:
        log.warning("no object table for tract %d; neighbour covariates will be "
                    "missing for its stamps", tract)
        return None
    try:
        table = butler.get(refs[0], parameters={"columns": list(columns)})
    except Exception as exc:
        log.warning("object table for tract %d unreadable (%r); neighbour "
                    "covariates will be missing", tract, exc)
        return None
    return _NeighbourIndex(table, bands)


class _NeighbourIndex:
    """Catalogue neighbours around a position, indexed once and searched in memory."""

    def __init__(self, catalogue, bands: Sequence[str]):
        self.ra = np.asarray(catalogue["coord_ra"], dtype=float)
        self.dec = np.asarray(catalogue["coord_dec"], dtype=float)
        self.ids = np.asarray(catalogue["objectId"], dtype=np.int64)
        self.flux = {
            b: np.asarray(catalogue[f"{b}_cModelFlux"], dtype=float)
            for b in bands
            if f"{b}_cModelFlux" in catalogue.colnames
        }
        self.extendedness = (
            np.asarray(catalogue["refExtendedness"], dtype=float)
            if "refExtendedness" in catalogue.colnames
            else np.full(len(self.ra), np.nan)
        )

    def near(self, ra: float, dec: float, radius_arcsec: float, band: str,
             host_id: int | None = None):
        cosd = max(np.cos(np.deg2rad(dec)), 1e-6)
        r_deg = radius_arcsec / 3600.0
        box = (np.abs(self.dec - dec) < r_deg) & (
            np.abs(self.ra - ra) * cosd < r_deg
        )
        idx = np.where(box)[0]
        if idx.size == 0:
            return []
        sep = np.hypot((self.ra[idx] - ra) * cosd, self.dec[idx] - dec) * 3600.0
        idx = idx[sep < radius_arcsec]
        sep = sep[sep < radius_arcsec]
        flux = self.flux.get(band)
        return [
            {
                "objectId": int(self.ids[i]),
                "ra": float(self.ra[i]),
                "dec": float(self.dec[i]),
                "sep_arcsec": float(s),
                "flux": float(flux[i]) if flux is not None else float("nan"),
                "extendedness": float(self.extendedness[i]),
                # The host is in the catalogue too, so it appears in its own
                # neighbour list.  Flagged rather than dropped: its separation
                # here is the jitter offset, which is worth being able to check.
                "is_host": bool(host_id is not None and int(self.ids[i]) == host_id),
            }
            for i, s in zip(idx, sep)
        ]


def _write_table(stem: Path, table) -> None:
    """Persist the selected host catalogue next to the shards.

    Size, magnitude, ellipticity and blendedness are known only at selection
    time and are not carried in the shard metadata, so without this the host
    population cannot be inspected after the fact.
    """
    try:
        df = table.to_pandas() if hasattr(table, "to_pandas") else table
        df.to_parquet(stem.with_suffix(".parquet"), index=False)
    except Exception as exc:  # pragma: no cover - depends on the environment
        log.warning("could not write %s.parquet (%s); trying CSV", stem.name, exc)
        try:
            table.write(stem.with_suffix(".csv"), format="ascii.csv", overwrite=True)
        except Exception as exc2:
            log.warning("could not write host table at all: %s", exc2)


def _read_table(path: Path):
    """Read back a table written by ``_write_table``."""
    from astropy.table import Table

    path = Path(path)
    for candidate in (path, path.with_suffix(".parquet"), path.with_suffix(".csv")):
        if candidate.exists() and candidate.is_file():
            if candidate.suffix == ".csv":
                return Table.read(candidate, format="ascii.csv")
            return Table.read(candidate)
    raise FileNotFoundError(path)


def _write_manifest(
    out_dir: Path, records: Iterable[dict], neighbour_rows: Iterable[dict]
) -> dict:
    """Parquet if pandas is available, CSV otherwise.  Never lose the records."""
    records = list(records)
    # A stamp can fail several gates at once, and ``gate`` returns them in a
    # fixed order.  Counting only the first blames whichever check happens to run
    # early -- which is why the summary used to disagree with the figure, and why
    # a plane gated last could account for a quarter of the rejections without
    # appearing in the counts at all.  Count every reason; the totals therefore
    # exceed the number of rejected stamps, which ``n_rejected`` gives.
    reasons: dict[str, int] = {}
    primary: dict[str, int] = {}
    n_rejected = 0
    for r in records:
        if r.get("status") == "accepted":
            continue
        n_rejected += 1
        parts = [p.split(":")[0] for p in str(r.get("reasons", "unknown")).split(";")]
        parts = [p for p in parts if p and p != "nan"]
        for key in dict.fromkeys(parts):
            reasons[key] = reasons.get(key, 0) + 1
        if parts:
            primary[parts[0]] = primary.get(parts[0], 0) + 1
    try:
        import pandas as pd

        pd.DataFrame(records).to_parquet(out_dir / "manifest.parquet", index=False)
        if neighbour_rows:
            pd.DataFrame(list(neighbour_rows)).to_parquet(
                out_dir / "neighbours.parquet", index=False
            )
        fmt = "parquet"
    except Exception as exc:
        import csv

        log.warning("parquet unavailable (%s); writing CSV", exc)
        fmt = "csv"
        if records:
            keys = sorted({k for r in records for k in r})
            with (out_dir / "manifest.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(records)
    return {
        "n_attempts": len(records),
        "n_rejected": n_rejected,
        "manifest_format": fmt,
        # Every reason a stamp failed for; a stamp failing three gates appears
        # three times, so these sum to more than n_rejected.
        "rejection_counts": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        # The first reason only, which is what the old counts were.  Kept so a
        # single blame can be assigned, but do not read it as "the cause".
        "first_rejection_counts": dict(sorted(primary.items(), key=lambda kv: -kv[1])),
        "diagnostic_percentiles": _diagnostic_percentiles(records),
    }


#: Diagnostics whose distribution over *every* attempt, accepted or not, is what
#: a threshold should be chosen from.
PERCENTILE_DIAGNOSTICS = (
    "diag_cell_depth_ratio",
    "diag_variance_step",
    "diag_frac_no_data",
    "diag_inner_frac_no_data",
    "diag_frac_COSMIC_RAY",
    "diag_inner_frac_COSMIC_RAY",
    "diag_frac_INTERPOLATED",
    "diag_inner_frac_INTERPOLATED",
    "diag_frac_SATURATED",
    "diag_inner_frac_SATURATED",
)


def _diagnostic_percentiles(records: list[dict]) -> dict[str, dict[str, float]]:
    """Percentiles of each gated diagnostic across every attempt.

    Thresholds should come from what the field actually looks like rather than
    from a guess, and this puts the numbers in the summary so choosing one does
    not mean writing code against the manifest.
    """
    out: dict[str, dict[str, float]] = {}
    for key in PERCENTILE_DIAGNOSTICS:
        vals = np.asarray([r[key] for r in records if key in r], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            continue
        pcts = np.percentile(vals, [5, 25, 50, 75, 90, 95, 99])
        out[key.removeprefix("diag_")] = {
            f"p{p}": float(v) for p, v in zip((5, 25, 50, 75, 90, 95, 99), pcts)
        }
    return out
