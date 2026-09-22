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
import requests

from ..config import BANDS
from ..selection import AB_ZEROPOINT, HostCuts, Selection
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

#: Attribute names on a ``CellCoadd``, and the names of the butler components
#: that serve them.  Note what is *absent*: ``grid`` and ``bounds`` are Python
#: properties reading through to ``psf.bounds``, not components, so they come
#: off the PSF object rather than from the butler.
DP2_ATTRS = {
    "image": "image",
    "variance": "variance",
    "mask": "mask",
    "psf": "psf",
    "wcs": "sky_projection",
    "bbox": "bbox",
    "schema": "schema",
    "origin": "yx0",
    "provenance": "provenance",
}

#: DP2 offers three WCS representations and they do **not** share a pixel
#: origin: ``sky_projection`` is in *tract* coordinates, ``astropy_wcs`` in
#: *patch-local*.  Mixing them is a position error of up to a full patch
#: (~4000 px) -- far enough to land in the wrong galaxy, close enough to look
#: plausible.  This module uses tract coordinates throughout, which is both the
#: precise representation and the frame ``Box.factory`` and ``bbox.contains``
#: expect.  ``yx0`` converts to patch-local if ever needed.
PIXEL_ORIGIN = "tract"

#: How many patches may hold none of their assigned hosts before the run says
#: the patch numbering is probably wrong rather than the hosts merely being near
#: an edge.
PATCH_CHECK_AFTER = 20

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

#: What the neighbour index needs, and nothing else.
#:
#: Deliberately not ``host_columns``.  The TAP ``dp2.Object`` view and the
#: butler ``object`` parquet are **not the same table**: TAP serves derived
#: columns the pipeline never wrote, ``{band}_cModelMag`` among them, and asking
#: the butler for one fails the whole read.  The host selection runs against
#: TAP; the neighbour index runs against the butler; so they get different
#: lists, chosen for what each source has and each caller needs.
NEIGHBOUR_COLUMNS = ["objectId", "coord_ra", "coord_dec", "refExtendedness"]
NEIGHBOUR_BAND_COLUMNS = ["{b}_cModelFlux"]

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
    "{b}_blendedness",
    # HSM adaptive moments, measured on the PSF-convolved coadd, and the same
    # moments for the PSF itself.  The difference is what makes a size cut mean
    # anything: a point source has ixx == ixxPSF, so an absolute pixel threshold
    # on the moments alone rejects nothing at median seeing.
    "{b}_ixx",
    "{b}_iyy",
    "{b}_ixy",
    "{b}_ixxPSF",
    "{b}_iyyPSF",
    # The real bright limit, and the real reason to drop a core.
    "{b}_pixelFlags_saturatedCenter",
    "{b}_pixelFlags_interpolatedCenter",
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


def read_component(butler, ref, role: str):
    """A ``deep_coadd`` component.  Raises if the repo will not serve it.

    Components move no pixels, so everything needed to place and characterise a
    stamp -- WCS, PSF, provenance -- comes this way and the pixels then come
    from a bbox read of just the stamp.  A component that does not answer is a
    bug in this mapping, not a condition to work around: ``grid`` and ``bounds``
    were assumed to be components here for a while, silently forced a whole-patch
    read on every stamp, and cost the run two orders of magnitude in I/O before
    anyone noticed.
    """
    name = DP2_ATTRS[role]
    try:
        return butler.get(f"{DATASET_TYPE}.{name}", dataId=ref.dataId)
    except Exception as exc:
        raise RuntimeError(
            f"{DATASET_TYPE}.{name} (role {role!r}) is not served by this repo: "
            f"{exc!r}. Fix DP2_ATTRS[{role!r}] in rubin/extract.py -- note that "
            f"CellCoadd.grid and .bounds are properties reading through to "
            f"psf.bounds, not components."
        ) from None


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
    """A ``DataCoordinate`` as a plain dict.

    Through ``.mapping``, not ``dict()``: ``DataCoordinate`` stopped being a
    ``Mapping`` in daf_butler v27, so ``dict()`` falls through to sequence
    iteration, asks for ``data_id[0]``, and dies with ``KeyError: 0``.
    """
    return {str(k): v for k, v in data_id.mapping.items()}


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
            len(names),
            max_planes,
            names[max_planes:],
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


def neighbour_columns(bands: Sequence[str] = BANDS) -> list[str]:
    """Columns for the per-tract neighbour index, read through the butler."""
    return list(NEIGHBOUR_COLUMNS) + [
        c.format(b=b) for b in bands if b in PHOTOMETRY_BANDS for c in NEIGHBOUR_BAND_COLUMNS
    ]


def host_columns(bands: Sequence[str] = BANDS, extra: Sequence[str] = ()) -> list[str]:
    """The column subset to read.  The table has 1248 columns, so this is not
    optional, and asking for one that does not exist fails the whole read."""
    usable = [b for b in bands if b in PHOTOMETRY_BANDS]
    unknown = [b for b in bands if b not in PHOTOMETRY_BANDS]
    if unknown:
        raise ValueError(f"band(s) {unknown} are not DP2 bands; choose from {PHOTOMETRY_BANDS}")
    columns = list(OBJECT_COLUMNS)
    for b in usable:
        columns += [c.format(b=b) for c in OBJECT_BAND_COLUMNS]
    return columns + list(extra)


def find_object_refs(
    butler,
    ra: float | None = None,
    dec: float | None = None,
    radius_deg: float | None = None,
    limit: int | None = None,
):
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
    refs = list(
        butler.query_datasets(
            "object",
            where="tract.region OVERLAPS :region",
            bind={"region": region},
            limit=limit,
        )
    )
    if not refs:
        raise RuntimeError(f"no object table within {radius_deg} deg of ({ra}, {dec})")
    return refs


#: The TAP-side table.  The host cuts are a selection, and a selection is what a
#: query service is for: the whole footprint is ~10^9 rows and the survivors are
#: ~10^4, so the difference between filtering there and filtering here is the
#: difference between moving the survivors and moving the catalogue.
TAP_TABLE = "dp2.Object"


def host_adql(
    bands=BANDS,
    cuts: HostCuts | None = None,
    ra: float | None = None,
    dec: float | None = None,
    radius_deg: float | None = None,
    top: int | None = None,
) -> str:
    """The ADQL for the host selection, built from ``HostCuts``.

    Arithmetic here is addition and multiplication only.  ``LOG10`` and
    ``POWER`` are not guaranteed across ADQL dialects, and a clause the service
    silently declines to apply is worse than one it refuses outright -- so the
    magnitude limits are written as fluxes and the surface-brightness limit as a
    flux against an area.

    The boolean flags are fetched and applied locally: how a boolean compares in
    ADQL is backend-specific and a wrong guess quietly returns nothing.  Nor is
    there an ``ORDER BY`` -- sorting burdens a shared service and the stratified
    draw happens here anyway.
    """
    cuts = cuts or HostCuts()
    band = cuts.band
    faint, bright = cuts.flux_range
    columns = ", ".join(host_columns(bands))
    where = [f"{band}_cModelFlux > {faint:.1f}",
             f"{band}_cModelFlux <= {bright:.1f}"]
    if cuts.min_extendedness is not None:
        where.append(f"refExtendedness > {float(cuts.min_extendedness)}")
    # NaN and NULL both fail a > comparison, which is what is wanted: an object
    # with no fit is not a large object.
    where.append(f"sersic_reff_major >= {float(cuts.min_reff_arcsec)}")
    where.append(f"sersic_reff_major <= {float(cuts.max_reff_arcsec)}")
    floor = cuts.surface_brightness_floor()
    if floor is not None:
        where.append(f"{band}_cModelFlux >= {floor:.1f} "
                     f"* sersic_reff_major * sersic_reff_minor")
    if cuts.max_sersic_index is not None:
        where.append(f"sersic_index <= {float(cuts.max_sersic_index)}")
    if cuts.min_deconvolved_px is not None:
        # T^2 = ((ixx+iyy) - (ixxPSF+iyyPSF))/2, so this is T >= the threshold.
        where.append(
            f"({band}_ixx + {band}_iyy - {band}_ixxPSF - {band}_iyyPSF) "
            f">= {2.0 * float(cuts.min_deconvolved_px) ** 2}"
        )
    if cuts.max_blendedness is not None:
        where.append(f"{band}_blendedness <= {float(cuts.max_blendedness)}")
    if radius_deg is not None and ra is not None and dec is not None:
        where.append(
            "CONTAINS(POINT('ICRS', coord_ra, coord_dec), "
            f"CIRCLE('ICRS', {float(ra)}, {float(dec)}, {float(radius_deg)})) = 1"
        )
    select = f"SELECT TOP {int(top)}" if top else "SELECT"
    return f"{select} {columns}\nFROM {TAP_TABLE}\nWHERE " + "\n  AND ".join(where)


#: Public, unauthenticated: it is how the RSP itself finds its service URLs, and
#: it means the endpoint below does not have to be hard-coded forever.
RSP_DISCOVERY_URL = "https://data.lsst.cloud/repertoire/discovery"

#: Where a Gafaelfawr token is looked for, in order.  Same precedence as
#: ``lsst.rsp`` uses, so a notebook and a login node behave the same.  Never a
#: command-line argument: that would put the token in shell history and in every
#: process listing on a shared machine.
TOKEN_ENV_VARS = ("ACCESS_TOKEN", "NUBLADO_TOKEN", "RSP_TOKEN")
TOKEN_PATHS = ("/etc/nublado/secrets/token", "~/.rsp-token", "~/.rsp_token")


class _BearerForPrefix(requests.auth.AuthBase):
    """Attach a bearer token, but only to URLs under the service.

    Restricting by prefix rather than setting a session header outright, because
    a session header follows redirects: one redirect off-host and the token has
    been handed to whoever answered.
    """

    def __init__(self, token: str, prefixes: Sequence[str]) -> None:
        self._token = token
        self._prefixes = tuple(p.rstrip("/") for p in prefixes)

    def __call__(self, request):
        from urllib.parse import urlparse, urlunparse

        url = urlunparse(urlparse(request.url or "")._replace(query="", fragment=""))
        if any(url == p or url.startswith(p + "/") for p in self._prefixes):
            request.headers["Authorization"] = f"Bearer {self._token}"
        return request


#: The scope a Gafaelfawr token needs to use TAP.
TAP_SCOPE = "read:tap"


def find_token(token: str | None = None) -> tuple[str, str]:
    """``(token, where it came from)``.  The source matters: ``ACCESS_TOKEN`` is
    a generic name that other software sets too, and picking up somebody else's
    value looks exactly like a rejected RSP token."""
    if token:
        return token, "the caller"
    import os

    for var in TOKEN_ENV_VARS:
        if value := os.environ.get(var):
            return value.strip(), f"${var}"
    for path in TOKEN_PATHS:
        candidate = Path(path).expanduser()
        try:
            if candidate.is_file() and (value := candidate.read_text().strip()):
                return value, str(candidate)
        except OSError:
            continue
    raise RuntimeError(
        "No RSP token found. Make one at https://data.lsst.cloud under Security "
        f"tokens with the {TAP_SCOPE!r} scope, then either export it as "
        "ACCESS_TOKEN or save it to ~/.rsp-token (chmod 600). Looked in "
        f"{TOKEN_ENV_VARS} and {TOKEN_PATHS}. To avoid TAP entirely, use "
        "source='butler'."
    )


def rsp_token(token: str | None = None) -> str:
    """A Gafaelfawr token from the environment or from disk."""
    value, source = find_token(token)
    # The source and the type prefix, never the token: a Gafaelfawr token is
    # "gt-<key>.<secret>", so the prefix alone says whether what was found is
    # even an RSP token, which is the usual answer when ACCESS_TOKEN was set by
    # something else entirely.
    log.info("RSP token from %s (looks like %r)", source, value.split("-", 1)[0] + "-...")
    return value


def token_info(token: str, base_url: str = "https://data.lsst.cloud") -> dict:
    """What Gafaelfawr says about a token: username, scopes, expiry."""
    url = base_url.rstrip("/") + "/auth/api/v1/token-info"
    response = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=15)
    if response.status_code in (401, 403):
        raise RuntimeError(
            f"Gafaelfawr rejected the token ({response.status_code}). It is "
            f"expired, revoked, or not an RSP token at all. Make a new one at "
            f"{base_url} under Security tokens with the {TAP_SCOPE!r} scope."
        )
    response.raise_for_status()
    return response.json()


def check_tap_scope(token: str, base_url: str = "https://data.lsst.cloud") -> None:
    """Fail now, with the reason, rather than as a 401 inside a TAP job."""
    info = token_info(token, base_url)
    scopes = list(info.get("scopes") or [])
    who = info.get("username", "?")
    if TAP_SCOPE not in scopes:
        raise RuntimeError(
            f"The token for {who!r} is valid but has scopes {scopes}, which do "
            f"not include {TAP_SCOPE!r}. TAP will answer 401. Make a new token "
            f"at {base_url} under Security tokens with that scope ticked."
        )
    log.info("token for %s carries %s", who, TAP_SCOPE)


def discover_tap_url(release: str = "dp2", discovery_url: str = RSP_DISCOVERY_URL) -> str:
    """The TAP endpoint for a release, from the RSP's own discovery document."""
    response = requests.get(discovery_url, timeout=15)
    response.raise_for_status()
    datasets = response.json().get("datasets", {})
    url = datasets.get(release, {}).get("services", {}).get("tap", {}).get("url")
    if not url:
        raise RuntimeError(
            f"{discovery_url} lists no TAP service for {release!r}; it offers "
            f"{sorted(datasets)}"
        )
    return url


def tap_client(release: str = "dp2", url: str | None = None, token: str | None = None):
    """A TAP client for the RSP, from anywhere -- no ``lsst.rsp`` needed.

    TAP is an IVOA standard and the RSP's endpoint is an ordinary TAP service
    behind a bearer token, so ``pyvo`` speaks to it directly.  ``lsst.rsp`` only
    exists on the RSP itself, where it wraps exactly this.
    """
    try:
        import pyvo
    except ImportError as exc:
        raise RuntimeError(
            "pyvo is needed to reach the TAP service (pip install pyvo), or use "
            "source='butler' to scan the object tables through the repo instead."
        ) from exc

    from urllib.parse import urlparse

    url = url or discover_tap_url(release)
    value = rsp_token(token)
    parts = urlparse(url)
    check_tap_scope(value, f"{parts.scheme}://{parts.netloc}")
    session = requests.Session()
    session.auth = _BearerForPrefix(value, [url])
    log.info("TAP service at %s", url)
    return pyvo.dal.TAPService(url, session=session)


def run_adql(service, query: str, timeout: float | None = None):
    """Submit an async ADQL job, wait for it, return an astropy table.

    Async rather than sync because a footprint-wide selection is a long query;
    the job is deleted afterwards either way, since an abandoned job sits on a
    shared service.
    """
    job = service.submit_job(query)
    try:
        job.run()
        job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=timeout)
        if job.phase != "COMPLETED":
            job.raise_if_error()
            raise RuntimeError(f"TAP job ended in phase {job.phase}")
        return job.fetch_result().to_table()
    finally:
        try:
            job.delete()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            log.debug("could not delete TAP job: %r", exc)


def unmask(table):
    """Masked TAP columns to plain values.  NULL floats become NaN.

    A VOTable NULL comes back as a masked entry, and ``np.asarray`` on a masked
    column hands back the raw buffer with no hint that part of it is not data.
    For a float that is usually NaN and harmless downstream; for an integer like
    ``patch`` it is whatever happened to be in memory, which would file a host
    under a patch it is nowhere near.
    """
    for name in list(getattr(table, "colnames", [])):
        column = table[name]
        mask = getattr(column, "mask", None)
        if mask is None or not np.any(mask):
            continue
        n = int(np.sum(mask))
        if column.dtype.kind == "f":
            table[name] = np.asarray(column.filled(np.nan), dtype=column.dtype)
        elif column.dtype.kind in "iu":
            log.warning(
                "%d row(s) have no %s; setting them to -1, which will " "not match any patch",
                n,
                name,
            )
            table[name] = np.asarray(column.filled(-1), dtype=column.dtype)
        else:
            continue
        log.debug("unmasked %d null(s) in %s", n, name)
    return table


def with_positions(table):
    """Rows that have a usable sky position, with a count of those that do not."""
    ra = np.asarray(table["coord_ra"], dtype=float)
    dec = np.asarray(table["coord_dec"], dtype=float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    if not ok.all():
        log.warning("dropping %d host candidate(s) with no sky position", int((~ok).sum()))
    return table[ok]


def build_host_catalogue(
    butler=None,
    bands: Sequence[str] = BANDS,
    ra: float | None = None,
    dec: float | None = None,
    radius_deg: float | None = None,
    limit_tracts: int | None = None,
    cache: str | Path | None = None,
    report_every: int = 25,
    source: str = "tap",
    tap_service=None,
    tap_url: str | None = None,
    top: int | None = None,
    cuts: HostCuts | None = None,
):
    """Host candidates from every object table in reach, cut but not sampled.

    Two ways to get there.  ``source="tap"`` sends the cuts to the TAP service as
    one ADQL query, which is what a query service is for: the footprint is ~10^9
    rows and the survivors are ~10^4, so the selection belongs where the
    catalogue already is.  That needs network and an RSP token, which a batch
    node may not have -- so the result is cached, and extraction can then run
    from the cache with no network at all.

    ``source="butler"`` is the offline route: scan the object tables through the
    butler instead.  It reads far more (every row of every tract, column-pruned)
    but needs nothing beyond the repo.  It is not a fallback that happens
    silently; ask for it.

    Either way the cuts run before anything is concatenated, so what is held in
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

    cuts = cuts or HostCuts()
    if cache is not None:
        cache = Path(cache)
        if cache.exists():
            table = _read_table(cache)
            log.info("host catalogue: %d candidates from cache %s", len(table), cache)
            return table

    if source not in ("tap", "butler"):
        raise ValueError(f"source must be 'tap' or 'butler', not {source!r}")

    if source == "tap":
        query = host_adql(bands=bands, cuts=cuts, ra=ra, dec=dec,
                          radius_deg=radius_deg, top=top)
        log.info("querying %s:\n%s", TAP_TABLE, query)
        pool = with_positions(unmask(run_adql(tap_service or tap_client(url=tap_url), query)))
        log.info("TAP returned %d usable rows", len(pool))
        # The service applied the numeric cuts; these are the rest -- the Sersic
        # failure flags, the point-source cross-check, and the dedupe across
        # tracts, which the query cannot do.
        pool = select_hosts(pool, cuts=cuts, n_hosts=None)
        log.info("host catalogue: %d candidates", len(pool))
        if cache is not None:
            _write_table(cache.with_suffix(""), pool)
            log.info("cached the host catalogue at %s", cache)
        return pool

    if butler is None:
        raise ValueError("source='butler' needs a butler")
    refs = find_object_refs(butler, ra, dec, radius_deg, limit=limit_tracts)
    # Through the butler, so the same caveat as the neighbour index: the parquet
    # is the pipeline's own output and does not carry TAP's derived columns.
    columns = host_columns(bands)
    log.info(
        "building host catalogue from %d object table(s)%s",
        len(refs),
        "" if radius_deg is None else f" within {radius_deg} deg",
    )

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
        survivors = select_hosts(table, cuts=cuts, n_hosts=None)
        if len(survivors):
            kept.append(survivors)
        if i % report_every == 0 or i == len(refs):
            log.info(
                "  %d/%d tables, %d rows scanned, %d candidates so far",
                i,
                len(refs),
                n_rows,
                sum(len(t) for t in kept),
            )

    if not kept:
        raise RuntimeError(
            f"no host passed the cuts in {len(refs)} object table(s) covering "
            f"{n_rows} rows; loosen the host cuts"
        )
    pool = vstack(kept, metadata_conflicts="silent") if len(kept) > 1 else kept[0]
    # Tracts overlap, so a host in an overlap appears in two tables under two
    # different objectIds.  The per-tract dedupe cannot see that; this can.
    pool = dedupe_hosts(with_positions(pool))
    log.info(
        "host catalogue: %d candidates from %d rows across %d tables", len(pool), n_rows, len(refs)
    )
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


def host_mu_e(table, band: str) -> np.ndarray:
    """Mean surface brightness inside the half-light ellipse, mag/arcsec^2."""
    flux = np.asarray(table[f"{band}_cModelFlux"], dtype=float)
    a = host_half_light_arcsec(table, "major")
    b = host_half_light_arcsec(table, "minor")
    with np.errstate(invalid="ignore", divide="ignore"):
        mag = AB_ZEROPOINT - 2.5 * np.log10(np.where(flux > 0, flux, np.nan))
        return mag + 2.5 * np.log10(2.0 * np.pi * a * b)


def host_deconvolved_px(table, band: str) -> np.ndarray:
    """Moment radius with the PSF removed in quadrature, in native pixels.

    ``T^2 = ((ixx + iyy) - (ixxPSF + iyyPSF)) / 2``.  A point source gives zero,
    which is the property an absolute cut on the raw moments does not have: at
    median DP2 seeing a star sits at 2.0 px, and ``min_trace_px = 1.75`` was
    therefore rejecting nothing at all.
    """
    needed = [f"{band}_{c}" for c in ("ixx", "iyy", "ixxPSF", "iyyPSF")]
    have = _colnames(table)
    if not set(needed) <= have:
        raise KeyError(
            f"{[c for c in needed if c not in have]} not in the object table; "
            f"the PSF moments are what make a size cut mean anything"
        )
    ixx, iyy, pxx, pyy = (np.asarray(table[c], dtype=float) for c in needed)
    excess = 0.5 * ((ixx + iyy) - (pxx + pyy))
    return np.sqrt(np.maximum(excess, 0.0))


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

    ``sersic_reff_major`` carries no band prefix: it is one morphology fit to
    all six bands at once, so it does not inherit the band-to-band scatter of a
    per-band fit and does not need blending across components the way cModel's
    separate exponential and de Vaucouleurs radii do.

    Two things to know about what it means.  It is in **arcsec** -- unlike
    ``sersic_reff_x``, which is the same quantity in pixels -- and it is measured
    *before* convolution with the PSF, so it is the galaxy's intrinsic size
    rather than its observed extent.  The seen object is a little larger.

    NaN where the fit failed, had no data, or is nonsensical, so a failure
    cannot compare its way through a size cut.
    """
    if axis not in ("major", "minor"):
        raise ValueError(f"axis must be 'major' or 'minor', not {axis!r}")
    col = f"sersic_reff_{axis}"
    have = _colnames(table)
    if col not in have:
        stale = sorted(c for c in have if "_cModel_" in c and "reff" in c)
        hint = (
            " This table has per-band cModel radii and no Sersic fit, so it was "
            "written before the switch to the multiband Sersic size -- it is an "
            "older extraction. Re-extract, or point at the newer output "
            "directory."
            if stale
            else ""
        )
        raise KeyError(
            f"{col!r} not in the object table. It is the multiband Sersic fit "
            f"and carries no band prefix.{hint} Columns present: "
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
    cuts: HostCuts | None = None,
    n_hosts: int | None = None,
    seed: int = 0,
    exclude_ids: set[int] | None = None,
):
    """Apply ``HostCuts`` to a catalogue and draw a size-stratified sample.

    The same cuts the ADQL already applied are applied again here, because the
    butler path does not go through ADQL at all and because booleans cannot be
    trusted to the query.  Re-applying a cut the service already made is cheap
    and keeps one definition of what a host is.

    ``size_stratified`` draws equally from bins of equal *width* in log size
    over the **fixed** range in the cuts, not over the sample's own min and max.
    Two ways to get this wrong, both of which were here: quantile bins hold
    equal numbers by construction, so drawing equally from them is exactly a
    uniform sample; and data-driven edges hand whole bins to whatever tail the
    sample has, which with runaway fits means stratification selects them
    preferentially.
    """
    cuts = cuts or HostCuts()
    band = cuts.band
    rng = np.random.default_rng(seed)
    if band not in PHOTOMETRY_BANDS:
        raise ValueError(
            f"band {band!r} has no DP2 Object photometry; choose from "
            f"{PHOTOMETRY_BANDS}"
        )
    t = dedupe_hosts(table, cuts.dedupe_radius_arcsec)
    have = _colnames(t)
    keep = np.ones(len(t), dtype=bool)
    if exclude_ids:
        # Topping up towards a target: these have been tried, and offering them
        # again would either duplicate a stamp or re-earn the same rejection.
        keep &= ~np.isin(np.asarray(t["objectId"], dtype=np.int64),
                         np.fromiter(exclude_ids, dtype=np.int64,
                                     count=len(exclude_ids)))
    if cuts.min_extendedness is not None and "refExtendedness" in have:
        ext = np.asarray(t["refExtendedness"], dtype=float)
        keep &= np.isfinite(ext) & (ext > cuts.min_extendedness)

    faint, bright = cuts.flux_range
    flux = np.asarray(t[f"{band}_cModelFlux"], dtype=float)
    keep &= np.isfinite(flux) & (flux > faint) & (flux <= bright)
    if cuts.max_blendedness is not None and f"{band}_blendedness" in have:
        bl = np.asarray(t[f"{band}_blendedness"], dtype=float)
        keep &= ~(np.isfinite(bl) & (bl > cuts.max_blendedness))
    for flag, wanted in (("saturatedCenter", cuts.reject_saturated_centre),
                         ("interpolatedCenter", cuts.reject_interpolated_centre)):
        column = f"{band}_pixelFlags_{flag}"
        if wanted and column in have:
            keep &= ~np.asarray(t[column], dtype=bool)

    reff = host_half_light_arcsec(t)
    if cuts.min_deconvolved_px is not None:
        keep &= host_deconvolved_px(t, band) >= cuts.min_deconvolved_px
    in_size = (np.isfinite(reff) & (reff >= cuts.min_reff_arcsec)
               & (reff <= cuts.max_reff_arcsec))
    if cuts.max_mu_e is not None:
        mu = host_mu_e(t, band)
        bright_enough = np.isfinite(mu) & (mu <= cuts.max_mu_e)
        log.info("mu_e <= %.1f: %d of %d survive", cuts.max_mu_e,
                 int((keep & bright_enough).sum()), int(keep.sum()))
        keep &= bright_enough
    log.info("size in [%.2f, %.2f]\": %d of %d survive; %d too small, %d too "
             "large, %d with no usable Sersic fit",
             cuts.min_reff_arcsec, cuts.max_reff_arcsec,
             int((keep & in_size).sum()), int(keep.sum()),
             int((keep & np.isfinite(reff) & (reff < cuts.min_reff_arcsec)).sum()),
             int((keep & np.isfinite(reff) & (reff > cuts.max_reff_arcsec)).sum()),
             int((keep & ~np.isfinite(reff)).sum()))
    keep &= in_size

    t = t[keep]
    reff = reff[keep]
    if n_hosts is None or n_hosts >= len(t):
        return t
    if not cuts.size_stratified:
        return t[rng.choice(len(t), size=n_hosts, replace=False)]

    edges = np.linspace(np.log10(cuts.min_reff_arcsec),
                        np.log10(cuts.max_reff_arcsec), cuts.n_size_bins + 1)
    edges[-1] += 1e-9
    log_size = np.log10(np.maximum(reff, 1e-6))
    per_bin = max(n_hosts // cuts.n_size_bins, 1)
    picks: list[int] = []
    for a, b in zip(edges[:-1], edges[1:]):
        idx = np.where((log_size >= a) & (log_size < b))[0]
        if len(idx) == 0:
            continue
        picks += list(rng.choice(idx, size=min(per_bin, len(idx)), replace=False))
    # Sparse bins at the large end leave the quota unfilled; top up uniformly
    # rather than returning fewer hosts than asked for.
    picks = list(dict.fromkeys(picks))
    if len(picks) < n_hosts:
        rest = np.setdiff1d(np.arange(len(t)), np.array(picks, dtype=int))
        extra = min(n_hosts - len(picks), len(rest))
        if extra:
            picks += list(rng.choice(rest, size=extra, replace=False))
    return t[np.array(picks[:n_hosts])]


def _tract_refs(butler, dataset_type: str, tract: int, bands=None) -> list:
    """Every ``dataset_type`` in one tract, constrained by data id, not by text.

    By data id because the expression language bit once and silently: in
    ``where="tract = :tract"`` the bind key shadows the dimension of the same
    name, so it resolved as ``tract = tract`` -- true for every row.  The query
    then returned the whole repo, truncated at the default 20000, and hosts were
    matched against same-numbered patches in other tracts, which projected a
    couple of hundred thousand pixels away.  ``data_id`` takes key-value equality
    constraints and cannot be read as anything else.
    """
    refs = butler.query_datasets(
        dataset_type,
        data_id={"skymap": SKYMAP, "tract": int(tract)},
        limit=None,
        explain=False,
    )
    kept, wrong = [], 0
    want = set(bands) if bands else None
    for ref in refs:
        fields = _data_id_dict(ref.dataId)
        if int(fields.get("tract", -1)) != int(tract):
            wrong += 1
            continue
        if want is not None and str(fields.get("band", "?")) not in want:
            continue
        kept.append(ref)
    if wrong:
        # Belt and braces: if a server-side constraint ever stops applying
        # again, this is the line that says so instead of a run of empty
        # patches and a confusing projection.
        log.warning(
            "%d %s refs came back for tracts other than %d and were "
            "dropped here; the query is not constraining tract",
            wrong,
            dataset_type,
            tract,
        )
    return kept


def coadd_refs_for_tract(
    butler, tract: int, patches: Iterable[int], bands: Sequence[str] = BANDS
) -> list:
    """``deep_coadd`` refs for the patches of one tract that hold a host.

    Driven by the host list rather than by a disc on the sky.  With a size cut
    this selective a patch holds one or two hosts, so sweeping every patch that
    overlaps a field loads a great many that hold none; asking for the patches
    the hosts are actually in does not.
    """
    patches = {int(p) for p in patches}
    if not patches:
        return []
    return [
        r
        for r in _tract_refs(butler, DATASET_TYPE, tract, bands)
        if int(_data_id_dict(r.dataId).get("patch", -1)) in patches
    ]


def object_refs_for_tract(butler, tract: int) -> list:
    """The ``object`` table of one tract."""
    return _tract_refs(butler, "object", tract)


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
        if not (np.isfinite(ra[i]) and np.isfinite(dec[i])):
            xs[i] = ys[i] = np.nan
            continue
        try:
            xy = wcs.sky_to_pixel(SkyCoord(ra=ra[i] * u.deg, dec=dec[i] * u.deg, frame="icrs"))
            xs[i], ys[i] = float(xy.x), float(xy.y)
        except Exception as exc:
            # A projection can refuse a position far outside what it covers.
            # That is an answer -- "not here" -- not a failure.
            log.debug("sky_to_pixel(%s, %s) failed: %r", ra[i], dec[i], exc)
            xs[i] = ys[i] = np.nan
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
    if not (np.isfinite(x) and np.isfinite(y)):
        raise ValueError(f"cannot centre a stamp on ({x}, {y})")
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
    # A position can arrive non-finite two ways: a catalogue row with no
    # coordinates, or a projection of somewhere this patch does not cover.  Both
    # mean "not in this patch", and neither is worth ending a run over.
    if not (np.isfinite(x) and np.isfinite(y)):
        return False
    ix, iy = int(round(x)), int(round(y))
    half = size // 2
    return bool(
        bbox.contains(x=ix - half, y=iy - half)
        and bbox.contains(x=ix - half + size - 1, y=iy - half + size - 1)
    )


def cells_in_stamp(source, x: float, y: float, size: int) -> list[tuple[int, int]]:
    """Every ``(i, j)`` cell index the stamp covers.  Empty if the grid is absent.

    Takes a CellCoadd or a cell grid on its own, so a component read serves.
    """
    try:
        grid = getattr(source, "grid", source)
        half = size // 2
        corners = [
            grid.index_of(x=int(round(x)) + dx, y=int(round(y)) + dy)
            for dx in (-half, half - 1)
            for dy in (-half, half - 1)
        ]
        ii = [c.i for c in corners]
        jj = [c.j for c in corners]
        return [(i, j) for i in range(min(ii), max(ii) + 1) for j in range(min(jj), max(jj) + 1)]
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


def cell_visit_counts(source) -> dict[tuple[int, int], int]:
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
    # Accepts a CellCoadd or the provenance on its own, so it works equally off
    # a component read and off a patch that had to be loaded whole.
    prov = getattr(source, "provenance", source)
    contributions = getattr(prov, "contributions", None)
    if contributions is None or len(contributions) == 0:
        log.debug("no coadd provenance; falling back to the measured variance step")
        return {}

    names = list(getattr(contributions, "colnames", None) or getattr(contributions, "columns", []))
    cell_cols = next((c for c in CONTRIB_CELL_COLUMNS if set(c) <= set(names)), None)
    if cell_cols is None or "visit" not in names:
        raise RuntimeError(
            f"provenance.contributions has columns {names}; expected a visit "
            f"column and one of {[list(c) for c in CONTRIB_CELL_COLUMNS]}. Add "
            f"the real spelling to CONTRIB_CELL_COLUMNS in rubin/extract.py."
        )

    # One row per (visit, detector, cell), so a visit crossing a detector
    # boundary inside a cell appears twice; count distinct visits, not rows.
    keys = np.stack(
        [
            np.asarray(contributions[cell_cols[0]], dtype=np.int64),
            np.asarray(contributions[cell_cols[1]], dtype=np.int64),
            np.asarray(contributions["visit"], dtype=np.int64),
        ],
        axis=1,
    )
    cells, counts = np.unique(np.unique(keys, axis=0)[:, :2], axis=0, return_counts=True)
    return {(int(i), int(j)): int(n) for (i, j), n in zip(cells, counts)}


def missing_cells(bounds, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Which of ``cells`` were never built.

    ``CellGridBounds.bbox`` is the populated *rectangle*; ``missing`` is the
    handful of cells inside it that are absent anyway.  A corner test catches a
    stamp hanging off the edge of coverage but not a hole in the middle of one,
    and slicing across either raises rather than returning empty pixels.
    """
    absent = getattr(bounds, "missing", None)
    if not absent or not cells:
        return []
    keys = {(int(c.i), int(c.j)) for c in absent}
    return [c for c in cells if c in keys]


def stamp_depth(
    counts: dict[tuple[int, int], int], cells: list[tuple[int, int]]
) -> tuple[int, int]:
    """``(min, max)`` visit count over the cells a stamp covers; ``(-1, -1)`` if
    unknown.  Cells missing from the table contributed nothing and count as 0."""
    if not counts or not cells:
        return -1, -1
    n = [counts.get(c, 0) for c in cells]
    return min(n), max(n)


# -- the driver ------------------------------------------------------------


def next_batch(
    batch: int | None,
    n_hosts: int,
    gained: int,
    shortfall: int,
    headroom: float = 1.3,
    blind_growth: int = 4,
    floor: int = 16,
) -> int:
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


class _Done(Exception):
    """Raised to leave the sweep the moment the target is reached."""


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
    host_source: str = "tap",
    tap_service=None,
    tap_url: str | None = None,
    limit_tracts: int | None = None,
    limit_hosts: int | None = None,
    selection=None,   # ExtractionConfig or Selection: anything with
                      # .hosts and .patches

    patches_per_shard: int = 1024,
    max_patches: int | None = None,
    neighbour_radius_arcsec: float = 30.0,
    seed: int = 0,
    prefix: str = "patches",
) -> dict:
    """Extract patches near selected hosts and write shards plus a manifest.

    The loop is organised **by patch, not by host**: one query finds every coadd
    overlapping the field, and each is loaded once and sliced for every host that
    falls inside it.  Iterating hosts instead reloads the same patch repeatedly
    and dominates the runtime.

    Stamps are **centred on the host**.  A prior trained on centred galaxies
    would learn that galaxies are always centred, which is useless for a
    transient that can sit anywhere in the scene -- but the decentring belongs in
    the loader, not here: it crops ``nominal_crop`` out of ``native_size`` at a
    random offset, so the same stamp is seen at a different offset every epoch
    instead of at one offset fixed at extraction time.  The reach of that is
    ``(native_size - nominal_crop)/2``; widen ``native_size`` for more.

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
    selection = selection if selection is not None else Selection()
    gate_kwargs = selection.patches.gate_kwargs()
    # ``n_patches`` is a target to work towards; ``max_patches`` is a hard stop.
    # Both end the sweep at the same place, only ``n_patches`` tops up.
    target = n_patches if n_patches is not None else max_patches

    field = build_host_catalogue(
        butler,
        bands=bands,
        cuts=selection.hosts,
        ra=ra,
        dec=dec,
        radius_deg=radius_deg,
        limit_tracts=limit_tracts,
        cache=host_cache,
        source=host_source,
        tap_service=tap_service,
        tap_url=tap_url,
        top=limit_hosts,
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
    depth_checked = False
    depth_usable = True
    n_empty_patches = 0
    n_matched_patches = 0

    n_refs = 0
    neighbour_cols = neighbour_columns(bands)
    depth_logged = False
    #: Every cell of every patch swept, so the run can report its own depth
    #: rather than whichever patch happened to be visited first.
    all_visit_counts: list[int] = []

    def depth_logged_once() -> bool:
        """True after the first time the per-cell depth line has been printed."""
        nonlocal depth_logged
        was, depth_logged = depth_logged, True
        return was

    def _sweep(hosts, host_id, tgt_ra, tgt_dec):
        """One pass for one batch of hosts, tract by tract.

        Tract-major, not patch-major: the object table is per tract, so the
        neighbour index is built once per tract and thrown away, which is what
        makes a footprint-wide host list affordable.  Within a tract only the
        patches that actually hold a host are asked for.

        Nothing here falls back.  Every read either answers or ends the run,
        because the alternatives -- a whole-patch read standing in for a
        component, a mask plane quietly absent -- are indistinguishable from
        working until you look at the clock or the training set.
        """
        nonlocal writer, mask_mapping, n_accepted, n_refs
        nonlocal depth_checked, depth_usable, n_empty_patches, n_matched_patches

        by_patch: dict[tuple[int, int], list[int]] = {}
        for h, (t, pa) in enumerate(
            zip(np.asarray(hosts["tract"], dtype=int), np.asarray(hosts["patch"], dtype=int))
        ):
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
            neighbours = _neighbour_index(butler, tract, neighbour_cols, bands)

            for ref in refs:
                fields = _data_id_dict(ref.dataId)
                band_name = str(fields["band"])
                base = {
                    "dataId": json.dumps({k: str(v) for k, v in fields.items()}),
                    "band": band_name,
                    "tract": int(fields["tract"]),
                    "patch": int(fields["patch"]),
                }

                candidates = [
                    h
                    for h in by_patch.get((tract, base["patch"]), [])
                    if (int(host_id[h]), band_name) not in seen
                ]
                if not candidates:
                    continue

                # Components only: no pixels move until a host is known to land
                # inside the cells.  `bounds` and the cell grid come off the PSF,
                # which is where CellCoadd reads them from too.
                wcs = read_component(butler, ref, "wcs")
                bounds = read_component(butler, ref, "psf").bounds
                visit_counts = cell_visit_counts(read_component(butler, ref, "provenance"))
                if visit_counts:
                    all_visit_counts.extend(visit_counts.values())
                    log.log(
                        logging.INFO if not depth_logged_once() else logging.DEBUG,
                        "tract %d patch %d: %d cells, %d-%d visits per cell",
                        tract, base["patch"], len(visit_counts),
                        min(visit_counts.values()), max(visit_counts.values()),
                    )

                xs, ys = _sky_to_pixel(wcs, tgt_ra[candidates], tgt_dec[candidates])
                # The cell grid, not the image bbox: a patch at the edge of
                # coverage has cells that were never built, and slicing outside
                # them raises rather than returning empty pixels.
                inside = [
                    (h, x, y)
                    for h, x, y in zip(candidates, xs, ys)
                    if _fits_in_patch(bounds, x, y, native_size)
                ]
                if not inside:
                    n_empty_patches += 1
                    if n_matched_patches == 0 and n_empty_patches == PATCH_CHECK_AFTER:
                        n_projected = int(np.sum(np.isfinite(xs) & np.isfinite(ys)))
                        log.warning(
                            "%d patches so far have held none of the hosts the "
                            "catalogue assigned to them (this one: %d of %d "
                            "positions even projected). If that continues, the "
                            "Object table's `patch` column and the deep_coadd "
                            "dataId `patch` are not the same numbering.",
                            n_empty_patches,
                            n_projected,
                            len(xs),
                        )
                    continue
                n_matched_patches += 1
                log.debug("patch %s band %s: %d hosts", base["patch"], band_name, len(inside))

                for h, x, y in inside:
                    if target is not None and n_accepted >= target:
                        raise _Done
                    rec = {
                        **base,
                        "host_id": int(host_id[h]),
                    }

                    sep = _verify_centre(wcs, x, y, float(tgt_ra[h]), float(tgt_dec[h]))
                    rec["centre_sep_arcsec"] = sep
                    if not np.isfinite(sep) or sep > CENTRE_TOLERANCE_ARCSEC:
                        rec.update(status="rejected", reasons=f"centre_mismatch:{sep:.2f}")
                        records.append(rec)
                        continue

                    cells = cells_in_stamp(bounds, x, y, native_size)
                    if visit_counts and cells and not depth_checked:
                        # The grid's (i, j) and the provenance table's cell
                        # columns are two independent conventions and nothing
                        # guarantees they agree on which one is x.  Transposed,
                        # every lookup misses and every stamp reads as
                        # zero-visit.
                        depth_checked = True
                        depth_usable = any(c in visit_counts for c in cells)
                        if not depth_usable:
                            log.warning(
                                "none of the cells a stamp covers %s appear in "
                                "provenance.contributions (which has e.g. %s); "
                                "per-cell depth ignored for this run",
                                cells[:4],
                                sorted(visit_counts)[:4],
                            )
                    absent = missing_cells(bounds, cells)
                    if absent:
                        rec.update(status="rejected", reasons=f"missing_cells:{len(absent)}")
                        records.append(rec)
                        continue

                    # One read, of just these pixels.
                    stamp = butler.get(ref, parameters={"bbox": _stamp_box(x, y, native_size)})
                    image = np.asarray(_attr(stamp, "image").array, dtype=np.float32)
                    if image.shape != (native_size, native_size):
                        rec.update(status="rejected", reasons=f"clipped:{image.shape}")
                        records.append(rec)
                        continue

                    # Variance and mask are read, used, and dropped: they are
                    # what the gate is made of, and the prior never sees them.
                    variance = np.asarray(_attr(stamp, "variance").array, dtype=np.float32)
                    packed, mapping = pack_mask(_attr(stamp, "mask"))
                    if mask_mapping is None:
                        mask_mapping = mapping
                    elif mapping != mask_mapping:
                        raise RuntimeError(
                            f"mask schema changed mid-run: {mapping} after "
                            f"{mask_mapping}. The packing is per-shard, so a "
                            f"changing schema would make the gate's plane names "
                            f"mean different bits in different stamps."
                        )

                    n_lo, n_hi = stamp_depth(visit_counts if depth_usable else {}, cells)
                    if n_lo > 0:
                        depth_ratio = n_hi / n_lo
                    elif n_lo == 0:
                        depth_ratio = np.inf  # a cell with no visits at all
                    else:
                        depth_ratio = None  # provenance carried no such cell
                    reasons, diag = gate(
                        image,
                        variance,
                        packed,
                        mask_mapping,
                        cell_depth_ratio=depth_ratio,
                        n_visits=n_lo,
                        **gate_kwargs,
                    )
                    rec.update({f"diag_{k}": v for k, v in diag.items()})
                    if reasons:
                        rec.update(status="rejected", reasons=";".join(reasons))
                        records.append(rec)
                        continue

                    if writer is None:
                        writer = ShardWriter(
                            out_dir / "shards",
                            native_size=native_size,
                            prefix=prefix,
                            patches_per_shard=patches_per_shard,
                            dataset_type=DATASET_TYPE,
                            attrs={
                                "release": "DP2",
                                "skymap": SKYMAP,
                                "bands": json.dumps(list(bands)),
                                "flux_units": "nJy",
                                "correlated_noise": 1,  # coadds are warped
                                "pixel_origin": PIXEL_ORIGIN,
                                # DP2 coadds get a final background subtraction
                                # that over-subtracts around extended galaxies.
                                # These are as delivered.
                                "background_restored": 0,
                            },
                        )

                    nb = neighbours.near(
                        float(tgt_ra[h]),
                        float(tgt_dec[h]),
                        neighbour_radius_arcsec,
                        band_name,
                        host_id=int(host_id[h]),
                    )
                    others = [n for n in nb if not n["is_host"]]
                    gal = [n["sep_arcsec"] for n in others if n["extendedness"] > 0.5]
                    star = [n["sep_arcsec"] for n in others if n["extendedness"] <= 0.5]
                    y0, x0 = _origin(stamp)
                    acf.add(image)
                    writer.add(
                        image,
                        meta={
                            "band_idx": BANDS.index(band_name),
                            "x0": x0,
                            "y0": y0,
                            "ra": float(tgt_ra[h]),
                            "dec": float(tgt_dec[h]),
                            "pixel_scale": _pixel_scale(wcs, x, y),
                            "sky_noise": diag["sky_noise"],
                            "host_id": int(host_id[h]),
                            "tract": base["tract"],
                            "patch": base["patch"],
                            "n_cells_spanned": len(cells),
                            "n_visits_min": n_lo,
                            "n_visits_max": n_hi,
                            "cell_depth_ratio": diag.get("cell_depth_ratio", np.nan),
                            "variance_step": diag["variance_step"],
                            "frac_no_data": diag["frac_no_data"],
                            "frac_inexact_psf": diag["frac_INEXACT_PSF"],
                            "frac_rejected": diag["frac_REJECTED"],
                            "n_neighbours": len(others),
                            "neighbour_flux_max": float(
                                max([n["flux"] for n in others], default=np.nan)
                            ),
                            "nearest_galaxy_arcsec": float(min(gal, default=np.nan)),
                            "nearest_star_arcsec": float(min(star, default=np.nan)),
                        },
                    )
                    for n in others:
                        neighbour_rows.append({"patch_index": n_accepted, **n})
                    rec.update(status="accepted", patch_index=n_accepted)
                    records.append(rec)
                    seen.add((int(host_id[h]), band_name))
                    n_accepted += 1

    host_tables = []
    batch = n_hosts
    rounds = 0
    while True:
        rounds += 1
        hosts = select_hosts(field, cuts=selection.hosts, n_hosts=batch,
                             exclude_ids=tried, seed=seed + rounds)
        if not len(hosts):
            log.info("no untried hosts left in the catalogue after %d round(s)", rounds - 1)
            break
        tried.update(int(i) for i in hosts["objectId"])
        host_tables.append(hosts)
        log.info("round %d: %d hosts selected from %d candidates", rounds, len(hosts), len(field))

        host_ra = np.asarray(hosts["coord_ra"], dtype=float)
        host_dec = np.asarray(hosts["coord_dec"], dtype=float)
        host_id = np.asarray(hosts["objectId"], dtype=np.int64)

        before = n_accepted
        try:
            _sweep(hosts, host_id, host_ra, host_dec)
        except _Done:
            log.info("reached the target of %d cutouts", target)
            break
        gained = n_accepted - before
        log.info(
            "round %d: %d cutouts from %d hosts (%.2f per host), %d total",
            rounds,
            gained,
            len(hosts),
            gained / max(len(hosts), 1),
            n_accepted,
        )

        if target is None or n_accepted >= target:
            break
        if rounds >= max_rounds:
            log.warning(
                "stopping after %d rounds with %d of %d cutouts; raise max_rounds, "
                "widen --radius-deg, or loosen the gate -- read rejection_counts "
                "in the summary first",
                rounds,
                n_accepted,
                target,
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
        visits_per_cell=_distribution(all_visit_counts),
        n_patches_with_hosts=n_matched_patches,
        n_patches_without_hosts=n_empty_patches,
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


def _distribution(values) -> dict:
    """Percentiles of a list, for the summary.  Empty in, empty out."""
    a = np.asarray(list(values), dtype=float)
    if not a.size:
        return {}
    pcts = np.percentile(a, [0, 25, 50, 75, 100])
    return {"n": int(a.size),
            **{k: float(v) for k, v in zip(("min", "p25", "p50", "p75", "max"), pcts)}}


def _within_radius(table, ra: float, dec: float, radius_deg: float):
    """Catalogue rows inside a disc on the sky.  Flat-sky; exact enough under a
    degree, where the error is a part in 10^5."""
    r = np.asarray(table["coord_ra"], dtype=float)
    d = np.asarray(table["coord_dec"], dtype=float)
    cosd = np.maximum(np.cos(np.deg2rad(dec)), 1e-6)
    return table[np.hypot((r - ra) * cosd, d - dec) <= radius_deg]


def _stack_tables(tables):
    """Concatenate the per-round host tables."""
    if len(tables) == 1:
        return tables[0]
    from astropy.table import vstack

    return vstack(tables, join_type="exact")


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


def _neighbour_index(butler, tract: int, columns: Sequence[str], bands: Sequence[str]):
    """Neighbour index for one tract.

    Neighbours must come from the *whole* tract, not from the host pool: a host
    is interesting precisely because of what sits near it, and almost nothing
    near it passed the host cuts.  One table per tract, held only while that
    tract is being swept.
    """
    refs = object_refs_for_tract(butler, tract)
    if not refs:
        raise RuntimeError(
            f"no object table for tract {tract}, yet hosts were selected from "
            f"it: the host catalogue and the repo disagree about what exists"
        )
    try:
        table = butler.get(refs[0], parameters={"columns": list(columns)})
    except Exception as exc:
        raise RuntimeError(
            f"could not read the object table for tract {tract} with columns "
            f"{list(columns)}: {exc}. Note the TAP dp2.Object view and the "
            f"butler object parquet are not the same table -- TAP serves "
            f"derived columns the pipeline never wrote, so a column that works "
            f"in the ADQL can still be absent here."
        ) from None
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

    def near(
        self, ra: float, dec: float, radius_arcsec: float, band: str, host_id: int | None = None
    ):
        cosd = max(np.cos(np.deg2rad(dec)), 1e-6)
        r_deg = radius_arcsec / 3600.0
        box = (np.abs(self.dec - dec) < r_deg) & (np.abs(self.ra - ra) * cosd < r_deg)
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
                # neighbour list.  Flagged rather than dropped, so a caller can
                # see that the centre really is the object it asked for.
                "is_host": bool(host_id is not None and int(self.ids[i]) == host_id),
            }
            for i, s in zip(idx, sep)
        ]


def _write_table(stem, table) -> None:
    """Persist a table as parquet.

    Size, magnitude and blendedness are known only at selection time and are not
    carried in the shard metadata, so without this the host population cannot be
    inspected after the fact.
    """
    df = table.to_pandas() if hasattr(table, "to_pandas") else table
    df.to_parquet(Path(stem).with_suffix(".parquet"), index=False)


def _read_table(path):
    """Read back a table written by ``_write_table``."""
    from astropy.table import Table

    path = Path(path)
    return Table.read(path if path.suffix else path.with_suffix(".parquet"))


def _write_manifest(out_dir: Path, records: Iterable[dict], neighbour_rows: Iterable[dict]) -> dict:
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
