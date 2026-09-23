"""Build a training set of host-galaxy cutouts from DP2 ``deep_coadd`` images.

Plain Python against the Butler as a data-access layer -- no ``pipetask``, no
BPS.  Import-safe without the LSST stack: everything stack-specific is imported
lazily by ``_lsst()``, so the rest of the package (and the test suite) works on a
laptop.

**The shape of a run.**  TAP answers one ADQL query with the host list; then each
host is visited in turn and a stamp is cut from the coadd centred on it, one per
band.  There is no tract sweep and no patch bookkeeping: a host knows its own
tract and patch, ``query_datasets`` turns those into refs, and
``butler.get(ref, parameters={"bbox": ...})`` moves only the stamp's pixels.
Because each host is visited once, by position, it cannot be extracted twice,
and nothing downstream has to de-duplicate stamps.

**One definition of each cut.**  The host cuts live in ``extraction.yaml``, are
compiled into the ADQL, and are applied by the service.  They are not re-applied
here; what happens locally is only what a query cannot do -- the Sersic failure
flags and the saturated/interpolated core flags, whose boolean comparison is
backend-specific, and the cross-tract dedupe, which needs the whole pool at once.

DP2 is not DP1 with more data.  It replaces ``lsst.afw.image`` with
``lsst.images``: a ``deep_coadd`` is a ``CellCoadd``, getters became attributes,
mask planes were renamed and are read through a schema, and there are two
different pixel-origin conventions.  DP1 code does not error on DP2, it
misbehaves quietly.  The things that bite:

* **Two pixel origins.**  ``sky_projection`` works in *tract* coordinates,
  ``astropy_wcs`` in *patch-local*.  Mixing them misplaces a position by up to a
  full patch (~4000 px) -- far enough to land on the wrong galaxy, close enough
  to look plausible.  Tract coordinates are used throughout, which is what
  ``Box.factory`` and ``bbox.contains`` expect, and every stamp centre is
  projected back to the sky and compared against the position asked for
  (``_verify_centre``).  That guard costs nothing and turns a whole class of
  silent geometry error into a loud one.
* **Coadds are cell-based** -- a 22x22 grid of 150-pixel cells, each built from a
  different set of input visits.  Depth and PSF are therefore piecewise constant
  with steps at cell edges, and a stamp larger than 150 native pixels *will*
  straddle them.  ``n_cells_spanned`` is recorded per stamp so the effect stays
  measurable, and ``provenance.contributions`` gives the step exactly.
* **``grid`` and ``bounds`` are not butler components.**  They are Python
  properties on ``CellCoadd`` reading through to ``psf.bounds``.  Asking the
  butler for them appears to work -- it silently falls back to reading the whole
  patch -- and cost two orders of magnitude in I/O before anyone noticed.
* **Mask planes are dynamic.**  Bit numbers are not stable across releases, so
  the mask is repacked into a ``uint32`` using a mapping derived from the
  coadd's own ``mask.schema``, and that mapping is stored with every shard.
  Nothing downstream hard-codes a bit.
* **Variance holds ``inf``** where there were no contributing exposures,
  including the cores of saturated stars.  That is measured as a fraction, not
  treated as corruption -- see ``quality.gate``.
* ``coadd[box]`` returns a **view**; it must be ``.copy()``-ed before the parent
  is released.  ``Box.factory`` takes ``[y, x]`` -- numpy order, not ``(x, y)``.
* **Pixels are already nanojanskys** and variance is nJy^2.  No calibration step
  belongs here.
* DP2 coadds are **over-subtracted around extended galaxies**, and the
  background is recoverable via ``backgrounds`` / ``apply_background``.  This
  module takes the image **as delivered** and records that choice in every shard
  as ``background_restored=0``, so a set made the other way is distinguishable.

Nothing here falls back.  A component that does not answer, a mask schema that
changes mid-run, a WCS that does not round-trip: each ends the run, because the
alternative -- a whole-patch read standing in for a component, a mask plane
quietly absent -- is indistinguishable from working until you look at the clock
or at the training set.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
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

#: Attribute names on a ``CellCoadd``, and the names of the butler components
#: that serve them.  Note what is *absent*: ``grid`` and ``bounds`` are Python
#: properties reading through to ``psf.bounds``, not components -- and ``psf``
#: itself is no longer read, because deserialising a 22x22 block of per-cell
#: PSFs to find a bounding box cost 52 s of a 400 s run and the bbox read
#: answers the same question by succeeding or failing.
DP2_ATTRS = {
    "image": "image",
    "variance": "variance",
    "mask": "mask",
    "wcs": "sky_projection",
    "schema": "schema",
    "origin": "yx0",
    "provenance": "provenance",
}

#: DP2 offers three WCS representations and they do **not** share a pixel
#: origin: ``sky_projection`` is in *tract* coordinates, ``astropy_wcs`` in
#: *patch-local*.  This module uses tract coordinates throughout, which is both
#: the precise representation and the frame ``Box.factory`` and ``bbox.contains``
#: expect.
PIXEL_ORIGIN = "tract"

#: Maximum separation between where a stamp landed and where it was asked for,
#: in arcsec.  A pixel-origin mix-up is off by far more than this.
CENTRE_TOLERANCE_ARCSEC = 1.0

#: How many hosts may fail to land inside the patch the catalogue assigned them
#: before the run concludes the two ``patch`` numberings disagree.
PATCH_CHECK_AFTER = 20

#: Band-independent columns of the ``dp2.Object`` view.  The table has 1248
#: columns, so a subset is not optional.
#:
#: The ``sersic_*`` columns are the multiband fit and carry no band prefix: one
#: morphology fit to all six bands at once, which is why it is the size
#: measurement here.  ``sersic_reff_major`` is in **arcsec** (unlike
#: ``sersic_reff_x``, which is in pixels) and is the radius *before* convolution
#: with the PSF, so it is the galaxy's intrinsic size rather than its observed
#: extent.  ``sersic_reff_minor`` is needed only for the surface brightness,
#: ``mu_e = m + 2.5 log10(2 pi a b)``.
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
]

#: Bands for which the DP2 Object table carries photometry.  Verified against
#: the schema YAML (``sdm_schemas``, ``drp_base.yaml``), which is the source to
#: use: the rendered HTML schema page is large enough that reading it in
#: excerpts gives a confidently wrong answer about which bands exist.
PHOTOMETRY_BANDS = ("u", "g", "r", "i", "z", "y")

#: Added per band in ``PHOTOMETRY_BANDS``.  Size is the band-independent
#: ``sersic_reff_major``; there are deliberately no second moments here, and no
#: non-parametric size cross-check.  One measurement of size, one cut on it.
OBJECT_BAND_COLUMNS = [
    "{b}_cModelFlux",
    "{b}_blendedness",
    # The real bright limit, and the real reason to drop a core.
    "{b}_pixelFlags_saturatedCenter",
    "{b}_pixelFlags_interpolatedCenter",
]

#: ECDFS, still in tract 5063 as on DP1, and the field the DP2 tutorials use
#: throughout.  ELAISS1 (10.26, -44.49) and EDFS (59.10, -48.73) also appear.
ECDFS = (53.13, -28.10)


def _lsst() -> SimpleNamespace:
    """Import the LSST stack once, lazily."""
    global _STACK
    if _STACK is None:
        from lsst.daf.butler import Butler
        from lsst.images import Box

        _STACK = SimpleNamespace(Butler=Butler, Box=Box)
    return _STACK


def read_component(butler, ref, role: str):
    """A ``deep_coadd`` component.  Raises if the repo will not serve it.

    Components move no pixels, so everything needed to place and characterise a
    stamp -- WCS, cell grid, provenance -- comes this way and the pixels then
    come from a bbox read of just the stamp.  A component that does not answer
    is a bug in this mapping, not a condition to work around.
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


def pack_mask(mask, max_planes: int = 32) -> tuple[np.ndarray, dict[str, int]]:
    """Flatten a DP2 plane-based mask into a ``uint32`` plus its own mapping.

    A DP2 mask pixel is a short byte array, not a single integer, so
    ``mask.array & bit`` does not work at all; ``mask.get(name)`` returns a plain
    boolean plane and is the only sane way in.  Bit numbers are assigned
    dynamically and are not stable across releases, so the bits used here are
    local and travel with the shard.  Nothing downstream hard-codes one.
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


# -- the host list ---------------------------------------------------------

#: The TAP-side table.  The host cuts are a selection, and a selection is what a
#: query service is for: the whole footprint is ~10^9 rows and the survivors are
#: ~10^4, so the difference between filtering there and filtering here is the
#: difference between moving the survivors and moving the catalogue.
TAP_TABLE = "dp2.Object"


def host_columns(bands: Sequence[str] = BANDS) -> list[str]:
    """The column subset to read.  Asking for one that does not exist fails the
    whole query, so this is the list and there is no ``extra``."""
    unknown = [b for b in bands if b not in PHOTOMETRY_BANDS]
    if unknown:
        raise ValueError(f"band(s) {unknown} are not DP2 bands; choose from {PHOTOMETRY_BANDS}")
    columns = list(OBJECT_COLUMNS)
    for b in bands:
        columns += [c.format(b=b) for c in OBJECT_BAND_COLUMNS]
    return columns


def host_adql(
    bands=BANDS,
    cuts: HostCuts | None = None,
    ra: float | None = None,
    dec: float | None = None,
    radius_deg: float | None = None,
    top: int | None = None,
) -> str:
    """The ADQL for the host selection, built from ``HostCuts``.

    This is where the cuts are applied.  They are not applied again after the
    rows arrive, so what this query says is what the catalogue is.

    Arithmetic here is addition and multiplication only.  ``LOG10`` and
    ``POWER`` are not guaranteed across ADQL dialects, and a clause the service
    silently declines to apply is worse than one it refuses outright -- so the
    magnitude limits are written as fluxes and the surface-brightness limit as a
    flux against an area.

    The boolean flags are the exception: how a boolean compares in ADQL is
    backend-specific and a wrong guess quietly returns nothing, so they are
    fetched and applied in ``select_hosts``.  Nor is there an ``ORDER BY`` --
    sorting burdens a shared service and the draw happens here anyway.
    """
    cuts = cuts or HostCuts()
    band = cuts.band
    faint, bright = cuts.flux_range
    columns = ", ".join(host_columns(bands))
    # NaN and NULL both fail a > comparison, which is what is wanted: an object
    # with no fit is not a large object.
    where = [
        f"{band}_cModelFlux > {faint:.1f}",
        f"{band}_cModelFlux <= {bright:.1f}",
        f"sersic_reff_major >= {float(cuts.min_reff_arcsec)}",
        f"sersic_reff_major <= {float(cuts.max_reff_arcsec)}",
    ]
    if cuts.min_extendedness is not None:
        where.append(f"refExtendedness > {float(cuts.min_extendedness)}")
    floor = cuts.surface_brightness_floor()
    if floor is not None:
        where.append(f"{band}_cModelFlux >= {floor:.1f} "
                     f"* sersic_reff_major * sersic_reff_minor")
    if cuts.max_sersic_index is not None:
        where.append(f"sersic_index <= {float(cuts.max_sersic_index)}")
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

#: The scope a Gafaelfawr token needs to use TAP.
TAP_SCOPE = "read:tap"


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
        f"{TOKEN_ENV_VARS} and {TOKEN_PATHS}."
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
            "pyvo is needed to reach the TAP service: pip install pyvo"
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
                "%d row(s) have no %s; setting them to -1, which will not match "
                "any patch", n, name,
            )
            table[name] = np.asarray(column.filled(-1), dtype=column.dtype)
        else:
            continue
        log.debug("unmasked %d null(s) in %s", n, name)
    return table


def build_host_catalogue(
    bands: Sequence[str] = BANDS,
    cuts: HostCuts | None = None,
    ra: float | None = None,
    dec: float | None = None,
    radius_deg: float | None = None,
    cache: str | Path | None = None,
    tap_service=None,
    tap_url: str | None = None,
    top: int | None = None,
):
    """Host candidates for the whole region, cut by the service.

    One ADQL query.  The footprint is ~10^9 rows and the survivors are ~10^4, so
    the selection belongs where the catalogue already is; what comes back over
    the wire is the host list rather than the footprint.

    That needs network and a token with ``read:tap``, which a batch node may not
    have -- so pass ``cache`` and the query happens once, after which extraction
    runs from the parquet with no network at all.

    The query that produced a cache is written beside it and checked against the
    one the current config asks for, because a cache has the cuts baked in and
    reusing one after editing ``extraction.yaml`` is a cut that looks applied and
    is not.  Delete both files to re-query.
    """
    cuts = cuts or HostCuts()
    query = host_adql(bands=bands, cuts=cuts, ra=ra, dec=dec,
                      radius_deg=radius_deg, top=top)
    sidecar = None if cache is None else Path(cache).with_suffix(".sql")

    if cache is not None and Path(cache).exists():
        cache = Path(cache)
        if not sidecar.exists():
            raise RuntimeError(
                f"{cache} has no {sidecar.name} beside it, so there is no way "
                f"to tell which cuts it was built with. It predates this check; "
                f"delete it and re-query."
            )
        was = sidecar.read_text()
        if was != query:
            raise RuntimeError(
                f"{cache} was built by a different query, so the cuts in "
                f"extraction.yaml are not the cuts it holds. Delete {cache} and "
                f"{sidecar} to re-query.\n\ncached:\n{was}\n\nasked for:\n{query}"
            )
        table = _read_table(cache)
        log.info("host catalogue: %d candidates from cache %s", len(table), cache)
        return table

    log.info("querying %s:\n%s", TAP_TABLE, query)
    table = unmask(run_adql(tap_service or tap_client(url=tap_url), query))
    log.info("TAP returned %d rows", len(table))
    pool = select_hosts(table, cuts=cuts)
    log.info("host catalogue: %d candidates", len(pool))
    if cache is not None:
        cache = Path(cache)
        cache.parent.mkdir(parents=True, exist_ok=True)
        _write_table(cache.with_suffix(""), pool)
        sidecar.write_text(query)
        log.info("cached the host catalogue at %s (query in %s)", cache, sidecar)
    return pool


def select_hosts(table, cuts: HostCuts | None = None):
    """What the query could not do: the boolean flags, and the dedupe.

    The numeric cuts ran on the service and are **not** repeated here.  Three
    things are left.

    A row needs a usable sky position, or it cannot be turned into a stamp at
    all.  A Sersic fit that failed or had nothing to fit leaves whatever was in
    the column, which would otherwise compare its way through the size cut the
    service applied.  And a saturated or interpolated core is a boolean, whose
    comparison in ADQL is backend-specific -- a wrong guess there returns
    nothing at all, silently.
    """
    cuts = cuts or HostCuts()
    band = cuts.band
    if band not in PHOTOMETRY_BANDS:
        raise ValueError(
            f"band {band!r} has no DP2 Object photometry; choose from "
            f"{PHOTOMETRY_BANDS}"
        )
    ra = np.asarray(table["coord_ra"], dtype=float)
    dec = np.asarray(table["coord_dec"], dtype=float)
    keep = np.isfinite(ra) & np.isfinite(dec)
    if not keep.all():
        log.info("dropping %d host candidate(s) with no sky position",
                 int((~keep).sum()))
    for flag in ("sersic_unknown_flag", "sersic_no_data_flag"):
        keep &= ~np.asarray(table[flag], dtype=bool)
    for flag, wanted in (("saturatedCenter", cuts.reject_saturated_centre),
                         ("interpolatedCenter", cuts.reject_interpolated_centre)):
        if wanted:
            keep &= ~np.asarray(table[f"{band}_pixelFlags_{flag}"], dtype=bool)
    log.info("flags and positions: %d of %d survive", int(keep.sum()), len(table))
    return dedupe_hosts(table[keep], cuts.dedupe_radius_arcsec)


def dedupe_hosts(table, radius_arcsec: float = 0.5):
    """Drop objects that are the same source seen twice.

    Not about patches -- a host is visited once, by position, so it cannot yield
    two stamps.  This is about the catalogue: DP2 has no ``detect_isPrimary``
    (no ``detect_*`` columns at all) and tracts overlap at their edges, so a
    galaxy in an overlap is listed twice under two different ``objectId``s.
    Deduplicating on id alone would not catch that, so near coincidences on the
    sky are collapsed too.  Without this such a galaxy is silently weighted up.
    """
    ra = np.asarray(table["coord_ra"], dtype=float)
    dec = np.asarray(table["coord_dec"], dtype=float)
    keep = np.zeros(len(table), dtype=bool)
    # Sort by declination so the sky search only has to look at a local window.
    order = np.argsort(dec)
    kept_ra: list[float] = []
    kept_dec: list[float] = []
    r_deg = radius_arcsec / 3600.0
    for i in order:
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
        kept_ra.append(ra[i])
        kept_dec.append(dec[i])
    if not keep.all():
        log.info("dedupe: dropped %d duplicate row(s)", int((~keep).sum()))
    return table[keep]


def host_mu_e(table, band: str) -> np.ndarray:
    """Mean surface brightness inside the half-light ellipse, mag/arcsec^2.

    ``mu_e = m + 2.5 log10(2 pi a b)``, with ``a``/``b`` the Sersic half-light
    axes in arcsec.  The cut on this runs in the ADQL as a flux against an area;
    this is the same quantity in the units it is thought about, for the
    diagnostic figures.
    """
    flux = np.asarray(table[f"{band}_cModelFlux"], dtype=float)
    a = np.asarray(table["sersic_reff_major"], dtype=float)
    b = np.asarray(table["sersic_reff_minor"], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mag = AB_ZEROPOINT - 2.5 * np.log10(np.where(flux > 0, flux, np.nan))
        return mag + 2.5 * np.log10(2.0 * np.pi * a * b)


# -- geometry --------------------------------------------------------------


def coadd_refs(butler, tract: int, patch: int, bands: Sequence[str]) -> list:
    """The ``deep_coadd`` refs for one patch: at most one per band.

    Constrained by data id, not by a ``where`` string.  The expression language
    bit once and silently: in ``where="tract = :tract"`` the bind key shadows the
    dimension of the same name, so it resolved as ``tract = tract`` -- true for
    every row.  The query returned the whole repo, truncated at the default
    20000, and hosts were matched against same-numbered patches in other tracts,
    which projected a couple of hundred thousand pixels away.  ``data_id`` takes
    key-value equality constraints and cannot be read as anything else.
    """
    refs = list(butler.query_datasets(
        DATASET_TYPE,
        data_id={"skymap": SKYMAP, "tract": int(tract), "patch": int(patch)},
        limit=None,
        explain=False,
    ))
    want = set(bands)
    kept = []
    for ref in refs:
        fields = _data_id_dict(ref.dataId)
        if (int(fields.get("tract", -1)) != int(tract)
                or int(fields.get("patch", -1)) != int(patch)):
            raise RuntimeError(
                f"asked for {DATASET_TYPE} in tract {tract} patch {patch} and "
                f"got {fields}: the query is not constraining the data id, so "
                f"every stamp would be cut from the wrong piece of sky."
            )
        if str(fields.get("band", "?")) in want:
            kept.append(ref)
    return kept


def _sky_to_pixel(wcs, ra: float, dec: float) -> tuple[float, float]:
    """(ra, dec) in degrees -> fractional **tract** pixel (x, y).

    ``sky_projection.sky_to_pixel`` takes a SkyCoord and returns an object with
    ``.x``/``.y``.  One position at a time, which is all a host-major walk needs
    and costs nothing beside the pixel read that follows.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    xy = wcs.sky_to_pixel(SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs"))
    return float(xy.x), float(xy.y)


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


def cells_in_stamp(source, x: float, y: float, size: int) -> list[tuple[int, int]]:
    """Every ``(i, j)`` cell index the stamp covers, in the patch's numbering.

    Takes a ``CellCoadd`` or a cell grid on its own.  In the walk it is handed
    the *stamp* -- a CellCoadd of just this region, already read -- rather than
    the patch's PSF, which cost 52 s of a 400 s run to deserialise for its
    bounding box.  Raises if neither answers: an empty list here would silently
    disable the depth check.
    """
    grid = getattr(source, "grid", source)
    half = size // 2
    try:
        corners = [
            grid.index_of(x=int(round(x)) + dx, y=int(round(y)) + dy)
            for dx in (-half, half - 1)
            for dy in (-half, half - 1)
        ]
    except Exception as exc:
        raise RuntimeError(
            f"{type(grid).__name__}.index_of did not answer for a stamp at "
            f"({x:.1f}, {y:.1f}): {exc!r}. The cell grid is what the depth and "
            f"missing-cell checks are made of."
        ) from None
    ii = [c.i for c in corners]
    jj = [c.j for c in corners]
    return [(i, j) for i in range(min(ii), max(ii) + 1) for j in range(min(jj), max(jj) + 1)]


#: Candidate spellings of the cell index in ``provenance.contributions``.  The
#: API documents the table as ``{visit, detector, cell}`` without pinning the
#: column names, and ``CellIJ`` does not survive into an astropy column as one
#: object, so the pair is resolved by trial and the real names are reported if
#: none of these match.
CONTRIB_CELL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("cell_i", "cell_j"),
    ("cell_x", "cell_y"),
    ("i", "j"),
    ("x", "y"),
)


def cell_visit_counts(provenance) -> dict[tuple[int, int], int]:
    """Distinct visits contributing to each ``(i, j)`` cell.

    This is the quantity that makes a depth step, and it is exact: DP2 exposures
    share an integration time, so a cell built from 12 visits is simply shallower
    than its neighbour built from 30, and the noise steps across the edge between
    them.  No mask plane says so, but ``CellCoadd.provenance.contributions`` is a
    table of ``{visit, detector, cell}`` -- which observation went into which
    cell -- so counting it gives the step before a single pixel is examined.

    ``deep_coadd_input_summary`` is *not* an alternative: Rubin documents it as
    patch-level and says outright that it does not record which visit-detector
    images contributed to each cell.
    """
    contributions = getattr(provenance, "contributions", None)
    if contributions is None or len(contributions) == 0:
        raise RuntimeError(
            "the coadd carries no provenance.contributions, so per-cell depth "
            "cannot be measured. Every coadd was built from visits, so an empty "
            "contributions table means the component is not what this expects."
        )

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


def stamp_depth(
    counts: dict[tuple[int, int], int], cells: list[tuple[int, int]]
) -> tuple[int, int]:
    """``(min, max)`` visit count over the cells a stamp covers.  A cell absent
    from the table contributed nothing and counts as zero."""
    n = [counts.get(c, 0) for c in cells]
    return min(n), max(n)


# -- the driver ------------------------------------------------------------


def extract_patches(
    butler,
    out_dir: str | Path,
    *,
    ra: float = ECDFS[0],
    dec: float = ECDFS[1],
    radius_deg: float | None = None,
    bands: Sequence[str] = BANDS,
    native_size: int = 512,
    patches_per_shard: int = 1024,
    host_cache: str | Path | None = None,
    tap_service=None,
    tap_url: str | None = None,
    limit_hosts: int | None = None,
    n_stamps: int | None = None,
    seed: int = 0,
    prefix: str = "patches",
    selection=None,   # ExtractionConfig or Selection: anything with
                      # .hosts and .patches
) -> dict:
    """Cut a stamp around each host and write shards plus a manifest.

    **Host-major.**  TAP answers one query with the host list; the list is
    shuffled and walked, and each host in turn is turned into up to one stamp per
    band, centred on it, read as a bbox out of its own patch.  Because a host is
    visited once, by position, it cannot be extracted twice -- the tract sweep,
    the patch grouping and the duplicate bookkeeping that used to guard against
    that are all gone with it.

    **Stamps are centred on the host.**  A prior trained on centred galaxies
    would learn that galaxies are always centred, which is useless for a
    transient that can sit anywhere in the scene -- but the decentring belongs in
    the loader, not here: it crops ``nominal_crop`` out of ``native_size`` at a
    random offset, so the same stamp is seen at a different offset every epoch
    instead of at one offset fixed at extraction time.  The reach of that is
    ``(native_size - nominal_crop)/2``; widen ``native_size`` for more.

    ``n_stamps`` is a target number of *accepted cutouts*, and the walk
    continues until it has them or the catalogue runs out.  The check happens
    between hosts, so a run can overshoot by up to one host's worth of bands
    rather than leaving the last host with an arbitrary subset of them.

    Every attempt is recorded in the manifest, rejections included, with the
    reason and the diagnostics.  Those statistics *are* the selection function,
    and the bias they reveal -- against dense bright centres -- is the regime
    this project exists to model.
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection = selection if selection is not None else Selection()
    gate_kwargs = selection.patches.gate_kwargs()
    clock = Stopwatch()

    with clock("host catalogue"):
        catalogue = build_host_catalogue(
            bands=bands,
            cuts=selection.hosts,
            ra=ra,
            dec=dec,
            radius_deg=radius_deg,
            cache=host_cache,
            tap_service=tap_service,
            tap_url=tap_url,
            top=limit_hosts,
        )
    if not len(catalogue):
        raise RuntimeError(
            "the host catalogue is empty. Loosen the host cuts in the config, "
            "or widen sky.radius_deg."
        )

    # Shuffled, because the walk stops the moment the target is reached; in
    # catalogue order it would fill the set from one corner of the footprint.
    #
    # Then *ordered* -- not grouped -- so that hosts sharing a patch are walked
    # consecutively.  Patches keep the shuffled order of the first host drawn
    # into each, so where the set is drawn from is unchanged; all this does is
    # put the repeats next to each other, which is what lets the component cache
    # below be one entry deep.  The loop is still one host at a time.
    order = rng.permutation(len(catalogue))
    seen_patches: dict[tuple[int, int], int] = {}
    for k in order:
        seen_patches.setdefault(
            (int(catalogue["tract"][k]), int(catalogue["patch"][k])),
            len(seen_patches))
    order = sorted(order, key=lambda k: seen_patches[
        (int(catalogue["tract"][k]), int(catalogue["patch"][k]))])
    order = np.asarray(order)
    host_id = np.asarray(catalogue["objectId"], dtype=np.int64)[order]
    host_ra = np.asarray(catalogue["coord_ra"], dtype=float)[order]
    host_dec = np.asarray(catalogue["coord_dec"], dtype=float)[order]
    host_tract = np.asarray(catalogue["tract"], dtype=int)[order]
    host_patch = np.asarray(catalogue["patch"], dtype=int)[order]

    records: list[dict] = []
    writer: ShardWriter | None = None
    mask_mapping: dict[str, int] | None = None
    acf = AutocorrelationAccumulator(native_size)
    all_visit_counts: list[int] = []
    #: Patches whose depth has already been folded into ``all_visit_counts``.
    #: Two hosts in one patch would otherwise count its cells twice and skew the
    #: run's depth distribution towards whichever patches happened to be busy.
    depth_counted: set[tuple[int, int, str]] = set()
    n_accepted = 0
    n_tried = 0
    n_no_coadd = 0
    n_off_the_grid = 0
    n_pixel_reads = 0
    n_off_the_grid_stamps = 0
    #: Exception type -> the first message seen with it.  Every distinct kind of
    #: stamp-read failure is reported once; see the read below for why.
    read_failures: dict[str, str] = {}
    depth_checked = False

    # One patch deep, which the patch-ordered walk makes sufficient: every host
    # in a patch is visited before the next patch is reached.  Everything in
    # here is a per-(tract, patch) or per-(tract, patch, band) quantity that the
    # old patch-major sweep amortised for free and this one would otherwise pay
    # per host: the ref query, and the three component reads.
    #
    # Keyed on the *patch*, holding every band of it, because the inner loop
    # runs over the bands of one host: a key that included the band would be
    # cleared on every band change and would never hit at all.
    cache_patch: tuple[int, int] | None = None
    cache: dict = {}

    def patch_cache() -> dict:
        """The cache for the patch being walked, emptied when it changes."""
        nonlocal cache_patch, cache
        if (tract, patch) != cache_patch:
            cache_patch, cache = (tract, patch), {}
        return cache

    def component(ref, band, role, clock_name):
        """A component of ``ref``, read once per patch rather than per host."""
        entry = patch_cache()
        if (band, role) not in entry:
            with clock(clock_name):
                entry[(band, role)] = read_component(butler, ref, role)
        return entry[(band, role)]

    for k in range(len(order)):
        if n_stamps is not None and n_accepted >= n_stamps:
            break
        n_tried += 1
        # A patch numbering mismatch looks exactly like a field with no
        # coverage: every host lands outside the cells of the patch it was filed
        # under, every stamp is rejected, and an empty set is written without
        # complaint.  Say so instead of walking the whole catalogue to find out.
        if n_tried > PATCH_CHECK_AFTER and not n_pixel_reads:
            raise RuntimeError(
                f"none of the first {PATCH_CHECK_AFTER} hosts produced a stamp "
                f"({n_no_coadd} had no deep_coadd for their patch, "
                f"{n_off_the_grid} landed outside its built cells). The most "
                f"likely cause is that the Object table's `patch` column and "
                f"the deep_coadd dataId `patch` are not the same numbering; the "
                f"next is a native_size ({native_size}) too large to fit inside "
                f"a patch anywhere."
            )
        hid = int(host_id[k])
        tract, patch = int(host_tract[k]), int(host_patch[k])
        target_ra, target_dec = float(host_ra[k]), float(host_dec[k])

        if n_tried % 25 == 0:
            log.info("host %d/%d (id %d): %d cutouts so far%s",
                     n_tried, len(order), hid, n_accepted,
                     f" of {n_stamps}" if n_stamps else "")

        # Cached like the components, and for the same reason: consecutive
        # hosts in one patch ask the butler an identical question.  Leaving this
        # out of the cache was worth 22% of a real run's wall time.
        entry = patch_cache()
        if "refs" not in entry:
            with clock("coadd ref queries"):
                entry["refs"] = coadd_refs(butler, tract, patch, bands)
        refs = entry["refs"]
        if not refs:
            n_no_coadd += 1
            records.append({"host_id": hid, "tract": tract, "patch": patch,
                            "band": "", "status": "rejected",
                            "reasons": "no_coadd"})
            continue

        fitted = False
        for ref in refs:
            fields = _data_id_dict(ref.dataId)
            band_name = str(fields["band"])
            rec = {
                "dataId": json.dumps({key: str(v) for key, v in fields.items()}),
                "host_id": hid,
                "band": band_name,
                "tract": tract,
                "patch": patch,
            }

            wcs = component(ref, band_name, "wcs", "read: wcs")
            x, y = _sky_to_pixel(wcs, target_ra, target_dec)
            sep = _verify_centre(wcs, x, y, target_ra, target_dec)
            rec["centre_sep_arcsec"] = sep
            if not np.isfinite(sep) or sep > CENTRE_TOLERANCE_ARCSEC:
                raise RuntimeError(
                    f"host {hid} was placed at tract pixel ({x:.1f}, {y:.1f}), "
                    f"which projects back {sep:.1f}\" from where it was asked "
                    f"for. DP2 has two pixel-origin conventions -- "
                    f"sky_projection is tract, astropy_wcs is patch-local -- and "
                    f"mixing them displaces a position by up to a whole patch."
                )

            # One read, of just these pixels -- and the read is also the test of
            # whether the stamp fits.  There used to be a pre-check here, from
            # the coadd's cell grid: whether both corners lay inside the cells
            # that were actually built, and whether any cell in the middle was
            # missing.  Both conditions make this read raise anyway, and getting
            # the grid meant deserialising a 22x22 block of per-cell PSFs for
            # every host, which measured 52 s of a 400 s run.
            #
            # What it cost to drop: the pre-check could distinguish "this stamp
            # is off the grid" from "this read failed", and the try cannot.  So
            # every distinct failure is reported once, loudly, with its type and
            # message -- a burst of OSError here is a broken repo being counted
            # as a selection effect, and that has to be visible.
            try:
                with clock("stamp pixels"):
                    stamp = butler.get(
                        ref, parameters={"bbox": _stamp_box(x, y, native_size)})
            except Exception as exc:
                kind = type(exc).__name__
                if kind not in read_failures:
                    read_failures[kind] = str(exc)
                    log.warning(
                        "stamp read failed with %s, counted as off_the_grid: %s. "
                        "Expected for a host near the edge of coverage or over a "
                        "cell that was never built; anything else means the read "
                        "is failing for a reason this is silently absorbing.",
                        kind, exc)
                n_off_the_grid_stamps += 1
                rec.update(status="rejected", reasons=f"off_the_grid:{kind}")
                records.append(rec)
                continue
            fitted = True
            n_pixel_reads += 1

            # The cell indices come off the stamp, which is a CellCoadd of just
            # this region and already in hand.  They must be the patch's own
            # (i, j) and not indices relative to the sub-region -- the depth
            # cross-check below is what says so, since provenance is keyed by
            # the patch's numbering.
            cells = cells_in_stamp(stamp, x, y, native_size)

            visit_counts = cell_visit_counts(
                component(ref, band_name, "provenance", "read: provenance"))
            if (tract, patch, band_name) not in depth_counted:
                depth_counted.add((tract, patch, band_name))
                all_visit_counts.extend(visit_counts.values())
            if not depth_checked:
                # The grid's (i, j) and the provenance table's cell columns are
                # two independent conventions and nothing guarantees they agree
                # on which one is x.  Transposed, every lookup misses and every
                # stamp reads as zero-visit.
                depth_checked = True
                if not any(c in visit_counts for c in cells):
                    raise RuntimeError(
                        f"none of the cells a stamp covers {cells[:4]} appear in "
                        f"provenance.contributions (which has e.g. "
                        f"{sorted(visit_counts)[:4]}). The grid index and the "
                        f"provenance cell columns are transposed relative to "
                        f"each other; fix CONTRIB_CELL_COLUMNS in "
                        f"rubin/extract.py."
                    )
                log.info("%d cells in this patch, %d-%d visits each",
                         len(visit_counts), min(visit_counts.values()),
                         max(visit_counts.values()))

            image = np.asarray(_attr(stamp, "image").array, dtype=np.float32)
            if image.shape != (native_size, native_size):
                rec.update(status="rejected", reasons=f"clipped:{image.shape}")
                records.append(rec)
                continue

            # Variance and mask are read, used, and dropped: they are what the
            # gate is made of, and the prior never sees them.
            variance = np.asarray(_attr(stamp, "variance").array, dtype=np.float32)
            packed, mapping = pack_mask(_attr(stamp, "mask"))
            if mask_mapping is None:
                mask_mapping = mapping
            elif mapping != mask_mapping:
                raise RuntimeError(
                    f"mask schema changed mid-run: {mapping} after "
                    f"{mask_mapping}. The packing is per-shard, so a changing "
                    f"schema would make the gate's plane names mean different "
                    f"bits in different stamps."
                )

            n_lo, n_hi = stamp_depth(visit_counts, cells)
            # A cell with no visits at all is infinitely shallower than its
            # neighbour, which is the worst case and must not read as 1.0.
            depth_ratio = (n_hi / n_lo) if n_lo > 0 else np.inf
            with clock("gate"):
                reasons, diag = gate(
                    image, variance, packed, mask_mapping,
                    cell_depth_ratio=depth_ratio, n_visits=n_lo, **gate_kwargs)
            rec.update({f"diag_{key}": v for key, v in diag.items()})
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
                        # DP2 coadds get a final background subtraction that
                        # over-subtracts around extended galaxies.  These are as
                        # delivered.
                        "background_restored": 0,
                    },
                )

            y0, x0 = _origin(stamp)
            acf.add(image)
            writer.add(image, meta={
                "band_idx": BANDS.index(band_name),
                "x0": x0,
                "y0": y0,
                "ra": target_ra,
                "dec": target_dec,
                "pixel_scale": _pixel_scale(wcs, x, y),
                "sky_noise": diag["sky_noise"],
                "host_id": hid,
                "tract": tract,
                "patch": patch,
                "n_cells_spanned": len(cells),
                "n_visits_min": n_lo,
                "n_visits_max": n_hi,
                "cell_depth_ratio": diag.get("cell_depth_ratio", np.nan),
                "variance_step": diag["variance_step"],
                "frac_no_data": diag["frac_no_data"],
                "frac_inexact_psf": diag["frac_INEXACT_PSF"],
                "frac_rejected": diag["frac_REJECTED"],
            })
            rec.update(status="accepted", patch_index=n_accepted)
            records.append(rec)
            n_accepted += 1

        if not fitted:
            n_off_the_grid += 1

    if n_stamps is not None and n_accepted < n_stamps:
        log.warning(
            "produced %d of the %d cutouts asked for, having walked the whole "
            "catalogue of %d hosts. Loosen the gate or widen the host cuts -- "
            "read rejection_counts in the summary first",
            n_accepted, n_stamps, len(order))

    paths = writer.close() if writer is not None else []
    acf_result = acf.result()
    hosts_tried = catalogue[order[:n_tried]]
    _write_table(out_dir / "hosts", hosts_tried)
    summary = _write_manifest(out_dir, records)
    summary.update(
        release="DP2",
        # A *host* is a catalogue object.  A *stamp* is one cutout of one host in
        # one band, so a single host yields up to len(bands) of them -- which is
        # why stamps_attempted runs several times hosts_tried.
        counts={
            "host_candidates_in_catalogue": len(catalogue),
            "hosts_tried": n_tried,
            "hosts_with_no_coadd_for_their_patch": n_no_coadd,
            "hosts_too_near_the_edge_of_coverage": n_off_the_grid,
            "stamps_off_the_grid": n_off_the_grid_stamps,
            "stamps_attempted": summary.pop("n_attempts"),
            "stamps_accepted": n_accepted,
            "stamps_rejected": summary.pop("n_rejected"),
            "stamps_requested": n_stamps,
            "shards_written": len(paths),
        },
        seconds=clock.summary(),
        # Empty unless a stamp read failed.  A type other than the geometry
        # error means reads are failing for a reason the walk is counting as a
        # selection effect.
        stamp_read_failures=read_failures,
        shards=[str(p) for p in paths],
        visits_per_cell=_distribution(all_visit_counts),
        field_radius_deg=radius_deg,
        dataset_type=DATASET_TYPE,
        mask_plane_dict=mask_mapping,
        correlation_length_native_flux_px=acf_result["xi"],
        correlation_length_noise_fraction=acf_result["noise_fraction"],
        correlation_length_n_patches=acf_result["n_patches"],
        correlation_profile_native_flux=acf_result["profile"],
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("\n%s", describe_summary(summary))
    return summary


class Stopwatch:
    """Cumulative wall time per stage.

    A run that takes half an hour with no output is a run you cannot tune.
    These totals go into the summary, so the answer to "what is it doing" is a
    measurement rather than a guess.
    """

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.totals[name] = (self.totals.get(name, 0.0)
                                 + time.perf_counter() - start)

    def summary(self) -> dict[str, float]:
        return {k: round(v, 1) for k, v in
                sorted(self.totals.items(), key=lambda kv: -kv[1])}


def describe_summary(summary: dict) -> str:
    """The run in a dozen lines, in the units the numbers are actually in.

    A host is a catalogue object; a stamp is one cutout of one host in one band.
    Keeping the two words apart is most of what makes the counts readable.
    """
    c = summary.get("counts", {})
    lines = [
        f"{c.get('hosts_tried', 0)} hosts walked, out of "
        f"{c.get('host_candidates_in_catalogue', 0)} in the catalogue",
        f"  {c.get('hosts_with_no_coadd_for_their_patch', 0)} had no coadd for "
        f"their patch; "
        f"{c.get('hosts_too_near_the_edge_of_coverage', 0)} sat too near the "
        f"edge of coverage for a stamp to fit",
        f"{c.get('stamps_attempted', 0)} stamps attempted (one host x one band) "
        f"-> {c.get('stamps_accepted', 0)} written, "
        f"{c.get('stamps_rejected', 0)} rejected",
    ]
    reasons = summary.get("rejection_counts") or {}
    if reasons:
        top = ", ".join(f"{k} {v}" for k, v in list(reasons.items())[:5])
        lines.append(f"  most common reasons: {top}")
        lines.append("  (a stamp can fail several gates, so these sum to more "
                     "than the rejections)")
    depth = summary.get("visits_per_cell") or {}
    if depth:
        lines.append(f"depth: {depth.get('min')}-{depth.get('max')} visits per "
                     f"cell, median {depth.get('p50')}, over {depth.get('n')} cells")
    seconds = summary.get("seconds") or {}
    if seconds:
        total = sum(seconds.values())
        spend = ", ".join(f"{k} {v:.0f}s" for k, v in seconds.items())
        lines.append(f"time: {total:.0f}s total -- {spend}")
    return "\n".join(lines)


# -- bookkeeping -----------------------------------------------------------


def _distribution(values) -> dict:
    """Percentiles of a list, for the summary.  Empty in, empty out."""
    a = np.asarray(list(values), dtype=float)
    if not a.size:
        return {}
    pcts = np.percentile(a, [0, 25, 50, 75, 100])
    return {"n": int(a.size),
            **{k: float(v) for k, v in zip(("min", "p25", "p50", "p75", "max"), pcts)}}


def _origin(stamp) -> tuple[int, int]:
    """``(y0, x0)`` of a stamp in tract coordinates.

    Mandatory, not optional: without the origin a saved stamp cannot be mapped
    back to the sky, so an absent one ends the run rather than writing -1.
    """
    yx0 = _attr(stamp, "origin")
    return int(yx0.y), int(yx0.x)


def _pixel_scale(wcs, x: float, y: float) -> float:
    """Arcsec per pixel, measured by stepping one pixel through the WCS itself.

    Avoids assuming any particular WCS introspection API.
    """
    ra0, dec0 = _pixel_to_sky(wcs, x, y)
    ra1, dec1 = _pixel_to_sky(wcs, x + 1.0, y)
    cosd = np.cos(np.deg2rad(dec0))
    return float(np.hypot((ra1 - ra0) * cosd, dec1 - dec0) * 3600.0)


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


def _write_manifest(out_dir: Path, records: Iterable[dict]) -> dict:
    """Write every attempt, accepted or not, and count why the rest failed."""
    import pandas as pd

    records = list(records)
    # A stamp can fail several gates at once, and ``gate`` returns them in a
    # fixed order.  Counting only the first blames whichever check happens to run
    # early -- which is why the summary used to disagree with the figure, and why
    # a plane gated last could account for a quarter of the rejections without
    # appearing in the counts at all.  Count every reason; the totals therefore
    # exceed the number of rejected stamps, which ``n_rejected`` gives.
    reasons: dict[str, int] = {}
    n_rejected = 0
    for r in records:
        if r.get("status") == "accepted":
            continue
        n_rejected += 1
        parts = [p.split(":")[0] for p in str(r.get("reasons", "unknown")).split(";")]
        for key in dict.fromkeys(p for p in parts if p and p != "nan"):
            reasons[key] = reasons.get(key, 0) + 1
    pd.DataFrame(records).to_parquet(out_dir / "manifest.parquet", index=False)
    return {
        "n_attempts": len(records),
        "n_rejected": n_rejected,
        "rejection_counts": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
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
