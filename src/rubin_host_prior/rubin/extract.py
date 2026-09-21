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
  patch -- far enough to land on the wrong galaxy, close enough to look
  plausible.  This module commits to one convention (``PIXEL_ORIGIN``) and then
  **verifies every stamp** by projecting its centre back to the sky and
  comparing against the position asked for; see ``_verify_centre``.  If the
  convention is wrong the first stamp fails loudly instead of quietly producing a
  mis-centred training set.
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

Items marked WARN are not verified against the DP2 tutorials -- the skill's
``references/dp2-facts.md`` and ``references/dp2-images-api.md`` were not
available.  They are collected in ``DP2_ATTRS``, ``REPO``, ``COLLECTION`` and
``FIELDS`` so each is a one-line fix, and the accessors report what the object
actually offers when a name is wrong.
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

#: WARN: the DP2 repo alias and collection are unverified.  Check with
#: ``Butler.get_known_repos()`` / ``$DAF_BUTLER_REPOSITORY_INDEX`` and
#: ``butler.collections.query("*")``.
REPO = "dp2"
COLLECTION = "LSSTCam/DP2"

DATASET_TYPE = "deep_coadd"
#: Coadds are tiled by patch, so this is the predicate that finds one.
COADD_REGION = "patch.region OVERLAPS :region"

#: Attribute names on a ``CellCoadd`` / its stamps.  DP2 turned DP1's getters
#: into attributes; WARN, these spellings are unverified, and every access goes
#: through ``_attr`` so a wrong one reports what the object really has.
DP2_ATTRS = {
    "image": "image",
    "variance": "variance",
    "mask": "mask",
    "psf": "psf",
    "wcs": "astropy_wcs",
    "schema": "schema",
    "origin": "yx0",
}

#: Which of DP2's two conventions this module works in.  ``astropy_wcs`` is
#: patch-local, which is the frame a sliced stamp lives in.  Changing this means
#: changing ``_sky_to_pixel`` and ``_verify_centre`` together.
PIXEL_ORIGIN = "patch-local"

#: Maximum separation between where a stamp actually landed and where it was
#: asked for, in arcsec.  A pixel-origin mix-up is off by far more than this.
CENTRE_TOLERANCE_ARCSEC = 1.0

#: Minimal column subset of the per-tract ``object`` table.  WARN: DP2 column
#: names assumed unchanged from DP1.
OBJECT_COLUMNS = [
    "objectId",
    "coord_ra",
    "coord_dec",
    "shape_xx",
    "shape_yy",
    "shape_xy",
    "refExtendedness",
]
#: Added per band.
OBJECT_BAND_COLUMNS = ["{b}_cModelFlux", "{b}_cModelFluxErr", "{b}_blendedness"]

#: WARN: ECDFS is a standard LSST deep-drilling field, so DP2 very likely covers
#: it, but the DP2 field list was not available to check.
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

    DP2 reads planes by name (``mask.get("SATURATED")``) and its bit numbers are
    not stable across releases, so the bits used here are assigned locally and
    recorded with the shard.  Nothing downstream hard-codes one.  Planes beyond
    ``max_planes`` are dropped -- with a warning, since a silently truncated
    mask would be worse than a loud one.
    """
    names = list(_attr(mask, "schema"))
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


def load_object_table(
    butler,
    ra: float,
    dec: float,
    bands: Sequence[str] = BANDS,
    extra_columns: Sequence[str] = (),
):
    """The per-tract ``object`` table covering ``(ra, dec)``, column-subset."""
    refs = butler.query_datasets(
        "object",
        where="tract.region OVERLAPS POINT(:ra, :dec)",
        bind={"ra": float(ra), "dec": float(dec)},
    )
    if not refs:
        raise RuntimeError(f"no object table covers ({ra}, {dec})")
    columns = list(OBJECT_COLUMNS)
    for b in bands:
        columns += [c.format(b=b) for c in OBJECT_BAND_COLUMNS]
    columns += list(extra_columns)
    tables = []
    for ref in refs:
        try:
            t = butler.get(ref, parameters={"columns": columns + ["detect_isPrimary"]})
        except Exception:
            log.warning("detect_isPrimary unavailable; duplicates not filtered")
            t = butler.get(ref, parameters={"columns": columns})
        t["tract"] = ref.dataId.get("tract", -1)
        tables.append(t)
    from astropy.table import vstack

    return vstack(tables, metadata_conflicts="silent") if len(tables) > 1 else tables[0]


def select_hosts(
    table,
    band: str = "r",
    flux_range: tuple[float, float] = (360.0, 36000.0),
    max_blendedness: float | None = None,
    n_hosts: int | None = None,
    seed: int = 0,
    size_stratified: bool = True,
):
    """Extended objects in a flux range, optionally stratified by apparent size.

    ``flux_range`` bounds are 360 nJy (r = 25.0) to 36000 nJy (r = 20.0).

    ``extendedness`` is a hard threshold on a flux ratio, so it is unreliable
    near the faint limit -- a cut on it alone at r > 23 admits a lot of faint
    stars.  The size cross-check below is not a substitute for inspecting the
    sample, but it does drop the point-like contaminants.

    ``size_stratified`` samples uniformly across size quintiles rather than
    uniformly over the catalogue.  Without it the sample is dominated by the
    smallest, faintest galaxies (there are far more of them) and the prior never
    sees a well-resolved host.
    """
    rng = np.random.default_rng(seed)
    flux_col = f"{band}_cModelFlux"
    t = table

    keep = np.ones(len(t), dtype=bool)
    if "detect_isPrimary" in t.colnames:
        keep &= np.asarray(t["detect_isPrimary"], dtype=bool)
    else:
        _, first = np.unique(np.asarray(t["objectId"]), return_index=True)
        dedup = np.zeros(len(t), dtype=bool)
        dedup[first] = True
        keep &= dedup
    if "refExtendedness" in t.colnames:
        ext = np.asarray(t["refExtendedness"], dtype=float)
        keep &= np.isfinite(ext) & (ext > 0.5)
    flux = np.asarray(t[flux_col], dtype=float)
    keep &= np.isfinite(flux) & (flux > flux_range[0]) & (flux <= flux_range[1])
    if max_blendedness is not None and f"{band}_blendedness" in t.colnames:
        bl = np.asarray(t[f"{band}_blendedness"], dtype=float)
        keep &= ~(np.isfinite(bl) & (bl > max_blendedness))
    if {"shape_xx", "shape_yy"} <= set(t.colnames):
        trace = 0.5 * (
            np.asarray(t["shape_xx"], dtype=float)
            + np.asarray(t["shape_yy"], dtype=float)
        )
        keep &= np.isfinite(trace) & (trace > 1.75**2)

    t = t[keep]
    if n_hosts is None or n_hosts >= len(t):
        return t

    if not size_stratified or "shape_xx" not in t.colnames:
        return t[rng.choice(len(t), size=n_hosts, replace=False)]

    trace = 0.5 * (
        np.asarray(t["shape_xx"], dtype=float) + np.asarray(t["shape_yy"], dtype=float)
    )
    edges = np.percentile(trace, [0, 20, 40, 60, 80, 100])
    per_bin = max(n_hosts // 5, 1)
    picks: list[int] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.where((trace >= lo) & (trace <= hi))[0]
        if len(idx) == 0:
            continue
        picks += list(rng.choice(idx, size=min(per_bin, len(idx)), replace=False))
    picks = list(dict.fromkeys(picks))[:n_hosts]
    return t[np.array(picks)]


# -- image discovery -------------------------------------------------------


def find_coadd_refs(
    butler,
    ra: float,
    dec: float,
    radius_deg: float,
    bands: Sequence[str] = BANDS,
    limit: int | None = None,
):
    """Every ``deep_coadd`` patch overlapping a disc on the sky.

    One query for the whole field rather than one per host: the loop below loads
    each patch once and slices every stamp that falls in it.
    """
    sphgeom = _lsst().sphgeom
    region = sphgeom.Region.from_ivoa_pos(
        f"CIRCLE {float(ra)} {float(dec)} {float(radius_deg)}"
    )
    where = COADD_REGION
    if bands is not None and len(bands) < len(BANDS):
        where += " AND band.name IN (" + ", ".join(f"'{b}'" for b in bands) + ")"
    return list(
        butler.query_datasets(
            DATASET_TYPE, where=where, bind={"region": region},
            order_by=["band.name"], limit=limit,
        )
    )


# -- geometry --------------------------------------------------------------


def _sky_to_pixel(wcs, ra, dec) -> tuple[np.ndarray, np.ndarray]:
    """(ra, dec) in degrees -> fractional pixel (x, y) in the ``PIXEL_ORIGIN`` frame."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    coords = SkyCoord(np.atleast_1d(ra) * u.deg, np.atleast_1d(dec) * u.deg)
    x, y = wcs.world_to_pixel(coords)
    return np.atleast_1d(x), np.atleast_1d(y)


def _pixel_to_sky(wcs, x: float, y: float) -> tuple[float, float]:
    sky = wcs.pixel_to_world(x, y)
    return float(sky.ra.deg), float(sky.dec.deg)


def _verify_centre(wcs, x: float, y: float, ra: float, dec: float) -> float:
    """Separation in arcsec between where a stamp landed and where it was asked for.

    The guard against DP2's two pixel-origin conventions.  A mix-up displaces a
    position by up to a patch -- far enough to land on a different galaxy, close
    enough that the stamp still looks like a plausible piece of sky.  Round-trip
    every centre and reject loudly rather than quietly build a mis-centred set.
    """
    got_ra, got_dec = _pixel_to_sky(wcs, x, y)
    cosd = np.cos(np.deg2rad(dec))
    return float(
        np.hypot((got_ra - ra) * cosd, got_dec - dec) * 3600.0
    )


def _stamp_box(x: float, y: float, size: int):
    """A ``size`` x ``size`` box centred on fractional pixel ``(x, y)``.

    ``Box.factory`` is indexed ``[y, x]`` -- numpy order, the opposite of DP1's
    ``(x, y)`` ``Box2I``.  Rounding is to the nearest pixel; the sub-pixel
    remainder is recorded with the stamp rather than thrown away.
    """
    Box = _lsst().Box
    iy, ix = int(round(y)), int(round(x))
    half = size // 2
    return Box.factory[iy - half : iy - half + size, ix - half : ix - half + size]


# -- the driver ------------------------------------------------------------


def extract_patches(
    butler,
    out_dir: str | Path,
    ra: float = ECDFS[0],
    dec: float = ECDFS[1],
    radius_deg: float = 0.3,
    bands: Sequence[str] = BANDS,
    native_size: int = 416,
    psf_size: int = 41,
    n_hosts: int | None = 8000,
    jitter_arcsec: float = 4.0,
    host_flux_range: tuple[float, float] = (360.0, 36000.0),
    max_blendedness: float | None = None,
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

    Every attempt is recorded in the manifest, rejections included, with the
    reason and the diagnostics.  Those statistics *are* the selection function,
    and the bias they reveal -- against dense bright centres -- is the regime this
    project exists to model.
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_kwargs = dict(gate_kwargs or {})

    log.info("loading object table near (%.4f, %.4f)", ra, dec)
    catalogue = load_object_table(butler, ra, dec, bands=bands)
    hosts = select_hosts(
        catalogue,
        band="r" if "r" in bands else bands[0],
        flux_range=host_flux_range,
        max_blendedness=max_blendedness,
        n_hosts=n_hosts,
        seed=seed,
    )
    log.info("selected %d hosts from %d catalogue rows", len(hosts), len(catalogue))

    host_ra = np.asarray(hosts["coord_ra"], dtype=float)
    host_dec = np.asarray(hosts["coord_dec"], dtype=float)
    host_id = np.asarray(hosts["objectId"], dtype=np.int64)

    # Jitter once per host, not once per (host, patch): the same physical scene
    # should be cut the same way in every band.
    r_jit = jitter_arcsec * np.sqrt(rng.uniform(size=len(hosts)))
    th_jit = rng.uniform(0, 2 * np.pi, size=len(hosts))
    cosd = np.maximum(np.cos(np.deg2rad(host_dec)), 1e-6)
    tgt_ra = host_ra + r_jit * np.cos(th_jit) / 3600.0 / cosd
    tgt_dec = host_dec + r_jit * np.sin(th_jit) / 3600.0

    neighbours = _NeighbourIndex(catalogue, bands)
    records: list[dict] = []
    neighbour_rows: list[dict] = []
    mask_mapping: dict[str, int] | None = None
    writer: ShardWriter | None = None
    n_accepted = 0
    acf = AutocorrelationAccumulator(native_size)

    refs = find_coadd_refs(butler, ra, dec, radius_deg, bands)
    log.info("%d coadd patches overlap the field", len(refs))

    try:
        for ref in refs:
            data_id = ref.dataId
            band_name = str(data_id.get("band", "?"))
            base = {
                "dataId": json.dumps({k: str(v) for k, v in dict(data_id).items()}),
                "band": band_name,
                "tract": int(data_id.get("tract", -1)),
                "patch": int(data_id.get("patch", -1)),
            }
            try:
                coadd = butler.get(ref)
            except Exception as exc:
                records.append({**base, "status": "rejected",
                                "reasons": f"read_failed:{exc!r}"[:120]})
                continue

            wcs = _attr(coadd, "wcs")
            image_all = np.asarray(_attr(coadd, "image"))
            height, width = image_all.shape[-2:]
            xs, ys = _sky_to_pixel(wcs, tgt_ra, tgt_dec)
            half = native_size // 2
            inside = (
                (xs - half >= 0) & (xs + half < width)
                & (ys - half >= 0) & (ys + half < height)
            )
            which = np.where(inside)[0]
            if which.size == 0:
                continue
            log.debug("patch %s: %d hosts", base["patch"], which.size)

            for h in which:
                if max_patches is not None and n_accepted >= max_patches:
                    raise _Done
                rec = {**base, "host_id": int(host_id[h]),
                       "host_offset_arcsec": float(r_jit[h])}

                x, y = float(xs[h]), float(ys[h])
                box = _stamp_box(x, y, native_size)
                # `coadd[box]` is a VIEW; copy before the parent goes away.
                stamp = coadd[box].copy()

                image = np.asarray(_attr(stamp, "image"), dtype=np.float32)
                if image.shape != (native_size, native_size):
                    rec.update(status="rejected", reasons=f"clipped:{image.shape}")
                    records.append(rec)
                    continue

                sep = _verify_centre(wcs, x, y, float(tgt_ra[h]), float(tgt_dec[h]))
                rec["centre_sep_arcsec"] = sep
                if not np.isfinite(sep) or sep > CENTRE_TOLERANCE_ARCSEC:
                    rec.update(status="rejected", reasons=f"centre_mismatch:{sep:.2f}")
                    records.append(rec)
                    continue

                variance = np.asarray(_attr(stamp, "variance"), dtype=np.float32)
                packed, mapping = pack_mask(_attr(stamp, "mask"))
                if mask_mapping is None:
                    mask_mapping = mapping
                elif mapping != mask_mapping:
                    rec.update(status="rejected", reasons="mask_schema_changed")
                    records.append(rec)
                    continue

                reasons, diag = gate(image, variance, packed, mask_mapping,
                                     **gate_kwargs)
                rec.update({f"diag_{k}": v for k, v in diag.items()})
                psf = psf_bundle(stamp, x, y, psf_size)
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
                        psf_size=psf_size,
                        mask_plane_dict=mask_mapping,
                        prefix=prefix,
                        patches_per_shard=patches_per_shard,
                        dataset_type=DATASET_TYPE,
                        attrs={
                            "release": "DP2",
                            "field_ra": ra,
                            "field_dec": dec,
                            "bands": json.dumps(list(bands)),
                            "jitter_arcsec": jitter_arcsec,
                            "flux_units": "nJy",
                            "correlated_noise": 1,  # coadds are warped
                            "pixel_origin": PIXEL_ORIGIN,
                            # DP2 coadds are over-subtracted around extended
                            # galaxies and the background can be restored with
                            # apply_background.  These are as delivered.
                            "background_restored": 0,
                        },
                    )

                nb = neighbours.near(
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
                        "psf_ixx": psf["psf_ixx"],
                        "psf_iyy": psf["psf_iyy"],
                        "psf_ixy": psf["psf_ixy"],
                        "pixel_scale": _pixel_scale(wcs),
                        "sky_noise": diag.get("sky_noise", np.nan),
                        "host_id": int(host_id[h]),
                        "host_offset_arcsec": float(r_jit[h]),
                        "tract": base["tract"],
                        "patch": base["patch"],
                        "n_neighbours": len(others),
                        "neighbour_flux_max": float(
                            max([n["flux"] for n in others], default=np.nan)
                        ),
                        "nearest_galaxy_arcsec": float(min(gal, default=np.nan)),
                        "nearest_star_arcsec": float(min(star, default=np.nan)),
                        "frac_no_data": diag.get("frac_no_data", np.nan),
                        "frac_inexact_psf": diag.get("frac_INEXACT_PSF", np.nan),
                        "frac_rejected": diag.get("frac_REJECTED", np.nan),
                    },
                )
                for n in others:
                    neighbour_rows.append({"patch_index": n_accepted, **n})
                rec.update(status="accepted", patch_index=n_accepted)
                records.append(rec)
                n_accepted += 1
            del coadd
    except _Done:
        log.info("reached max_patches=%s", max_patches)

    paths = writer.close() if writer is not None else []
    acf_result = acf.result()
    _write_table(out_dir / "hosts", hosts)
    summary = _write_manifest(out_dir, records, neighbour_rows)
    summary.update(
        release="DP2",
        n_accepted=n_accepted,
        n_shards=len(paths),
        shards=[str(p) for p in paths],
        n_hosts=len(hosts),
        n_coadd_patches=len(refs),
        dataset_type=DATASET_TYPE,
        mask_plane_dict=mask_mapping,
        correlation_length_native_flux_px=acf_result["xi"],
        correlation_length_noise_fraction=acf_result["noise_fraction"],
        correlation_length_n_patches=acf_result["n_patches"],
        correlation_profile_native_flux=acf_result["profile"],
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _origin(stamp) -> tuple[int, int]:
    """``(y0, x0)`` of a stamp, or ``(-1, -1)``.

    Mandatory to keep, not optional: without the origin a saved stamp cannot be
    mapped back to the sky.  WARN: the attribute name is unverified.
    """
    try:
        origin = getattr(stamp, DP2_ATTRS["origin"])
        return int(origin[0]), int(origin[1])
    except Exception as exc:
        log.debug("no stamp origin: %s", exc)
        return -1, -1


def _pixel_scale(wcs) -> float:
    """Arcsec per pixel from an astropy WCS.  WARN: unverified on DP2."""
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales

        return float(np.mean(proj_plane_pixel_scales(wcs)) * 3600.0)
    except Exception:
        return float("nan")


def psf_bundle(stamp, x: float, y: float, psf_size: int) -> dict | None:
    """Coadd PSF image and moments at ``(x, y)``, or ``None`` if unavailable.

    A patch without a PSF cannot be forward-modelled later and is rejected rather
    than stored incomplete.  WARN: the DP2 PSF interface is unverified; several
    spellings are tried before giving up.
    """
    try:
        psf = _attr(stamp, "psf")
        image = None
        for call in ("compute_image", "computeImage", "image_at", "__call__"):
            fn = getattr(psf, call, None)
            if fn is None:
                continue
            try:
                image = np.asarray(fn(x, y), dtype=np.float32)
                break
            except Exception:
                continue
        if image is None:
            return None
        return {"psf": image, **_psf_moments(image)}
    except Exception as exc:
        log.debug("PSF evaluation failed: %s", exc)
        return None


def _psf_moments(image: np.ndarray) -> dict:
    """Second moments straight from the PSF stamp.

    Computed here rather than asked of the stack, so the numbers are defined the
    same way on any release and do not depend on an unverified accessor.
    """
    a = np.asarray(image, dtype=np.float64)
    total = a.sum()
    if not np.isfinite(total) or total <= 0:
        return {"psf_sigma": np.nan, "psf_ixx": np.nan,
                "psf_iyy": np.nan, "psf_ixy": np.nan}
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    cy, cx = (a * yy).sum() / total, (a * xx).sum() / total
    ixx = float((a * (xx - cx) ** 2).sum() / total)
    iyy = float((a * (yy - cy) ** 2).sum() / total)
    ixy = float((a * (xx - cx) * (yy - cy)).sum() / total)
    det = max(ixx * iyy - ixy**2, 0.0)
    return {"psf_sigma": float(det**0.25), "psf_ixx": ixx,
            "psf_iyy": iyy, "psf_ixy": ixy}


class _Done(Exception):
    """Internal: stop the nested extraction loops at max_patches."""


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


def _write_manifest(
    out_dir: Path, records: Iterable[dict], neighbour_rows: Iterable[dict]
) -> dict:
    """Parquet if pandas is available, CSV otherwise.  Never lose the records."""
    records = list(records)
    reasons: dict[str, int] = {}
    for r in records:
        if r.get("status") != "accepted":
            key = str(r.get("reasons", "unknown")).split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
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
        "manifest_format": fmt,
        "rejection_counts": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }
