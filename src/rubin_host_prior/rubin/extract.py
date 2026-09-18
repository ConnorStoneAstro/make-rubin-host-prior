"""Build a patch training set from the DP1 repo.

Plain Python against the Butler as a data-access layer -- no ``pipetask``, no
BPS.  Import-safe without the LSST stack: everything stack-specific is imported
lazily by ``_lsst()``, so the rest of the package (and the test suite) works on a
laptop.

DP1 specifics that this file depends on, and that are easy to get wrong:

* Dataset types are ``visit_image`` / ``deep_coadd``, not ``calexp`` /
  ``deepCoadd``.  Queries go through ``butler.query_datasets``.
* Band constraints are ``band.name = 'r'``, not ``band = 'r'``.
* **Pixels are already nanojanskys** and variance is nJy^2; ``photoCalib`` is
  spatially constant with mean 1.0.  No calibration step belongs here.  The
  header may report ``'adu'`` (DM-51270) -- it lies.
* Spatial predicates differ by dataset: ``visit_detector_region.region`` for
  visit/difference images, ``patch.region`` for coadds, ``tract.region`` for the
  object table.
* The ``object`` table is per-tract and has >1000 columns; always subset with
  ``parameters={'columns': ...}``.  It has no ``[f]_psfMag`` columns -- those
  exist only in TAP.  Magnitudes: ``m = -2.5*log10(f_nJy) + 31.4``.

Items marked WARN below are not verified against the DP1 tutorials and are
written defensively.  Check them on the cluster; the code degrades rather than
crashes if a name is wrong.
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

#: Spatial predicate per dataset type.
REGION_PREDICATE = {
    "visit_image": "visit_detector_region.region OVERLAPS POINT(:ra, :dec)",
    "difference_image": "visit_detector_region.region OVERLAPS POINT(:ra, :dec)",
    "deep_coadd": "patch.region OVERLAPS POINT(:ra, :dec)",
    "template_coadd": "patch.region OVERLAPS POINT(:ra, :dec)",
}

#: Minimal column subset of the per-tract ``object`` table.
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

#: ECDFS: best cadence in DP1, all six bands, low stellar density.
ECDFS = (53.13, -28.10)


def _lsst() -> SimpleNamespace:
    """Import the LSST stack once, lazily."""
    global _STACK
    if _STACK is None:
        import lsst.geom as geom
        from lsst.daf.butler import Butler

        _STACK = SimpleNamespace(geom=geom, Butler=Butler)
    return _STACK


# -- repo ------------------------------------------------------------------


def open_butler(repo: str = "dp1", collection: str = "LSSTComCam/DP1"):
    """Open the DP1 repo read-only.

    ``"dp1"`` is the RSP alias; at NERSC confirm it with
    ``Butler.get_known_repos()`` or ``$DAF_BUTLER_REPOSITORY_INDEX`` and pass the
    path if it differs.  Never open the shared mirror writeable.
    """
    butler = _lsst().Butler(repo, collections=collection)
    if butler is None:
        raise RuntimeError(f"could not open butler repo {repo!r}")
    return butler


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
    # detect_isPrimary is standard in LSST object tables and is how deblend
    # duplicates and tract-overlap repeats are dropped, but it does not appear in
    # the DP1 tutorials.  WARN: ask for it, and carry on without it if absent.
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

    ``flux_range`` defaults to the bounds the DP1 tutorials use: 360 nJy is
    r = 25.0 and 36000 nJy is r = 20.0.

    ``extendedness`` is a hard threshold on ``cModelFlux * 0.985 < psfFlux``, so
    it is unreliable near the faint limit -- a cut on it alone at r > 23 admits a
    lot of faint stars.  The size cross-check below is not a substitute for
    inspecting the sample, but it does drop the point-like contaminants.

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
        # Without it, deduplicate on objectId so tract-overlap regions are not
        # silently oversampled.
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
    # Size cross-check: trace radius above the typical DP1 PSF (sigma ~ 1.75 px).
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


def find_image_refs(
    butler,
    ra: float,
    dec: float,
    bands: Sequence[str] = BANDS,
    dataset_type: str = "visit_image",
    limit: int | None = None,
):
    """Refs for images covering ``(ra, dec)``, time-ordered where meaningful."""
    if dataset_type not in REGION_PREDICATE:
        raise ValueError(
            f"unsupported dataset_type {dataset_type!r}; "
            f"expected one of {sorted(REGION_PREDICATE)}"
        )
    where = REGION_PREDICATE[dataset_type]
    bind: dict = {"ra": float(ra), "dec": float(dec)}
    if bands is not None and len(bands) < len(BANDS):
        where += " AND band.name IN (" + ", ".join(f"'{b}'" for b in bands) + ")"
    # Ordering in the query, not by sorting visitInfo afterwards.
    order_by = (
        ["visit.timespan.begin"] if dataset_type.endswith("_image") else ["band.name"]
    )
    return list(
        butler.query_datasets(
            dataset_type, where=where, bind=bind, order_by=order_by, limit=limit
        )
    )


class StreakCache:
    """Per-(visit, detector) satellite-trail verdict from the difference image.

    The ``visit_image`` ``STREAK`` plane is **not reliably populated in DP1** -- it
    is set during difference imaging and only propagates back sometimes.  A gate
    built on the visit image's own STREAK plane silently passes every trail.  The
    cheap correct route is to read the matching ``difference_image`` mask, which
    uses Rubin's own detection; the same (visit, detector) dataId works for both.

    Faint trails that escaped masking are exactly the ones that would teach a
    diffusion model to generate straight lines, so this is worth the extra read.
    Where no difference image exists the verdict is ``nan`` and the patch is not
    rejected on this basis.
    """

    def __init__(self, butler, enabled: bool = True):
        self.butler = butler
        self.enabled = enabled
        self._cache: dict[tuple, float] = {}

    def fraction(self, data_id) -> float:
        if not self.enabled:
            return float("nan")
        key = (data_id.get("visit"), data_id.get("detector"))
        if key in self._cache:
            return self._cache[key]
        value = float("nan")
        try:
            mask = self.butler.get(
                "difference_image.mask",
                visit=key[0],
                detector=key[1],
                instrument=data_id.get("instrument", "LSSTComCam"),
            )
            planes = dict(mask.getMaskPlaneDict())
            if "STREAK" in planes:
                bit = 1 << int(planes["STREAK"])
                value = float(np.mean((mask.array & bit) != 0))
        except Exception as exc:  # no difference image, or component unavailable
            log.debug("no difference_image mask for %s: %s", key, exc)
        self._cache[key] = value
        return value


# -- stamps ----------------------------------------------------------------


def stamp_bbox(wcs, ra: float, dec: float, size: int):
    """Centred ``Box2I`` of ``size`` pixels at ``(ra, dec)``.

    ``Point2D``, not ``Point2I``: galaxies are not centred on pixel centres and
    rounding throws away the sub-pixel position.  ``makeCenteredBox`` avoids
    hand-computing a lower-left corner, which is the classic off-by-one here.
    """
    geom = _lsst().geom
    sky = geom.SpherePoint(ra * geom.degrees, dec * geom.degrees)
    xy = geom.Point2D(wcs.skyToPixel(sky))
    return geom.Box2I.makeCenteredBox(xy, geom.Extent2I(size, size)), xy


def get_component(butler, dataset_type: str, data_id, component: str):
    """Component read (``'visit_image.wcs'``), which moves no pixels.

    The ``.wcs`` form is the verified DP1 idiom; ``.bbox``/``.psf`` follow the
    same pattern.  WARN: not every component name is confirmed, so callers
    should tolerate failure.
    """
    return butler.get(f"{dataset_type}.{component}", dataId=data_id)


def psf_bundle(stamp, xy, psf_size: int) -> dict | None:
    """PSF image and moments at ``xy``, or ``None`` if the PSF cannot be evaluated.

    ``computeImage`` carries the sub-pixel offset of the requested position and
    raises near detector edges -- caught here, because a patch without a PSF
    cannot be forward-modelled later and should be rejected rather than stored
    incomplete.
    """
    try:
        psf = stamp.getPsf()
        image = np.asarray(psf.computeImage(xy).array, dtype=np.float32)
        shape = psf.computeShape(xy)
        return {
            "psf": image,
            "psf_sigma": float(shape.getDeterminantRadius()),
            "psf_ixx": float(shape.getIxx()),
            "psf_iyy": float(shape.getIyy()),
            "psf_ixy": float(shape.getIxy()),
        }
    except Exception as exc:
        log.debug("PSF evaluation failed: %s", exc)
        return None


def _mjd(stamp) -> float:
    """Mid-exposure MJD.  WARN: accessor chain not verified; degrades to nan."""
    for getter in (
        lambda: stamp.visitInfo.date.toAstropy().mjd,
        lambda: stamp.getInfo().getVisitInfo().getDate().toAstropy().mjd,
    ):
        try:
            return float(getter())
        except Exception:
            continue
    return float("nan")


# -- the driver ------------------------------------------------------------


def extract_patches(
    butler,
    out_dir: str | Path,
    ra: float = ECDFS[0],
    dec: float = ECDFS[1],
    bands: Sequence[str] = BANDS,
    dataset_type: str = "visit_image",
    native_size: int = 416,
    psf_size: int = 41,
    n_hosts: int | None = 2000,
    max_images_per_host: int = 20,
    jitter_arcsec: float = 4.0,
    host_flux_range: tuple[float, float] = (360.0, 36000.0),
    max_blendedness: float | None = None,
    patches_per_shard: int = 1024,
    max_patches: int | None = None,
    use_difference_streak: bool = True,
    max_streak_fraction: float = 1e-4,
    neighbour_radius_arcsec: float = 30.0,
    gate_kwargs: dict | None = None,
    seed: int = 0,
    prefix: str = "patches",
) -> dict:
    """Extract patches near selected hosts and write shards plus a manifest.

    Ordering is deliberate and matters for cost: catalogue selection and the
    cached streak verdict come before any pixel read, and the bbox containment
    test comes before the stamp read.

    Positions are **jittered** around each host rather than centred on it.  A
    prior trained on centred galaxies learns that galaxies are always centred,
    which is useless for a transient that can sit anywhere in the scene.

    Every attempt is recorded in the manifest, rejections included, with the
    reason and the diagnostics.  Those statistics are the only way to find out
    whether the selection function is biased against bright dense centres -- the
    regime this project exists to model.
    """
    geom = _lsst().geom
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

    neighbours = _NeighbourIndex(catalogue, bands)
    streaks = StreakCache(butler, enabled=use_difference_streak)
    records: list[dict] = []
    neighbour_rows: list[dict] = []
    # Streaming, so xi is measured over every accepted patch rather than a
    # subsample held in memory.
    acf = AutocorrelationAccumulator(native_size)
    mask_plane_dict: dict[str, int] | None = None
    writer: ShardWriter | None = None
    n_accepted = 0

    host_ra = np.asarray(hosts["coord_ra"], dtype=float)
    host_dec = np.asarray(hosts["coord_dec"], dtype=float)
    host_id = np.asarray(hosts["objectId"], dtype=np.int64)

    try:
        for h in range(len(hosts)):
            refs = find_image_refs(
                butler, host_ra[h], host_dec[h], bands, dataset_type
            )
            if not refs:
                records.append(
                    {"host_id": int(host_id[h]), "status": "rejected",
                     "reasons": "no_images"}
                )
                continue
            if len(refs) > max_images_per_host:
                refs = [refs[i] for i in rng.choice(len(refs), max_images_per_host,
                                                    replace=False)]

            for ref in refs:
                if max_patches is not None and n_accepted >= max_patches:
                    raise _Done
                data_id = ref.dataId
                rec = {
                    "host_id": int(host_id[h]),
                    "dataId": json.dumps({k: str(v) for k, v in dict(data_id).items()}),
                    "band": str(data_id.get("band", "?")),
                    "visit": int(data_id.get("visit", -1)),
                    "detector": int(data_id.get("detector", -1)),
                }

                streak = streaks.fraction(data_id)
                rec["streak_fraction"] = streak
                if np.isfinite(streak) and streak > max_streak_fraction:
                    rec.update(status="rejected", reasons=f"streak:{streak:.5f}")
                    records.append(rec)
                    continue

                try:
                    wcs = get_component(butler, dataset_type, data_id, "wcs")
                except Exception as exc:
                    rec.update(status="rejected", reasons=f"no_wcs:{exc!r}"[:120])
                    records.append(rec)
                    continue

                # Jitter the target position: uniform within a disc, so the host
                # is somewhere in the patch rather than always at its centre.
                r = jitter_arcsec * np.sqrt(rng.uniform())
                theta = rng.uniform(0, 2 * np.pi)
                d_ra = r * np.cos(theta) / 3600.0 / max(
                    np.cos(np.deg2rad(host_dec[h])), 1e-6
                )
                d_dec = r * np.sin(theta) / 3600.0
                tgt_ra, tgt_dec = host_ra[h] + d_ra, host_dec[h] + d_dec
                rec["host_offset_arcsec"] = float(r)

                bbox, xy = stamp_bbox(wcs, tgt_ra, tgt_dec, native_size)
                try:
                    full = get_component(butler, dataset_type, data_id, "bbox")
                    contained = full.contains(bbox)
                except Exception:
                    contained = None  # WARN: component name unconfirmed
                if contained is False:
                    rec.update(status="rejected", reasons="bbox_not_contained")
                    records.append(rec)
                    continue

                try:
                    handle = butler.getDeferred(dataset_type, dataId=data_id)
                    stamp = handle.get(parameters={"bbox": bbox})
                except Exception as exc:
                    rec.update(status="rejected", reasons=f"read_failed:{exc!r}"[:120])
                    records.append(rec)
                    continue

                image = np.asarray(stamp.image.array, dtype=np.float32)
                if image.shape != (native_size, native_size):
                    # Silent clipping at the detector edge.  Never pad.
                    rec.update(
                        status="rejected", reasons=f"clipped:{image.shape}"
                    )
                    records.append(rec)
                    continue
                variance = np.asarray(stamp.variance.array, dtype=np.float32)
                mask = np.asarray(stamp.mask.array, dtype=np.uint32)
                planes = dict(stamp.getMask().getMaskPlaneDict())
                if mask_plane_dict is None:
                    mask_plane_dict = planes
                elif planes != mask_plane_dict:
                    rec.update(status="rejected", reasons="mask_plane_dict_changed")
                    records.append(rec)
                    continue

                reasons, diag = gate(image, variance, mask, planes, **gate_kwargs)
                rec.update({f"diag_{k}": v for k, v in diag.items()})
                psf = psf_bundle(stamp, xy, psf_size)
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
                        mask_plane_dict=mask_plane_dict,
                        prefix=prefix,
                        patches_per_shard=patches_per_shard,
                        dataset_type=dataset_type,
                        attrs={
                            "field_ra": ra,
                            "field_dec": dec,
                            "bands": json.dumps(list(bands)),
                            "jitter_arcsec": jitter_arcsec,
                            "flux_units": "nJy",
                        },
                    )

                nb = neighbours.near(tgt_ra, tgt_dec, neighbour_radius_arcsec,
                                     str(data_id.get("band", "r")))
                bb = stamp.getBBox()
                band_name = str(data_id.get("band", "r"))
                sky = wcs.pixelToSky(xy)
                writer.add(
                    image,
                    variance,
                    mask,
                    psf["psf"],
                    meta={
                        "band_idx": BANDS.index(band_name) if band_name in BANDS else 255,
                        "visit": int(data_id.get("visit", -1)),
                        "detector": int(data_id.get("detector", -1)),
                        "x0": int(bb.getMinX()),
                        "y0": int(bb.getMinY()),
                        "center_x": float(xy.getX()),
                        "center_y": float(xy.getY()),
                        "ra": float(sky.getRa().asDegrees()),
                        "dec": float(sky.getDec().asDegrees()),
                        "mjd": _mjd(stamp),
                        "psf_sigma": psf["psf_sigma"],
                        "psf_ixx": psf["psf_ixx"],
                        "psf_iyy": psf["psf_iyy"],
                        "psf_ixy": psf["psf_ixy"],
                        "pixel_scale": float(wcs.getPixelScale().asArcseconds()),
                        "sky_noise": diag.get("sky_noise", np.nan),
                        "host_id": int(host_id[h]),
                        "host_offset_arcsec": float(r),
                        "tract": int(data_id.get("tract", -1)),
                        "patch": int(data_id.get("patch", -1)),
                        "n_neighbours": len(nb),
                        "neighbour_flux_max": float(
                            max([n["flux"] for n in nb], default=np.nan)
                        ),
                    },
                )
                acf.add(image)
                for n in nb:
                    neighbour_rows.append({"patch_index": n_accepted, **n})
                rec.update(status="accepted", patch_index=n_accepted)
                records.append(rec)
                n_accepted += 1
    except _Done:
        log.info("reached max_patches=%s", max_patches)

    paths = writer.close() if writer is not None else []
    summary = _write_manifest(out_dir, records, neighbour_rows)
    acf_result = acf.result()
    summary.update(
        n_accepted=n_accepted,
        n_shards=len(paths),
        shards=[str(p) for p in paths],
        n_hosts=len(hosts),
        # Native-resolution, flux-space correlation length: provenance and a
        # sanity check, NOT the number to act on.  At native resolution the
        # small lags are dominated by the PSF, and the log transform changes the
        # correlation structure anyway.  The number that decides how much
        # context the model needs is measured on the pooled, log-space training
        # representation -- scripts/prepare_config.py reports that one.
        correlation_length_native_flux_px=acf_result["xi"],
        correlation_length_noise_fraction=acf_result["noise_fraction"],
        correlation_length_n_patches=acf_result["n_patches"],
        correlation_profile_native_flux=acf_result["profile"],
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


class _Done(Exception):
    """Internal: stop the nested extraction loops at max_patches."""


class _NeighbourIndex:
    """Catalogue neighbours around a position.

    One TAP/Butler query per cutout would be far too slow, so the tract table is
    indexed once and searched in memory.
    """

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

    def near(self, ra: float, dec: float, radius_arcsec: float, band: str):
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
            }
            for i, s in zip(idx, sep)
        ]


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
