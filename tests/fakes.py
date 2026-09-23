"""A butler and a coadd small enough to read, real enough to run the sweep on.

The extraction sweep is where every runtime failure in this project has
happened -- a data id that stopped being a Mapping, a bind key that shadowed a
dimension, a component that turned out to be a property, a patch with cells that
were never built -- and none of them were catchable without a butler.  So this
is a butler: a few tracts of catalogue, a WCS that actually inverts, and coadds
sliced by bounding box, in about two hundred lines.

The geometry is a tangent plane about each tract centre, which round-trips
exactly.  That matters: ``_verify_centre`` compares a position against its own
projection, so a fake WCS that does not invert cleanly would fail every stamp
for the wrong reason.
"""

from __future__ import annotations

import numpy as np

PIXEL_SCALE = 0.2 / 3600.0  # degrees per pixel
CELL = 150
CELLS_PER_PATCH = 22
PATCH = CELL * CELLS_PER_PATCH  # 3300
PLANES = ("NO_DATA", "DETECTION_EDGE", "SATURATED", "COSMIC_RAY",
          "INTERPOLATED", "DETECTED", "INEXACT_PSF", "REJECTED")


# -- geometry ---------------------------------------------------------------


class XY:
    def __init__(self, x, y):
        self.x, self.y = x, y


class CellIJ:
    def __init__(self, i, j):
        self.i, self.j = i, j


class Interval:
    def __init__(self, start, stop):
        self.start, self.stop = int(start), int(stop)

    @property
    def min(self):
        return self.start

    @property
    def max(self):
        return self.stop - 1

    @property
    def size(self):
        return self.stop - self.start


class Box:
    """``Box.factory[y, x]`` -- numpy order, the opposite of DP1's Box2I(x, y)."""

    def __init__(self, x: Interval, y: Interval):
        self.x, self.y = x, y

    class _Factory:
        def __getitem__(self, key):
            ys, xs = key
            return Box(Interval(xs.start, xs.stop), Interval(ys.start, ys.stop))

    factory = _Factory()

    def contains(self, *, x, y):
        return (self.x.start <= x < self.x.stop) and (self.y.start <= y < self.y.stop)

    def __repr__(self):
        return f"Box[y={self.y.start}:{self.y.stop}, x={self.x.start}:{self.x.stop}]"


class CellGrid:
    def __init__(self, bbox: Box):
        self.bbox = bbox

    def index_of(self, *, x, y):
        return CellIJ(i=(int(x) - self.bbox.x.start) // CELL,
                      j=(int(y) - self.bbox.y.start) // CELL)


class CellGridBounds:
    """The populated region, minus individually missing cells."""

    def __init__(self, bbox: Box, missing=()):
        self.bbox = bbox
        self.grid = CellGrid(bbox)
        self.missing = frozenset(missing)

    def contains(self, *, x, y):
        if not self.bbox.contains(x=x, y=y):
            return False
        c = self.grid.index_of(x=x, y=y)
        return all((c.i, c.j) != (m.i, m.j) for m in self.missing)


class Wcs:
    """A tangent plane about the tract centre.  Inverts exactly."""

    def __init__(self, ra0, dec0, x0, y0):
        self.ra0, self.dec0, self.x0, self.y0 = ra0, dec0, x0, y0
        self._cosd = np.cos(np.deg2rad(dec0))

    def sky_to_pixel(self, coord):
        ra, dec = float(coord.ra.deg), float(coord.dec.deg)
        return XY(x=self.x0 + (ra - self.ra0) * self._cosd / PIXEL_SCALE,
                  y=self.y0 + (dec - self.dec0) / PIXEL_SCALE)

    def pixel_to_sky(self, *, x, y):
        from astropy.coordinates import SkyCoord

        return SkyCoord(ra=self.ra0 + (x - self.x0) * PIXEL_SCALE / self._cosd,
                        dec=self.dec0 + (y - self.y0) * PIXEL_SCALE, unit="deg")


# -- pixels -----------------------------------------------------------------


class Plane:
    def __init__(self, array):
        self.array = array


class Mask:
    """DP2 masks are plane-based: no integer array, only named boolean planes."""

    def __init__(self, shape, set_planes=()):
        self.shape = shape
        self.schema = type("Schema", (), {"names": list(PLANES)})()
        self._set = {n: np.zeros(shape, bool) for n in PLANES}
        for name, region in set_planes:
            self._set[name][region] = True

    def get(self, name):
        return self._set[name]


class Stamp:
    def __init__(self, image, variance, mask, y0, x0):
        self.image = Plane(image)
        self.variance = Plane(variance)
        self.mask = mask
        self.yx0 = XY(x=x0, y=y0)


class Psf:
    def __init__(self, bounds):
        self.bounds = bounds


class Provenance:
    def __init__(self, table):
        self.contributions = table


class DataCoordinate:
    """Not a Mapping: daf_butler v27 dropped that, and dict() on one raises."""

    def __init__(self, values):
        self.mapping = dict(values)

    def __getitem__(self, key):
        return self.mapping[key]


class Ref:
    def __init__(self, **values):
        self.dataId = DataCoordinate(values)


# -- the butler -------------------------------------------------------------


class FakeButler:
    """Serves object tables, coadd components and bbox reads for a few tracts."""

    def __init__(self, tracts: dict, objects: dict, n_visits=20, missing=(),
                 seed=0, skymap="lsst_cells_v2"):
        self.tracts = tracts        # tract -> (ra0, dec0) centre
        self.objects = objects      # tract -> astropy Table
        self.n_visits = n_visits
        self.missing = dict(missing)  # (tract, patch) -> [CellIJ, ...]
        self.skymap = skymap
        self.rng = np.random.default_rng(seed)
        self.reads = {"component": 0, "bbox": 0, "whole": 0}
        # Built once per patch and handed out again: the read is still counted,
        # so a test can see how many there were, but 484 cells x n_visits rows
        # do not get rebuilt for every stamp.
        self._contrib_cache: dict = {}

    # geometry of a patch inside its tract
    def _patch_box(self, patch):
        row, col = divmod(int(patch), 10)
        return Box(Interval(col * PATCH, (col + 1) * PATCH),
                   Interval(row * PATCH, (row + 1) * PATCH))

    def _bounds(self, tract, patch):
        return CellGridBounds(self._patch_box(patch),
                              self.missing.get((int(tract), int(patch)), ()))

    def _wcs(self, tract):
        ra0, dec0 = self.tracts[int(tract)]
        return Wcs(ra0, dec0, x0=0.0, y0=0.0)

    def query_datasets(self, kind, data_id=None, limit=None, explain=True, **kw):
        """Constrained by data id.  A real repo answers a tract+patch query with
        one ref per band; anything wider would let a wrong-patch ref through."""
        data_id = dict(data_id or {})
        tract, patch = data_id.get("tract"), data_id.get("patch")
        tracts = [tract] if tract is not None else sorted(self.tracts)
        refs = []
        for t in tracts:
            table = self.objects.get(t)
            if table is None:
                continue
            present = {int(p) for p in table["patch"]}
            for pa in sorted(present if patch is None else present & {int(patch)}):
                for band in ("u", "g", "r", "i", "z", "y"):
                    refs.append(Ref(skymap=self.skymap, tract=t, patch=pa,
                                    band=band))
        return refs

    def get(self, what, dataId=None, parameters=None):
        if isinstance(what, str):
            kind, _, component = what.partition(".")
            self.reads["component"] += 1
            tract = int(dataId["tract"])
            patch = int(dataId["patch"])
            if component == "sky_projection":
                return self._wcs(tract)
            if component == "psf":
                return Psf(self._bounds(tract, patch))
            if component == "provenance":
                return Provenance(self._contributions(tract, patch))
            raise RuntimeError(f"no component {component!r}")

        fields = what.dataId.mapping
        if not parameters or "bbox" not in parameters:
            self.reads["whole"] += 1
            raise AssertionError(
                "the sweep must read stamps by bbox, not whole patches"
            )
        self.reads["bbox"] += 1
        return self._stamp(int(fields["tract"]), int(fields["patch"]),
                           parameters["bbox"])

    def _contributions(self, tract, patch):
        from astropy.table import Table

        key = (int(tract), int(patch))
        if key in self._contrib_cache:
            return self._contrib_cache[key]
        rows = []
        for i in range(CELLS_PER_PATCH):
            for j in range(CELLS_PER_PATCH):
                for v in range(self.n_visits):
                    rows.append((i, j, 1000 + v, 1))
        a = np.asarray(rows, dtype=np.int64)
        table = Table({"cell_i": a[:, 0], "cell_j": a[:, 1],
                       "visit": a[:, 2], "detector": a[:, 3]})
        self._contrib_cache[key] = table
        return table

    def _stamp(self, tract, patch, box):
        bounds = self._bounds(tract, patch)
        if not (bounds.contains(x=box.x.start, y=box.y.start)
                and bounds.contains(x=box.x.max, y=box.y.max)):
            raise ValueError(
                f"grid bounding box {bounds.bbox} does not contain {box}"
            )
        ny, nx = box.y.size, box.x.size
        rng = np.random.default_rng((tract * 1000 + patch) % 2**32)
        sky = 12.0
        image = rng.normal(0.0, sky, (ny, nx)).astype(np.float32)
        yy, xx = np.mgrid[0:ny, 0:nx]
        r2 = (xx - nx / 2) ** 2 + (yy - ny / 2) ** 2
        image += (400.0 * np.exp(-r2 / (2 * 9.0**2))).astype(np.float32)
        variance = np.full((ny, nx), sky**2, dtype=np.float32)
        return Stamp(image, variance, Mask((ny, nx)), box.y.start, box.x.start)


class FakeTap:
    """A TAP service that answers every query with one table.

    The host cuts run *on the service*, so a fake that applied them would be
    testing itself.  What the extraction code does with the rows it gets back is
    what the sweep tests are about.
    """

    def __init__(self, table):
        self.table = table
        self.queries: list[str] = []

    def submit_job(self, query):
        self.queries.append(query)
        return _FakeJob(self.table)


class _FakeJob:
    def __init__(self, table):
        self.table = table
        self.phase = "COMPLETED"
        self.deleted = False

    def run(self):
        pass

    def wait(self, phases=None, timeout=None):
        pass

    def raise_if_error(self):
        pass

    def fetch_result(self):
        return type("Result", (), {"to_table": lambda _self: self.table})()

    def delete(self):
        self.deleted = True


def install(monkeypatch, module):
    """Point the extraction module's lazy stack import at these fakes."""
    from types import SimpleNamespace

    monkeypatch.setattr(module, "_lsst",
                        lambda: SimpleNamespace(Box=Box, Butler=None))
