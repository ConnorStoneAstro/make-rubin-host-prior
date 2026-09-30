"""Sharded HDF5 storage for native-resolution patches.

Design rules, all of them learned the hard way on parallel filesystems and in
forward models:

* **A few large files, not millions of small ones.**  NERSC's filesystem
  punishes small-file access patterns severely.
* **Store physical units at native resolution.**  Pooling, the log transform and
  the per-band offsets all happen in the loader, so any of them can change
  without re-extracting terabytes.
* **Store the image and nothing else.**  The prior is a distribution over
  pixels; it never sees a variance plane, a mask or a PSF, so carrying them
  triples the storage to no purpose.  They are still *read* during extraction --
  they are what the quality gate is made of -- and then discarded.  What
  survives of them is a handful of scalars per stamp, which is what a later cut
  from the manifest needs.
* **Keep the origin.**  ``x0``/``y0`` is the stamp origin in tract pixels;
  without it an array index cannot be mapped back to the sky.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from ..config import BANDS

#: Per-patch scalar columns and their on-disk dtypes.  ``-1`` is the convention
#: for "not applicable".
META_DTYPES: dict[str, str] = {
    # Index into the **global** ``config.BANDS``, not into whatever subset a
    # given run extracted: ``BANDS.index(band_name)``, so g is always 1 whether
    # or not u was collected.  That is what lets shards from runs with different
    # band lists be merged and read together.  Read it with
    # ``ShardSet.band_counts()``; enumerating a shard's own ``bands`` and
    # comparing the position to this is an off-by-one, and was one.
    "band_idx": "u1",
    "x0": "i4",
    "y0": "i4",
    "ra": "f8",
    "dec": "f8",
    "pixel_scale": "f4",
    "sky_noise": "f4",  # sqrt(median variance) at native resolution, nJy
    "host_id": "i8",
    "tract": "i4",
    "patch": "i4",
    # Depth.  Cells are coadded from different input visits, so a stamp over
    # 150 native px straddles a boundary between two of them; exposure times are
    # equal, so the ratio of visit counts is the depth step exactly, and
    # variance_step is the same thing measured off the pixels.
    "n_cells_spanned": "i2",
    "n_visits_min": "i2",
    "n_visits_max": "i2",
    "cell_depth_ratio": "f4",
    "variance_step": "f4",
    # What the gate saw, kept so a later cut can be made from the manifest
    # without re-reading pixels.
    "frac_no_data": "f4",
    "frac_inexact_psf": "f4",
    "frac_rejected": "f4",
}

IMAGE_KEYS = ("image",)
IMAGE_DTYPES = {"image": "f4"}

#: Bumped whenever the shard layout or the meaning of its metadata changes.
#: Older shards still *open* -- a missing metadata column fills with -1 -- which
#: is precisely the problem, because a stale set then trains or plots without
#: complaint.  ``ShardSet.open`` refuses them instead.  Schema 1 carried
#: variance, mask and PSF arrays; schema 2 added four neighbour columns that
#: nothing trained on and that cost an object-table read per tract.
SHARD_SCHEMA = 3


class ShardWriter:
    """Buffer patches in memory and flush a shard when it is full."""

    def __init__(
        self,
        out_dir: str | Path,
        native_size: int,
        prefix: str = "patches",
        patches_per_shard: int = 1024,
        dataset_type: str = "deep_coadd",
        attrs: dict | None = None,
        compression: str | None = "lzf",
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.native_size = native_size
        self.prefix = prefix
        self.patches_per_shard = patches_per_shard
        self.compression = compression
        self.attrs = {
            "bands": json.dumps(list(BANDS)),
            "native_size": native_size,
            "schema": SHARD_SCHEMA,
            "dataset_type": dataset_type,
            **{k: json.dumps(v) if isinstance(v, (dict, list)) else v
               for k, v in (attrs or {}).items()},
        }
        self._buf: list[dict] = []
        self._shard_index = 0
        self.paths: list[Path] = []
        self.n_written = 0

    def add(self, image: np.ndarray, meta: dict) -> None:
        n = self.native_size
        if image.shape != (n, n):
            raise ValueError(f"image has shape {image.shape}, expected {(n, n)}")
        unknown = set(meta) - set(META_DTYPES)
        if unknown:
            raise ValueError(f"unknown meta keys {sorted(unknown)}")
        self._buf.append(
            {"image": np.asarray(image, dtype=np.float32), "meta": meta}
        )
        if len(self._buf) >= self.patches_per_shard:
            self.flush()

    def flush(self) -> Path | None:
        if not self._buf:
            return None
        path = self.out_dir / f"{self.prefix}-{self._shard_index:05d}.h5"
        n = len(self._buf)
        with h5py.File(path, "w") as f:
            for k, v in self.attrs.items():
                f.attrs[k] = v
            f.attrs["n_patches"] = n
            for key in IMAGE_KEYS:
                f.create_dataset(
                    key,
                    data=np.stack([b[key] for b in self._buf]),
                    dtype=IMAGE_DTYPES[key],
                    compression=self.compression,
                    chunks=(1, self.native_size, self.native_size),
                )
            g = f.create_group("meta")
            for name, dtype in META_DTYPES.items():
                fill = -1 if dtype[0] in "iu" else np.nan
                g.create_dataset(
                    name,
                    data=np.array(
                        [b["meta"].get(name, fill) for b in self._buf], dtype=dtype
                    ),
                )
        self.paths.append(path)
        self.n_written += n
        self._shard_index += 1
        self._buf.clear()
        return path

    def close(self) -> list[Path]:
        self.flush()
        return self.paths

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@dataclass
class ShardSet:
    """Random access across a list of shard files."""

    paths: list[Path]
    counts: np.ndarray
    offsets: np.ndarray
    attrs: dict
    meta: dict[str, np.ndarray]

    @classmethod
    def open(cls, paths) -> "ShardSet":
        if isinstance(paths, (str, Path)):
            paths = [Path(paths)]
        else:
            paths = sorted(Path(p) for p in paths)
        if not paths:
            raise ValueError("no shard files given")
        counts, attrs = [], None
        meta: dict[str, list[np.ndarray]] = {k: [] for k in META_DTYPES}
        for p in paths:
            with h5py.File(p, "r") as f:
                counts.append(int(f.attrs["n_patches"]))
                a = dict(f.attrs)
                found = int(a.get("schema", 1))
                if found != SHARD_SCHEMA:
                    raise ValueError(
                        f"{p} is shard schema {found}, this code writes and "
                        f"reads {SHARD_SCHEMA}. An older shard opens here with "
                        f"the metadata it lacks silently filled with -1. "
                        f"Re-extract, or point at the newer output directory."
                    )
                if attrs is None:
                    attrs = a
                elif a.get("native_size") != attrs.get("native_size"):
                    raise ValueError(
                        f"{p} has native_size {a.get('native_size')} but "
                        f"{paths[0]} has {attrs.get('native_size')}"
                    )
                missing = [k for k in META_DTYPES if k not in f["meta"]]
                if missing:
                    raise ValueError(f"{p} has no metadata for {missing}")
                for k in META_DTYPES:
                    meta[k].append(f["meta"][k][:])
        counts = np.asarray(counts)
        return cls(
            paths=paths,
            counts=counts,
            offsets=np.concatenate([[0], np.cumsum(counts)]),
            attrs=attrs or {},
            meta={k: np.concatenate(v) for k, v in meta.items()},
        )

    @classmethod
    def from_dir(cls, directory, pattern: str = "*.h5") -> "ShardSet":
        return cls.open(sorted(Path(directory).glob(pattern)))

    def __len__(self) -> int:
        return int(self.offsets[-1])

    @property
    def native_size(self) -> int:
        return int(self.attrs["native_size"])

    @property
    def bands(self) -> tuple[str, ...]:
        return tuple(json.loads(self.attrs.get("bands", json.dumps(list(BANDS)))))

    def band_counts(self) -> dict[str, int]:
        """Patches per band name, over the global ``BANDS``.

        ``band_idx`` is a global index, so this cannot be computed by walking
        ``self.bands`` -- that is the subset this run extracted, and its
        positions are not the stored values.  Doing exactly that made
        ``prepare_config.py`` report every band's count against the next band's
        name, and the first band of the subset as absent.
        """
        idx = np.asarray(self.meta["band_idx"], dtype=int)
        return {b: int(np.sum(idx == i)) for i, b in enumerate(BANDS)}

    def nbytes(self, key: str = "image") -> int:
        n = self.native_size
        itemsize = np.dtype(IMAGE_DTYPES.get(key, "f4")).itemsize
        return len(self) * n * n * itemsize

    def load(self, key: str = "image") -> np.ndarray:
        """Read one whole dataset across all shards into RAM.

        Filled into one preallocated array rather than concatenated from a list
        of per-shard arrays.  The list form peaked at twice the final size --
        every shard still referenced while ``concatenate`` built the copy -- so
        caching a 30 GiB set needed 60 GiB to get started, which is exactly the
        size where someone is reaching for this.
        """
        n = len(self)
        side = self.native_size
        out = np.empty((n, side, side), dtype=IMAGE_DTYPES.get(key, "f4"))
        for i, p in enumerate(self.paths):
            with h5py.File(p, "r") as f:
                f[key].read_direct(out[self.offsets[i]:self.offsets[i + 1]])
        return out

    def gather(self, indices: np.ndarray, key: str = "image") -> np.ndarray:
        """Read the given global indices, one sorted pass per shard.

        **Not with h5py's fancy indexing.**  ``f[key][list_of_rows]`` goes
        through HDF5's point-selection machinery and is catastrophically slow on
        a chunked dataset: measured on a warm local SSD, where filesystem
        latency is nil, it costs 32 ms per row against 1.0 ms for the same rows
        read one at a time with ``read_direct`` -- 32x, for byte-identical
        output, and all of it CPU rather than I/O.  On a training step reading
        128 stamps that is the difference between seconds and a tenth of one.

        So the rows are walked instead, in contiguous runs: a run is one
        hyperslab read, and isolated rows fall back to a single-row read, which
        is still 32x better than the selection path.  Sorting first is what
        makes runs appear at all, and it is also the order the chunks sit in on
        disk.
        """
        indices = np.asarray(indices)
        shard_of = np.searchsorted(self.offsets, indices, side="right") - 1
        side = self.native_size
        out = np.empty((len(indices), side, side),
                       dtype=IMAGE_DTYPES.get(key, "f4"))
        for s in np.unique(shard_of):
            sel = np.where(shard_of == s)[0]
            local = indices[sel] - self.offsets[s]
            order = np.argsort(local)
            local, sel = local[order], sel[order]
            # Split the sorted rows into contiguous runs, so neighbours become
            # one read rather than several.
            breaks = np.where(np.diff(local) != 1)[0] + 1
            with h5py.File(self.paths[s], "r") as f:
                d = f[key]
                start = 0
                for run in np.split(local, breaks):
                    stop = start + len(run)
                    # Read the run into a buffer and place it: the caller's
                    # order is arbitrary, so `sel` is not contiguous and cannot
                    # be a `read_direct` destination selection.
                    buf = np.empty((len(run), side, side), dtype=out.dtype)
                    d.read_direct(buf, np.s_[int(run[0]):int(run[-1]) + 1])
                    out[sel[start:stop]] = buf
                    start = stop
        return out
