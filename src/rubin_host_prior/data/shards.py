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
    "band_idx": "u1",
    "x0": "i4",
    "y0": "i4",
    "ra": "f8",
    "dec": "f8",
    "pixel_scale": "f4",
    "sky_noise": "f4",  # sqrt(median variance) at native resolution, nJy
    "host_id": "i8",
    "host_offset_arcsec": "f4",
    "tract": "i4",
    "patch": "i4",
    "n_neighbours": "i2",
    "neighbour_flux_max": "f4",
    # Separation to the nearest catalogue object other than the host itself,
    # split by extendedness.  NaN where there is none inside the search radius.
    "nearest_galaxy_arcsec": "f4",
    "nearest_star_arcsec": "f4",
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
#: Shards written before this existed carried variance, mask and PSF arrays and
#: a different metadata set; they still *open* -- missing metadata columns fill
#: with -1 -- which is precisely the problem, because a stale set then trains or
#: plots without complaint.  ``ShardSet.open`` refuses them instead.
SHARD_SCHEMA = 2


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
                        f"reads {SHARD_SCHEMA}. Schema 1 stored variance, mask "
                        f"and PSF arrays alongside a different metadata set; it "
                        f"would open here with the new columns silently filled "
                        f"with -1. Re-extract, or point at the newer output "
                        f"directory."
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

    def nbytes(self, key: str = "image") -> int:
        n = self.native_size
        itemsize = np.dtype(IMAGE_DTYPES.get(key, "f4")).itemsize
        return len(self) * n * n * itemsize

    def load(self, key: str = "image") -> np.ndarray:
        """Read one whole dataset across all shards into RAM."""
        out = []
        for p in self.paths:
            with h5py.File(p, "r") as f:
                out.append(f[key][:])
        return np.concatenate(out)

    def gather(self, indices: np.ndarray, key: str = "image") -> np.ndarray:
        """Read the given global indices, one sorted pass per shard."""
        indices = np.asarray(indices)
        shard_of = np.searchsorted(self.offsets, indices, side="right") - 1
        out = None
        for s in np.unique(shard_of):
            sel = np.where(shard_of == s)[0]
            local = np.sort(indices[sel] - self.offsets[s])
            order = np.argsort(indices[sel] - self.offsets[s])
            with h5py.File(self.paths[s], "r") as f:
                chunk = f[key][local]
            if out is None:
                out = np.empty((len(indices),) + chunk.shape[1:], dtype=chunk.dtype)
            out[sel[order]] = chunk
        return out
