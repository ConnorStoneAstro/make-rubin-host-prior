"""Sharded HDF5 storage for native-resolution patches.

Design rules, all of them learned the hard way on parallel filesystems and in
forward models:

* **A few large files, not millions of small ones.**  NERSC's filesystem
  punishes small-file access patterns severely.
* **Store physical units at native resolution.**  Pooling, the log transform and
  the per-band offsets all happen in the loader, so any of them can change
  without re-extracting terabytes.
* **Keep the variance plane, the mask, the mask plane dictionary, the PSF and
  the origin.**  A patch without variance and PSF cannot be forward-modelled,
  and a mask without its plane dictionary is uninterpretable -- DP2 bit
  assignments are dynamic, so extraction repacks the mask and records the
  mapping it used.  ``x0``/``y0`` is the stamp origin; without it an array index
  cannot be mapped back to the sky.
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
    "center_x": "f8",
    "center_y": "f8",
    "ra": "f8",
    "dec": "f8",
    "psf_sigma": "f4",
    "psf_fwhm": "f4",
    "psf_ixx": "f4",
    "psf_iyy": "f4",
    "psf_ixy": "f4",
    "pixel_scale": "f4",
    "sky_noise": "f4",  # sqrt(median variance), native resolution, nJy
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
    # DP2 covariates: recorded for every stamp, never gated on.  INEXACT_PSF and
    # REJECTED cover a large fraction of the coadd, so a cut on them keeps
    # almost nothing; frac_no_data is DP2's inf-variance regions, which include
    # the cores of saturated stars.
    # Cells are coadded from different input visits, so depth and PSF step at
    # cell edges; any stamp over 150 native px straddles them.
    "n_cells_spanned": "i2",
    "frac_no_data": "f4",
    "frac_inexact_psf": "f4",
    "frac_rejected": "f4",
}

IMAGE_KEYS = ("image", "variance", "mask")
IMAGE_DTYPES = {"image": "f4", "variance": "f4", "mask": "u4"}


class ShardWriter:
    """Buffer patches in memory and flush a shard when it is full."""

    def __init__(
        self,
        out_dir: str | Path,
        native_size: int,
        psf_size: int,
        mask_plane_dict: dict[str, int],
        prefix: str = "patches",
        patches_per_shard: int = 1024,
        dataset_type: str = "visit_image",
        attrs: dict | None = None,
        compression: str | None = "lzf",
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.native_size = native_size
        self.psf_size = psf_size
        self.prefix = prefix
        self.patches_per_shard = patches_per_shard
        self.compression = compression
        self.attrs = {
            "mask_plane_dict": json.dumps(dict(mask_plane_dict)),
            "bands": json.dumps(list(BANDS)),
            "native_size": native_size,
            "psf_size": psf_size,
            "dataset_type": dataset_type,
            **{k: json.dumps(v) if isinstance(v, (dict, list)) else v
               for k, v in (attrs or {}).items()},
        }
        self._buf: list[dict] = []
        self._shard_index = 0
        self.paths: list[Path] = []
        self.n_written = 0

    def add(
        self,
        image: np.ndarray,
        variance: np.ndarray,
        mask: np.ndarray,
        psf: np.ndarray,
        meta: dict,
    ) -> None:
        n = self.native_size
        for name, arr in (("image", image), ("variance", variance), ("mask", mask)):
            if arr.shape != (n, n):
                raise ValueError(f"{name} has shape {arr.shape}, expected {(n, n)}")
        unknown = set(meta) - set(META_DTYPES)
        if unknown:
            raise ValueError(f"unknown meta keys {sorted(unknown)}")
        self._buf.append(
            {
                "image": np.asarray(image, dtype=np.float32),
                "variance": np.asarray(variance, dtype=np.float32),
                "mask": np.asarray(mask, dtype=np.uint32),
                "psf": _pad_to(np.asarray(psf, dtype=np.float32), self.psf_size),
                "meta": meta,
            }
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
            f.create_dataset(
                "psf",
                data=np.stack([b["psf"] for b in self._buf]),
                dtype="f4",
                compression=self.compression,
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


def _pad_to(psf: np.ndarray, size: int) -> np.ndarray:
    """Centre-pad or centre-crop a PSF stamp to a common size.

    LSST PSF kernel size varies with focal-plane position, so stamps from one
    visit are not all the same shape.  The true size is recoverable from the
    non-zero support; pad rather than resample.
    """
    h, w = psf.shape
    if (h, w) == (size, size):
        return psf
    out = np.zeros((size, size), dtype=np.float32)
    ch, cw = min(h, size), min(w, size)
    sy, sx = (h - ch) // 2, (w - cw) // 2
    dy, dx = (size - ch) // 2, (size - cw) // 2
    out[dy : dy + ch, dx : dx + cw] = psf[sy : sy + ch, sx : sx + cw]
    return out


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
                if attrs is None:
                    attrs = a
                elif a.get("native_size") != attrs.get("native_size"):
                    raise ValueError(
                        f"{p} has native_size {a.get('native_size')} but "
                        f"{paths[0]} has {attrs.get('native_size')}"
                    )
                for k in META_DTYPES:
                    meta[k].append(f["meta"][k][:] if k in f["meta"] else
                                   np.full(counts[-1], -1))
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
    def mask_plane_dict(self) -> dict[str, int]:
        return json.loads(self.attrs["mask_plane_dict"])

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
