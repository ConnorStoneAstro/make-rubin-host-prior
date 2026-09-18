"""The training data loader.

Pooling happens in **flux** space and the log transform is applied afterwards.
That ordering is not interchangeable and it is the physically correct one: the
pre-noise scene averages linearly in flux, and averaging log-fluxes would
compute a geometric mean, biasing every patch low wherever there is structure.

Two modes:

``native`` (default)
    Serves from native-resolution stamps, so translation and scale jitter are
    available.  Integer native-pixel translations are exact and give
    sub-output-pixel positional augmentation for free.

``pooled``
    Serves from a cached, already-pooled-and-transformed array.  Faster and
    smaller, but only the dihedral augmentations remain available, since the
    crop is baked in.  Use it for validation batches and for quick experiments.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from ..config import Config
from .augment import random_dihedral
from .diagnostics import correlation_length
from .pooling import pool_to_training_grid
from .shards import ShardSet
from .transform import LogFluxTransform


def cache_key(config: Config, transform: LogFluxTransform, shards: ShardSet) -> str:
    """Hash of everything that affects the cached pooled array."""
    payload = {
        "patch": asdict(config.patch),
        "transform": {
            "offsets": list(transform.offsets),
            "log_scale": transform.log_scale,
            "floor_ratio": transform.floor_ratio,
            "bands": list(transform.bands),
        },
        "shards": [p.name for p in shards.paths],
        "counts": [int(c) for c in shards.counts],
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


class PatchDataset:
    def __init__(
        self,
        config: Config,
        transform: LogFluxTransform,
        band_idx: np.ndarray,
        shards: ShardSet | None = None,
        native: np.ndarray | None = None,
        pooled: np.ndarray | None = None,
    ):
        if pooled is None and shards is None and native is None:
            raise ValueError("need shards, native stamps, or a pooled array")
        self.config = config
        self.transform = transform
        self.band_idx = np.asarray(band_idx)
        self.shards = shards
        self._native = native
        self._pooled = pooled

    # -- construction -----------------------------------------------------

    @classmethod
    def from_shards(
        cls,
        shards: ShardSet,
        config: Config,
        transform: LogFluxTransform,
        in_memory: bool | str = "auto",
        max_in_memory_gb: float = 16.0,
    ) -> "PatchDataset":
        if shards.native_size < config.patch.native_size:
            raise ValueError(
                f"shards hold {shards.native_size}-pixel stamps but the config "
                f"asks for {config.patch.native_size}"
            )
        gb = shards.nbytes("image") / 1024**3
        if in_memory == "auto":
            in_memory = gb <= max_in_memory_gb
        native = shards.load("image") if in_memory else None
        return cls(
            config=config,
            transform=transform,
            band_idx=shards.meta["band_idx"],
            shards=shards,
            native=native,
        )

    @classmethod
    def from_pooled_cache(
        cls,
        path: str | Path,
        config: Config,
        transform: LogFluxTransform,
        expect_key: str | None = None,
    ) -> "PatchDataset":
        with h5py.File(path, "r") as f:
            key = f.attrs.get("cache_key")
            if expect_key is not None and key != expect_key:
                raise ValueError(
                    f"cache at {path} was built for key {key!r} but the current "
                    f"config hashes to {expect_key!r}; rebuild it"
                )
            pooled = f["x"][:]
            band_idx = f["band_idx"][:]
        return cls(
            config=config, transform=transform, band_idx=band_idx, pooled=pooled
        )

    @property
    def mode(self) -> str:
        return "pooled" if self._pooled is not None else "native"

    def __len__(self) -> int:
        return (
            len(self._pooled) if self._pooled is not None else len(self.band_idx)
        )

    # -- batch construction ----------------------------------------------

    def _native_stamps(self, indices: np.ndarray) -> np.ndarray:
        if self._native is not None:
            return self._native[indices]
        return self.shards.gather(indices, "image")

    def _pool(
        self,
        indices: np.ndarray,
        rng: np.random.Generator | None,
        translate: bool,
        scale_jitter: float,
        out_size: int | None = None,
    ) -> np.ndarray:
        """Pooled flux in nJy, before the log transform."""
        p = self.config.patch
        out_size = p.out_size if out_size is None else out_size
        stamps = self._native_stamps(indices)
        out = np.empty((len(indices), out_size, out_size), dtype=np.float32)
        for i in range(len(indices)):
            out[i] = pool_to_training_grid(
                stamps[i],
                out_size=out_size,
                pool_factor=p.pool_factor,
                rng=rng,
                translate=translate,
                scale_jitter=scale_jitter,
                max_translate=p.max_translate_native,
            )
        return out

    def _pool_and_transform(
        self,
        indices: np.ndarray,
        rng: np.random.Generator | None,
        translate: bool,
        scale_jitter: float,
        out_size: int | None = None,
    ) -> np.ndarray:
        out = self._pool(indices, rng, translate, scale_jitter, out_size)
        # Pool in flux, THEN take the log.  The other order computes a geometric
        # mean and biases every structured patch low.
        return self.transform.forward(out, self.band_idx[indices])

    def make_batch(
        self,
        indices: np.ndarray,
        rng: np.random.Generator | None = None,
        augment: bool = True,
        out_size: int | None = None,
    ) -> np.ndarray:
        """``(B, 1, out_size, out_size)`` float32 in the log representation."""
        aug = self.config.augment
        out_size = self.config.patch.out_size if out_size is None else out_size
        if self._pooled is not None:
            x = self._pooled[indices]
            cached = x.shape[-1]
            if out_size > cached:
                raise ValueError(
                    f"pooled cache holds {cached}px images; cannot serve "
                    f"{out_size}px. Rebuild the cache or use the native loader."
                )
            if out_size < cached:
                # A sub-crop of a pooled, transformed image is exactly the
                # pooled transform of the corresponding native sub-region:
                # pooling is local and the transform is pointwise.  So the
                # cache serves every size at or below the one it was built at.
                room = cached - out_size
                if augment and aug.translate and rng is not None:
                    y0, x0 = rng.integers(0, room + 1, size=2)
                else:
                    y0 = x0 = room // 2
                x = x[:, y0 : y0 + out_size, x0 : x0 + out_size]
        else:
            x = self._pool_and_transform(
                indices,
                rng,
                translate=augment and aug.translate,
                scale_jitter=aug.scale_jitter if augment else 0.0,
                out_size=out_size,
            )
        if augment and aug.dihedral:
            if rng is None:
                raise ValueError("dihedral augmentation requires an rng")
            x = random_dihedral(x, rng)
        return np.ascontiguousarray(x[:, None].astype(np.float32))

    def batches(
        self,
        batch_size: int,
        seed: int = 0,
        shuffle: bool = True,
        sizes: tuple[int, ...] | None = None,
    ) -> Iterator[np.ndarray]:
        """Infinite stream of augmented batches, reshuffled every epoch.

        A batch is shape-homogeneous (JAX needs that), so when several sizes are
        configured they are cycled round-robin across batches -- deterministic,
        equal coverage, and a predictable number of jit compilations (one per
        distinct size).
        """
        rng = np.random.default_rng(seed)
        sizes = tuple(sizes) if sizes else self.config.patch.training_sizes
        n = len(self)
        if n < batch_size:
            raise ValueError(f"{n} patches is fewer than batch_size {batch_size}")
        k = 0
        while True:
            order = rng.permutation(n) if shuffle else np.arange(n)
            for start in range(0, n - batch_size + 1, batch_size):
                yield self.make_batch(
                    order[start : start + batch_size],
                    rng,
                    out_size=sizes[k % len(sizes)],
                )
                k += 1

    def validation_batch(self, n: int, seed: int = 12345) -> np.ndarray:
        """Fixed, un-augmented, nominally pooled batch -- comparable across runs."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self), size=min(n, len(self)), replace=False)
        return self.make_batch(np.sort(idx), rng=rng, augment=False)

    # -- cache ------------------------------------------------------------

    def build_pooled_cache(
        self, path: str | Path, chunk: int = 512
    ) -> Path:
        """Write the nominal pooled + transformed array to ``path``."""
        if self._pooled is not None:
            raise ValueError("already serving from a pooled cache")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        key = cache_key(self.config, self.transform, self.shards)
        p = self.config.patch
        n = len(self)
        with h5py.File(path, "w") as f:
            f.attrs["cache_key"] = key
            f.attrs["config"] = json.dumps(self.config.to_dict(), sort_keys=True)
            dset = f.create_dataset(
                "x", shape=(n, p.out_size, p.out_size), dtype="f4"
            )
            f.create_dataset("band_idx", data=self.band_idx)
            for start in range(0, n, chunk):
                idx = np.arange(start, min(start + chunk, n))
                dset[start : start + len(idx)] = self._pool_and_transform(
                    idx, rng=None, translate=False, scale_jitter=0.0
                )
        return path

    # -- diagnostics ------------------------------------------------------

    def flux_headroom(self, n: int = 512, seed: int = 0) -> dict:
        """How far negative the pooled flux goes, in units of pooled sky noise.

        This is what sets ``k_sigma``.  The transform's offset
        ``b_band = k_sigma * sigma_pooled`` is a hard bound on representable
        flux -- the model spans ``(-b_band, +inf)`` and nothing below -- and the
        clip at ``floor_ratio`` bites at ``-0.9 * k_sigma`` sigma.  So
        ``k_sigma`` must exceed the deepest negative excursion you intend to
        keep, and if you are keeping background-subtraction artefacts (dark
        haloes, dark edges) rather than gating them out, that is deeper than the
        noise alone would suggest.

        Note the asymmetry that makes this bind: pooling divides the *noise* by
        ``pool_factor`` but leaves a smooth negative offset untouched, so in
        pooled-sigma units an over-subtracted region is ``pool_factor`` times
        deeper than it was natively.
        """
        if self._pooled is not None:
            raise ValueError(
                "flux_headroom needs native stamps; the pooled cache stores the "
                "already-transformed, already-clipped representation. Build the "
                "dataset from shards instead."
            )
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(self), size=min(n, len(self)), replace=False))
        flux = self._pool(idx, rng=None, translate=False, scale_jitter=0.0)
        sigma_pooled = (
            np.asarray(self.shards.meta["sky_noise"])[idx] / self.config.patch.pool_factor
        )
        good = np.isfinite(sigma_pooled) & (sigma_pooled > 0)
        if not np.any(good):
            return {"n": 0}
        worst = flux[good].min(axis=(1, 2)) / sigma_pooled[good]
        pcts = {str(q): float(np.percentile(worst, q)) for q in (0.1, 1, 5, 50)}
        # The clip sits at -0.9 * k_sigma, so covering a depth D needs
        # k_sigma > D / 0.9.
        needed = float(abs(np.percentile(worst, 0.1)) / 0.9)
        return {
            "n": int(good.sum()),
            "min_flux_sigma_percentiles": pcts,
            "deepest": float(worst.min()),
            "k_sigma_current": float(self.config.transform.k_sigma),
            "k_sigma_needed": needed,
        }

    def correlation_length(self, n: int = 512, seed: int = 0) -> dict:
        """Structural correlation length of the training representation, in
        *pooled* pixels -- the units the loss crop is measured in.

        This is the authoritative version of the number: it is measured on the
        pooled, log-space patches the model actually sees, not on native flux.
        Compare it against the model's ``loss_margin``; see ``data.diagnostics``.
        """
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(self), size=min(n, len(self)), replace=False))
        x = self.make_batch(idx, rng=rng, augment=False)[:, 0]
        return correlation_length(x)

    def stats(self, n: int = 512, seed: int = 0) -> dict:
        """Summary of the log-space data. Read this before setting sigma_min/max."""
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(self), size=min(n, len(self)), replace=False))
        x = self.make_batch(idx, rng=rng, augment=False)[:, 0]
        p16, p84 = np.percentile(x, [16, 84])
        return {
            "n": int(len(idx)),
            "mean": float(x.mean()),
            "std": float(x.std()),
            "percentiles": {
                str(q): float(np.percentile(x, q))
                for q in (0.1, 1, 16, 50, 84, 99, 99.9)
            },
            "min": float(x.min()),
            "max": float(x.max()),
            # Most pixels in a patch are sky, so the 16-84 half-width measures
            # the sky scatter in log space.  It should come out near
            # 1 / k_sigma; if it does not, the band offsets are wrong.
            "sky_scatter": float(0.5 * (p84 - p16)),
            # Per-patch spread is what sigma_max has to cover: the reverse
            # process starts from N(0, sigma_max^2) and must be able to reach
            # the brightest structure in a single patch.
            "per_patch_range_p99": float(
                np.percentile(x.max(axis=(1, 2)) - x.min(axis=(1, 2)), 99)
            ),
            # Measured on x, not on the inverted flux: expm1(log1p(.)) is only
            # accurate to ~1e-8, so a comparison against floor_ratio downstream
            # of the round trip misses the pixels that were actually clipped.
            "clipped_fraction": float(
                np.mean(x <= self.transform.x_floor + 1e-6)
            ),
            "x_floor": self.transform.x_floor,
        }


def suggest_sigma_range(stats: dict, sky_scatter_hint: float | None = None) -> tuple[float, float]:
    """Heuristic ``(sigma_min, sigma_max)`` from ``PatchDataset.stats``.

    ``sigma_max`` must dominate the data's own scale or the ``t = 1`` marginal is
    not really Gaussian and sampling starts from the wrong distribution.
    ``sigma_min`` should sit comfortably below the sky scatter in log space
    (``~ 1 / k_sigma``), since below that the score is dominated by the noise the
    data already contains and there is nothing left to learn.
    """
    sigma_max = float(max(2.0 * stats["std"], stats["per_patch_range_p99"]))
    floor = sky_scatter_hint
    if floor is None:
        floor = stats.get("sky_scatter", stats["std"])
    sigma_min = float(max(1e-3, 0.05 * floor))
    return sigma_min, sigma_max
