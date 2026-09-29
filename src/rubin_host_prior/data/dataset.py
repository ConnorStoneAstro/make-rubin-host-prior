"""The training data loader.

Pooling happens in **flux** space and the log transform is applied afterwards.
That ordering is not interchangeable and it is the physically correct one: the
pre-noise scene averages linearly in flux, and averaging log-fluxes would
compute a geometric mean, biasing every patch low wherever there is structure.

Every batch is built from the native-resolution stamps, so translation and scale
jitter are always available -- integer native-pixel translations are exact and
give sub-output-pixel positional augmentation for free.  There was once a second
path serving a pre-pooled, pre-transformed cache; it was faster and it baked the
crop in, so it lost the translation augmentation and it was one more derived
file to fall out of step with the shards.  The shards load fast enough.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from ..config import Config
from .augment import random_dihedral
from .diagnostics import correlation_length
from .pooling import pool_to_training_grid
from .shards import ShardSet
from .transform import LogFluxTransform


class PatchDataset:
    def __init__(
        self,
        config: Config,
        transform: LogFluxTransform,
        band_idx: np.ndarray,
        shards: ShardSet | None = None,
        native: np.ndarray | None = None,
    ):
        if shards is None and native is None:
            raise ValueError("need shards or native stamps")
        self.config = config
        self.transform = transform
        self.band_idx = np.asarray(band_idx)
        self.shards = shards
        self._native = native

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
        # No per-band check any more: there is one softening scale and
        # ``from_config`` refuses a config without it, so a shard set cannot
        # contain a band the transform has no scale for.
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

    def __len__(self) -> int:
        return len(self.band_idx)

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
        context: int = 0,
    ) -> np.ndarray:
        """Pooled flux in nJy, before the log transform.

        ``context`` is the border the loss will crop away again; 0 gives the
        nominal crop alone, which is what the diagnostics want.
        """
        p = self.config.patch
        out_size = p.out_size if out_size is None else out_size
        stamps = self._native_stamps(indices)
        side = out_size + 2 * context
        out = np.empty((len(indices), side, side), dtype=np.float32)
        for i in range(len(indices)):
            out[i] = pool_to_training_grid(
                stamps[i],
                out_size=out_size,
                pool_factor=p.pool_factor,
                rng=rng,
                translate=translate,
                scale_jitter=scale_jitter,
                max_translate=self.config.max_translate_native(),
                context=context,
            )
        return out

    def _pool_and_transform(
        self,
        indices: np.ndarray,
        rng: np.random.Generator | None,
        translate: bool,
        scale_jitter: float,
        out_size: int | None = None,
        context: int = 0,
    ) -> np.ndarray:
        out = self._pool(indices, rng, translate, scale_jitter, out_size, context)
        # Pool in flux, THEN take the log.
        return self.transform.forward(out)

    def make_batch(
        self,
        indices: np.ndarray,
        rng: np.random.Generator | None = None,
        augment: bool = True,
        out_size: int | None = None,
        context: int | None = None,
    ) -> np.ndarray:
        """``(B, 1, S, S)`` float32 in the log representation, ``S = out_size +
        2 * context``.

        ``context`` defaults to the architecture's ``loss_margin``, so the loss
        lands on exactly ``out_size``.  Every pixel of it is real sky, context
        included -- the loader raises rather than reflect a shortfall, so a
        config whose reach outruns the stamp fails at the first batch instead of
        training the coarse score on a mirror symmetry.  The diagnostics pass 0,
        which needs no context at all.
        """
        aug = self.config.augment
        out_size = self.config.patch.out_size if out_size is None else out_size
        if context is None:
            context = self.config.energy.loss_margin
        x = self._pool_and_transform(
            indices,
            rng,
            translate=augment and aug.translate,
            scale_jitter=aug.scale_jitter if augment else 0.0,
            out_size=out_size,
            context=context,
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

    # -- diagnostics ------------------------------------------------------

    def correlation_length(self, n: int = 512, seed: int = 0) -> dict:
        """Structural correlation length of the training representation, in
        *pooled* pixels -- the units the loss crop is measured in.

        This is the authoritative version of the number: it is measured on the
        pooled, log-space patches the model actually sees, not on native flux.
        Compare it against the model's ``loss_margin``; see ``data.diagnostics``.
        """
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(self), size=min(n, len(self)), replace=False))
        x = self.make_batch(idx, rng=rng, augment=False, context=0)[:, 0]
        return correlation_length(x)

    def stats(self, n: int = 512, seed: int = 0) -> dict:
        """Summary of the log-space data. Read this before setting sigma_min/max."""
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(self), size=min(n, len(self)), replace=False))
        x = self.make_batch(idx, rng=rng, augment=False, context=0)[:, 0]
        # Most pixels in a patch are sky, so the 16-84 half-width is the sky
        # scatter.  Taken across all the bands at once, which is meaningful
        # again now there is a single softening scale: every band's sky sits at
        # the same log(s * log 2), so this measures the width of one peak rather
        # than the spread between six of them.  The bands do have different
        # widths about that shared level, so what this gives is a typical one.
        p16, p84 = np.percentile(x, [16, 84])
        sky_scatter = float(0.5 * (p84 - p16))
        deepest_sigma = np.nan
        flux = self._pool(idx, rng=None, translate=False, scale_jitter=0.0)
        sigma_pooled = (
            np.asarray(self.shards.meta["sky_noise"])[idx] / self.config.patch.pool_factor
        )
        good = np.isfinite(sigma_pooled) & (sigma_pooled > 0)
        if np.any(good):
            deepest_sigma = float((flux[good].min(axis=(1, 2)) / sigma_pooled[good]).min())
        return {
            "n": int(len(idx)),
            "mean": float(x.mean()),
            "std": float(x.std()),
            "percentiles": {
                str(q): float(np.percentile(x, q)) for q in (0.1, 1, 16, 50, 84, 99, 99.9)
            },
            "min": float(x.min()),
            "max": float(x.max()),
            # Compare against transform.expected_sky_scatter(softening_sigma);
            # a large disagreement means the per-band softening scales are
            # wrong.  It is a check on the width -- the per-band sky *level*
            # differs by design.
            "sky_scatter": sky_scatter,
            # Per-patch spread is what sigma_max has to cover: the reverse
            # process starts from N(0, sigma_max^2) and must be able to reach
            # the brightest structure in a single patch.
            "per_patch_range_p99": float(
                np.percentile(x.max(axis=(1, 2)) - x.min(axis=(1, 2)), 99)
            ),
            # Skew of the log-space values.  The softening is nonlinear across
            # the noise range, so sky pixels are left-skewed; that is expected,
            # not a fault, and it grows as softening_sigma falls.
            "skew": float(np.mean(((x - x.mean()) / max(x.std(), 1e-12)) ** 3)),
            # How far negative the measured flux goes, in pooled sky noise.
            # Informational only: softplus has no floor, so however deep this
            # goes the pixel is representable.
            "deepest_flux_sigma": deepest_sigma,
            # One number: with a single softening scale every band's sky sits
            # at the same log(s * log 2).
            "sky_level": self.transform.sky_level,
        }


def suggest_sigma_range(stats: dict, sky_scatter_hint: float | None = None) -> tuple[float, float]:
    """Heuristic ``(sigma_min, sigma_max)`` from ``PatchDataset.stats``.

    ``sigma_max`` must dominate the data's own scale or the ``t = 1`` marginal is
    not really Gaussian and sampling starts from the wrong distribution.
    ``sigma_min`` should sit comfortably below the sky scatter in log space,
    since below that the score is dominated by the noise the data already
    contains and there is nothing left to learn.
    """
    sigma_max = float(max(2.0 * stats["std"], stats["per_patch_range_p99"]))
    floor = sky_scatter_hint
    if floor is None:
        floor = stats.get("sky_scatter", stats["std"])
    sigma_min = float(max(1e-3, 0.05 * floor))
    return sigma_min, sigma_max


def pool_shards(
    shards: ShardSet,
    config: Config,
    n: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Nominally pooled flux (nJy) and band index, straight from shards.

    Breaks the chicken-and-egg in setting up the transform: the softening scale
    needs the pooled sky noise, the pooled sky noise needs pooled patches, and
    pooling needs no transform at all.
    """
    p = config.patch
    idx = np.arange(len(shards))
    if n is not None and n < len(idx):
        idx = np.sort(np.random.default_rng(seed).choice(len(idx), n, replace=False))
    stamps = shards.gather(idx, "image") if len(idx) < len(shards) else shards.load("image")
    out = np.empty((len(idx), p.out_size, p.out_size), dtype=np.float32)
    for i in range(len(idx)):
        out[i] = pool_to_training_grid(stamps[i], out_size=p.out_size, pool_factor=p.pool_factor)
    return out, np.asarray(shards.meta["band_idx"])[idx]
