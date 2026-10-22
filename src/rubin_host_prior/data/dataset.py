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
from .repair import fill_batch
from .shards import ShardSet
from .transform import LogFluxTransform


def usable_rows(shards: ShardSet, config: Config) -> np.ndarray:
    """Shard rows that survive the no-data cut: ``frac_no_data <= max_no_data``.

    **The metadata is the only honest handle on these.**  A no-data pixel in
    DP2 carries no sentinel -- measured across 1093 flagged stamps, not one is
    exactly zero or non-finite, and the values are ordinary negatives between
    -5 and -18 sigma that overlap the real sky's own tail.  Nothing in the
    image separates them, so there is nothing to detect and nothing to repair;
    what there is, is a per-stamp fraction recorded at extraction, and a cut on
    it is exact.

    Note what this does **not** catch.  ``frac_no_data`` comes from the
    inf-variance plane, which marks pixels where *zero* exposures contributed.
    A star that saturates in only some exposures still has the rest, so its
    variance is finite and its ``frac_no_data`` is zero -- while its core is
    built from the subset that did not saturate, which is the worse-seeing one,
    so the peak reads low.  That is a different defect with a different flag:
    ``SATURATED``, which extraction records as ``diag_frac_SATURATED`` in
    ``manifest.parquet`` and not in the shards.
    """
    frac = np.asarray(shards.meta["frac_no_data"], dtype=float)
    if config.patch.max_no_data < 0:
        return np.arange(len(frac))
    # -1 is the fill for a column an older shard schema never wrote.  Treated
    # as unknown rather than as a pass: the point of this cut is exactness, and
    # a silent pass would be the opposite.
    unknown = ~np.isfinite(frac) | (frac < 0)
    if unknown.any():
        raise ValueError(
            f"{int(unknown.sum())} of {len(frac)} shard rows carry no "
            f"frac_no_data, so the no-data cut cannot be applied to them. "
            f"These shards predate the column: re-extract them, or set "
            f"patch.max_no_data below zero to disable the cut."
        )
    return np.flatnonzero(frac <= config.patch.max_no_data)


class PatchDataset:
    def __init__(
        self,
        config: Config,
        transform: LogFluxTransform,
        band_idx: np.ndarray,
        shards: ShardSet | None = None,
        native: np.ndarray | None = None,
        pool_shards: int = 8,
        pool_refill_every: int = 64,
        prefetch_depth: int = 3,
    ):
        if shards is None and native is None:
            raise ValueError("need shards or native stamps")
        self.config = config
        self.transform = transform
        self.band_idx = np.asarray(band_idx)
        self.shards = shards
        self._native = native
        #: Machine properties, not config fields -- the same run on a node with
        #: more RAM should hold more shards, exactly as ``--devices`` and
        #: ``--max-in-memory-gb`` are the machine and not the model.
        self.pool_shards = pool_shards
        self.pool_refill_every = pool_refill_every
        self.prefetch_depth = prefetch_depth
        self._shard_pool = None
        #: Global shard rows this dataset presents, after the no-data cut.  All
        #: indexing goes through here, so a dataset index is a position in this
        #: array and never a raw shard row -- which is what keeps the cut from
        #: having to be repeated at every call site.
        self._rows = usable_rows(shards, config) if shards is not None else None
        #: What the last batch's no-data fill touched.  A report rather than
        #: state: if a future release writes its holes with a value the
        #: detector does not know, every count here stays zero, and that is
        #: visible instead of invisible.
        self._last_fill: dict = {"n_stamps": 0, "n_stamps_filled": 0,
                                 "n_pixels_filled": 0, "largest_hole_px": 0}

    @property
    def in_memory(self) -> bool:
        """Whether the whole set is cached, or served from a pool of shards.

        Cached, a batch is a numpy fancy-index and every patch is equally
        available.  Otherwise ``batches`` draws from ``ShardPool`` -- whole
        shards held in RAM, one replaced at a time by a sequential read -- which
        costs a fraction of the RAM and, measured, about the same per step.
        What is *not* an option any more is a row-by-row read of the whole set
        per batch: on a parallel filesystem that measured 22x the cost of
        reading a shard in one pass, and it silently became the fallback the
        moment a campaign grew past ``max_in_memory_gb``.
        """
        return self._native is not None

    def storage_note(self, max_in_memory_gb: float | None = None) -> str:
        """One line on where batches come from."""
        gb = self.shards.nbytes("image") / 1024**3 if self.shards else 0.0
        if self.in_memory:
            return (f"dataset: {gb:.2f} GiB of stamps cached in RAM, "
                    f"prefetch depth {self.prefetch_depth}")
        over = (f", over the {max_in_memory_gb:g} GiB cap"
                if max_in_memory_gb is not None else "")
        held = min(self.pool_shards, len(self.shards.paths))
        per = self.shards.nbytes("image") / max(len(self.shards.paths), 1) / 1024**3
        return (
            f"dataset: {gb:.2f} GiB of stamps in {len(self.shards.paths)} "
            f"shards{over}; batches come from a pool of {held} resident "
            f"(~{held * per:.1f} GiB), one replaced every "
            f"{self.pool_refill_every} batches by a background sequential read, "
            f"prefetch depth {self.prefetch_depth}. "
            f"--max-in-memory-gb {max(gb + 2, 2):.0f} would cache it outright."
        )

    # -- construction -----------------------------------------------------

    @classmethod
    def from_shards(
        cls,
        shards: ShardSet,
        config: Config,
        transform: LogFluxTransform,
        in_memory: bool | str = "auto",
        max_in_memory_gb: float = 16.0,
        pool_shards: int = 8,
        pool_refill_every: int = 64,
        prefetch_depth: int = 3,
    ) -> "PatchDataset":
        if shards.native_size < config.patch.native_size:
            raise ValueError(
                f"shards hold {shards.native_size}-pixel stamps but the config "
                f"asks for {config.patch.native_size}"
            )
        # No per-band check any more: there is one softening scale and
        # ``from_config`` refuses a config without it, so a shard set cannot
        # contain a band the transform has no scale for.
        # A silent threshold, until it was not: the same config on 2,051
        # patches cached and on 30,737 streamed, and nothing in the startup
        # output said which.  ``storage_note`` now does.
        gb = shards.nbytes("image") / 1024**3
        if in_memory == "auto":
            in_memory = gb <= max_in_memory_gb
        native = shards.load("image") if in_memory else None
        return cls(
            config=config,
            transform=transform,
            # Cut to match: `band_idx` is indexed by *dataset* position and
            # `__len__` reads its length, so handing over the uncut array would
            # put every index out by however many rows the cut removed.
            band_idx=np.asarray(shards.meta["band_idx"])[usable_rows(shards, config)],
            shards=shards,
            native=native,
            pool_shards=pool_shards,
            pool_refill_every=pool_refill_every,
            prefetch_depth=prefetch_depth,
        )

    def __len__(self) -> int:
        return len(self.band_idx)

    @property
    def n_rejected(self) -> int:
        """Shard rows the no-data cut removed.  Reported, never silent."""
        if self._rows is None or self.shards is None:
            return 0
        return len(self.shards.meta["band_idx"]) - len(self._rows)

    # -- batch construction ----------------------------------------------

    def _rows_for(self, indices: np.ndarray) -> np.ndarray:
        """Dataset indices -> shard rows, through the no-data cut."""
        if self._rows is None:
            return np.asarray(indices)
        return self._rows[np.asarray(indices)]

    def _native_stamps(self, indices: np.ndarray) -> np.ndarray:
        rows = self._rows_for(indices)
        if self._native is not None:
            return self._native[rows]
        return self.shards.gather(rows, "image")

    def _sky_noise(self, indices: np.ndarray) -> np.ndarray:
        """Per-stamp native sky noise, nJy, for the rows named by ``indices``."""
        return np.asarray(self.shards.meta["sky_noise"],
                          dtype=np.float32)[self._rows_for(indices)]

    def sky_floor(
        self, sky_noise: np.ndarray, rng: np.random.Generator | None
    ) -> np.ndarray | float:
        """The per-stamp floor in nJy: ``k * pooled sigma``, ``k`` drawn per stamp.

        ``sky_noise`` is the native-resolution number the shards carry, so it is
        divided by ``pool_factor`` and corrected by the measured
        ``pooled_noise_ratio`` -- coadd noise is correlated, so pooling reduces
        it by less than ``pool_factor`` and the uncorrected figure would put the
        floor too low.

        Returns ``0.0`` when floors are switched off, which is exactly the
        pre-floor transform.  Without an rng -- the unaugmented path, which has
        to be reproducible -- ``k`` is the midpoint rather than a draw, and the
        per-stamp variation comes from each cutout's own sky noise alone.
        """
        t = self.config.transform
        if not t.floors_enabled:
            return 0.0
        ratio = t.pooled_noise_ratio
        if ratio is None:
            raise ValueError(
                "the config has no transform.pooled_noise_ratio, so a native "
                "sky_noise cannot be turned into the pooled one the floor "
                "needs. Run scripts/prepare_config.py on the shards; it "
                "measures the pooled noise and the ratio together."
            )
        sigma = np.asarray(sky_noise, dtype=np.float64) / self.config.patch.pool_factor
        sigma = sigma * float(ratio)
        if rng is None or t.floor_sigma_hi == t.floor_sigma_lo:
            k = 0.5 * (t.floor_sigma_lo + t.floor_sigma_hi)
        else:
            k = rng.uniform(t.floor_sigma_lo, t.floor_sigma_hi, size=sigma.shape)
        return (k * sigma)[:, None, None]

    def _pool_stamps(
        self,
        stamps: np.ndarray,
        rng: np.random.Generator | None,
        translate: bool,
        scale_jitter: float,
        out_size: int | None = None,
    ) -> np.ndarray:
        """Pooled flux in nJy, before the log transform.

        Takes stamps rather than indices, because the shard pool has native
        stamps to hand and no global index to offer: it holds whole shards and
        draws rows out of them.
        """
        p = self.config.patch
        out_size = p.out_size if out_size is None else out_size
        if self.config.transform.fill_no_data:
            # Before pooling, not after: a dead pixel averaged into its
            # neighbours would smear the hole over the whole pooled cell rather
            # than being replaced by what belongs there.
            stamps, self._last_fill = fill_batch(stamps)
        out = np.empty((len(stamps), out_size, out_size), dtype=np.float32)
        for i in range(len(stamps)):
            out[i] = pool_to_training_grid(
                stamps[i],
                out_size=out_size,
                pool_factor=p.pool_factor,
                rng=rng,
                translate=translate,
                scale_jitter=scale_jitter,
                max_translate=p.shift_bounds_native[1],
                min_translate=p.shift_bounds_native[0],
            )
        return out

    def to_log_space(
        self,
        pooled: np.ndarray,
        sky_noise: np.ndarray | None,
        rng: np.random.Generator | None,
        augment: bool,
    ) -> tuple[np.ndarray, np.ndarray | float]:
        """Pooled flux -> ``x``, and the floor that was used.  ``(x, floor)``.

        **The one place the floor is applied.**  Everything that needs to show
        or reproduce the training representation goes through here -- the
        loader and ``plots.plot_transform`` both -- because a figure that
        reimplemented the chain is exactly how the transform figure came to be
        drawing an unfloored map of a floored dataset.

        The ``k`` draw is an *augmentation*, so it follows ``augment`` and not
        merely whether an rng exists: ``validation_batch`` and the diagnostic
        figures pass an rng with augmentation off and have to be reproducible.
        Each stamp still floors at its own level either way -- that comes from
        its own sky noise -- it is only the jitter on ``k`` that freezes.
        """
        floor = (0.0 if sky_noise is None
                 else self.sky_floor(sky_noise, rng if augment else None))
        return self.transform.forward(pooled, floor), floor

    def batch_from_stamps(
        self,
        stamps: np.ndarray,
        rng: np.random.Generator | None = None,
        augment: bool = True,
        out_size: int | None = None,
        sky_noise: np.ndarray | None = None,
    ) -> np.ndarray:
        """``(B, 1, S, S)`` float32 in the log representation, from raw stamps.

        Everything ``make_batch`` does except deciding which stamps: crop and
        pool, take the log, then the dihedral draw.  Split out so the shard pool
        can hand its own rows in -- it holds whole shards, so it has stamps and
        not global indices.
        """
        aug = self.config.augment
        out_size = self.config.patch.out_size if out_size is None else out_size
        x = self._pool_stamps(
            stamps,
            rng,
            translate=augment and aug.translate,
            scale_jitter=aug.scale_jitter if augment else 0.0,
            out_size=out_size,
        )
        x, _ = self.to_log_space(x, sky_noise, rng, augment)
        if augment and aug.dihedral:
            if rng is None:
                raise ValueError("dihedral augmentation requires an rng")
            x = random_dihedral(x, rng)
        return np.ascontiguousarray(x[:, None].astype(np.float32))

    def make_batch(
        self,
        indices: np.ndarray,
        rng: np.random.Generator | None = None,
        augment: bool = True,
        out_size: int | None = None,
    ) -> np.ndarray:
        """``(B, 1, S, S)`` float32 in the log representation, ``S = out_size``.

        By global index, which means a read per patch when the set is not
        cached -- right for the diagnostics and for ``validation_batch``, which
        need particular patches, and wrong for training, which needs a great
        many and does not care which.  ``batches`` uses the pool instead.
        """
        return self.batch_from_stamps(
            self._native_stamps(indices), rng, augment, out_size,
            sky_noise=self._sky_noise(indices))

    def pool(self) -> "ShardPool":
        """The resident-shard pool, built on first use.

        Lazy because it costs ``n_resident`` whole-shard reads and a thread, and
        most things that hold a dataset -- the diagnostics, ``prepare_config``,
        a validation batch -- never ask for a training stream.
        """
        from .pool import ShardPool

        if self._shard_pool is None:
            if self.shards is None:
                raise ValueError("a pool needs shards to read from")
            self._shard_pool = ShardPool(
                self.shards,
                n_resident=self.pool_shards,
                refill_every=self.pool_refill_every,
                seed=self.config.train.seed,
                max_no_data=self.config.patch.max_no_data,
            )
        return self._shard_pool

    def _raw_batches(
        self,
        batch_size: int,
        seed: int,
        shuffle: bool,
        sizes: tuple[int, ...],
    ) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(seed)
        k = 0
        if self._native is None:
            # Not cached: draw from whole shards held in RAM rather than reading
            # scattered rows, which measured 22x slower per row on a parallel
            # filesystem.  There is no epoch here -- residency is the shuffle.
            pool = self.pool()
            while True:
                stamps, noise = pool.draw(rng, batch_size)
                yield self.batch_from_stamps(
                    stamps, rng, out_size=sizes[k % len(sizes)],
                    sky_noise=noise)
                k += 1
        n = len(self)
        if n < batch_size:
            raise ValueError(f"{n} patches is fewer than batch_size {batch_size}")
        while True:
            order = rng.permutation(n) if shuffle else np.arange(n)
            for start in range(0, n - batch_size + 1, batch_size):
                yield self.make_batch(
                    order[start : start + batch_size],
                    rng,
                    out_size=sizes[k % len(sizes)],
                )
                k += 1

    def batches(
        self,
        batch_size: int,
        seed: int = 0,
        shuffle: bool = True,
        sizes: tuple[int, ...] | None = None,
        prefetch: int | None = None,
    ) -> Iterator[np.ndarray]:
        """Infinite stream of augmented batches, produced on a background thread.

        A batch is shape-homogeneous (JAX needs that), so when several sizes are
        configured they are cycled round-robin across batches -- deterministic,
        equal coverage, and a predictable number of jit compilations (one per
        distinct size).

        **The thread is not an optimisation of the I/O alone.**  Building a
        batch is 154 ms of pooling and transform against a 234 ms step, so even
        a fully cached run spends 40% of its wall clock with the GPU idle.
        Both numbers are numpy and h5py, which release the GIL over the work,
        so the overlap is real.  ``prefetch=0`` runs inline, which is what the
        determinism tests want.
        """
        from .pool import prefetch as _prefetch

        sizes = tuple(sizes) if sizes else self.config.patch.training_sizes
        raw = self._raw_batches(batch_size, seed, shuffle, sizes)
        depth = self.prefetch_depth if prefetch is None else prefetch
        return _prefetch(raw, depth) if depth else raw

    def close(self) -> None:
        """Stop the pool's reader thread.  Idempotent; safe if there is no pool."""
        if self._shard_pool is not None:
            self._shard_pool.close()
            self._shard_pool = None

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
        x = self.make_batch(idx, rng=rng, augment=False)[:, 0]
        return correlation_length(x)

    def pair_distances(self, n: int = 512, seed: int = 0) -> dict:
        """Euclidean distances between random pairs of training images.

        **The standard way to set ``sigma_max``.**  Song & Ermon (2020),
        *Improved Techniques for Training Score-Based Generative Models*,
        Technique 1: take ``sigma_max`` to be the maximum pairwise distance
        between training points.  The argument is about mode coverage.  The
        reverse process starts at ``t = 1`` from ``N(data, sigma_max^2 I)``, and
        for that start to be able to reach *any* mode the noise has to be able
        to carry a sample from one datum to another.  If it cannot, the chain
        is committed to whichever basin it was initialised near before it has
        taken a single step, and the model's large-scale structure is decided
        by the initial draw rather than by the score.

        **Why a per-pixel sigma is compared against a whole-image distance.**
        It looks like a unit error and it is not.  Take two scenes separated by
        ``Delta``, ``||Delta|| = d``, and perturb one of them: ``y = x_a +
        sigma z`` with ``z ~ N(0, I_D)``.  The log-likelihood ratio between the
        two components at ``y`` is

            log N(y|x_b) - log N(y|x_a)
                = -<z, Delta> / sigma - d^2 / (2 sigma^2)

        and ``<z, Delta> ~ N(0, d^2)``, because an isotropic unit normal
        projected onto any one direction has unit variance however many
        directions there are.  So the ratio is distributed as
        ``N(-r^2/2, r^2)`` with ``r = d / sigma`` -- **a function of ``d/sigma``
        alone, with no ``D`` in it at all.**  The noise is isotropic in ``D``
        dimensions, but only its one-dimensional projection onto the separation
        can tell the two scenes apart, and that projection has spread ``sigma``
        whatever ``D`` is.  Mixing needs the ratio to be ``O(1)``, i.e.
        ``sigma >~ d``, and ``sigma_max ~ d_max`` is the exact criterion rather
        than a loose bound.  ``tests/test_data.py`` pins the independence.

        Where ``D`` *does* enter is how big ``d`` is for a given per-pixel
        spread: ``d ~ sqrt(2 D) s`` if the pixels are independent, so a 128x128
        patch multiplies per-pixel intuition by ~180.  That is why a
        ``sigma_max`` read off a histogram of pixel values is short, and why
        the same factor cannot be recovered by taking "a multiple of the
        brightest pixel" unless the multiple is itself ``~sqrt(D)``.  Real
        scenes are correlated -- see ``correlation_length`` -- so the effective
        count of independent directions is well below ``D`` and ``d`` falls
        short of ``sqrt(2 D) s``.  Measuring it is the point: the correlations
        are exactly what a dimensional argument cannot supply.

        Reported alongside is the same distance divided by ``sqrt(D)`` -- the
        per-pixel rms difference between two scenes -- because that is the
        number that can be compared against ``data_std`` and the sky scatter,
        and keeping both in view is what stops the two being confused.

        One caveat worth keeping in view: Technique 1 models the data as an
        empirical set of isolated points, and ``d_max`` is then the width of
        the whole set.  Galaxy scenes are a connected distribution rather than
        a handful of Diracs, so the strict maximum is the conservative end of
        the range and a high percentile is defensible; ``p99`` is reported for
        that reason.  Going lower than that is where the mode-coverage
        argument starts to bite.
        """
        rng = np.random.default_rng(seed)
        k = min(n, len(self))
        idx = np.sort(rng.choice(len(self), size=k, replace=False))
        x = self.make_batch(idx, rng=rng, augment=False)[:, 0]
        flat = x.reshape(len(x), -1).astype(np.float64)
        dim = flat.shape[1]

        # Centred first.  ``x`` is absolute log flux, so it sits near +3 and
        # ||x||^2 is ~50x the squared distances being read off it; the Gram
        # identity below is a difference of those, and subtracting the mean --
        # which moves no pair relative to another -- is what keeps the
        # cancellation from eating the answer.
        flat = flat - flat.mean()

        # Every unordered pair of the sample, not a random subset of them: at
        # n = 512 that is 130k distances for the price of one matrix, and the
        # maximum is the statistic most sensitive to how many pairs you look
        # at, so it is worth taking all of them.
        sq = np.sum(flat**2, axis=1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (flat @ flat.T)
        iu = np.triu_indices(len(flat), k=1)
        d = np.sqrt(np.maximum(d2[iu], 0.0))
        return {
            "n_patches": int(k),
            "n_pairs": int(d.size),
            "dim": int(dim),
            "mean": float(d.mean()),
            "std": float(d.std()),
            "max": float(d.max()),
            "min": float(d.min()),
            "percentiles": {
                str(q): float(np.percentile(d, q)) for q in (50, 90, 99)
            },
            # Per-pixel rms difference: distance / sqrt(D).  Comparable with
            # data_std and sky_scatter; the raw distance is not.
            "max_per_pixel": float(d.max() / np.sqrt(dim)),
            "mean_per_pixel": float(d.mean() / np.sqrt(dim)),
            # The largest value the loader emits, which is the other thing
            # sigma has to be able to move across.
            "x_max": float(x.max()),
            "x_min": float(x.min()),
        }

    def stats(self, n: int = 512, seed: int = 0) -> dict:
        """Summary of the log-space data. Read this before setting sigma_min/max."""
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(self), size=min(n, len(self)), replace=False))
        x = self.make_batch(idx, rng=rng, augment=False)[:, 0]
        # Most pixels in a patch are sky, so the 16-84 half-width is the sky
        # scatter.  Taken across all the bands at once, which is meaningful
        # again now there is a single softening scale: every band's sky sits at
        # the same log(s * log 2), so this measures the width of one peak rather
        # than the spread between six of them.  The bands do have different
        # widths about that shared level, so what this gives is a typical one.
        p16, p84 = np.percentile(x, [16, 84])
        sky_scatter = float(0.5 * (p84 - p16))
        deepest_sigma = np.nan
        flux = self._pool_stamps(self._native_stamps(idx), rng=None,
                                 translate=False, scale_jitter=0.0)
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


def suggest_sigma_range(
    stats: dict,
    pairs: dict | None = None,
    sky_scatter_hint: float | None = None,
    rule: str = "pairs",
) -> tuple[float, float]:
    """``(sigma_min, sigma_max)`` from ``PatchDataset.stats``.

    ``sigma_min`` sits comfortably below the sky scatter in log space, since
    below that the score is dominated by the noise the data already contains
    and there is nothing left to learn.  That end has never been in doubt.

    ``sigma_max`` is **the largest distance between two training images**:
    Song & Ermon (2020) Technique 1, which wants ``pairs`` from
    ``PatchDataset.pair_distances``.  That is the condition for the ``t = 1``
    marginal to bridge between modes, and ``pair_distances`` carries the
    derivation -- including why a per-pixel ``sigma`` is legitimately compared
    against a whole-image distance, which looks like a unit error and is not.

    ``rule="per-pixel"`` returns the old ``max(2 std, per-patch range)``.  It is
    kept only to rebuild a config written before this, and it is wrong: it
    compares ``sigma``, which is added to every pixel independently, against
    the spread of a *single* pixel.  It therefore misses the whole ``sqrt(D)``
    between a per-pixel scale and a whole-image one -- a factor of 181 on a
    128x128 grid -- and the symptom is a model that reproduces texture and
    small sources while **omitting structure above some size entirely**.  That
    is the mechanism in ``plots.schedule_headroom``: a Fourier mode of power
    ``P`` survives to ``t = 1`` whenever ``P > sigma_max^2``, and a mode that
    survives is one the sampler is never asked to build, because
    ``VESDE.prior_sample`` starts from white noise that does not contain it.
    The deficit is a floor of ``1/sqrt(1 + P/sigma_max^2)`` on that mode's rms
    and no amount of training removes it.  Coarse scales hold the most power
    per mode, so they go first -- big objects, missing.

    There is deliberately **no silent fallback**: asking for the default rule
    without the measurement raises rather than quietly returning the number
    this function used to return.
    """
    if rule not in ("pairs", "per-pixel"):
        raise ValueError(
            f"rule must be 'pairs' or 'per-pixel', got {rule!r}")
    if rule == "pairs":
        if pairs is None:
            raise ValueError(
                "rule='pairs' needs a PatchDataset.pair_distances() result. "
                "Pass pairs=dataset.pair_distances(), or ask for "
                "rule='per-pixel' to reproduce a config written before the "
                "pairwise rule -- which under-sets sigma_max by ~sqrt(D) and "
                "drops structure above some size from the samples."
            )
        sigma_max = float(pairs["max"])
    else:
        sigma_max = float(max(2.0 * stats["std"], stats["per_patch_range_p99"]))
    floor = sky_scatter_hint
    if floor is None:
        floor = stats.get("sky_scatter", stats["std"])
    sigma_min = float(max(1e-3, 0.05 * floor))
    return sigma_min, sigma_max


def suggest_n_steps(
    dim: int, sigma_min: float, sigma_max: float, overlap: float = 0.5
) -> dict:
    """The fewest sampler steps the schedule can be crossed in.

    Song & Ermon (2020) Technique 2, the companion to the ``sigma_max`` rule in
    ``suggest_sigma_range``.  Two adjacent noise levels in a geometric ladder
    differ by a ratio ``gamma``, and for a sample annealed down the ladder to
    stay in the typical set of the next level the two marginals have to
    overlap.  In ``D`` dimensions the perturbed data concentrates on a thin
    shell of radius ``sigma sqrt(D)``, and the shells for ``sigma`` and
    ``gamma sigma`` overlap in the fraction

        Phi(sqrt(D) (gamma - 1) + 3 gamma) - Phi(sqrt(D) (gamma - 1) - 3 gamma)

    which the paper asks to be about ``1/2``.  Solving it for ``gamma`` and
    taking ``log(sigma_max / sigma_min) / log(gamma)`` gives the step count.

    The ``sqrt(D)`` is the same factor that makes ``sigma_max`` large, acting
    the other way: high dimension makes the shells thin, so the ladder has to
    be fine.  At a 128x128 grid ``gamma`` comes out near 1.02, which is a few
    hundred steps across a realistic range -- more than a sampler default of
    128 or 256 supplies, and a sampler that takes too few steps produces
    exactly the blurred, structureless draws that a too-small ``sigma_max``
    does.  The two failures look alike and this separates them.
    """
    from math import erf, log, sqrt

    root_d = sqrt(float(dim))
    phi = lambda z: 0.5 * (1.0 + erf(z / sqrt(2.0)))
    mass = lambda g: phi(root_d * (g - 1.0) + 3.0 * g) - phi(
        root_d * (g - 1.0) - 3.0 * g)

    # mass is 1 at gamma = 1 and falls monotonically; bisect for the crossing.
    lo, hi = 1.0, 2.0
    while mass(hi) > overlap and hi < 1e6:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mass(mid) > overlap:
            lo = mid
        else:
            hi = mid
    gamma = 0.5 * (lo + hi)
    span = log(max(sigma_max, 1e-30) / max(sigma_min, 1e-30))
    n = span / log(gamma) if gamma > 1.0 else float("inf")
    return {"gamma": float(gamma), "n_steps": int(np.ceil(max(n, 1.0))),
            "overlap": float(overlap), "dim": int(dim)}


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
