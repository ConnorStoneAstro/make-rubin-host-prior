#!/usr/bin/env python
"""Time the loader's I/O paths against real shards, on the machine that matters.

    python scripts/bench_loader.py --shards /path/to/shards

Everything here is filesystem-dependent, so numbers from a laptop say nothing
about a parallel filesystem and vice versa.  Run it on a compute node of the
machine you train on, against the shards you train from.

It decomposes a batch into the pieces that a different sampling strategy would
change:

* **open**            one ``h5py.File`` open+close.  Whether this or the reads
                      dominate is what decides if "one random shard per batch"
                      is worth anything: that strategy removes opens and keeps
                      the scattered reads.
* **scattered/all**   what happens now -- ``batch_size`` random global indices,
                      which touch nearly every shard of a large set.
* **scattered/one**   ``batch_size`` random indices inside ONE shard.
* **sequential**      one whole shard read in a single pass, which is what a
                      resident pool would do every N steps instead.
* **make_batch**      the full path from cache, so the CPU cost of pooling and
                      the transform is separated from the I/O.

**Page cache is the trap.**  A shard read once is in the kernel's cache and the
second read is free, which flatters every strategy that reuses a shard.  So each
repeat uses a *different* shard, and ``--rounds 2`` re-measures the same shards
to show you how big the caching effect is on this filesystem.  If round 2 is
much faster than round 1, the node is caching and a long run may behave like
round 2 -- or may not, if the set is larger than the cache.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rubin_host_prior.config import Config  # noqa: E402
from rubin_host_prior.data import (  # noqa: E402
    LogFluxTransform,
    PatchDataset,
    ShardSet,
    estimate_softening,
    pool_shards,
)


def _time(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return time.perf_counter() - t0, out


def _fmt(seconds: float) -> str:
    return f"{seconds * 1e3:8.1f} ms" if seconds < 1 else f"{seconds:8.2f} s "


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", required=True, help="directory of *.h5 shards")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--repeats", type=int, default=8,
                   help="distinct shards to sample per measurement")
    p.add_argument("--rounds", type=int, default=2,
                   help="passes over the same shards; round 2 shows page cache")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool-shards", type=int, default=8)
    p.add_argument("--pool-refill-every", type=int, default=64)
    p.add_argument("--pool-batches", type=int, default=0,
                   help="also time this many batches through the real loader "
                        "in pool mode, which is what training will do")
    return p


def main() -> None:
    args = parser().parse_args()
    shards = ShardSet.from_dir(args.shards)
    n, b = len(shards), args.batch_size
    gb = shards.nbytes("image") / 1024**3
    per_shard = shards.counts
    print(f"{n:,} patches in {len(shards.paths)} shards, {gb:.2f} GiB total, "
          f"{shards.native_size} px native")
    print(f"shard sizes: min {per_shard.min():,} max {per_shard.max():,}")
    print(f"batch of {b} stamps = {b * shards.native_size ** 2 * 4 / 1024**2:.0f} MiB\n")

    rng = np.random.default_rng(args.seed)
    reps = min(args.repeats, len(shards.paths))
    picks = rng.choice(len(shards.paths), size=reps, replace=False)

    for rnd in range(1, args.rounds + 1):
        tag = "cold-ish" if rnd == 1 else f"round {rnd} (page cache warm)"
        print(f"--- {tag} " + "-" * (58 - len(tag)))

        opens = [_time(lambda p: h5py.File(p, "r").close(),
                       shards.paths[i])[0] for i in picks]
        print(f"  open+close one shard      {_fmt(np.median(opens))}"
              f"   (x{len(shards.paths)} would be {_fmt(np.median(opens) * len(shards.paths))})")

        glob = [_time(shards.gather, np.sort(rng.choice(n, b, replace=False)),
                      "image")[0] for _ in range(reps)]
        print(f"  scattered, all shards     {_fmt(np.median(glob))}"
              f"   <- what the streaming path does now")

        local = []
        for i in picks:
            lo, hi = shards.offsets[i], shards.offsets[i + 1]
            k = min(b, hi - lo)
            idx = np.sort(rng.choice(np.arange(lo, hi), k, replace=False))
            local.append(_time(shards.gather, idx, "image")[0])
        print(f"  scattered, one shard      {_fmt(np.median(local))}"
              f"   <- 'random shard, random {b} from it'")

        seq = []
        for i in picks:
            def read(p=shards.paths[i]):
                with h5py.File(p, "r") as f:
                    return f["image"][:]
            seq.append(_time(read)[0])
        med_seq = np.median(seq)
        rows = np.median(per_shard)
        print(f"  whole shard, sequential   {_fmt(med_seq)}"
              f"   ({rows:,.0f} rows, {med_seq / rows * b * 1e3:.0f} ms per {b} equivalent)")
        print()

    # CPU floor: pooling + transform, with no I/O at all.
    config = Config()
    config.patch.native_size = shards.native_size
    small = min(len(shards), 512)
    sample, _ = pool_shards(shards, config, n=small)
    config.transform.softening = estimate_softening(
        sample, config.transform.softening_sigma)
    cached = PatchDataset.from_shards(
        shards, config, LogFluxTransform.from_config(config.transform),
        in_memory=False)
    cached._native = shards.gather(np.arange(min(len(shards), 4 * b)), "image")
    idx = rng.integers(0, len(cached._native), b)
    r = np.random.default_rng(0)
    cpu = [_time(cached.make_batch, idx, r)[0] for _ in range(5)]
    print(f"--- CPU floor " + "-" * 53)
    print(f"  make_batch from RAM       {_fmt(np.median(cpu))}"
          f"   (pool + transform + augment, no I/O)")
    if args.pool_batches:
        pooled = PatchDataset.from_shards(
            shards, config, LogFluxTransform.from_config(config.transform),
            max_in_memory_gb=1e-9, pool_shards=args.pool_shards,
            pool_refill_every=args.pool_refill_every, prefetch_depth=3)
        print(f"\n--- the real loader, pool mode " + "-" * 36)
        print(f"  {pooled.storage_note(1e-9)}")
        t0 = time.perf_counter()
        first = _time(lambda: next(pooled.batches(b, seed=0)))[0]
        fill = time.perf_counter() - t0
        it = pooled.batches(b, seed=0)
        next(it)
        per = []
        for _ in range(args.pool_batches):
            per.append(_time(next, it)[0])
        pooled.close()
        per = np.asarray(per)
        print(f"  first batch (includes the pool's initial fill) {_fmt(fill)}")
        print(f"  median batch              {_fmt(np.median(per))}"
              f"   <- what a training step waits for")
        print(f"  slowest of {len(per):>4}           {_fmt(per.max())}"
              f"   (a refill that the prefetch queue did not cover)")
        print(f"  over the {args.pool_refill_every}-batch refill period, "
              f"{100 * (per > 2 * np.median(per)).mean():.0f}% of batches were "
              f"more than twice the median")

    print("\nA resident pool only pays off if 'whole shard, sequential' divided")
    print("by the batches you serve from it beats 'scattered, all shards'.")
    print("'Random shard per batch' only pays off if 'scattered, one shard' is")
    print("much cheaper than 'scattered, all shards' -- i.e. if opens dominate.")
    print("The 'real loader' block is the one that settles it: it is the code")
    print("training runs, prefetch thread and all.")


if __name__ == "__main__":
    main()
