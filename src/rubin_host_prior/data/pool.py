"""Getting batches to the GPU without it waiting: a shard pool and a prefetcher.

Two measurements shape this, both taken on a NERSC GPU node against a 30 GiB
campaign of 41 shards, reading 128 stamps:

    scattered across all shards   6.03 s cold / 3.18 s warm   47.1 ms per row
    scattered inside one shard    1.08 s cold / 592 ms warm    8.4 ms per row
    one whole shard, sequential   2.14 s for 1,024 rows        2.09 ms per row
    make_batch from RAM                          154 ms        1.20 ms per row

So random access across the set costs **22x** what reading a shard in one pass
does, and it is not the file opens -- 41 of those are 86 ms, under 1.5% of the
6.03 s.  It is locality: a parallel filesystem's readahead, client cache and
striping all work within a file and not across forty of them.  (On a local SSD
the two are indistinguishable, because there is no seek to pay for.  That is why
this is measured on the machine that trains.)

``ShardPool`` therefore never reads a row on its own.  It keeps ``n_resident``
whole shards in RAM, draws every batch uniformly from all of them, and replaces
one shard at a time from a shuffled cycle -- each replacement a single
sequential read, issued by a background thread that stays one shard ahead.  The
cost per step falls from 3-6 s to ``2.14 s / refill_every``: 33 ms at 64, which
is under the 154 ms the pooling and transform take anyway.

``prefetch`` then hides that 154 ms too.  Batch construction is numpy and h5py,
both of which drop the GIL over the expensive parts, so a producer thread really
does overlap the JAX step rather than merely interleaving with it.

**Uniform shards matter here.**  A slot holds one shard for ``n_resident *
refill_every`` batches whatever its length, so rows in a short shard are drawn
more often than rows in a full one.  With shards of equal length the scheme is
exactly uniform; a split campaign's partial tails make it approximately so.
``scripts/merge_parts.py --repack`` is what makes them equal.
"""

from __future__ import annotations

import queue
import threading
from typing import Iterator

import h5py
import numpy as np


def prefetch(source: Iterator, depth: int = 3) -> Iterator:
    """Run ``source`` in a background thread, yielding through a bounded queue.

    ``depth`` is how many batches may sit ready.  It buys two different things:
    steady-state overlap needs only 1-2, but a pool refill is a one-off spike
    (2.14 s against a 234 ms step), so a deeper queue is what stops that
    becoming a stall.  Each slot is a batch -- 8 MiB at the defaults -- so depth
    is cheap.

    Exceptions from the thread are re-raised in the consumer, which matters
    because the alternative is a training run that hangs on an empty queue with
    the traceback in a thread nobody is watching.
    """
    if depth < 1:
        raise ValueError(f"prefetch depth must be at least 1, got {depth}")
    q: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()
    DONE = object()

    def worker():
        try:
            for item in source:
                while not stop.is_set():
                    try:
                        q.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
        except BaseException as exc:  # noqa: BLE001 -- re-raised in the consumer
            try:
                q.put(exc, timeout=1.0)
            except queue.Full:
                pass
        else:
            try:
                q.put(DONE, timeout=1.0)
            except queue.Full:
                pass

    thread = threading.Thread(target=worker, daemon=True, name="batch-prefetch")
    thread.start()
    try:
        while True:
            item = q.get()
            if item is DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()


class ShardPool:
    """``n_resident`` whole shards in RAM, refilled one at a time, sequentially.

    ``draw`` is **not** thread-safe: it mutates the residency, and it is meant to
    be called from one producer (the ``prefetch`` thread).  The background reader
    it owns is a second thread, and the only thing they share is a one-slot
    queue.
    """

    def __init__(
        self,
        shards,
        n_resident: int = 8,
        refill_every: int = 64,
        seed: int = 0,
        key: str = "image",
    ):
        n_files = len(shards.paths)
        if n_resident < 1 or refill_every < 1:
            raise ValueError(
                f"n_resident and refill_every must be positive, got "
                f"{n_resident} and {refill_every}"
            )
        self.shards = shards
        self.key = key
        #: Clamped rather than refused: a set with fewer shards than the pool
        #: wants is simply held whole, which is the right answer for it.
        self.n_resident = min(n_resident, n_files)
        self.refill_every = refill_every
        self._rng = np.random.default_rng(seed)
        self._cycle: list[int] = []
        self._slots: list[np.ndarray] = []
        self._slot_shard: list[int] = []
        self._next_slot = 0
        self._since_refill = 0
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._inbox: queue.Queue = queue.Queue(maxsize=1)

        # The first fill is synchronous: there is nothing to overlap it with,
        # and a pool that started empty would serve its first batches from a
        # fraction of the shards it is meant to mix.
        for _ in range(self.n_resident):
            i = self._next_index()
            self._slots.append(self._read(i))
            self._slot_shard.append(i)

        self._reader = threading.Thread(
            target=self._read_ahead, daemon=True, name="shard-reader")
        self._reader.start()

    # -- residency --------------------------------------------------------

    def _next_index(self) -> int:
        """Next shard from a shuffled cycle, reshuffled each pass.

        A cycle rather than an independent draw each time: with replacement a
        shard can go unread for far longer than the cycle length, and over a run
        the coverage is uneven for no benefit.
        """
        if not self._cycle:
            self._cycle = list(self._rng.permutation(len(self.shards.paths)))
        return int(self._cycle.pop())

    def _read(self, i: int) -> np.ndarray:
        """One whole shard, in a single sequential pass."""
        with h5py.File(self.shards.paths[i], "r") as f:
            d = f[self.key]
            out = np.empty(d.shape, dtype=d.dtype)
            d.read_direct(out)
        return out

    def _read_ahead(self) -> None:
        """Keep exactly one shard staged.  The queue's size of 1 is the throttle."""
        try:
            while not self._stop.is_set():
                i = self._next_index()
                arr = self._read(i)
                while not self._stop.is_set():
                    try:
                        self._inbox.put((i, arr), timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # noqa: BLE001 -- surfaced by `draw`
            self._error = exc

    def _maybe_refill(self) -> None:
        if self._since_refill < self.refill_every:
            return
        try:
            i, arr = self._inbox.get_nowait()
        except queue.Empty:
            return  # reader still working; try again next batch rather than block
        self._slots[self._next_slot] = arr
        self._slot_shard[self._next_slot] = i
        self._next_slot = (self._next_slot + 1) % self.n_resident
        self._since_refill = 0

    # -- drawing ----------------------------------------------------------

    @property
    def resident_rows(self) -> int:
        return sum(len(s) for s in self._slots)

    @property
    def resident_bytes(self) -> int:
        return sum(s.nbytes for s in self._slots)

    def draw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """``n`` stamps drawn uniformly from every resident shard at once."""
        if self._error is not None:
            raise RuntimeError(
                f"the pool's shard reader failed: {self._error!r}"
            ) from self._error
        total = self.resident_rows
        if total < n:
            raise ValueError(
                f"the pool holds {total} rows but a batch of {n} was asked for. "
                f"Raise n_resident (now {self.n_resident} shards) or lower the "
                f"batch size."
            )
        sizes = np.array([len(s) for s in self._slots])
        offsets = np.concatenate([[0], np.cumsum(sizes)])
        flat = rng.choice(total, size=n, replace=False)
        slot_of = np.searchsorted(offsets, flat, side="right") - 1
        out = np.empty((n,) + self._slots[0].shape[1:], dtype=self._slots[0].dtype)
        for s in np.unique(slot_of):
            sel = np.where(slot_of == s)[0]
            out[sel] = self._slots[s][flat[sel] - offsets[s]]
        self._since_refill += 1
        self._maybe_refill()
        return out

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        self._stop.set()
        try:
            self._inbox.get_nowait()
        except queue.Empty:
            pass
        self._reader.join(timeout=2.0)

    def __enter__(self) -> "ShardPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def note(self) -> str:
        gb = self.resident_bytes / 1024**3
        return (
            f"pool of {self.n_resident} resident shards ({gb:.2f} GiB, "
            f"{self.resident_rows:,} rows), one replaced every "
            f"{self.refill_every} batches by a background sequential read"
        )
